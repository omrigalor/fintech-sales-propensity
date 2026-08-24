"""
End-to-end scoring pipeline → outputs/ranked_companies.csv + artifacts.

Priority = blend(FIT, WIN, VALUE):
  WIN   = calibrated win-propensity, an ML model trained on the 100 won/lost.
  FIT   = transparent ICP-fit score from firmographics + LLM website signals.
  VALUE = expected account size (the middle-market multiplier that rescues the whales).
The three sub-scores (as percentiles) are stored so the app can re-blend live with a dial.

Run:  python src/pipeline.py     (run AFTER src/enrich.py has produced enrichment.csv)
"""
from __future__ import annotations
import ast, json, warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import common, features
warnings.filterwarnings("ignore")
RNG = 7

# ---------- helpers ----------
def pct(x):  # robust 0-1 percentile rank
    s = pd.Series(x).rank(pct=True)
    return s.fillna(s.median()).values

def norm01(x):
    x = pd.Series(x).astype(float)
    lo, hi = x.quantile(0.02), x.quantile(0.98)
    return ((x - lo) / (hi - lo + 1e-9)).clip(0, 1).fillna(0).values


# ---------- WIN-propensity model ----------
def build_estimator(kind):
    if kind == "logistic":
        clf = LogisticRegression(penalty="l2", C=0.5, max_iter=2000, class_weight="balanced")
    else:
        clf = GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05,
                                         subsample=0.9, random_state=RNG)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", clf)])

def train_win(Xtr, ytr, feats):
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    results = {}
    for kind in ["logistic", "gb"]:
        est = build_estimator(kind)
        oof = cross_val_predict(est, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        # fold-wise AUC for an honest confidence interval (small N!)
        aucs = []
        for _, te in cv.split(Xtr, ytr):
            try: aucs.append(roc_auc_score(ytr[te], oof[te]))
            except Exception: pass
        results[kind] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                         "brier": float(brier_score_loss(ytr, oof)), "oof": oof}
    best = max(results, key=lambda k: results[k]["auc_mean"])
    # calibrate the winner (isotonic needs data; use sigmoid/Platt for small N safety)
    final = CalibratedClassifierCV(build_estimator(best), method="sigmoid", cv=cv)
    final.fit(Xtr, ytr)
    # a plain logistic for transparent per-company reasons (coef * z-score)
    lin = build_estimator("logistic").fit(Xtr, ytr)
    coef = lin.named_steps["clf"].coef_[0]
    scaler = lin.named_steps["sc"]; imp = lin.named_steps["imp"]
    return final, results, best, {"coef": coef, "scaler": scaler, "imp": imp, "feats": feats}


def win_reasons(row_feats, lin, k=3):
    """Transparent 'why' for propensity: top signed contributions coef*z-score."""
    x = lin["imp"].transform(pd.DataFrame([row_feats])[lin["feats"]])
    z = lin["scaler"].transform(x)[0]
    contrib = z * lin["coef"]
    order = np.argsort(-np.abs(contrib))[:k]
    nice = {"emp_log": "SMB-scale (win pattern)", "rev_log": "revenue fit", "msp_minus_acct": "MSP-type (vs accounting)",
            "msp_probability": "MSP profile", "acct_probability": "accounting profile",
            "finance_ops_complexity": "finance-ops complexity", "is_b2b_int": "B2B services",
            "recurring_int": "recurring revenue", "icp_fit_score": "ICP signals",
            "b2b_confidence": "B2B confidence", "tech_stack_confidence": "tech-stack signals",
            "size_ord": "company size", "recent_press_count": "market activity"}
    out = []
    for i in order:
        f = lin["feats"][i]
        label = nice.get(f, f)
        out.append(("+" if contrib[i] > 0 else "−") + label)
    return out


# ---------- too-big gate: true enterprise is OUT of Meridian's mid-market ICP ----------
def too_big(d):
    rev = pd.to_numeric(d.get("annual_revenue"), errors="coerce").fillna(0)
    emp = pd.to_numeric(d.get("employee_count"), errors="coerce").fillna(0)
    sig = d.get("llm_size_signal", pd.Series("", index=d.index)).astype(str).str.lower()
    return ((rev > 1e9) | (emp > 3000) | (sig == "enterprise")).astype(float)


# ---------- product need: Meridian wins if a company needs EITHER product ----------
def product_need(d):
    ar = pd.to_numeric(d.get("llm_ar_need_score"), errors="coerce").fillna(0) / 100
    ap = pd.to_numeric(d.get("llm_ap_need_score"), errors="coerce").fillna(0) / 100
    if "scr_has_payment_portal" in d:   # live-scraped AR structural signal
        ar = np.minimum(1.0, ar + 0.15 * d["scr_has_payment_portal"].fillna(False).astype(float))
    if "scr_has_vendor_page" in d:      # live-scraped AP structural signal
        ap = np.minimum(1.0, ap + 0.10 * d["scr_has_vendor_page"].fillna(False).astype(float))
    return ar, ap

def recommended_product(d):
    ar, ap = product_need(d)
    both = (ar >= 0.6) & (ap >= 0.6)
    return np.where(both, "AR + AP", np.where(ap > ar + 0.10, "AP (beta)", "AR"))


# ---------- FIT (transparent ICP score, driven by product need) ----------
def fit_score(d):
    b2b = d["is_b2b_int"].fillna(0)
    ar, ap = product_need(d)
    need = np.maximum(ar, ap)                 # fit on the stronger of the two products
    rec = d["recurring_int"].fillna(0)
    raw = 0.50*need + 0.20*b2b + 0.15*norm01(d.get("msp_minus_acct")) + 0.15*rec
    raw = raw * (0.35 + 0.65*b2b)            # heavy penalty if not a B2B service business
    raw = raw * (1 - 0.45*too_big(d))         # penalty if too big for Meridian's motion
    return (100 * pct(raw)).round(1), pct(raw)


# ---------- VALUE (middle-market sweet spot, NOT "bigger = always better") ----------
def value_score(d):
    # the LLM mid-market score is designed to peak in the mid-market and drop for tiny AND Fortune-500
    mm = pd.to_numeric(d.get("llm_midmarket_score"), errors="coerce").fillna(30) / 100
    rev = norm01(d.get("rev_log")); emp = norm01(d.get("emp_log"))
    raw = 0.55*mm + 0.25*rev + 0.20*emp
    # third-party evidence (external.py): PE-backed / multi-entity = the multi-entity holdco prize
    def _yes(col):
        return (d[col].astype(str).str.lower() == "yes").astype(float) if col in d else np.zeros(len(d))
    raw = np.minimum(1.0, raw + 0.10 * _yes("ext_pe_backed") + 0.10 * _yes("ext_multi_entity"))
    raw = raw * (1 - 0.6*too_big(d))          # a global mega-corp is not a valuable *Meridian* target
    return (100 * pct(raw)).round(1), pct(raw)


# ---------- "why now" trigger score: buying signals, not just fit (fit x TIMING) ----------
def _count_list(d, col):
    if col not in d: return np.zeros(len(d))
    def n(x):
        try:
            v = ast.literal_eval(x) if isinstance(x, str) else x
            return len(v) if isinstance(v, (list, tuple)) else (1 if v else 0)
        except Exception:
            return 0
    return d[col].map(n).values.astype(float)

def trigger_score(d):
    """A separate TIMING lens: how many buying signals are firing right now. Kept OUT of the
    core fit/win/value blend on purpose — it answers 'why now', not 'is it a fit'."""
    growth = np.minimum(_count_list(d, "llm_growth_signals"), 3)
    pain = np.minimum(_count_list(d, "llm_pain_signals"), 3)
    hiring = d["scr_hiring"].fillna(False).astype(float).values if "scr_hiring" in d else np.zeros(len(d))
    acq = (d.get("ext_acquisitions", pd.Series("", index=d.index)).astype(str) == "yes").astype(float).values
    comp = d["scr_competitor_tool"].fillna(False).astype(float).values if "scr_competitor_tool" in d else np.zeros(len(d))
    raw = 1.5*growth + 1.0*pain + 1.0*hiring + 2.0*acq + 1.5*comp
    score = np.minimum(100, raw * 12).round(0)
    reasons = []
    for i in range(len(d)):
        r = []
        if growth[i] > 0: r.append("growth/expansion")
        if hiring[i] > 0: r.append("actively hiring")
        if acq[i] > 0: r.append("acquisition activity")
        if comp[i] > 0: r.append("on a competitor → displacement")
        if pain[i] > 0: r.append("cash-flow / AR-AP pain")
        reasons.append(" · ".join(r))
    return score, reasons


# ---------- deterministic, GTM-actionable segments (stable across runs — no KMeans label churn) ----------
def _size_tier(e):
    if pd.isna(e): return "size n/a"
    if e >= 1000: return "1000+ emp"
    if e >= 200: return "200–999 emp"
    if e >= 50: return "50–199 emp"
    if e >= 10: return "10–49 emp"
    return "<10 emp"

def segment(d, feats=None, k=None):
    """Rule-based segments a GTM rep can act on, from vertical, size tier, product need,
    and enrichment confidence. Deterministic: same input -> same labels (no cluster churn)."""
    emp  = pd.to_numeric(d.get("employee_count"), errors="coerce")
    msp  = pd.to_numeric(d.get("msp_probability"), errors="coerce").fillna(0)
    acct = pd.to_numeric(d.get("acct_probability"), errors="coerce").fillna(0)
    conf = pd.to_numeric(d.get("llm_extraction_confidence"), errors="coerce").fillna(0)
    vert = d.get("llm_vertical", pd.Series("", index=d.index)).astype(str).str.lower()
    ar, ap = product_need(d); tb = too_big(d)
    # The LLM-read vertical is authoritative. The msp/acct probabilities are on a 0-100 scale
    # and are missing for most rows, so they only act as a fallback when the vertical is blank
    # or generic. Anything that doesn't clearly read as MSP/IT or accounting defaults to a
    # generic B2B service business — never MSP.
    MSP_KEYS = ("msp", "it serv", "it service", "managed it", "managed serv", "technology serv",
                "it support", "it solution", "network serv", "cybersecurity")
    ACCT_KEYS = ("account", "bookkeep", "cpa", "tax", "audit", "advisory")
    GENERIC_VERT = ("", "nan", "none", "other", "unknown", "n/a", "b2b services")
    names = []
    for i in d.index:
        if conf[i] < 0.4:
            names.append("Low-confidence enrichment queue"); continue
        if tb[i] >= 1:
            names.append("Strategic enterprise service business"); continue
        st = _size_tier(emp[i])
        prod = "AR+AP" if (ar[i] >= 0.6 and ap[i] >= 0.6) else ("AP-led" if ap[i] > ar[i] + 0.1 else "AR-led")
        v = vert[i].strip()
        has_vert = v not in GENERIC_VERT
        msp_i, acct_i = msp[i], acct[i]
        msp_vert = any(kw in v for kw in MSP_KEYS)
        acct_vert = any(kw in v for kw in ACCT_KEYS)
        if msp_vert or (not has_vert and msp_i >= 50 and msp_i >= acct_i):
            tier = "Core MSP" if (pd.notna(emp[i]) and emp[i] < 50) else "Scaling MSP"
            names.append(f"{tier} · {st} · {prod}")
        elif acct_vert or (not has_vert and acct_i >= 50 and acct_i > msp_i):
            names.append(f"Accounting firm · {st} · {prod}")
        else:
            names.append(f"B2B service business · {st} · {prod}")
    d = d.copy(); d["segment_name"] = names
    codes, uniques = pd.factorize(d["segment_name"])
    d["segment"] = codes
    return d, {i: lbl for i, lbl in enumerate(uniques)}


# ---------- matched-association robustness check (NOT causal — observational, small-N) ----------
def matched_association(Xtr, ytr, feats):
    """For each feature: split high/low by median, match each unit to its nearest twin
    in the OTHER group on all OTHER features, compare win-rates. A stable positive gap is
    ASSOCIATIONAL evidence (suggestive) — a real causal estimate needs the randomized
    GTM roll-out described in the recommendation, not this."""
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(Xtr))
    Z = pd.DataFrame(Z, columns=feats)
    y = np.array(ytr); out = {}
    for f in feats:
        hi = Z[f] >= Z[f].median()
        A, B = Z[hi], Z[~hi]
        if len(A) < 5 or len(B) < 5: continue
        others = [c for c in feats if c != f]
        nn = NearestNeighbors(n_neighbors=1).fit(B[others].values)
        _, idx = nn.kneighbors(A[others].values)
        eff = y[hi.values].mean() - y[~hi.values][idx.flatten()].mean()
        out[f] = round(float(eff), 3)
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


# ---------- validation suite: baselines + repeated-CV instability (honest small-N story) ----------
def validation_suite(Xtr, ytr, feats, best):
    from sklearn.model_selection import RepeatedStratifiedKFold
    Xdf = pd.DataFrame(np.asarray(Xtr), columns=feats)
    y = np.array(ytr); base = float(y.mean())
    const_brier = float(np.mean((y - base) ** 2))
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RNG)
    aucs = []
    for tr, te in rcv.split(Xdf, y):
        est = build_estimator(best).fit(Xdf.iloc[tr], y[tr])
        p = est.predict_proba(Xdf.iloc[te])[:, 1]
        try: aucs.append(roc_auc_score(y[te], p))
        except Exception: pass
    aucs = np.array(aucs) if aucs else np.array([np.nan])
    size_feats = [f for f in feats if f in ("emp_log", "rev_log", "size_ord")]
    size_auc = None
    if size_feats:
        sa = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RNG).split(Xdf, y):
            est = build_estimator("logistic").fit(Xdf[size_feats].iloc[tr], y[tr])
            p = est.predict_proba(Xdf[size_feats].iloc[te])[:, 1]
            try: sa.append(roc_auc_score(y[te], p))
            except Exception: pass
        size_auc = round(float(np.mean(sa)), 3) if sa else None
    return {
        "base_rate": round(base, 3),
        "constant_brier": round(const_brier, 3),
        "repeated_cv": {"n": int(len(aucs)), "auc_mean": round(float(np.nanmean(aucs)), 3),
                        "auc_std": round(float(np.nanstd(aucs)), 3),
                        "auc_min": round(float(np.nanmin(aucs)), 3),
                        "auc_p10": round(float(np.nanpercentile(aucs, 10)), 3)},
        "size_only_baseline_auc": size_auc,
        "note": "50/50 label balance != true base rate, so scores are RELATIVE win propensity, not absolute probabilities.",
    }


# ---------- rank stability: how often does each account stay in the top cohort? ----------
def rank_stability(Xtr, ytr, Xsc, fit_pct, val_pct, weights, B=40, seed=RNG):
    """Bootstrap the 100 labels B times, refit the win model, re-score & re-rank all
    companies, and record how often each lands in the top-50 / top-100. This exposes
    which high ranks are stable vs. lucky — the honest answer to a small, noisy sample."""
    wf, ww, wv = weights
    Xtr_a, ytr_a, Xsc_a = np.asarray(Xtr), np.asarray(ytr), np.asarray(Xsc)
    fitp, valp = np.asarray(fit_pct), np.asarray(val_pct)
    rng = np.random.default_rng(seed); n = len(ytr_a)
    incl50 = np.zeros(len(Xsc_a)); incl100 = np.zeros(len(Xsc_a))
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(ytr_a[idx])) < 2:      # need both classes
            continue
        est = build_estimator("gb").fit(Xtr_a[idx], ytr_a[idx])
        wpp = pd.Series(est.predict_proba(Xsc_a)[:, 1]).rank(pct=True).values
        pr = np.power(fitp, wf) * np.power(wpp, ww) * np.power(valp, wv)
        order = np.argsort(-pr)
        incl50[order[:50]] += 1; incl100[order[:100]] += 1
    return np.round(incl50 / B, 3), np.round(incl100 / B, 3)


# ---------- conformal prediction: distribution-free selective abstention ----------
def conformal_uncertain(oof, ytr, test_p_win, alpha=0.1):
    """Split-conformal (LAC / score method; Vovk; Angelopoulos & Bates 2021).
    Nonconformity on the 100 labeled out-of-fold preds = 1 - p(true class). A test account is
    'model-uncertain' (abstain) when BOTH win and lose fall inside the conformal set at coverage
    1-alpha — i.e. the model can't confidently commit. Gives a coverage GUARANTEE, unlike a
    hand-set threshold. (Small-sample caveat: n=100, so coverage is marginal/loose.)
    Returns: uncertain flag per test account, empirical coverage, qhat."""
    y = np.asarray(ytr); p = np.asarray(oof)
    s = np.where(y == 1, 1 - p, p)                     # nonconformity = 1 - p(true class)
    n = len(s); qlevel = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = float(np.quantile(s, qlevel, method="higher"))
    pw = np.asarray(test_p_win)
    win_in = (1 - pw) <= qhat                           # win plausible
    lose_in = pw <= qhat                                # lose plausible
    uncertain = ((win_in.astype(int) + lose_in.astype(int)) >= 2).astype(int)
    covered = float((((y == 1) & (1 - p <= qhat)) | ((y == 0) & (p <= qhat))).mean())
    return uncertain, covered, qhat


# ---------- main ----------
def main():
    labeled = common.load_labeled()
    enriched = common.load_enriched()
    n_enr = enriched["llm_extraction_confidence"].notna().sum() if "llm_extraction_confidence" in enriched else 0
    print(f"labeled={len(labeled)}  companies={len(enriched)}  LLM-enriched={n_enr}")

    Xtr, ytr, d, feats = features.build(labeled, enriched)

    win_model, cv, best, lin = train_win(Xtr, ytr, feats)
    print(f"WIN model: {best}  CV-AUC={cv[best]['auc_mean']:.3f}±{cv[best]['auc_std']:.3f}  Brier={cv[best]['brier']:.3f}")

    Xsc = features.engineer(enriched)[feats].astype(float)
    d["win_propensity"] = win_model.predict_proba(Xsc)[:, 1]

    # conformal prediction -> a coverage-guaranteed MODEL-uncertainty flag (separate from data quality)
    conf_unc, conf_cov, conf_q = conformal_uncertain(cv[best]["oof"], ytr, d["win_propensity"], alpha=0.1)
    d["model_uncertain"] = conf_unc

    d["fit_score"], fit_pct = fit_score(d)
    d["value_score"], val_pct = value_score(d)
    d["recommended_product"] = recommended_product(d)
    _ar, _ap = product_need(d)
    d["ar_need"] = (_ar * 100).round(0); d["ap_need"] = (_ap * 100).round(0)
    d["trigger_score"], d["trigger_reasons"] = trigger_score(d)   # "why now" timing lens
    win_pct = pct(d["win_propensity"])
    d["fit_pct"], d["win_pct"], d["value_pct"] = fit_pct, win_pct, val_pct

    # default blend (app re-blends live with the Fit<->Value dial)
    # Fit and Value drive the ranking; Win is a lighter tie-breaker because the win model is
    # honestly weak on 100 labels (CV-AUC ~0.64). See README "Key decisions & honesty".
    wf, ww, wv = 0.45, 0.20, 0.35
    d["priority_score"] = (100 * (np.power(fit_pct, wf) * np.power(win_pct, ww) *
                                  np.power(val_pct, wv))).round(1)

    # rank stability (bootstrap the labels, re-rank, count top-cohort inclusion)
    incl50, incl100 = rank_stability(Xtr, ytr, Xsc, fit_pct, val_pct, (wf, ww, wv))
    d["top50_incl"], d["top100_incl"] = incl50, incl100

    d, seg_names = segment(d, feats)

    # selective prediction: flag thin/low-confidence records for human review
    conf = pd.to_numeric(d.get("llm_extraction_confidence"), errors="coerce").fillna(0)
    thin = enriched[["employee_count", "annual_revenue"]].isna().mean(axis=1) if set(["employee_count","annual_revenue"]).issubset(enriched) else 0
    d["review_flag"] = ((conf < 0.4) | (pd.Series(thin) > 0.5)).astype(int)

    # transparent reasons per company (win contributions + fit/value + LLM one-liner)
    reasons = []
    for _, r in d.iterrows():
        wr = win_reasons({f: r.get(f) for f in feats}, lin, k=2)
        extra = []
        if pd.to_numeric(r.get("llm_finance_ops_complexity"), errors="coerce") and float(r.get("llm_finance_ops_complexity") or 0) >= 55:
            extra.append("+high finance-ops complexity")
        if r.get("value_score", 0) >= 70:
            extra.append("+mid-market value")
        if str(r.get("llm_ap_competitor_mentions", "[]")) not in ("[]", "nan", ""):
            extra.append("−already uses an AP competitor")
        reasons.append("; ".join(wr + extra[:2]))
    d["top_reasons"] = reasons
    d["why"] = d.apply(lambda r: f"{r.get('segment_name','')}. " +
                       (str(r.get('llm_one_line',''))[:120]) + f" | {r['top_reasons']}", axis=1)

    d = d.sort_values("priority_score", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)

    cols = ["rank", "company_id", "source_company_id", "company_name", "domain", "priority_score",
            "fit_score", "win_propensity", "value_score", "recommended_product",
            "ar_need", "ap_need", "trigger_score", "trigger_reasons", "segment_name",
            "possible_duplicate", "entity_cluster_id", "canonical_record", "duplicate_reason",
            "review_flag", "critic_disagreement", "model_uncertain", "llm_one_line", "top_reasons", "why",
            "industry", "employee_count", "annual_revenue",
            "fit_pct", "win_pct", "value_pct", "top50_incl", "top100_incl",
            "llm_finance_ops_complexity", "llm_midmarket_score", "llm_vertical",
            "llm_recurring_revenue_model", "llm_pain_signals", "llm_ap_competitor_mentions",
            "scr_scrape_ok", "scr_num_tools", "scr_tools_detected",
            "scr_has_payment_portal", "scr_has_vendor_page", "scr_competitor_tool",
            "ext_pe_backed", "ext_multi_entity", "ext_acquisitions", "ext_employee_estimate",
            "ext_notable", "ext_sources", "vr_verify_confidence", "vr_agree_with_prior",
            "sc_consistency_confidence"]
    cols = [c for c in cols if c in d.columns]
    out = d[cols].copy()
    out["win_propensity"] = (out["win_propensity"] * 100).round(1)
    out.to_csv(common.ROOT / "ranked_output.csv", index=False)   # the single deliverable (top level)

    # artifacts for app + write-up
    assoc = matched_association(Xtr, ytr, feats)
    validation = validation_suite(Xtr, ytr, feats, best)
    art = {
        "cv": {best: {k: v for k, v in cv[best].items() if k != "oof"}},
        "chosen_model": best,
        "target": "relative win propensity (labels are 50 won / 50 lost — NOT a natural base rate)",
        "validation": validation,
        "explanation_method": "per-company reasons are coef×z from an interpretable LOGISTIC SURROGATE; "
                              "the scoring model is calibrated gradient-boosting (label surrogate reasons as approximate).",
        "feature_coefficients": dict(zip(feats, [round(float(c), 3) for c in lin["coef"]])),
        "matched_association_effects": assoc,
        "segments": seg_names,
        "blend_weights_default": {"fit": wf, "win": ww, "value": wv},
        "n_companies": int(len(d)), "n_review_flag": int(d["review_flag"].sum()),
        "n_possible_duplicate": int(d["possible_duplicate"].sum()) if "possible_duplicate" in d else 0,
        "conformal": {"alpha": 0.1, "target_coverage": 0.9, "empirical_coverage": round(conf_cov, 3),
                      "n_model_uncertain": int(d["model_uncertain"].sum()),
                      "note": "distribution-free selective prediction (Angelopoulos & Bates 2021); coverage is marginal at n=100."},
        "critic_agreement_rate": round(float(1 - d["critic_disagreement"].mean()), 2) if "critic_disagreement" in d else None,
        "n_pe_backed": int((d.get("ext_pe_backed", pd.Series(dtype=str)).astype(str) == "yes").sum()) if "ext_pe_backed" in d else 0,
        "n_multi_entity": int((d.get("ext_multi_entity", pd.Series(dtype=str)).astype(str) == "yes").sum()) if "ext_multi_entity" in d else 0,
        "rank_stability": {
            "top50_median_inclusion": float(d.sort_values("priority_score", ascending=False)
                                            .head(50)["top50_incl"].median()),
            "top100_median_inclusion": float(d.sort_values("priority_score", ascending=False)
                                             .head(100)["top100_incl"].median()),
            "note": "share of bootstrap re-rankings in which a current top-50/100 account stays in that cohort.",
        },
    }
    (common.ART / "model_report.json").write_text(json.dumps(art, indent=2))
    import joblib
    joblib.dump({"win_model": win_model, "lin": lin, "feats": feats}, common.ART / "models.joblib")
    print(f"\nWrote ranked_output.csv ({len(out)} rows) + outputs/artifacts/model_report.json")
    print(f"Validation: repeated-CV AUC={validation['repeated_cv']['auc_mean']}±{validation['repeated_cv']['auc_std']} "
          f"(min {validation['repeated_cv']['auc_min']}), size-only baseline={validation['size_only_baseline_auc']}")
    print("Top 5:\n", out.head(5)[["rank","company_name","priority_score","fit_score","win_propensity","value_score","segment_name"]].to_string(index=False))
    print("\nMatched-association effects (suggestive, NOT causal):", assoc)

if __name__ == "__main__":
    main()
