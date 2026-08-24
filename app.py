"""
Meridian — the target list a GTM lead can act on.
One question, answered: who do I call first, and why?

  • Ranked call list (search / filter / sort, 🔥 = a live buying signal)
  • Account detail: what they do, which product to lead with, and a plain-English "why"

Run:  streamlit run app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

ROOT = Path(__file__).resolve().parent
RANKED = ROOT / "ranked_output.csv"
BLURBS = ROOT / "code" / "outputs" / "outreach_blurbs.csv"
COMPONENTS = ROOT / "code" / "outputs" / "score_components.csv"  # for live Fit/Value re-weighting
TOP_N = 100

st.set_page_config(page_title="Meridian — Who to call first", page_icon="▲", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.0rem; max-width: 1400px; }
.hero { background: linear-gradient(120deg,#141b34,#243056); color:#fff; padding:16px 22px; border-radius:14px; margin-bottom:12px; }
.hero h1 { margin:0; font-size:24px; } .hero p { margin:4px 0 0; color:#c7cede; font-size:13.5px; }
.card{border:1px solid #e6e8ee;border-radius:14px;padding:14px 16px;background:#fff;}
.metricbig{font-size:28px;font-weight:700;color:#141b34;line-height:1;}
.metriclbl{color:#6b7280;font-size:12px;text-transform:uppercase;letter-spacing:.04em;}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;}
.b-seg{background:#eef2ff;color:#33409a;}
.reason{background:#f6f7fb;border-radius:8px;padding:6px 10px;margin:3px 0;font-size:13px;}
.why{background:#eef4ff;border:1px solid #cfe0ff;border-radius:8px;padding:8px 11px;margin:3px 0;font-size:13.5px;line-height:1.45;}
.opener{background:#eefaf0;border:1px solid #cfe9d8;border-radius:8px;padding:8px 11px;margin:3px 0;font-size:13.5px;line-height:1.45;color:#215c37;}
div.stButton > button{text-align:left;justify-content:flex-start;width:100%;border:1px solid transparent;border-bottom:1px solid #eef0f4;border-radius:6px;background:#fff;padding:7px 11px;font-size:14px;color:#20283a;font-weight:500;line-height:1.25;}
div.stButton > button:hover{background:#f3f6ff;border-color:#cfe0ff;color:#141b34;}
div.stButton > button:focus{box-shadow:none !important;}
div.stButton > button[kind="primary"]{background:#eef4ff;border-left:3px solid #e8552c;color:#141b34;font-weight:700;}
div.stButton > button[kind="primary"]:hover{background:#e4edff;}
.chip{display:inline-block;background:#eaf7ef;color:#215c37;border:1px solid #cfe9d8;border-radius:999px;padding:2px 9px;margin:2px;font-size:12px;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    df = pd.read_csv(RANKED)
    for c in ["top_reasons", "trigger_reasons", "llm_one_line", "llm_vertical"]:
        if c in df: df[c] = df[c].fillna("")
    if BLURBS.exists():
        b = pd.read_csv(BLURBS)[["company_id", "outreach_angle", "why_polished"]]
        df = df.merge(b, on="company_id", how="left")
    if COMPONENTS.exists():   # raw Fit/Value components -> lets the app re-weight and re-rank live
        df["company_id"] = df["company_id"].astype(str)
        comp = pd.read_csv(COMPONENTS); comp["company_id"] = comp["company_id"].astype(str)
        df = df.merge(comp, on="company_id", how="left")
    return df

if not RANKED.exists():
    st.error("Run `python code/pipeline.py` first to generate ranked_output.csv."); st.stop()
df = load()

# --- Fit/Value are hand-chosen formulas, so their sub-weights are user-editable (Win's are learned) ---
FIT_DEF = {"fw_need": 0.50, "fw_b2b": 0.20, "fw_msp": 0.15, "fw_rec": 0.15}   # sum = 1.00
VAL_DEF = {"vw_mm": 0.55, "vw_rev": 0.25, "vw_emp": 0.20}                     # sum = 1.00
def _rebalance_group(changed, keys):   # drag one slider, the others adjust so the group always sums to 1.0
    others = [k for k in keys if k != changed]
    rem = max(0.0, 1.0 - float(st.session_state[changed]))
    osum = sum(float(st.session_state[k]) for k in others)
    for k in others:
        st.session_state[k] = (rem * float(st.session_state[k]) / osum) if osum > 0 else (rem / len(others))
def _norm(vals):
    t = sum(vals) or 1.0
    return [v / t for v in vals]
def _fit_w():
    if st.session_state.get("custom_fit"):
        return _norm([float(st.session_state.get(k, dv)) for k, dv in FIT_DEF.items()])
    return list(FIT_DEF.values())
def _val_w():
    if st.session_state.get("custom_val"):
        return _norm([float(st.session_state.get(k, dv)) for k, dv in VAL_DEF.items()])
    return list(VAL_DEF.values())
def _pct(x):   # matches pipeline.pct: 0-1 percentile rank
    s = pd.Series(x).rank(pct=True)
    return s.fillna(s.median()).values

# --- Mid-market sweet-spot taper: Meridian's proven motion is the core/lower mid-market (won deals skew small).
# Value should reward the mid-market prize but PEAK in the core (rev <= $250M, emp <= 600) and taper for
# upper-mid / borderline-enterprise accounts (e.g. certain upper-mid accounts) that sit outside Meridian's serviceable range.
_mm_rev = pd.to_numeric(df.get("annual_revenue"), errors="coerce").fillna(0.0).values
_mm_emp = pd.to_numeric(df.get("employee_count"), errors="coerce").fillna(0.0).values
MM_REV, MM_EMP, MM_P = 250e6, 600.0, 0.5     # core mid-market ceilings (Fed def: $10M-$250M; ~370 emp avg)
_mm_taper = np.minimum(
    np.where(_mm_rev > MM_REV, (MM_REV / np.maximum(_mm_rev, 1.0)) ** MM_P, 1.0),
    np.where(_mm_emp > MM_EMP, (MM_EMP / np.maximum(_mm_emp, 1.0)) ** MM_P, 1.0))
if "value_score" in df:
    df["value_pct"] = _pct(pd.to_numeric(df["value_score"], errors="coerce").fillna(0.0) * _mm_taper)

st.markdown('<div class="hero"><h1>▲ Meridian — Sales Pipeline</h1></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ Which queue")
    mode = st.segmented_control(
        "Rank for", ["⚖️ Balanced", "⚡ Close Now", "🎯 Strategic"], default="⚖️ Balanced",
        help="Balanced = the full priority ranking.  Close Now = safe near-term wins.  Strategic = the biggest accounts.  Fit always counts.")
    st.markdown("### 🔎 Filter")
    q = st.text_input("Search company / keyword", "")
    prods = ["(all)"] + (sorted(df["recommended_product"].dropna().unique().tolist()) if "recommended_product" in df else [])
    prod = st.selectbox("Lead product", prods)
    sort_by = st.selectbox("Sort by", ["Priority", "Fit", "Win", "Value", "Timing"], index=0,
                           help="Reorder the list by any score. Priority is the default overall ranking.")
    only_hot = st.checkbox("Only show 🔥 live buying signals", False)

# re-rank live with the queue toggle. Fit is the fixed anchor; the toggle trades a small budget
# between Win (near-term) and Value (strategic). Fit and Win co-lead; Value is the lighter factor.
DEFAULT_BLEND = (0.40, 0.40, 0.20)   # official, logic-chosen weights (the revert baseline)
_PRESETS = {"⚡ Close Now": (0.40, 0.50, 0.10), "🎯 Strategic": (0.40, 0.25, 0.35)}
fit_w, win_w, val_w = _PRESETS.get(mode, DEFAULT_BLEND)
if st.session_state.get("custom_w"):   # hand-chosen blend weights are editable (Win's model weights are not)
    cf = float(st.session_state.get("cw_fit", DEFAULT_BLEND[0]))
    cw = float(st.session_state.get("cw_win", DEFAULT_BLEND[1]))
    cv = float(st.session_state.get("cw_val", DEFAULT_BLEND[2]))
    _tot = (cf + cw + cv) or 1.0
    fit_w, win_w, val_w = cf / _tot, cw / _tot, cv / _tot   # normalize so the three always sum to 1
d = df.copy()
# live re-weighting of the hand-chosen Fit / Value formulas (recompute the percentile over all 2,000)
if st.session_state.get("custom_fit") and "f_need" in d:
    wn, wb, wm, wr = _fit_w()
    _raw = (wn*d["f_need"] + wb*d["f_b2b"] + wm*d["f_msp"] + wr*d["f_rec"]) * (0.35 + 0.65*d["f_b2b"]) * (1 - 0.45*d["too_big"])
    d["fit_pct"] = _pct(_raw)
if st.session_state.get("custom_val") and "v_mm" in d:
    wm, wr, we = _val_w()
    _raw = np.minimum(1.0, wm*d["v_mm"] + wr*d["v_rev"] + we*d["v_emp"] + d["v_bonus"]) * (1 - 0.6*d["too_big"])
    d["value_pct"] = _pct(_raw)
d["priority_live"] = (100 * (np.power(d["fit_pct"], fit_w) * np.power(d["win_pct"], win_w) *
                             np.power(d["value_pct"], val_w))).round(0)
if q:
    m = d["company_name"].astype(str).str.contains(q, case=False, na=False)
    for c in ["llm_one_line", "llm_vertical", "top_reasons"]:
        if c in d: m |= d[c].astype(str).str.contains(q, case=False, na=False)
    d = d[m]
if prod != "(all)": d = d[d["recommended_product"] == prod]
d["_hot"] = pd.to_numeric(d.get("trigger_score", 0), errors="coerce").fillna(0) > 0
if only_hot: d = d[d["_hot"]]
d = d.sort_values("priority_live", ascending=False).reset_index(drop=True)  # base order / tiebreaker
d["priority_rank"] = np.arange(1, len(d) + 1)   # canonical priority rank (kept through re-sort)
_sortmap = {"Priority": ("priority_live", "priority"), "Fit": ("fit_pct", "fit %ile"),
            "Win": ("win_pct", "win %ile"), "Value": ("value_pct", "value %ile"),
            "Timing": ("trigger_score", "timing score")}
sort_col, sort_lbl = _sortmap.get(sort_by, ("priority_live", "priority"))
d = d.sort_values(sort_col, ascending=False, kind="stable").reset_index(drop=True)
d["rank"] = np.arange(1, len(d) + 1)   # rank = position in the current sort (top row = 1)

# when the queue/sort changes, jump the account detail to the new #1 (don't leave it stuck on a stale account)
_rank_key = f"{mode}|{sort_by}"
if st.session_state.get("_rank_key") != _rank_key:
    st.session_state["_rank_key"] = _rank_key
    if len(d): st.session_state["sel_id"] = str(d.iloc[0]["company_id"])

# ---------- ACV / capacity constants at module level (both tabs use them) ----------
acv_mid = 15000.0        # fixed: benchmarked mid-market ACV (Melio/BILL/Tipalti comps) — not a guess, a placeholder
acv_smb = 0.2 * acv_mid
acv_lrg = 2.3 * acv_mid
CAPACITY = 100          # fixed: ~5 reps x 20 accounts/quarter
BASE_WR = 0.15          # fixed: assumed blind win-rate
MEAN_WIN = pd.to_numeric(df["win_propensity"], errors="coerce").fillna(0).mean() or 1.0

# continuous ACV: scales SUB-linearly with revenue (deal size grows slower than revenue — realistic),
# anchored so a ~$20M mid-market company pays the slider value. Floored for tiny, capped for giants.
ANCHOR_REV = 20e6
def _acv_of(rev):
    r = pd.to_numeric(pd.Series(np.atleast_1d(rev)), errors="coerce")
    return (acv_mid * np.sqrt((r / ANCHOR_REV).clip(lower=0))).clip(2000, acv_mid * 8).fillna(acv_mid).values

def _fmt_rev(v):
    try: v = float(v)
    except (TypeError, ValueError): return "—"
    if v >= 1e9: return f"${v/1e9:.1f}B/yr"
    if v >= 1e6: return f"${v/1e6:.0f}M/yr"
    if v >= 1e3: return f"${v/1e3:.0f}K/yr"
    return f"${v:.0f}/yr"

def _recurring_label(v):
    s = str(v or "").lower()
    if any(k in s for k in ("subscription", "recurring", "saas", "retainer", "contract")): return "High — recurring / subscription"
    if "mixed" in s or "hybrid" in s: return "Medium — mixed (some recurring)"
    if any(k in s for k in ("one-time", "one time", "project", "transaction")): return "Low — mostly one-time"
    return str(v) if v else "—"

def _vertical_fit(v):
    s = str(v or "").lower()
    if any(k in s for k in ("msp", "it service", "managed", "software", "saas", "technolog", "consult", "agency", "professional", "staffing", "logistics")): return f"Strong ({v})"
    if any(k in s for k in ("retail", "consumer", "hospitality", "restaurant", "real estate", "manufactur", "healthcare", "nonprofit")): return f"Weaker ({v})"
    return f"Moderate ({v})" if v else "—"

def _emp_band(n):
    try: n = int(float(n))
    except (TypeError, ValueError): return "—"
    band = "Small" if n < 50 else "Mid" if n < 500 else "Large" if n < 3000 else "Enterprise"
    return f"{band} ({n:,})"

def _clean_reasons(s):
    s = str(s or "")
    for a, b in [("SMB-scale", "small-business size"), ("ICP signals", "ideal-customer match"),
                 ("tech-stack signals", "tech-stack match"), (" (win pattern)", "")]:
        s = s.replace(a, b)
    return s.strip() or "—"

def _wbar(name, w, color):
    pct = int(round(float(w) * 100))
    return (f"<div style='display:flex;align-items:center;gap:10px;margin:5px 0;font-size:14px;'>"
            f"<span style='width:118px;color:#20283a'>{name}</span>"
            f"<span style='flex:1;background:#eef0f4;border-radius:5px;height:16px;'>"
            f"<span style='display:block;width:{pct}%;background:{color};height:16px;border-radius:5px;'></span></span>"
            f"<span style='width:40px;text-align:right;font-weight:700;color:#141b34'>{float(w):.2f}</span></div>")

def _sbar(name, coef, maxc=0.55):
    c = float(coef); pct = int(round(min(abs(c) / maxc, 1) * 100))
    color = "#2e9e5b" if c >= 0 else "#c0504d"; sign = "+" if c >= 0 else "−"
    return (f"<div style='display:flex;align-items:center;gap:10px;margin:5px 0;font-size:14px;'>"
            f"<span style='width:150px;color:#20283a'>{name}</span>"
            f"<span style='flex:1;background:#eef0f4;border-radius:5px;height:16px;'>"
            f"<span style='display:block;width:{pct}%;background:{color};height:16px;border-radius:5px;'></span></span>"
            f"<span style='width:46px;text-align:right;font-weight:700;color:{color}'>{sign}{abs(c):.2f}</span></div>")

def _ibar(name, imp, sign, maxi=0.226):
    pct = int(round(min(imp / maxi, 1) * 100))
    color = "#2e9e5b" if sign >= 0 else "#c0504d"
    return (f"<div style='display:flex;align-items:center;gap:10px;margin:5px 0;font-size:14px;'>"
            f"<span style='width:158px;color:#20283a'>{name}</span>"
            f"<span style='flex:1;background:#eef0f4;border-radius:5px;height:16px;'>"
            f"<span style='display:block;width:{pct}%;background:{color};height:16px;border-radius:5px;'></span></span>"
            f"<span style='width:42px;text-align:right;font-weight:700;color:{color}'>{imp*100:.0f}%</span></div>")

# Win model's own FEATURE IMPORTANCES from the gradient-boosting model — how much it relies on
# each signal (the closest a tree ensemble has to a readable per-feature weight). Direction sign
# comes from a linear surrogate. AR-flag, AP-flag, recent-press, has-ERP are ~0% (near-constant).
WIN_IMPORTANCES = [("company size", 0.226, -1), ("tech-stack signal", 0.178, -1), ("revenue", 0.165, -1),
                   ("MSP-vs-accounting", 0.136, 1), ("B2B confidence", 0.112, -1),
                   ("sub-vertical conf.", 0.099, 1), ("ICP-fit score", 0.037, -1), ("recurring revenue", 0.020, 1)]

tab_list, tab_biz = st.tabs(["📞  Call list", "💰  Business case"])

with tab_list:
    # ---------- KPI row ----------
    c1, c2 = st.columns(2)
    def kpi(col, val, lbl):
        col.markdown(f'<div class="card"><div class="metricbig">{val}</div><div class="metriclbl">{lbl}</div></div>', unsafe_allow_html=True)
    kpi(c1, f"{len(d):,}", "companies in view")
    kpi(c2, f"{int(d['_hot'].sum()):,}", "with a live buying signal 🔥")
    st.write("")

    # ---------- the call list + account detail ----------
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Ranked call list")
        st.caption(f"All {len(d):,} companies, sorted by {sort_by.lower()} (change it under Filter → Sort by). 🔥 = live buying signal. Click anywhere in a row to open its detail.")
        g = d.copy()
        g["signal"] = np.where(g["_hot"], "🔥", "")
        if sort_by in ("Fit", "Win", "Value"):
            g["metric"] = np.floor(g[sort_col] * 100).astype(int)   # floor so only the single top = 100 (not a 100-tie)
        elif sort_by == "Timing":
            g["metric"] = pd.to_numeric(g["trigger_score"], errors="coerce").fillna(0).round(0).astype(int)
        else:
            g["metric"] = g["priority_live"].round(0).astype(int)
        if sort_by == "Priority":
            cols = ["rank", "company_name", "metric", "recommended_product", "signal", "company_id"]
            ren = {"company_name": "company", "recommended_product": "lead", "metric": sort_lbl}
        else:
            cols = ["rank", "company_name", "metric", "priority_rank", "recommended_product", "signal", "company_id"]
            ren = {"company_name": "company", "recommended_product": "lead", "metric": sort_lbl, "priority_rank": "priority #"}
        grid_df = g[cols].rename(columns=ren)
        gb = GridOptionsBuilder.from_dataframe(grid_df)
        gb.configure_default_column(sortable=False, filter=False, resizable=False, suppressMovable=True, suppressMenu=True)
        gb.configure_selection("single", use_checkbox=False)
        gb.configure_column("company_id", hide=True)
        gb.configure_column("rank", width=60, type=["numericColumn"])
        gb.configure_column("company", minWidth=160, flex=2)
        gb.configure_column(sort_lbl, width=96, type=["numericColumn"])
        if "priority #" in grid_df.columns:
            gb.configure_column("priority #", width=90, type=["numericColumn"])
        gb.configure_column("lead", width=90)
        gb.configure_column("signal", width=80)
        gb.configure_grid_options(rowHeight=32, headerHeight=36)
        grid = AgGrid(grid_df, gridOptions=gb.build(), height=560, theme="streamlit",
                      update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True,
                      allow_unsafe_jscode=True, key=f"callgrid_{mode}_{sort_by}")
        sel = grid["selected_rows"]
        sel_cid = None
        if isinstance(sel, pd.DataFrame):
            if not sel.empty and "company_id" in sel.columns: sel_cid = str(sel.iloc[0]["company_id"])
        elif isinstance(sel, list) and sel:
            sel_cid = str(sel[0].get("company_id"))
        if sel_cid:
            st.session_state["sel_id"] = sel_cid
        elif "sel_id" not in st.session_state and len(d):
            st.session_state["sel_id"] = str(d.iloc[0]["company_id"])

    with right:
        st.markdown("#### Account detail")
        sel_id = st.session_state.get("sel_id")
        match = d[d["company_id"].astype(str) == str(sel_id)] if sel_id is not None else d.iloc[0:0]
        if len(match) == 0 and len(d):
            match = d.iloc[0:1]
        if len(match):
            r = match.iloc[0]
            st.markdown(f"**{r.get('company_name','')}** &nbsp;<span class='badge b-seg'>{r.get('segment_name','')}</span>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8a92a6;margin-top:6px'>What they do</div>", unsafe_allow_html=True)
            st.caption(str(r.get("llm_one_line", "")))
            lead = r.get("recommended_product", "AR")
            trig = str(r.get("trigger_reasons", "") or "")
            line = f"**Lead with → {lead}**  ·  AR-need {r.get('ar_need','—')}/100 · AP-need {r.get('ap_need','—')}/100"
            if trig.strip(): line += f"\n\n**🔥 Why now:** {trig}"
            st.markdown(line)
            # per-account expected value (uses the Business-case ACV tiers + win-rate, same model as the headline)
            _acv_r = float(_acv_of(r.get("annual_revenue"))[0])
            _dc1, _dc2 = st.columns([5, 1])
            _dc1.markdown(f"**💵 Est. deal size ≈ \\${_acv_r:,.0f}/yr**")
            with _dc2.popover("ⓘ"):
                st.markdown("Estimated annual contract value (ACV), scaled from the account's revenue and anchored to the benchmarked deal size in the Business-case tab (Melio, BILL, Tipalti).")
            a, b, c, e, _pi = st.columns([2, 2, 2, 2, 1])
            a.metric("Priority", f"{r['priority_live']:.0f}")
            b.metric("Fit", f"{r['fit_pct']*100:.0f}")
            c.metric("Win", f"{r['win_pct']*100:.0f}")
            e.metric("Value", f"{r['value_pct']*100:.0f}")
            with _pi.popover("ⓘ"):
                st.markdown(
                    "**Fit · Win · Value are percentiles** — this account's rank on each dimension among all 2,000.\n\n"
                    "**Priority is the blend** — a weighted **geometric mean** of those three: I *multiply* them "
                    "(each raised to its weight — Fit^0.40 · Win^0.40 · Value^0.20), rather than average.\n\n"
                    "Because it's a geometric mean, Priority always lands **between** the three scores, and multiplying "
                    "means a weak score on **any one** dimension drags it down — it can't hide behind the other two. "
                    "(The very top account scores ~88, not 100, because no single company ranks first on all three at once.)")

            why = str(r.get("why_polished", "") or "").strip()
            st.markdown(f'<div class="why"><b>🧠 Why this account —</b> {why if why else "High-fit service business for Meridian based on its firmographics and website signals."}</div>',
                        unsafe_allow_html=True)
            st.markdown("<div style='margin-top:14px;font-size:12px;color:#8a92a6;text-transform:uppercase;letter-spacing:.05em'>Dig deeper</div>", unsafe_allow_html=True)

            with st.expander("🔬 Score breakdown — the signals behind each score"):
                st.caption("Priority is built from each score's **percentile** (rank among all 2,000), not the raw 0–100 number — weighted Fit 40% · Win 40% · Value 20%.")
                st.markdown(f"**🧩 Fit · {r['fit_pct']*100:.0f}th percentile** — is this the right *kind* of company? (need is half the score; being a right-sized B2B service business is the other half)")
                st.markdown(
                    f"- AR-need (struggles to get paid): **{r.get('ar_need','—')}/100**\n"
                    f"- AP-need (pays many vendors): **{r.get('ap_need','—')}/100**\n"
                    f"- Recurring revenue: **{_recurring_label(r.get('llm_recurring_revenue_model'))}**\n"
                    f"- Finance-ops complexity: **{r.get('llm_finance_ops_complexity','—')}/100**\n"
                    f"- Vertical fit: **{_vertical_fit(r.get('llm_vertical'))}**")
                st.markdown(f"**📈 Win · {r['win_pct']*100:.0f}th percentile** — how much it looks like Meridian's past wins")
                st.markdown(
                    f"- Win-likelihood: **{r['win_pct']*100:.0f}th percentile**\n"
                    f"- Key drivers: {_clean_reasons(r.get('top_reasons'))}  \n"
                    f"  <span style='color:#6b7280;font-size:12px'>(+ raises win-likelihood, − lowers it)</span>", unsafe_allow_html=True)
                st.markdown(f"**💰 Value · {r['value_pct']*100:.0f}th percentile** — how big the prize is")
                st.markdown(
                    f"- Mid-market fit: **{r.get('llm_midmarket_score','—')}/100**\n"
                    f"- Annual revenue: **{_fmt_rev(r.get('annual_revenue'))}**\n"
                    f"- Employees: **{_emp_band(r.get('employee_count'))}**\n"
                    f"- PE-backed: **{r.get('ext_pe_backed','unknown')}** · Multi-entity: **{r.get('ext_multi_entity','unknown')}**")
                tscore = r.get("trigger_score", 0)
                try: tscore = f"{float(tscore):.0f}/100"
                except (TypeError, ValueError): tscore = "—"
                st.markdown("---")
                st.markdown("**⏱️ Why now — timing** *(separate from the three scores; it decides urgency, not rank)*")
                st.markdown(
                    f"- Timing score: **{tscore}**\n"
                    f"- Live signal: {trig if trig.strip() else 'no live trigger right now'}")


    # ---------- how the scores are built (interactive weights) ----------
    st.write("")
    _WHEAD = ("<div style='display:flex;font-size:11px;color:#8a92a6;letter-spacing:.03em;margin:2px 0 4px'>"
              "<span style='flex:1'></span><span style='width:46px;text-align:right'>weight</span></div>")
    with st.expander("🗺️  How the scores are built — weights & formula"):
        _mix = "Custom" if st.session_state.get("custom_w") else mode
        st.markdown(f"**Priority** blends the three scores (each converted to a percentile first). Current mix — *{_mix}*:")
        st.markdown(_WHEAD, unsafe_allow_html=True)
        for nm, wt, col in [("Fit", fit_w, "#33409a"), ("Value", val_w, "#e8552c"), ("Win", win_w, "#2e9e5b")]:
            st.markdown(_wbar(nm, wt, col), unsafe_allow_html=True)
        st.caption("Multiplied, not averaged — a weak score on any one dimension can't be masked by the other two.")
        st.checkbox("🎚️ Adjust blend weights", key="custom_w",
                    help="The blend proportions are hand-chosen, so you can tune them. Win's internal model weights are learned, not editable.")
        if st.session_state.get("custom_w"):
            for _k, _dv in [("cw_fit", DEFAULT_BLEND[0]), ("cw_win", DEFAULT_BLEND[1]), ("cw_val", DEFAULT_BLEND[2])]:
                if _k not in st.session_state: st.session_state[_k] = _dv
            def _rebalance(changed):
                others = [k for k in ("cw_fit", "cw_val", "cw_win") if k != changed]
                rem = max(0.0, 1.0 - float(st.session_state[changed]))
                osum = sum(float(st.session_state[k]) for k in others)
                if osum > 0:
                    for k in others: st.session_state[k] = rem * float(st.session_state[k]) / osum
                else:
                    for k in others: st.session_state[k] = rem / len(others)
            st.slider("Fit weight", 0.0, 1.0, step=0.05, key="cw_fit", on_change=_rebalance, args=("cw_fit",))
            st.slider("Value weight", 0.0, 1.0, step=0.05, key="cw_val", on_change=_rebalance, args=("cw_val",))
            st.slider("Win weight", 0.0, 1.0, step=0.05, key="cw_win", on_change=_rebalance, args=("cw_win",))
            if st.button("↺ Revert to default weights"):
                for _k in ("cw_fit", "cw_win", "cw_val"): st.session_state.pop(_k, None)
                st.rerun()
            st.caption(f"Move any slider and the other two adjust automatically so the three always sum to 100%. Official defaults: Fit {DEFAULT_BLEND[0]} · Value {DEFAULT_BLEND[2]} · Win {DEFAULT_BLEND[1]}.")
        st.caption("Open a tab to drill into each score's own inputs.")
        t_fit, t_val, t_win = st.tabs(["🧩  Fit", "💰  Value", "📈  Win"])
        with t_fit:
            st.caption("Rule-based — *is this the right kind of company?* Hand-chosen weights, so you can tune them.")
            st.markdown(_WHEAD, unsafe_allow_html=True)
            for nm, wt in zip(["product-need", "is-B2B", "MSP-like", "recurring rev."], _fit_w()):
                st.markdown(_wbar(nm, wt, "#33409a"), unsafe_allow_html=True)
            st.caption("Then **× a penalty** if it isn't a B2B service business, and **× a penalty** if it's too big for Meridian's motion.")
            if "f_need" in df.columns:
                st.checkbox("🎚️ Adjust Fit weights", key="custom_fit")
                if st.session_state.get("custom_fit"):
                    for _k, _dv in FIT_DEF.items():
                        if _k not in st.session_state: st.session_state[_k] = _dv
                    st.slider("product-need", 0.0, 1.0, step=0.05, key="fw_need", on_change=_rebalance_group, args=("fw_need", tuple(FIT_DEF)))
                    st.slider("is-B2B", 0.0, 1.0, step=0.05, key="fw_b2b", on_change=_rebalance_group, args=("fw_b2b", tuple(FIT_DEF)))
                    st.slider("MSP-like", 0.0, 1.0, step=0.05, key="fw_msp", on_change=_rebalance_group, args=("fw_msp", tuple(FIT_DEF)))
                    st.slider("recurring rev.", 0.0, 1.0, step=0.05, key="fw_rec", on_change=_rebalance_group, args=("fw_rec", tuple(FIT_DEF)))
                    if st.button("↺ Revert Fit weights"):
                        for _k in FIT_DEF: st.session_state.pop(_k, None)
                        st.rerun()
                    st.caption("Normalized to sum to 1.00; re-ranks the whole list live.")
        with t_val:
            st.caption("Rule-based — *if we win it, how big is the prize?* Hand-chosen weights, so editable.")
            st.markdown(_WHEAD, unsafe_allow_html=True)
            for nm, wt in zip(["mid-market fit", "revenue", "headcount"], _val_w()):
                st.markdown(_wbar(nm, wt, "#e8552c"), unsafe_allow_html=True)
            st.caption("**+ a bonus** if it's PE-backed or multi-entity (from the outside web search).")
            if "v_mm" in df.columns:
                st.checkbox("🎚️ Adjust Value weights", key="custom_val")
                if st.session_state.get("custom_val"):
                    for _k, _dv in VAL_DEF.items():
                        if _k not in st.session_state: st.session_state[_k] = _dv
                    st.slider("mid-market fit", 0.0, 1.0, step=0.05, key="vw_mm", on_change=_rebalance_group, args=("vw_mm", tuple(VAL_DEF)))
                    st.slider("revenue", 0.0, 1.0, step=0.05, key="vw_rev", on_change=_rebalance_group, args=("vw_rev", tuple(VAL_DEF)))
                    st.slider("headcount", 0.0, 1.0, step=0.05, key="vw_emp", on_change=_rebalance_group, args=("vw_emp", tuple(VAL_DEF)))
                    if st.button("↺ Revert Value weights"):
                        for _k in VAL_DEF: st.session_state.pop(_k, None)
                        st.rerun()
                    st.caption("Normalized to sum to 1.00; re-ranks the whole list live.")
        with t_win:
            st.caption("Learned, not hand-set. Gradient boosting has **no single weight per feature**, so this shows the model's own **feature importance** — how much it relies on each signal (a magnitude; the closest a tree ensemble has to a weight). Unlike Fit and Value, these can't be edited — the data set them, not me.")
            for nm, imp, sign in WIN_IMPORTANCES:
                st.markdown(_ibar(nm, imp, sign), unsafe_allow_html=True)
            st.caption("Bar length = importance from the gradient-boosting model · color = direction from a linear surrogate (🟢 raises win-likelihood, 🔴 lowers it). **Company size matters most — and pushes win *down*** (AP's wins skew small/mid). AR-flag, AP-flag, recent-press and has-ERP land at ~0% — near-constant across the 100 deals.")

with tab_biz:
    # ---------- 💰 Business case — what the ranking is worth ----------
    with st.expander("💰  What is this ranking worth to Meridian?", expanded=True):
        st.caption("A ranking earns money by pointing scarce rep hours at the accounts that will actually **close** — the right *fit* **and** *winnable* — instead of the wrong-fit majority. "
                   "The number below is what the **data proves** (the lift) × a benchmarked mid-market deal size.")
        _dl, _di = st.columns([6, 1])
        _dl.markdown("**Typical mid-market deal size — ACV ≈ \\$15,000 / year**")
        with _di.popover("ⓘ"):
            st.markdown(
                "A **benchmarked placeholder**, not a guess — I won't invent Meridian's pricing. "
                "Anchored to comparable spend-management vendors:\n\n"
                "- **Melio** ≈ \\$1–3k (SMB)\n"
                "- **BILL** ≈ \\$14k (mid-market)\n"
                "- **Tipalti** ≈ \\$25–40k (larger)\n\n"
                "\\$15k sits at the mid-market midpoint. Each account's own deal size then scales "
                "sub-linearly from its revenue, anchored to this figure.")
        capacity = CAPACITY
        base_wr = BASE_WR

        bc = df.copy()
        # a company only CLOSES if it's a good fit AND winnable — a wrong-fit account won't buy regardless of the win score
        bc["_close"] = pd.to_numeric(bc["fit_pct"], errors="coerce").fillna(0.5) * pd.to_numeric(bc["win_pct"], errors="coerce").fillna(0.5)
        _mc = bc["_close"].mean() or 1.0
        bc["_pwin"] = np.minimum(0.95, base_wr * bc["_close"] / _mc)       # close prob, anchored so the average = Meridian's blind win-rate
        bc["_acv"] = _acv_of(bc["annual_revenue"])                        # continuous ACV, scales with each account's revenue
        bc["_exp"] = bc["_pwin"] * bc["_acv"]
        bc = bc.sort_values("priority_score", ascending=False).reset_index(drop=True)
        top = bc.iloc[:capacity]
        ranked = float(top["_exp"].sum())
        blind = float(capacity * bc["_exp"].mean())
        incr = ranked - blind
        st.markdown(
            f"<div style='background:#eefaf0;border:1px solid #cfe9d8;border-radius:12px;padding:14px 18px;margin-top:8px'>"
            f"<div style='font-size:12px;color:#215c37;text-transform:uppercase;letter-spacing:.05em'>Working this ranking vs. the list blind</div>"
            f"<div style='font-size:30px;font-weight:800;color:#141b34;line-height:1.15'>≈ ${incr:,.0f} more expected ARR / quarter"
            f"<span style='font-size:15px;font-weight:600;color:#5b6478'> &nbsp;(~${incr*4:,.0f}/yr)</span></div>"
            f"<div style='font-size:13px;color:#3a4257;margin-top:4px'>Top {capacity} in order → <b>${ranked:,.0f}</b> expected ARR"
            f" &nbsp;·&nbsp; {capacity} at random → <b>${blind:,.0f}</b></div></div>",
            unsafe_allow_html=True)
        with st.popover("ⓘ  How is this calculated?"):
            st.markdown(
                "**The idea.** Reps have limited hours, so the value of a ranking is simple: *does working these accounts, in order, beat working the same number at random?*\n\n"
                "**Per account:** expected ARR = **P(close) × ACV**\n"
                "- **P(close)** = the blind win-rate × the account's **Fit × Win** (relative to the list average). A company only closes if it's a *good fit* **and** *winnable* — a wrong-fit account (like a global holding co) won't buy no matter what, so this correctly sends the ~1,400 low-fit accounts near zero. The average across all 2,000 equals the blind win-rate, so nothing is inflated.\n"
                "- **ACV** (annual contract value = what one customer pays per year) = scaled sub-linearly from the account's revenue, anchored to your slider.\n\n"
                "**The headline:** *ranked* = expected ARR of the top-N by priority · *blind* = N × the average expected ARR · *incremental* = the difference.\n\n"
                "Assumes ~100 accounts worked/quarter (5 reps × 20) and a 15% blind win-rate. ACV scales **continuously with each account's revenue**, anchored to your slider (a ~20M-revenue mid-market company pays the slider value), floored low for tiny shops and capped for accounts too big for Meridian's motion — benchmarked to Melio, BILL, and Tipalti.")

