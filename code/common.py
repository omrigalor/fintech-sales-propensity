"""Shared utilities: paths, data loading, firmographic cleaning, LLM calls."""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import requests

CODE = Path(__file__).resolve().parent          # the code/ folder (holds data/ and outputs/)
ROOT = CODE.parent                              # submission root (holds ranked_output.csv)
DATA = CODE / "data"
OUT = CODE / "outputs"
ART = OUT / "artifacts"
ENV = ROOT / ".env"
for p in (OUT, ART):
    p.mkdir(parents=True, exist_ok=True)

LABELED = DATA / "case_study_labeled_wonlost_100.csv"
ENRICHED = DATA / "case_study_enriched_2k.csv"
ENRICHMENT = OUT / "enrichment.csv"          # produced by code/enrich.py (pro-model site enrichment)
SCRAPE = OUT / "scrape.csv"                   # produced by code/scrape.py (live technographic/AR-AP signals)
VERIFY = OUT / "verify.csv"                   # produced by code/verify.py (critic-pass calibrated scores)
EXTERNAL = OUT / "external.csv"               # produced by code/external.py (third-party web evidence)
CONSISTENCY = OUT / "consistency.csv"         # produced by code/consistency.py (self-consistency AR/AP)
RANKED_OUTPUT = ROOT / "ranked_output.csv"   # the deliverable


def api_key() -> str:
    """Resolve the LLM API key: prefer the LLM_API_KEY env var, else a local
    .env file. Never printed or logged. Clear, actionable error if it's missing."""
    key = os.environ.get("LLM_API_KEY")
    if key and key.strip():
        return key.strip()
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("LLM_API_KEY=") and not line.startswith("#"):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    raise SystemExit(
        "LLM_API_KEY not found. Either `export LLM_API_KEY=...` in your shell, "
        "or copy code/.env.example to .env and paste your key there. "
        "(Only enrich.py / llm_blurbs.py need it — pipeline.py and app.py run from the shipped outputs.)"
    )


def api_url() -> str:
    """The chat-completions endpoint. Any OpenAI-compatible provider works — set LLM_API_URL
    in the environment (or .env). Defaults to the OpenAI Chat Completions URL."""
    return (os.environ.get("LLM_API_URL")
            or "https://api.openai.com/v1/chat/completions")


def llm_chat(messages, model="reasoning-flash",
                   json_mode=False, max_tokens=500, temperature=0.2, timeout=90) -> str:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(api_url(),
                      headers={"Authorization": f"Bearer {api_key()}"}, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---- cleaning helpers --------------------------------------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def to_bool(s):
    return s.map(lambda v: True if str(v).strip().lower() in ("true", "1", "yes")
                 else (False if str(v).strip().lower() in ("false", "0", "no") else np.nan))


def clean_firmographics(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, reconcile disagreeing sources, engineer data-quality signals.
    The case explicitly says the data is messy: fields missing, sources disagree,
    formats inconsistent, some duplicates. We handle each here."""
    d = df.copy()
    for c in ["employee_count", "annual_revenue", "msp_probability", "acct_probability",
              "b2b_confidence", "tech_stack_confidence", "icp_fit_score", "sub_vertical_confidence",
              "services_focus", "recent_press_count", "finance_leader_count", "employees_alt",
              "founded_year", "source_count"]:
        if c in d:
            d[c] = to_num(d[c])
    for c in ["is_b2b_services", "recurring_revenue_signal", "erp_implementer",
              "ar_team_mentioned", "ap_team_mentioned", "erp_migration_announced",
              "psa_migration_announced"]:
        if c in d:
            d[c] = to_bool(d[c])

    # --- source disagreement = a data-quality signal (imperfect data is real signal) ---
    if "employees_alt" in d and "employee_count" in d:
        both = d["employee_count"].notna() & d["employees_alt"].notna()
        d["emp_disagreement"] = 0.0
        denom = (d["employee_count"].abs() + d["employees_alt"].abs()).replace(0, np.nan)
        d.loc[both, "emp_disagreement"] = ((d["employee_count"] - d["employees_alt"]).abs() / denom)[both]
    if "industry_alt" in d and "industry" in d:
        d["industry_disagreement"] = ((d["industry"].astype(str).str.lower().str.strip()
                                       != d["industry_alt"].astype(str).str.lower().str.strip())
                                      & d["industry_alt"].notna()).astype(float)

    # --- entity resolution WITHOUT deleting rows ---
    # The brief says "score every company." Shared-domain rows are usually distinct
    # franchises / local offices (CMIT Solutions, TeamLogic IT, NerdsToGo), not accidental
    # dupes — so we FLAG related records and still rank all of them, rather than dropping any.
    d["source_company_id"] = d["company_id"].astype(str) if "company_id" in d else d.index.astype(str)
    if "domain" in d:
        _dom = d["domain"].astype(str).str.strip().str.lower()
        _dom = _dom.where(~_dom.isin(["", "nan", "none"]), other=np.nan)
        counts = _dom.map(_dom.value_counts())
        d["possible_duplicate"] = (counts > 1).fillna(False).astype(bool)
        d["duplicate_reason"] = np.where(d["possible_duplicate"], "shares domain with sibling record(s)", "")
        uniq = {v: f"C{i:04d}" for i, v in enumerate(sorted(_dom.dropna().unique()))}
        d["entity_cluster_id"] = _dom.map(uniq).fillna("")
        # canonical = richest-source row per shared domain (others still ranked, just tagged)
        ordv = (d["source_count"] if "source_count" in d else pd.Series(0.0, index=d.index)).fillna(-1)
        d["_ord"] = ordv
        d["canonical_record"] = True
        dup_mask = _dom.notna() & (counts > 1)
        if dup_mask.any():
            canon_idx = (d[dup_mask].assign(_d=_dom[dup_mask])
                         .sort_values("_ord", ascending=False)
                         .drop_duplicates(subset="_d", keep="first").index)
            d.loc[dup_mask, "canonical_record"] = False
            d.loc[canon_idx, "canonical_record"] = True
        d = d.drop(columns=["_ord"])
    return d.reset_index(drop=True)


def load_labeled() -> pd.DataFrame:
    d = clean_firmographics(pd.read_csv(LABELED))
    d["y"] = (d["outcome"].astype(str).str.lower() == "closed_won").astype(int)
    return d


def load_enriched() -> pd.DataFrame:
    d = clean_firmographics(pd.read_csv(ENRICHED))
    if ENRICHMENT.exists():
        e = pd.read_csv(ENRICHMENT)
        e["company_id"] = e["company_id"].astype(str)
        d["company_id"] = d["company_id"].astype(str)
        d = d.merge(e.add_prefix("llm_").rename(columns={"llm_company_id": "company_id"}),
                    on="company_id", how="left")
    if SCRAPE.exists():
        s = pd.read_csv(SCRAPE)
        s["company_id"] = s["company_id"].astype(str)
        d["company_id"] = d["company_id"].astype(str)
        d = d.merge(s.add_prefix("scr_").rename(columns={"scr_company_id": "company_id"}),
                    on="company_id", how="left")
    for path, pfx in [(VERIFY, "vr_"), (EXTERNAL, "ext_"), (CONSISTENCY, "sc_")]:   # critic + external + self-consistency
        if path.exists():
            x = pd.read_csv(path); x["company_id"] = x["company_id"].astype(str)
            d["company_id"] = d["company_id"].astype(str)
            d = d.merge(x.add_prefix(pfx).rename(columns={f"{pfx}company_id": "company_id"}),
                        on="company_id", how="left")

    # --- reconcile the critic pass as a CALIBRATION signal (not an override, not a blend) ---
    # A second independent pro read disagrees on ~2/3 of AR/AP scores — evidence that LLM
    # AR/AP-need scoring is inherently noisy. We KEEP the extraction score as primary (averaging
    # two mismatched scales just adds noise), and record disagreement as an honesty signal.
    d["critic_disagreement"] = 0
    for base, ver in [("llm_ar_need_score", "vr_ar_need_v"), ("llm_ap_need_score", "vr_ap_need_v")]:
        if base in d and ver in d:
            a = pd.to_numeric(d[base], errors="coerce"); b = pd.to_numeric(d[ver], errors="coerce")
            d["critic_disagreement"] = np.maximum(
                d["critic_disagreement"], (b.notna() & ((a - b).abs() > 20)).astype(int))
    # self-consistency (median of N same-prompt samples) is a lower-variance estimate than a
    # single read — use it as the AR/AP score wherever we computed it (the top accounts).
    for base, sc in [("llm_ar_need_score", "sc_ar_need_sc"), ("llm_ap_need_score", "sc_ap_need_sc")]:
        if base in d and sc in d:
            v = pd.to_numeric(d[sc], errors="coerce")
            d[base] = np.where(v.notna(), v, pd.to_numeric(d[base], errors="coerce"))
    return d
