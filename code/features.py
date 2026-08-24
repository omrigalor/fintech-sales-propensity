"""Feature engineering.

The WIN-PROPENSITY model can only learn from features present in BOTH the labeled
100 and the 2k (so it can be applied to the 2k). Those are the shared, well-populated
firmographic/technographic columns. The LLM-only signals (finance_ops_complexity,
midmarket_score, ...) exist only for the 2k (the labeled deals have no website text),
so they feed the transparent Fit and Value layers instead — and they OVERRIDE bad
firmographics (e.g. the mislabeled `industry`)."""
from __future__ import annotations
import numpy as np
import pandas as pd

# features used by the trained propensity model (present in labeled AND 2k)
PROP_FEATURES = [
    "emp_log", "rev_log", "msp_probability", "acct_probability", "msp_minus_acct",
    "b2b_confidence", "tech_stack_confidence", "icp_fit_score", "sub_vertical_confidence",
    "services_focus", "recent_press_count", "size_ord",
    "is_b2b_int", "recurring_int", "erp_int", "ar_int", "ap_int",
]

_SIZE_ORD = {"micro": 0, "small": 1, "smb": 1, "mid_market": 2, "midmarket": 2,
             "mid": 2, "large": 3, "enterprise": 4}


def _bool_int(s):
    return s.map(lambda v: 1.0 if v is True else (0.0 if v is False else np.nan))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["emp_log"] = np.log1p(pd.to_numeric(d.get("employee_count"), errors="coerce"))
    d["rev_log"] = np.log1p(pd.to_numeric(d.get("annual_revenue"), errors="coerce"))
    for c in ["msp_probability", "acct_probability", "b2b_confidence", "tech_stack_confidence",
              "icp_fit_score", "sub_vertical_confidence", "services_focus", "recent_press_count"]:
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["msp_minus_acct"] = d["msp_probability"] - d["acct_probability"]
    d["size_ord"] = (d.get("company_size_signal", pd.Series(index=d.index, dtype=object))
                     .astype(str).str.lower().str.strip().map(_SIZE_ORD))
    d["is_b2b_int"] = _bool_int(d.get("is_b2b_services"))
    d["recurring_int"] = _bool_int(d.get("recurring_revenue_signal"))
    d["erp_int"] = _bool_int(d.get("erp_implementer"))
    d["ar_int"] = _bool_int(d.get("ar_team_mentioned"))
    d["ap_int"] = _bool_int(d.get("ap_team_mentioned"))
    return d


def apply_llm_overrides(d: pd.DataFrame) -> pd.DataFrame:
    """Use confident LLM readings of the website text to correct/fill the 2k firmographics."""
    d = d.copy()
    conf = pd.to_numeric(d.get("llm_extraction_confidence"), errors="coerce").fillna(0)
    # correct is_b2b_services when the LLM is confident (fixes ViaWest-type garbage)
    if "llm_is_b2b_services" in d:
        good = conf >= 0.6
        d.loc[good, "is_b2b_services"] = d.loc[good, "llm_is_b2b_services"]
    # fill missing size signal from the LLM's read
    if "llm_size_signal" in d and "company_size_signal" in d:
        miss = d["company_size_signal"].isna() | (d["company_size_signal"].astype(str).str.strip() == "")
        d.loc[miss, "company_size_signal"] = d.loc[miss, "llm_size_signal"]
    return d


def build(labeled: pd.DataFrame, enriched: pd.DataFrame):
    """Returns Xtr, ytr (labeled) and score_df (2k with features + LLM signals)."""
    lab = engineer(labeled)
    Xtr = lab[PROP_FEATURES].astype(float)
    ytr = lab["y"].astype(int).values

    enr = apply_llm_overrides(enriched)
    enr = engineer(enr)
    return Xtr, ytr, enr, PROP_FEATURES
