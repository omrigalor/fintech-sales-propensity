"""
Build a stratified ~50-account sample for MANUAL review of the LLM enrichment, so the
AR-need / AP-need extraction can be validated by a human rather than trusted blindly
(the AR-first thesis is partly encoded in the rubric, so it must be spot-checked).

Writes code/outputs/validation_sample.csv with the model's claims + blank reviewer
columns. Run from submission/:  python code/make_validation_sample.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "ranked_output.csv")
rng = np.random.default_rng(7)

conf = pd.to_numeric(d.get("llm_extraction_confidence", 0), errors="coerce").fillna(
    d.get("review_flag", 0)) if "llm_extraction_confidence" in d else d.get("review_flag", 0)
d["_msp"] = d["segment_name"].astype(str).str.contains("MSP")
d["_hi_score"] = d["priority_score"] >= d["priority_score"].quantile(0.9)
d["_lo_score"] = d["priority_score"] <= d["priority_score"].quantile(0.1)
d["_no_web"] = (pd.to_numeric(d.get("scr_num_tools", 0), errors="coerce").fillna(0) == 0)
d["_flag"] = d.get("review_flag", 0) == 1

def take(mask, n, seen):
    pool = d[mask & ~d["company_id"].isin(seen)]
    k = min(n, len(pool))
    return pool.sample(k, random_state=int(rng.integers(1e6))) if k else pool.head(0)

strata = [("high-priority", d["_hi_score"], 10), ("low-priority", d["_lo_score"], 8),
          ("MSP-type", d["_msp"], 8), ("non-MSP", ~d["_msp"], 8),
          ("needs-review/low-conf", d["_flag"], 8), ("no-website-signal", d["_no_web"], 8)]

rows, seen = [], set()
for label, mask, n in strata:
    s = take(mask, n, seen)
    for _, r in s.iterrows():
        seen.add(r["company_id"])
        rows.append({
            "stratum": label,
            "company_id": r["company_id"], "company_name": r.get("company_name"),
            "domain": r.get("domain"),
            "model_vertical": r.get("llm_vertical"), "model_one_line": r.get("llm_one_line"),
            "model_ar_need": r.get("ar_need"), "model_ap_need": r.get("ap_need"),
            "model_recommended_product": r.get("recommended_product"),
            "model_confidence": r.get("llm_extraction_confidence", ""),
            # ---- blanks for the human reviewer ----
            "reviewer_vertical_correct(Y/N)": "", "reviewer_ar_need_reasonable(Y/N)": "",
            "reviewer_ap_need_reasonable(Y/N)": "", "reviewer_product_agree(Y/N)": "",
            "reviewer_notes": "",
        })

out = pd.DataFrame(rows)
out.to_csv(ROOT / "code" / "outputs" / "validation_sample.csv", index=False)
print(f"Wrote validation_sample.csv — {len(out)} accounts across {out['stratum'].nunique()} strata "
      f"for manual precision check of the enrichment.")
