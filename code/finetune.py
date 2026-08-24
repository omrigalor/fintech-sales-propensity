"""
Fine-tuning demonstration (honest scope).

We fine-tune a small model to predict won/lost from a company profile. IMPORTANT HONEST NOTE:
the labeled won/lost deals contain NO website text — only structured features — so a true
"website-text -> outcome" fine-tune is not possible with this data. With only 100 rows and
structured features, the calibrated classic-ML model (src/pipeline.py) is the correct PRIMARY
scorer; a fine-tuned LLM would overfit. So this script:
  1) builds a proper supervised fine-tuning (SFT) dataset from the 100 labeled deals,
  2) prints the exact provider commands to launch the fine-tune,
  3) documents why we treat it as a demonstration / optional ensemble signal, and what data
     the company should collect (website text for won/lost deals) to make text-based fine-tuning worthwhile.

Run:  python src/finetune.py   (builds data/finetune_train.jsonl + prints launch commands)
"""
from __future__ import annotations
import json
import pandas as pd
import common

FEATS = ["industry", "employee_count", "annual_revenue", "msp_probability", "acct_probability",
         "b2b_confidence", "is_b2b_services", "recurring_revenue_signal", "company_size_signal",
         "services_focus", "tech_stack_confidence", "icp_fit_score"]

def profile_text(r) -> str:
    parts = []
    for f in FEATS:
        v = r.get(f)
        if pd.notna(v):
            parts.append(f"{f.replace('_',' ')}: {v}")
    return "; ".join(parts)

def main():
    lab = common.load_labeled()
    out = common.DATA / "finetune_train.jsonl"
    sys = ("Given a B2B company profile, predict whether Meridian would WIN or LOSE the deal. "
           "Answer with exactly one word: won or lost.")
    n = 0
    with open(out, "w") as fh:
        for _, r in lab.iterrows():
            msg = {"messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": profile_text(r)},
                {"role": "assistant", "content": "won" if r["y"] == 1 else "lost"}]}
            fh.write(json.dumps(msg) + "\n"); n += 1
    print(f"Wrote {out} ({n} examples).\n")
    print("To launch a real supervised fine-tune (needs a fine-tuning-enabled key + provider CLI):")
    print("  llmctl create dataset wonlost data/finetune_train.jsonl")
    print("  llmctl create sftj --base-model reasoning-flash \\")
    print("      --dataset wonlost --epochs 3 --output-model wonlost-ft")
    print("  llmctl deploy wonlost-ft   # then call it like any model as an ensemble signal\n")
    print("HONEST NOTE: with 100 structured rows this fine-tune would overfit; the calibrated ML model")
    print("in src/pipeline.py is the validated primary. Fine-tuning becomes worthwhile once the company logs the")
    print("website text (and richer signals) for won/lost deals — a concrete data-collection recommendation.")

if __name__ == "__main__":
    main()
