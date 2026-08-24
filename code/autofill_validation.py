"""
Auto-review the manual-validation sample with a SECOND model (LLM-as-a-judge).

make_validation_sample.py picks 50 accounts for a human to spot-check. This script has a
strong model (reasoning-pro) independently read each company's text and judge whether the
extraction's vertical / AR-need / AP-need look right — a machine first-pass that a human can
then confirm, and a quick precision read on the enrichment.

Run from submission/:  python code/autofill_validation.py
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, requests
import common

CODE = Path(__file__).resolve().parent
SAMPLE = CODE / "outputs" / "validation_sample.csv"
DATA = CODE / "data" / "case_study_enriched_2k.csv"
OUT = CODE / "outputs" / "validation_sample_autoreviewed.csv"
MODEL = "reasoning-pro"
API = common.api_url()

SYSTEM = ("You are an independent reviewer. Given a company's website text and a prior model's claims "
          "(vertical, AR-need 0-100, AP-need 0-100), judge each. Return STRICT JSON only: "
          "{\"vertical_ok\": true/false, \"ar_need_ok\": true/false, \"ap_need_ok\": true/false, "
          "\"reviewer_note\": \"<=140 chars\"}.")


def parse(s):
    try: return json.loads(s)
    except Exception:
        for c in reversed(re.findall(r"\{.*\}", s or "", re.S)):
            try: return json.loads(c)
            except Exception: continue
    return {}


def review(row, text, key):
    claims = {"vertical": row.get("model_vertical"), "ar_need": row.get("model_ar_need"),
              "ap_need": row.get("model_ap_need")}
    body = {"model": MODEL, "response_format": {"type": "json_object"}, "max_tokens": 1200, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Company: {row.get('company_name','')}\n\nPrior claims: "
                          f"{json.dumps(claims)}\n\nWebsite text:\n{str(text)[:7000]}"}]}
    for attempt in range(3):
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
            if r.status_code == 200:
                return parse(r.json()["choices"][0]["message"]["content"])
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {}


def main():
    key = common.api_key()
    s = pd.read_csv(SAMPLE)
    txt = pd.read_csv(DATA)[["company_id", "website_text"]]
    txt["company_id"] = txt["company_id"].astype(str); s["company_id"] = s["company_id"].astype(str)
    s = s.merge(txt, on="company_id", how="left")
    print(f"Auto-reviewing {len(s)} validation accounts with {MODEL}...", flush=True)
    rows = s.to_dict("records")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(review, r, r.get("website_text", ""), key): i for i, r in enumerate(rows)}
        for f in as_completed(futs):
            v = f.result(); i = futs[f]
            rows[i]["reviewer_vertical_correct(Y/N)"] = "Y" if v.get("vertical_ok") else "N"
            rows[i]["reviewer_ar_need_reasonable(Y/N)"] = "Y" if v.get("ar_need_ok") else "N"
            rows[i]["reviewer_ap_need_reasonable(Y/N)"] = "Y" if v.get("ap_need_ok") else "N"
            rows[i]["reviewer_notes"] = v.get("reviewer_note", "")
    out = pd.DataFrame(rows).drop(columns=["website_text"], errors="ignore")
    out.to_csv(OUT, index=False)
    agree = (out["reviewer_vertical_correct(Y/N)"] == "Y").mean()
    print(f"DONE -> {OUT}  ({len(out)} reviewed; vertical-agreement {agree:.0%})", flush=True)


if __name__ == "__main__":
    main()
