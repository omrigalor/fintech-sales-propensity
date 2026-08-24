"""
For the top-ranked companies, use the LLM to turn the model's structured reasons into
(a) a polished plain-English 'why', and (b) a suggested outreach opening angle for a rep.
This is the LLM 'wording the model's why': a transparent logistic surrogate (signed
coef * standardized feature value) produces the approximate factual reasons; the LLM only
phrases them and never changes the ranking. Output -> outputs/outreach_blurbs.csv (used by the app).

Run:  python src/llm_blurbs.py --top 150
"""
from __future__ import annotations
import argparse, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import common

SYS = ("You write crisp GTM notes for Meridian, which sells AR/AP automation, "
       "invoicing, collections and spend management to service businesses. Return ONLY compact JSON "
       "with exactly two keys: 'why_polished' and 'outreach_angle'. "
       "why_polished = why this is/ isn't a strong Meridian target, MAX 25 WORDS. "
       "outreach_angle = a specific opening hook a rep could use, MAX 20 WORDS. "
       "Be concrete and SHORT. No preamble.")

def extract_fields(out: str) -> dict:
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    # fallback: pull each field even if JSON is truncated/malformed
    def g(k):
        mm = re.search(rf'"{k}"\s*:\s*"([^"]+)"', out)
        return mm.group(1) if mm else ""
    return {"why_polished": g("why_polished"), "outreach_angle": g("outreach_angle")}

def one(row, key_unused):
    prof = (f"Company: {row.get('company_name')}\nWhat they do: {row.get('llm_one_line')}\n"
            f"Vertical: {row.get('llm_vertical')} | Revenue model: {row.get('llm_recurring_revenue_model')}\n"
            f"Finance-ops complexity: {row.get('llm_finance_ops_complexity')}/100 | Mid-market: {row.get('llm_midmarket_score')}/100\n"
            f"Pain signals: {row.get('llm_pain_signals')} | Competitor: {row.get('llm_ap_competitor_mentions')}\n"
            f"Scores — priority {row.get('priority_score')}, fit {row.get('fit_score')}, win% {row.get('win_propensity')}, value {row.get('value_score')}\n"
            f"Model reasons: {row.get('top_reasons')}")
    try:
        out = common.llm_chat(
            [{"role": "system", "content": SYS}, {"role": "user", "content": prof}],
            model="reasoning-pro",
            json_mode=True, max_tokens=900, temperature=0.2)
        j = extract_fields(out)
        return {"company_id": str(row["company_id"]),
                "why_polished": str(j.get("why_polished", ""))[:200],
                "outreach_angle": str(j.get("outreach_angle", ""))[:200]}
    except Exception as e:
        return {"company_id": str(row["company_id"]), "why_polished": "", "outreach_angle": ""}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=150); a = ap.parse_args()
    d = pd.read_csv(common.ROOT / "ranked_output.csv").head(a.top)
    print(f"Generating blurbs for top {len(d)} companies...")
    rows = d.to_dict("records"); res = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(one, r, None) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 50 == 0: print(f"  {i}/{len(rows)}")
    pd.DataFrame(res).to_csv(common.OUT / "outreach_blurbs.csv", index=False)
    print(f"Wrote outputs/outreach_blurbs.csv ({len(res)} rows)")

if __name__ == "__main__":
    main()
