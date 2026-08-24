"""
Self-consistency for AR/AP-need (Wang et al., 2022; SelfCheckGPT, Manakul et al., 2023).

A single LLM read of AR/AP-need is noisy (the critic pass agreed only ~35%). Self-consistency
fixes that the RIGHT way: sample the SAME extraction prompt several times at a non-zero
temperature and AGGREGATE (median) — same rubric, so the samples are on the same scale and CAN
be combined (unlike averaging two different prompts). The spread across samples is itself a
confidence signal (tight agreement = trustworthy; wide spread = uncertain).

Run on the accounts that matter (top-N by rank) to stay budget-aware.
Run from submission/:  python code/consistency.py --top 400 --samples 4
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests
import common

CODE = Path(__file__).resolve().parent
DATA = CODE / "data" / "case_study_enriched_2k.csv"
CACHE = CODE / "outputs" / "consistency_cache"
OUT = CODE / "outputs" / "consistency.csv"
MODEL = "reasoning-pro"
API = common.api_url()

SYSTEM = ("Score two needs for a US mid-market company from its website text, 0-100. "
          "AR-need = how much they struggle to GET PAID by clients (many client invoices, recurring "
          "billing, collections/late-payment pain). AP-need = how much they PAY MANY VENDORS "
          "(suppliers, bill-pay, multi-entity, procurement). Return STRICT JSON only: "
          "{\"ar_need\": int, \"ap_need\": int}.")


def _parse(s):
    try: return json.loads(s)
    except Exception:
        for c in reversed(re.findall(r"\{.*\}", s or "", re.S)):
            try: return json.loads(c)
            except Exception: continue
    return None


def sample_scores(name, text, key, n_samples):
    """n independent reads at temperature 0.5 -> lists of ar, ap."""
    body = {"model": MODEL, "response_format": {"type": "json_object"}, "max_tokens": 800,
            "temperature": 0.5, "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Company: {name}\n\nWebsite text:\n{str(text)[:6000]}"}]}
    ars, aps, pt, ct = [], [], 0, 0
    for _ in range(n_samples):
        for attempt in range(2):
            try:
                r = requests.post(API, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
                if r.status_code == 200:
                    d = r.json(); u = d.get("usage", {}); pt += u.get("prompt_tokens", 0); ct += u.get("completion_tokens", 0)
                    o = _parse(d["choices"][0]["message"]["content"])
                    if o and "ar_need" in o and "ap_need" in o:
                        ars.append(float(o["ar_need"])); aps.append(float(o["ap_need"]))
                    break
            except Exception:
                time.sleep(1.5)
    return ars, aps, pt, ct


def one(row, key, n_samples):
    cid = str(row["company_id"]); cf = CACHE / f"{cid}.json"
    if cf.exists():
        return {"company_id": cid, **json.loads(cf.read_text())}
    ars, aps, pt, ct = sample_scores(row.get("company_name", ""), row.get("website_text", ""), key, n_samples)
    if ars:
        # aggregate = median; confidence = 1 - normalized dispersion across samples
        disp = (np.std(ars) + np.std(aps)) / 2.0
        out = {"ar_need_sc": float(np.median(ars)), "ap_need_sc": float(np.median(aps)),
               "consistency_confidence": round(float(max(0.0, 1 - disp / 30.0)), 3),
               "n_samples": len(ars), "_pt": pt, "_ct": ct}
    else:
        out = {"ar_need_sc": None, "ap_need_sc": None, "consistency_confidence": 0.0,
               "n_samples": 0, "_pt": pt, "_ct": ct}
    cf.write_text(json.dumps(out))
    return {"company_id": cid, **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=400); ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    key = common.api_key()
    ranked = pd.read_csv(common.ROOT / "ranked_output.csv")
    top_ids = ranked.sort_values("rank").head(a.top)["company_id"].astype(str).tolist()
    df = pd.read_csv(DATA); df["company_id"] = df["company_id"].astype(str)
    rows = df[df["company_id"].isin(top_ids)][["company_id", "company_name", "website_text"]].to_dict("records")
    print(f"Self-consistency on top {len(rows)} accounts x {a.samples} samples ({MODEL})...", flush=True)
    res, done, pt, ct = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(one, r, key, a.samples) for r in rows]):
            x = f.result(); res.append(x); done += 1; pt += x.get("_pt", 0); ct += x.get("_ct", 0)
            if done % 50 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}  tokens in/out={pt}/{ct}", flush=True)
    pd.DataFrame(res).drop(columns=["_pt", "_ct"]).to_csv(OUT, index=False)
    print(f"DONE -> {OUT}  ({len(res)} rows; tokens={pt+ct})", flush=True)


if __name__ == "__main__":
    main()
