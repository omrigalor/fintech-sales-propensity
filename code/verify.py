"""
Critic / verification pass (LLM-as-a-judge, second opinion).

Pass 1 extracts AR-need / AP-need / is-B2B from each site. This pass has a SECOND model
re-read the same text plus Pass 1's answer and ask "is this actually supported by the
text?" — correcting the scores it can't justify and emitting a CALIBRATED confidence.
This is how we trade compute for reliability instead of trusting a single extraction.

Cached & resumable; tokens metered. Run from submission/:  python code/verify.py [--limit N]
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, requests
import common

CODE = Path(__file__).resolve().parent
DATA = CODE / "data" / "case_study_enriched_2k.csv"
ENRICH = CODE / "outputs" / "enrichment.csv"
CACHE = CODE / "outputs" / "verify_cache"
OUT = CODE / "outputs" / "verify.csv"
MODEL = "reasoning-pro"
API = common.api_url()
MAX_TEXT = 8000
WORKERS = 12

SYSTEM = (
    "You are a strict fact-checker. You are given a company's website text and a prior model's "
    "extracted scores (AR-need = how much they struggle to GET PAID by clients; AP-need = how much "
    "they PAY MANY VENDORS; both 0-100). Re-read the text and judge whether each score is SUPPORTED. "
    "Correct any score the text doesn't justify, and rate your own certainty. Return STRICT JSON only: "
    "{\"ar_need\": int, \"ap_need\": int, \"is_b2b_services\": true/false, "
    "\"verify_confidence\": float 0-1, \"agree_with_prior\": true/false, \"note\": \"<=120 chars\"}.")


def parse_json(s):
    if not s: return None
    try: return json.loads(s)
    except Exception: pass
    for c in reversed(re.findall(r"\{.*\}", s, re.S)):
        try: return json.loads(c)
        except Exception: continue
    return None


def verify_one(row, key):
    cid = str(row["company_id"])
    cf = CACHE / f"{cid}.json"
    if cf.exists():
        return {"company_id": cid, **json.loads(cf.read_text())}
    prior = {"ar_need": row.get("ar_need_score"), "ap_need": row.get("ap_need_score"),
             "is_b2b_services": row.get("is_b2b_services")}
    user = (f"Company: {row.get('company_name','')}\n\nPrior extraction: {json.dumps(prior)}\n\n"
            f"Website text:\n{str(row.get('website_text',''))[:MAX_TEXT]}")
    body = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}], "response_format": {"type": "json_object"},
            "max_tokens": 1500, "temperature": 0}
    out = {"ar_need_v": prior["ar_need"], "ap_need_v": prior["ap_need"],
           "verify_confidence": 0.0, "agree_with_prior": None, "verify_note": "",
           "_pt": 0, "_ct": 0}
    for attempt in range(3):
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
            if r.status_code == 200:
                d = r.json(); obj = parse_json(d["choices"][0]["message"]["content"]) or {}
                u = d.get("usage", {}); out["_pt"], out["_ct"] = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                if obj:
                    out["ar_need_v"] = obj.get("ar_need", prior["ar_need"])
                    out["ap_need_v"] = obj.get("ap_need", prior["ap_need"])
                    out["verify_confidence"] = obj.get("verify_confidence", 0.0)
                    out["agree_with_prior"] = obj.get("agree_with_prior")
                    out["verify_note"] = str(obj.get("note", ""))[:120]
                break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    cf.write_text(json.dumps(out))
    return {"company_id": cid, **out}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    key = common.api_key()
    d = pd.read_csv(DATA)[["company_id", "company_name", "website_text"]]
    e = pd.read_csv(ENRICH)[["company_id", "ar_need_score", "ap_need_score", "is_b2b_services"]]
    d["company_id"] = d["company_id"].astype(str); e["company_id"] = e["company_id"].astype(str)
    df = d.merge(e, on="company_id", how="left")
    if a.limit: df = df.head(a.limit)
    rows = df.to_dict("records")
    print(f"Verifying {len(rows)} extractions with {MODEL} ({WORKERS} workers)...", flush=True)
    res, done, pt, ct, changed = [], 0, 0, 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in as_completed([ex.submit(verify_one, r, key) for r in rows]):
            x = f.result(); res.append(x); done += 1; pt += x.get("_pt", 0); ct += x.get("_ct", 0)
            if x.get("agree_with_prior") is False: changed += 1
            if done % 100 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}  tokens in/out={pt}/{ct}  corrected={changed}", flush=True)
    pd.DataFrame(res).drop(columns=["_pt", "_ct"]).to_csv(OUT, index=False)
    print(f"DONE -> {OUT}  ({len(res)} rows; {changed} scores corrected; tokens={pt+ct})", flush=True)


if __name__ == "__main__":
    main()
