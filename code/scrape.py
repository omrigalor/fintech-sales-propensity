"""
Live website scraper — the one genuinely NEW data source beyond the provided text blob.

For each company (using its `domain`), fetch the homepage and detect:
  • TECHNOGRAPHICS: which accounting / PSA / payments / CRM tools they run (from the HTML/scripts).
  • AR structural signals: a "Pay Invoice" / "Client Portal" / "Make a Payment" link ⇒ they collect
    payments online ⇒ relevant to Meridian's *receivables* product.
  • AP structural signals: a "Vendors" / "Suppliers" / "Procurement" page ⇒ relevant to AP's *payables* product.
  • hiring signal (careers / open roles) ⇒ growth/intent.

Reproducible (requests + the domain column). Live scraping is best-effort: some sites block or time out
(that's real-world) — those are marked scrape_ok=False and simply contribute no signal.

Run:  python src/scrape.py            (resumable — caches each fetch)
      python src/scrape.py --limit 30
"""
from __future__ import annotations
import argparse, json, re, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests
warnings.filterwarnings("ignore")

CODE = Path(__file__).resolve().parent
CACHE = CODE / "outputs" / "scrape_cache"
OUT = CODE / "outputs" / "scrape.csv"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122 Safari/537.36"}

# technographic fingerprints: label -> regexes to look for in the page HTML
TOOLS = {
    "quickbooks": r"quickbooks|intuit", "xero": r"\bxero\b", "netsuite": r"netsuite",
    "sage": r"\bsage\b", "freshbooks": r"freshbooks",
    "connectwise": r"connectwise", "autotask": r"autotask", "kaseya": r"kaseya", "datto": r"\bdatto\b",
    "bill_com": r"bill\.com", "ramp": r"ramp\.com|\bramp\b\s*(?:card|spend|bill\s*pay)", "brex": r"\bbrex\b",
    "melio": r"melio", "tipalti": r"tipalti", "stampli": r"stampli", "avidxchange": r"avidxchange",
    "stripe": r"stripe\.com|js\.stripe|\bstripe\b\s*(?:payments?|checkout|connect)",
    "square": r"squareup|\bsquare\b\s*(?:payments?|invoices?|pos|checkout|point of sale)",
    "paypal": r"paypal", "authorize_net": r"authorize\.net",
    "hubspot": r"hubspot", "salesforce": r"salesforce|force\.com", "shopify": r"shopify|woocommerce",
}
AR_PAT = r"pay (my )?(invoice|bill)|client portal|make a payment|online payment|billing portal|pay online|payment portal"
AP_PAT = r"become a vendor|vendor portal|supplier portal|vendor onboarding|supplier onboarding|pay (?:your )?vendors|accounts payable|procure(?:ment)? portal"
HIRE_PAT = r"careers|we're hiring|were hiring|open positions|join our team|job openings"


def fetch(domain: str) -> str | None:
    urls = []
    dom = str(domain).strip()
    if not dom or dom.lower() == "nan":
        return None
    if not dom.startswith("http"):
        urls = [f"https://{dom}", f"http://{dom}"]
    else:
        urls = [dom]
    for u in urls:
        for verify in (True, False):
            try:
                r = requests.get(u, headers=UA, timeout=8, verify=verify, allow_redirects=True)
                if r.status_code < 400 and r.text:
                    return r.text[:200000]
            except requests.exceptions.SSLError:
                continue
            except Exception:
                break
    return None


def analyze(html: str) -> dict:
    h = html.lower()
    tools = [name for name, pat in TOOLS.items() if re.search(pat, h)]
    return {
        "scrape_ok": True,
        "tools_detected": tools,
        "num_tools": len(tools),
        "has_payment_portal": bool(re.search(AR_PAT, h)),     # AR structural signal
        "has_vendor_page": bool(re.search(AP_PAT, h)),        # AP structural signal
        "hiring": bool(re.search(HIRE_PAT, h)),
        "competitor_tool": bool(set(tools) & {"bill_com", "ramp", "brex", "melio", "tipalti", "stampli", "avidxchange"}),
    }


def scrape_one(row) -> dict:
    cid = str(row["company_id"]); cf = CACHE / f"{cid}.json"
    if cf.exists():
        return {"company_id": cid, **json.loads(cf.read_text())}
    html = fetch(row.get("domain"))
    res = analyze(html) if html else {"scrape_ok": False, "tools_detected": [], "num_tools": 0,
                                      "has_payment_portal": False, "has_vendor_page": False,
                                      "hiring": False, "competitor_tool": False}
    cf.write_text(json.dumps(res))
    return {"company_id": cid, **res}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CODE / "data" / "case_study_enriched_2k.csv")[["company_id", "domain"]]
    if a.limit: df = df.head(a.limit)
    rows = df.to_dict("records")
    print(f"Scraping {len(rows)} sites (best-effort, 20 workers)...", flush=True)
    res, done, ok = [], 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=20) as ex:
        for f in as_completed([ex.submit(scrape_one, r) for r in rows]):
            r = f.result(); res.append(r); done += 1; ok += int(r["scrape_ok"])
            if done % 100 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}  reachable={ok}  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(res).to_csv(OUT, index=False)
    print(f"DONE. {OUT}  ({len(res)} rows, {ok} reachable = {100*ok/len(res):.0f}%).", flush=True)


if __name__ == "__main__":
    main()
