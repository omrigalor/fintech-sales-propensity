"""
LLM enrichment of the 2,000-company list.

WHY: the structured 'enrichment' columns we were given are mostly empty (founded_year 100%
missing, finance/psa/accounting fields 97-100% missing) and sometimes wrong (the `industry`
column labels a real-estate developer as "IT & services"). The one rich, reliable field is
`website_text`. So we use a strong reasoning-grade LLM (reasoning-pro) to READ each
company's website text and extract clean, structured GTM signals — validated with Pydantic.

These signals are used to (a) fill gaps in the 2k, (b) OVERRIDE garbage firmographics,
(c) feed the ICP-Fit and Value scoring layers and the app's explanations.

Run:  python src/enrich.py            (resumable — caches every result to disk)
      python src/enrich.py --limit 25 (quick sample)
"""
from __future__ import annotations
import os, sys, json, re, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

CODE = Path(__file__).resolve().parent
DATA = CODE / "data" / "case_study_enriched_2k.csv"
CACHE = CODE / "outputs" / "enrich_cache_pro"
OUT = CODE / "outputs" / "enrichment.csv"
MODEL = "reasoning-pro"
# OpenAI-compatible endpoint; override with the LLM_API_URL env var (or .env)
API = os.environ.get("LLM_API_URL") or "https://api.openai.com/v1/chat/completions"
MAX_TEXT = 10000
WORKERS = 16

def api_key() -> str:
    env = os.environ.get("LLM_API_KEY")
    if env and env.strip():
        return env.strip()
    for line in (CODE.parent / ".env").read_text().splitlines():
        if line.startswith("LLM_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("No LLM_API_KEY in .env")

# ---- the extraction schema (what we pull from the website text) --------------
class Enrichment(BaseModel):
    one_line: str = Field("", description="<=20 words: what the company does")
    is_b2b_services: bool = False          # corrected B2B-services flag (overrides bad firmographics)
    primary_services: str = ""
    vertical: str = "other"                # MSP/IT, accounting, marketing agency, construction, staffing, logistics, legal, healthcare services, other
    recurring_revenue_model: str = "unclear"   # subscription | retainer | project_based | mixed | unclear
    size_signal: str = "unclear"           # micro | small | mid_market | enterprise | unclear
    midmarket_score: int = 0               # 0-100 how clearly a valuable mid-market account
    finance_ops_complexity: int = 0        # 0-100 overall finance-operations complexity
    ar_need_score: int = 0                  # 0-100 need for Accounts RECEIVABLE automation (getting paid: many client invoices, recurring billing, collections/late-payment pain)
    ap_need_score: int = 0                  # 0-100 need for Accounts PAYABLE automation (paying vendors: many suppliers, bill-pay, multi-entity, procurement)
    client_volume_signal: str = "unclear"   # low | medium | high | unclear (how many clients they invoice)
    serves_enterprise_clients: bool = False
    multi_location: bool = False
    pain_signals: list[str] = []           # billing/collections/cash-flow/manual-invoicing/AR/AP mentions
    growth_signals: list[str] = []         # hiring/new offices/funding/awards/expansion
    ap_competitor_mentions: list[str] = [] # Bill.com, Ramp, Melio, Tipalti, Stampli, ...
    fit_rationale: str = ""                 # <=25 words why Meridian would/wouldn't want them
    extraction_confidence: float = 0.5

    @field_validator("midmarket_score", "finance_ops_complexity", "ar_need_score", "ap_need_score", mode="before")
    @classmethod
    def _clip100(cls, v):
        try: return max(0, min(100, int(round(float(v)))))
        except Exception: return 0

    @field_validator("extraction_confidence", mode="before")
    @classmethod
    def _clip1(cls, v):
        try: return max(0.0, min(1.0, float(v)))
        except Exception: return 0.5

    @field_validator("pain_signals", "growth_signals", "ap_competitor_mentions", mode="before")
    @classmethod
    def _aslist(cls, v):
        if v is None: return []
        if isinstance(v, str): return [v] if v.strip() else []
        return [str(x) for x in v][:8]

    @field_validator("one_line", "primary_services", "vertical", "recurring_revenue_model",
                     "size_signal", "client_volume_signal", "fit_rationale", mode="before")
    @classmethod
    def _asstr(cls, v):
        # pro returns some of these as lists — join instead of failing validation
        if isinstance(v, list): return ", ".join(str(x) for x in v)
        return v if v is not None else ""

SYSTEM = """You are a B2B go-to-market analyst for Meridian, a fintech for service-based businesses. Meridian has TWO products:
  • Accounts RECEIVABLE automation (its core) — helps a company GET PAID by its clients: invoicing, payment collection, chasing late payments, cash flow.
  • Accounts PAYABLE automation (newer) — helps a company PAY its vendors/suppliers: bill pay, approvals, spend management.
A company is a strong Meridian target if it has strong need for EITHER product. Meridian most wants MID-MARKET service companies (not tiny shops, not Fortune-500).

Read the company's scraped website text and return a SINGLE JSON object with EXACTLY these keys:
one_line, is_b2b_services, primary_services, vertical, recurring_revenue_model, size_signal, midmarket_score, finance_ops_complexity, ar_need_score, ap_need_score, client_volume_signal, serves_enterprise_clients, multi_location, pain_signals, growth_signals, ap_competitor_mentions, fit_rationale, extraction_confidence.

Rules:
- vertical: one of "MSP/IT services","accounting","marketing agency","construction","staffing","logistics","legal services","healthcare services","other".
- recurring_revenue_model: one of "subscription","retainer","project_based","mixed","unclear".
- size_signal: one of "micro","small","mid_market","enterprise","unclear". mid_market = multiple offices, named enterprise clients, ~50-1000 staff, scale language.
- midmarket_score (0-100): higher = more clearly a valuable mid-market account (not a tiny shop, not a Fortune-500).
- finance_ops_complexity (0-100): overall finance-operations complexity.
- ar_need_score (0-100): need for RECEIVABLE automation — HIGH if they invoice MANY clients, have recurring/monthly billing, or mention collections/late payments/cash-flow pain.
- ap_need_score (0-100): need for PAYABLE automation — HIGH if they pay MANY vendors/suppliers, mention procurement/bill-pay, multiple entities, or international payments.
- client_volume_signal: one of "low","medium","high","unclear" — roughly how many clients they invoice.
- is_b2b_services: true ONLY if they sell services to OTHER BUSINESSES (not a product company, not consumer/B2C, not real estate).
- ap_competitor_mentions: array of any competing finance/payments tools they mention using — look for: Bill.com/BILL, Ramp, Brex, Melio, Tipalti, Stampli, AvidXchange, Corpay, Airbase, Coupa, Routable, Plooto, Invoiced, Versapay, Quadient. [] if none. (Note: most sites won't reveal their finance tools.)
- pain_signals/growth_signals: arrays of short strings, [] if none.
- fit_rationale: <=25 words, plain English.
- extraction_confidence: 0.0-1.0 based on how informative the text was.
Base everything ONLY on the text. Output ONLY the JSON object."""

def extract_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)   # reasoning models sometimes prepend text
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {}

def call(name: str, text: str, key: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Company: {name}\n\nWebsite text:\n{(text or '')[:MAX_TEXT]}"},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 2500, "temperature": 0,   # pro is a reasoning model — needs room before the JSON
    }
    last = None
    for attempt in range(4):
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
            if r.status_code == 200:
                d = r.json()
                content = d["choices"][0]["message"]["content"]
                usage = d.get("usage", {})
                return {"raw": extract_json(content), "usage": usage}
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (attempt + 1))
    return {"raw": {}, "usage": {}, "error": last}

def enrich_one(row, key) -> dict:
    cid = str(row["company_id"])
    cache_f = CACHE / f"{cid}.json"
    if cache_f.exists():
        return {"company_id": cid, **json.loads(cache_f.read_text()), "_cached": True}
    res = call(str(row.get("company_name", "")), row.get("website_text", ""), key)
    raw = res.get("raw", {})
    try:
        obj = Enrichment(**raw).model_dump()
    except ValidationError:
        obj = Enrichment().model_dump()  # graceful fallback, low confidence
        obj["extraction_confidence"] = 0.0
    obj["_prompt_tokens"] = res.get("usage", {}).get("prompt_tokens", 0)
    obj["_completion_tokens"] = res.get("usage", {}).get("completion_tokens", 0)
    obj["_error"] = res.get("error", "")
    cache_f.write_text(json.dumps(obj))
    return {"company_id": cid, **obj, "_cached": False}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    key = api_key()
    df = pd.read_csv(DATA)
    if args.limit:
        df = df.head(args.limit)
    rows = df[["company_id", "company_name", "website_text"]].to_dict("records")
    print(f"Enriching {len(rows)} companies with {MODEL} ({WORKERS} workers)...", flush=True)

    results, done, tin, tout, errs = [], 0, 0, 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(enrich_one, r, key): r for r in rows}
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            done += 1
            tin += res.get("_prompt_tokens", 0); tout += res.get("_completion_tokens", 0)
            if res.get("_error"): errs += 1
            if done % 100 == 0 or done == len(rows):
                dt = time.time() - t0
                print(f"  {done}/{len(rows)}  ({dt:.0f}s)  tokens in/out={tin}/{tout}  errors={errs}", flush=True)

    out = pd.DataFrame(results).drop(columns=[c for c in ["_cached", "_error"] if c in pd.DataFrame(results).columns], errors="ignore")
    out.to_csv(OUT, index=False)
    # rough cost note (reasoning-pro is a reasoning model; exact price varies)
    print(f"\nDONE. Wrote {OUT}  ({len(out)} rows). Tokens in={tin:,} out={tout:,}. Errors={errs}.", flush=True)

if __name__ == "__main__":
    main()
