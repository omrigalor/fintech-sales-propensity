"""
External-evidence enrichment (third-party, not the company's own promotional site).

For each company we search the web (DuckDuckGo — no API key), keep NON-owned results
(LinkedIn / Crunchbase / Bloomberg / directories / news / job boards), and LLM-read the
aggregated search snippets + any fetchable pages into OBJECTIVE fields the company won't
say about itself: PE-backed?, parent/holdco?, acquisitions?, multi-entity?, external
headcount, funding — each with a confidence + evidence + source URL.

This is the evidence behind the "multi-entity / PE-backed platform" thesis. Cached &
resumable; results ship pre-computed so the app/reviewer never wait on the network.

Run from submission/:  python code/external.py --top 300
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import common

CODE = Path(__file__).resolve().parent
CACHE = CODE / "outputs" / "external_cache"
OUT = CODE / "outputs" / "external.csv"
MODEL = "reasoning-pro"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
# pages that usually block scraping — we still use their search SNIPPET, just don't fetch them
NO_FETCH = ("linkedin.com", "crunchbase.com", "bloomberg.com", "facebook.com",
            "instagram.com", "twitter.com", "x.com", "zoominfo.com")

SYSTEM = (
    "You extract OBJECTIVE, third-party facts about a US mid-market company from web SEARCH "
    "RESULTS (titles+snippets) and page text. Use ONLY what the evidence supports; when unsupported, "
    "say \"unknown\". Do NOT infer from the company's marketing tone. Return STRICT JSON with keys: "
    "pe_backed (\"yes\"/\"no\"/\"unknown\"), parent_or_holdco (string or \"\"), "
    "acquisitions (\"yes\"/\"no\"/\"unknown\"), multi_entity (\"yes\"/\"no\"/\"unknown\"), "
    "employee_estimate (integer or null), funding_or_investors (string or \"\"), "
    "notable (<=140 chars), external_confidence (0-1 float), evidence (<=160 char quote), "
    "sources (array of up to 3 urls). No prose, JSON only.")


def domain_root(domain: str) -> str:
    d = str(domain).lower().split("//")[-1].replace("www.", "").split("/")[0]
    return d.split(".")[0] if d else ""


def ddg(query: str, n: int = 7):
    try:
        from ddgs import DDGS
    except Exception:
        from duckduckgo_search import DDGS
    for attempt in range(3):
        try:
            with DDGS() as d:
                return list(d.text(query, max_results=n))
        except Exception:
            time.sleep(2 + attempt * 2)
    return []


def _parse_json(content: str):
    """Robust: reasoning models may emit prose before the JSON. Try direct, then the last {...}."""
    if not content:
        return None
    try:
        return json.loads(content)
    except Exception:
        pass
    m = re.findall(r"\{.*\}", content, re.S)
    for cand in reversed(m):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def strip_html(t: str) -> str:
    t = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def gather(name: str, domain: str):
    """Return (aggregated_external_text, source_urls). Snippets always; fetch a couple pages."""
    root = domain_root(domain)
    results = ddg(f'"{name}" {root}') or ddg(f'{name} company {root}') or ddg(f'{name} company')
    own = f"{root}." if root else "\x00"
    ext = [r for r in results if own not in r.get("href", "").lower()]
    blob, srcs = [], []
    for r in ext[:7]:                       # snippets from every external result (dodges blocking)
        blob.append(f"[{r.get('title','')}] {r.get('body','')}  ({r.get('href','')})")
        srcs.append(r.get("href", ""))
    for r in ext[:4]:                       # fetch clean text from the non-blocked ones
        href = r.get("href", "")
        if href and not any(b in href for b in NO_FETCH):
            try:
                resp = requests.get(href, headers=UA, timeout=7)
                if resp.status_code < 400 and resp.text:
                    blob.append(strip_html(resp.text)[:2500])
            except Exception:
                pass
    return "\n".join(blob)[:12000], srcs[:5]


DEFAULT = {"pe_backed": "unknown", "parent_or_holdco": "", "acquisitions": "unknown",
           "multi_entity": "unknown", "employee_estimate": None, "funding_or_investors": "",
           "notable": "", "external_confidence": 0.0, "evidence": "", "sources": []}


def enrich_one(row):
    cid = str(row["company_id"])
    cf = CACHE / f"{cid}.json"
    if cf.exists():
        return {"company_id": cid, **json.loads(cf.read_text())}
    name, domain = str(row.get("company_name", "")), str(row.get("domain", ""))
    text, srcs = gather(name, domain)
    out = dict(DEFAULT)
    if text.strip():
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Company: {name}  (site: {domain})\n\nWeb search evidence:\n{text}"}]
        obj = None
        for attempt in range(2):
            try:
                content = common.llm_chat(msgs, model=MODEL, json_mode=True, max_tokens=900)
                obj = _parse_json(content)
                if obj:
                    break
            except Exception:
                pass
        if obj:
            for k in DEFAULT:
                if k in obj and obj[k] not in (None, ""):
                    out[k] = obj[k]
        if not out["sources"]:
            out["sources"] = srcs
    out["_searched"] = bool(text.strip())
    cf.write_text(json.dumps(out))
    return {"company_id": cid, **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300, help="deep-enrich the top N by rank")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(common.ROOT / "ranked_output.csv") if (common.ROOT / "ranked_output.csv").exists() \
        else common.load_enriched()
    src = ranked.sort_values("rank").head(a.top) if "rank" in ranked else ranked.head(a.top)
    rows = src[["company_id", "company_name", "domain"]].to_dict("records")
    print(f"External enrichment for top {len(rows)} (DuckDuckGo, {a.workers} workers)...", flush=True)
    res, done = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(enrich_one, r) for r in rows]):
            res.append(f.result()); done += 1
            if done % 25 == 0 or done == len(rows):
                got = sum(1 for x in res if x.get("_searched"))
                print(f"  {done}/{len(rows)}  with-evidence={got}", flush=True)
    pd.DataFrame(res).to_csv(OUT, index=False)
    pe = sum(1 for x in res if x.get("pe_backed") == "yes")
    me = sum(1 for x in res if x.get("multi_entity") == "yes")
    print(f"DONE -> {OUT}  ({len(res)} rows; PE-backed={pe}, multi-entity={me})", flush=True)


if __name__ == "__main__":
    main()
