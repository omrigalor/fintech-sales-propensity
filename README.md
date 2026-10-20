# Fintech Sales Propensity

A go-to-market target-scoring system that turns a large, messy, enriched list of ~2,000
companies into a **ranked view of who to pursue and why**, delivered as a lightweight local
app a non-technical GTM teammate can use.

This is a **sanitized, public** version of a take-home project. The company is fictionalized
here as **"Meridian"** — a fintech selling **AR (accounts-receivable)** and **AP
(accounts-payable)** automation to service businesses. All real prospect/client names, the
real company identity, the input datasets, and pre-computed outputs have been removed. The
**algorithms and engineering are intact.**

> **The target, in one line.** Most of Meridian's customers are smaller service businesses,
> but a small number of middle-market accounts are worth many multiples of a typical deal.
> Ranking purely by *who converts* would surface small MSP-type shops and bury that prize. So
> the priority score deliberately blends **Fit × Win-likelihood × Value**, and the app lets a
> user dial between *"easiest to win now"* and *"biggest strategic accounts."*

---

## What it scores

Every company gets a **priority score** built from three transparent sub-scores, each stored
as a percentile so the app can re-blend live:

- **Fit** — is this the right *kind* of company? A rule-based ICP score from firmographics and
  enriched signals: has an AR/AP need, is a B2B service business, right size.
- **Win-likelihood** — a calibrated, cross-validated ML model trained on 100 labeled won/lost
  deals; transparent (per-company reasons) and reported honestly, because 100 labels is a
  small sample.
- **Value** — the middle-market sweet spot: rewards the prize accounts but peaks in the core
  mid-market and tapers for both tiny shops and borderline-enterprise accounts.

**Priority = geometric blend( Fit^0.45 · Win^0.20 · Value^0.35 )** — multiplied, not averaged,
so a weak score on any one dimension can't hide behind the other two. Each company is also
tagged with a **recommended lead product** (AR / AP / both), a descriptive GTM **segment** (used
only for routing, never as a scoring input), and a separate **"why-now" timing** score. Low
confidence records are flagged for human review.

## How the score is built (plain English)

1. **AI enrichment.** The structured columns provided are largely empty and occasionally wrong.
   A reasoning-grade LLM reads each company's website text and extracts clean,
   schema-validated signals — including a separate **AR-need** (do they struggle to get *paid*?)
   and **AP-need** (do they pay many *vendors*?). A **critic pass** re-checks those scores, and a
   **third-party web-evidence pass** adds signals a company won't state about itself (PE-backed,
   multi-entity).
2. **Website scraping (a second, independent signal).** Each company's live site is fetched to
   detect its tech stack (accounting / PSA / payment tools) and structural AR/AP cues (a "Pay
   Invoice" portal implies AR relevance; a "Vendors" page implies AP relevance).
3. **Win-likelihood (ML).** A calibrated, cross-validated model learns which firmographics
   predict a win. Transparent per-company reasons; cross-validated AUC reported as-is.
4. **Fit and Value** are transparent, rule-based scores from firmographics and enriched signals.
5. **Priority** blends the three; Win is deliberately a lighter tie-breaker because on 100
   labels the win model is honestly weak (CV-AUC ~0.64) and shouldn't dominate.

## Techniques & research grounding

Deliberately more than "ask an LLM for a score." What's actually implemented:

- **Pydantic-validated extraction** — every LLM read is parsed into a typed schema; malformed
  or out-of-range values are rejected, not silently trusted (`enrich.py`).
- **LLM-as-a-judge critic pass** — an independent second read re-scores AR/AP-need; on
  disagreement we lower confidence and flag, rather than average two answers together
  (`verify.py`).
- **Self-consistency** *(Wang et al., 2022)* — noisy AR/AP scores are sampled N times and
  aggregated by median; the spread across samples is itself an uncertainty signal
  (`consistency.py`).
- **Conformal prediction** *(Angelopoulos & Bates, 2021)* — distribution-free selective
  abstention flags low-confidence accounts with coverage guarantees, not a hand-tuned cutoff
  (`pipeline.py`).
- **Calibrated, cross-validated win model** — gradient boosting with Platt/sigmoid calibration
  *(Guo et al., 2017)*, scored by repeated 5×10 CV against constant and size-only baselines
  (`pipeline.py`).
- **Bootstrap rank-stability** — resample the labels, re-rank, and measure how often a
  top-50/100 account stays in the cohort — an honesty check for a small-sample model.
- **Matched-association robustness** — nearest-neighbour matching for suggestive (not causal)
  feature effects (`pipeline.py`).
- **Third-party web research** — an independent DuckDuckGo pass adds PE-backed / multi-entity
  evidence a company won't state about itself (`external.py`).

**Deliberately *not* used:** RAG, BM25, embeddings, or fine-tuning as the primary. This is a
**structured-scoring** problem, not retrieval or generation — those tools would add machinery
without lift. `finetune.py` documents that call honestly and only builds an optional SFT demo.

## Key modules

| File | Role |
|------|------|
| `code/common.py` | Paths, data loading, firmographic cleaning, entity resolution, LLM client |
| `code/features.py` | Feature engineering + confident-LLM firmographic overrides |
| `code/enrich.py` | LLM website-text enrichment → typed, Pydantic-validated GTM signals |
| `code/scrape.py` | Live homepage scraper → technographics + AR/AP structural signals |
| `code/external.py` | Third-party web-evidence pass (PE-backed / multi-entity / acquisitions) |
| `code/verify.py` | LLM-as-a-judge critic pass over the AR/AP extraction |
| `code/consistency.py` | Self-consistency sampling for AR/AP-need |
| `code/pipeline.py` | End-to-end scoring: Fit / Win / Value, blend, conformal, rank-stability, segments |
| `code/finetune.py` | Honest fine-tuning demonstration + SFT dataset builder |
| `code/llm_blurbs.py` | LLM wording of the model's reasons into rep-ready "why" + outreach angle |
| `code/make_architecture.py` | Renders the pipeline architecture diagram |
| `code/make_validation_sample.py` / `code/autofill_validation.py` | Stratified human-review sample + LLM auto-review |
| `app.py` | Streamlit GTM explorer: ranked call list, account detail, live re-weighting, business case |
| `code/tests/test_submission.py` | Acceptance tests (every company scored, ranks contiguous, app boots) |

## Data omitted

This public version ships **code only**. The input datasets
(`code/data/case_study_*.csv`), all pre-computed outputs and caches (`code/outputs/`), the
generated `ranked_output.csv`, the model artifacts, and the written deliverables are **not
included** — they contain real company and prospect data. As a result the pipeline and app
expect those inputs to exist locally; the algorithms are fully readable as-is, and running
end-to-end requires supplying your own equivalently-shaped data and an LLM API key
(see `code/.env.example`).

## LLM provider

Any OpenAI-compatible chat-completions provider works. Set `LLM_API_KEY` (and optionally
`LLM_API_URL`) in your environment or a local `.env`. The model ids `reasoning-pro` /
`reasoning-flash` are placeholders — map them to your provider's reasoning-grade and fast
models.

## Run (with your own data + key)

```bash
python -m venv .venv && source .venv/bin/activate       # optional
pip install -r code/requirements.txt
python code/enrich.py            # LLM enrichment (needs LLM_API_KEY)
python code/scrape.py            # live technographic scrape
python code/pipeline.py          # build the ranking → ranked_output.csv
streamlit run app.py             # explore
```

---

By Omri Galor
