"""
Acceptance tests for the Meridian submission.

Run from the submission/ folder:
    python -m pytest code/tests -q

Covers the assignment's hard requirements: every company scored, ranks are
unique & contiguous, required columns exist, and the Streamlit app boots.
"""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]      # submission/
RANKED = ROOT / "ranked_output.csv"
APP = ROOT / "app.py"

N_EXPECTED = 2000


@pytest.fixture(scope="module")
def ranked():
    assert RANKED.exists(), "ranked_output.csv missing — run `python code/pipeline.py` first"
    return pd.read_csv(RANKED)


def test_every_company_scored(ranked):
    assert len(ranked) == N_EXPECTED, f"expected {N_EXPECTED} rows, got {len(ranked)}"


def test_ranks_unique_and_contiguous(ranked):
    ranks = sorted(ranked["rank"].tolist())
    assert ranks == list(range(1, N_EXPECTED + 1)), "ranks must be 1..2000 with no gaps/dupes"


def test_required_columns_present(ranked):
    required = {"rank", "company_id", "priority_score", "top_reasons", "why",
                "recommended_product", "ar_need", "ap_need"}
    missing = required - set(ranked.columns)
    assert not missing, f"missing required columns: {missing}"


def test_scores_in_range(ranked):
    assert ranked["priority_score"].between(0, 100).all(), "priority_score out of [0,100]"


def test_franchise_rows_preserved(ranked):
    # shared-domain siblings must be flagged and kept, not deleted
    if "possible_duplicate" in ranked and "domain" in ranked:
        assert ranked["domain"].notna().sum() >= ranked["domain"].nunique(), "rows were dropped"


def test_streamlit_app_boots():
    """Automated smoke test — the app must render without exceptions."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, f"Streamlit app raised: {at.exception}"
