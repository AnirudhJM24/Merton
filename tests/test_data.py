"""Data-layer logic, all offline. No test here touches the network."""

import numpy as np
import pandas as pd
import pytest

from merton.data.panel import add_accounting_ratios, add_labels
from merton.data.prices import realized_volatility, symbol_variants
from merton.data.sec import as_of, bankruptcy_date, filing_frame
from merton.data.symbols import name_similarity, normalize_name


# --- point-in-time discipline ------------------------------------------------

def _facts(rows):
    frame = pd.DataFrame(rows, columns=["end", "filed", "val"])
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    return frame


def test_as_of_uses_filing_date_not_period_end():
    """The defining no-look-ahead test.

    December's balance sheet covers 31 December but is not filed until
    February. An observer standing on 31 January cannot know it, so the panel
    must still be carrying September's figure on that date.
    """
    facts = _facts([
        ("2023-09-30", "2023-11-02", 100.0),
        ("2023-12-31", "2024-02-15", 250.0),
    ])
    dates = pd.to_datetime(["2023-12-31", "2024-01-31", "2024-02-29"])
    values = as_of(facts, dates)

    assert values.iloc[0] == 100.0     # December's own month-end: not filed yet
    assert values.iloc[1] == 100.0     # January: still not filed
    assert values.iloc[2] == 250.0     # February: now public


def test_as_of_is_empty_before_the_first_filing():
    facts = _facts([("2023-12-31", "2024-02-15", 250.0)])
    values = as_of(facts, pd.to_datetime(["2020-06-30", "2024-03-31"]))
    assert np.isnan(values.iloc[0])
    assert values.iloc[1] == 250.0


def test_as_of_drops_stale_figures():
    """A filer that goes dark must not carry its last balance sheet forever."""
    facts = _facts([("2016-12-31", "2017-02-15", 10.0)])
    values = as_of(facts, pd.to_datetime(["2017-06-30", "2021-06-30"]),
                   max_staleness_days=450)
    assert values.iloc[0] == 10.0
    assert np.isnan(values.iloc[1])


def test_as_of_prefers_the_latest_filing():
    facts = _facts([
        ("2023-03-31", "2023-05-01", 10.0),
        ("2023-06-30", "2023-08-01", 20.0),
    ])
    values = as_of(facts, pd.to_datetime(["2023-09-30"]))
    assert values.iloc[0] == 20.0


def test_as_of_on_empty_facts():
    dates = pd.to_datetime(["2023-01-31"])
    assert as_of(pd.DataFrame(columns=["end", "filed", "val"]), dates).isna().all()


# --- default labels ----------------------------------------------------------

def _submissions(forms, dates, items):
    return {"filings": {"recent": {"form": forms, "filingDate": dates, "items": items}}}


def test_bankruptcy_date_reads_item_103():
    sub = _submissions(
        ["10-Q", "8-K", "8-K"],
        ["2020-04-30", "2020-05-26", "2020-06-01"],
        ["", "1.01,1.03,2.04", "2.02"],
    )
    assert bankruptcy_date(sub) == pd.Timestamp("2020-05-26")


def test_bankruptcy_date_takes_the_first_filing():
    """A firm can file twice; the forward-looking label needs the earlier one."""
    sub = _submissions(["8-K", "8-K"], ["2021-06-16", "2020-05-26"],
                       ["1.03,9.01", "1.03"])
    assert bankruptcy_date(sub) == pd.Timestamp("2020-05-26")


def test_bankruptcy_date_ignores_other_items():
    """Item 1.01 is a material agreement, not a bankruptcy."""
    sub = _submissions(["8-K"], ["2020-05-26"], ["1.01,2.03"])
    assert bankruptcy_date(sub) is None


def test_bankruptcy_date_ignores_non_8k_forms():
    sub = _submissions(["10-K"], ["2020-05-26"], ["1.03"])
    assert bankruptcy_date(sub) is None


def test_filing_frame_survives_missing_items():
    sub = {"filings": {"recent": {"form": ["10-K"], "filingDate": ["2020-01-01"]}}}
    assert len(filing_frame(sub)) == 1


# --- labels and ratios -------------------------------------------------------

def test_labels_are_forward_looking_only():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2019-12-31", "2020-01-31", "2020-05-31"]),
        "default_date": [pd.Timestamp("2020-05-26")] * 3,
    })
    labelled = add_labels(panel, horizon_months=12)
    # Twelve months ahead of the filing: flagged. On a date after it: not,
    # because there is no longer a future default to predict.
    assert labelled["default_12m"].tolist() == [1, 1, 0]


def test_labels_ignore_defaults_beyond_the_horizon():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2018-01-31"]),
        "default_date": [pd.Timestamp("2020-05-26")],
    })
    assert add_labels(panel, horizon_months=12)["default_12m"].tolist() == [0]


def test_survivors_are_never_flagged():
    panel = pd.DataFrame({"date": pd.to_datetime(["2020-01-31"]), "default_date": [pd.NaT]})
    assert add_labels(panel)["default_12m"].tolist() == [0]


def test_altman_z_matches_the_published_weights():
    panel = pd.DataFrame({
        "current_assets": [500.0], "current_liabilities": [200.0],
        "retained_earnings": [300.0], "ebit": [120.0], "revenue": [900.0],
        "assets": [1000.0], "liabilities": [600.0], "E": [1200.0], "D": [400.0],
    })
    row = add_accounting_ratios(panel).iloc[0]
    expected = (1.2 * 0.3) + (1.4 * 0.3) + (3.3 * 0.12) + (0.6 * 2.0) + (1.0 * 0.9)
    assert row["altman_z"] == pytest.approx(expected)


def test_ratios_do_not_divide_by_zero():
    panel = pd.DataFrame({
        "current_assets": [1.0], "current_liabilities": [1.0], "retained_earnings": [1.0],
        "ebit": [1.0], "revenue": [1.0], "assets": [0.0], "liabilities": [0.0],
        "E": [1.0], "D": [1.0],
    })
    assert np.isnan(add_accounting_ratios(panel).iloc[0]["altman_z"])


# --- prices ------------------------------------------------------------------

def test_realized_volatility_recovers_a_known_sigma():
    rng = np.random.default_rng(0)
    sigma, n = 0.40, 2000
    steps = rng.normal(0, sigma / np.sqrt(252), n)
    prices = pd.Series(100 * np.exp(np.cumsum(steps)),
                       index=pd.bdate_range("2015-01-01", periods=n))
    estimate = realized_volatility(prices).dropna().mean()
    assert estimate == pytest.approx(sigma, rel=0.12)


def test_realized_volatility_needs_enough_history():
    prices = pd.Series(np.linspace(100, 110, 30),
                       index=pd.bdate_range("2020-01-01", periods=30))
    assert realized_volatility(prices).isna().all()


def test_symbol_variants_covers_share_classes():
    assert symbol_variants("BRK-B") == ["brk-b", "brk.b", "brkb"]
    assert symbol_variants("AAPL") == ["aapl"]


# --- name matching -----------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("HERTZ GLOBAL HOLDINGS, INC", "HERTZ GLOBAL"),
    ("WELLS FARGO & COMPANY/MN", "WELLS FARGO"),
    ("Party City Holdco Inc.", "PARTY CITY HOLDCO"),
])
def test_normalize_name_strips_legal_form(raw, expected):
    assert normalize_name(raw) == expected


def test_name_similarity_separates_real_matches_from_coincidences():
    assert name_similarity("RITE AID CORP", "Rite Aid Corporation") > 0.9
    assert name_similarity("BIG LOTS INC", "BIG Shopping Centers Ltd") < 0.72
    assert name_similarity("YELLOW CORP", "Yellow Cake PLC") < 0.72
    assert name_similarity("SUNRUN INC", "Sunrun Neptune Holdings") < 0.72


def test_short_prefix_does_not_count_as_a_match():
    """The bar to clear: one firm's prices must never land on another's books."""
    assert name_similarity("DELTA APPAREL", "Delta Air Lines") < 0.72
    assert name_similarity("WEWORK", "WeWork Inc.") > 0.9
