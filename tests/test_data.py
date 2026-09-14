"""Data-layer logic, all offline. No test here touches the network."""

import numpy as np
import pandas as pd
import pytest

from merton.data.panel import add_accounting_ratios, add_labels
from merton.data.prices import realized_volatility, symbol_variants
from merton.data.sec import as_of, bankruptcy_date, filing_frame
from merton.data.symbols import MATCH_THRESHOLD, name_similarity, normalize_name


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
        "cik": 1,
        "date": pd.to_datetime(["2019-12-31", "2020-01-31", "2020-05-31"]),
        "close": [10.0, 10.0, 10.0],
        "default_date": [pd.Timestamp("2020-05-26")] * 3,
    })
    labelled = add_labels(panel, horizon_months=12)
    # Twelve months ahead of the filing: flagged. On a date after it: not,
    # because there is no longer a future default to predict.
    assert labelled["default_12m"].tolist() == [1, 1, 0]


def test_labels_ignore_defaults_beyond_the_horizon():
    panel = pd.DataFrame({
        "cik": 1,
        "date": pd.to_datetime(["2018-01-31"]),
        "close": [10.0],
        "default_date": [pd.Timestamp("2020-05-26")],
    })
    assert add_labels(panel, horizon_months=12)["default_12m"].tolist() == [0]


def test_survivors_are_never_flagged():
    panel = pd.DataFrame({
        "cik": 1,
        "date": pd.to_datetime(["2020-01-31"]),
        "close": [10.0],
        "default_date": [pd.NaT],
    })
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


@pytest.mark.parametrize("sec_name,provider_name", [
    ("ENERPLUS Corp", "Energous Corporation"),          # matched WATT at 0.75
    ("SOUTHWESTERN ENERGY CO", "NorthWestern Energy Group"),
    ("RED HAT INC", "Red Cat Holdings, Inc."),
    ("HAWAIIAN HOLDINGS INC", "First Hawaiian, Inc."),
    ("WEB.COM GROUP, INC.", "Weber Inc."),
    ("CONTANGO OIL & GAS CO", "Contango ORE, Inc."),
    ("TUCSON ELECTRIC POWER CO", "Korea Electric Power Corporation"),
])
def test_near_miss_names_are_rejected(sec_name, provider_name):
    """Every one of these slipped through at a lower threshold.

    A false match is not a missing row -- it silently attaches one firm's price
    history to another firm's balance sheet and yields a plausible-looking
    distance-to-default for a company that does not exist.
    """
    assert name_similarity(sec_name, provider_name) < MATCH_THRESHOLD


def test_name_similarity_separates_real_matches_from_coincidences():
    for sec_name, provider_name in [
        ("RITE AID CORP", "Rite Aid Corporation"),
        ("WEWORK INC.", "WeWork Inc."),
        ("REVLON INC", "Revlon, Inc."),
        ("PARTY CITY HOLDCO INC.", "Party City Holdco Inc."),
        ("AMYRIS, INC.", "Amyris, Inc."),
    ]:
        assert name_similarity(sec_name, provider_name) >= MATCH_THRESHOLD
    assert name_similarity("BIG LOTS INC", "BIG Shopping Centers Ltd") < 0.72
    assert name_similarity("YELLOW CORP", "Yellow Cake PLC") < 0.72
    assert name_similarity("SUNRUN INC", "Sunrun Neptune Holdings") < 0.72


def test_short_prefix_does_not_count_as_a_match():
    """The bar to clear: one firm's prices must never land on another's books."""
    assert name_similarity("DELTA APPAREL", "Delta Air Lines") < 0.72
    assert name_similarity("WEWORK", "WeWork Inc.") > 0.9


# --- companyfacts extraction -------------------------------------------------

def _blob(tag, unit, rows, taxonomy="us-gaap"):
    return {"facts": {taxonomy: {tag: {"units": {unit: rows}}}}}


def test_facts_concept_keeps_instant_facts():
    from merton.data.sec import facts_concept

    blob = _blob("Assets", "USD", [
        {"end": "2023-03-31", "filed": "2023-05-01", "val": 100},
        {"end": "2023-06-30", "filed": "2023-08-01", "val": 110},
    ])
    frame = facts_concept(blob, "Assets")
    assert len(frame) == 2
    assert frame["val"].tolist() == [100, 110]


def test_facts_concept_separates_flows_from_stocks():
    """A quarter's revenue must never be read as a year's.

    Filers report both spans under the same tag, so the period filter is the
    only thing standing between a correct ratio and one that is four times
    too small.
    """
    from merton.data.sec import facts_concept

    blob = _blob("Revenues", "USD", [
        {"start": "2023-01-01", "end": "2023-03-31", "filed": "2023-05-01", "val": 25},
        {"start": "2022-07-01", "end": "2023-06-30", "filed": "2023-08-01", "val": 100},
    ])
    annual = facts_concept(blob, "Revenues", period="annual")
    assert annual["val"].tolist() == [100]
    assert facts_concept(blob, "Revenues", period="instant").empty


def test_facts_concept_prefers_the_latest_filing_of_a_period():
    """A restatement supersedes the original -- but only from its own filing date."""
    from merton.data.sec import facts_concept

    blob = _blob("Assets", "USD", [
        {"end": "2023-03-31", "filed": "2023-05-01", "val": 100},
        {"end": "2023-03-31", "filed": "2023-11-01", "val": 95},
    ])
    frame = facts_concept(blob, "Assets")
    assert len(frame) == 1
    assert frame["val"].iloc[0] == 95
    assert frame["filed"].iloc[0] == pd.Timestamp("2023-11-01")


def test_facts_concept_is_empty_for_an_untagged_concept():
    from merton.data.sec import facts_concept

    assert facts_concept(_blob("Assets", "USD", []), "Liabilities").empty
    assert facts_concept(None, "Assets").empty
    assert facts_concept({}, "Assets").empty


def test_tag_ladder_takes_the_first_tag_that_reports():
    from merton.data.sec import first_available_in

    blob = {"facts": {"us-gaap": {
        "LongTermDebt": {"units": {"USD": [
            {"end": "2023-03-31", "filed": "2023-05-01", "val": 50}]}},
    }}}
    frame, tag = first_available_in(
        blob, ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermNotesPayable"])
    assert tag == "LongTermDebt"
    assert frame["val"].tolist() == [50]


def test_tag_ladder_reports_nothing_when_no_tag_hits():
    from merton.data.sec import first_available_in

    frame, tag = first_available_in({"facts": {}}, ["Assets", "Liabilities"])
    assert tag is None
    assert frame.empty


def test_tag_ladder_crosses_taxonomies():
    """Shares outstanding live in the dei taxonomy, not us-gaap."""
    from merton.data.sec import first_available_in

    blob = _blob("EntityCommonStockSharesOutstanding", "shares",
                 [{"end": "2023-03-31", "filed": "2023-05-01", "val": 1000}],
                 taxonomy="dei")
    frame, tag = first_available_in(
        blob, [("dei", "EntityCommonStockSharesOutstanding")])
    assert tag == "EntityCommonStockSharesOutstanding"
    assert frame["val"].tolist() == [1000]


# --- the distress label ------------------------------------------------------

def _price_panel(closes, default_date=pd.NaT, start="2020-01-31"):
    dates = pd.date_range(start, periods=len(closes), freq="ME")
    return pd.DataFrame({
        "cik": 1, "date": dates, "close": closes, "default_date": default_date,
    })


def test_distress_flags_an_equity_wipeout():
    """A 90% fall within the horizon is the event, whether or not a court is involved."""
    closes = [100.0] * 6 + [5.0] + [5.0] * 11
    labelled = add_labels(_price_panel(closes), horizon_months=12)
    assert labelled["distress_12m"].iloc[0] == 1      # the collapse is 6 months ahead
    assert labelled["distress_12m"].iloc[5] == 1


def test_distress_ignores_a_fall_beyond_the_horizon():
    closes = [100.0] * 20 + [5.0] * 6
    labelled = add_labels(_price_panel(closes), horizon_months=12)
    assert labelled["distress_12m"].iloc[0] == 0      # collapse is 20 months out


def test_distress_ignores_a_merely_bad_year():
    closes = [100.0] * 6 + [55.0] * 12
    labelled = add_labels(_price_panel(closes), horizon_months=12)
    assert (labelled["distress_12m"].dropna() == 0).all()


def test_distress_is_censored_not_zero_at_the_end_of_the_data():
    """The last year of any firm's data cannot be scored: the event may simply
    not have happened yet. Calling that a zero would invent healthy firms."""
    labelled = add_labels(_price_panel([100.0] * 18), horizon_months=12)
    assert labelled["distress_12m"].iloc[:6].notna().all()
    assert labelled["distress_12m"].iloc[-11:].isna().all()


def test_distress_includes_the_bankruptcy_filing():
    """A filing counts even when the share price has not yet collapsed."""
    labelled = add_labels(
        _price_panel([100.0] * 18, default_date=pd.Timestamp("2020-07-31")),
        horizon_months=12)
    assert labelled["distress_12m"].iloc[0] == 1
    assert labelled["default_12m"].iloc[0] == 1


def test_labels_are_computed_per_firm():
    """One firm's collapse must not label another firm's months."""
    healthy = _price_panel([100.0] * 18).assign(cik=1)
    doomed = _price_panel([100.0] * 6 + [2.0] * 12).assign(cik=2)
    labelled = add_labels(pd.concat([healthy, doomed], ignore_index=True),
                          horizon_months=12)
    assert (labelled.loc[labelled["cik"] == 1, "distress_12m"].dropna() == 0).all()
    assert labelled.loc[labelled["cik"] == 2, "distress_12m"].iloc[0] == 1
