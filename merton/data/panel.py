"""Assemble the firm-month panel that everything downstream runs on.

One row is one firm in one month, carrying the three Merton inputs (market
value of equity, equity volatility, face value of debt), the risk-free rate,
accounting ratios for the benchmark model, and a forward-looking default flag.

Two rules govern the construction:

*No look-ahead.* An accounting figure enters a month only if it had already
been filed with the SEC by the end of that month. Market data enters only from
the trailing window. Nothing in a row could have been unknown to an observer
standing at that month-end.

*No quiet survivorship.* Months are kept right up to the bankruptcy filing.
A firm that defaults contributes exactly the distressed observations a credit
model needs to be judged on, and then stops.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from merton.data import sec
from merton.data.prices import monthly_equity
from merton.data.rates import risk_free_curve

log = logging.getLogger(__name__)

DEFAULT_START = "2016-09-30"   # the price source serves ten years; earlier months would carry no equity data
DEFAULT_HORIZON_MONTHS = 12

# Size ranking is taken from several historical dates rather than just the
# latest, so a firm that was large in 2016 and shrank is ranked on what it was.
FRAME_PERIODS = ["CY2016Q4I", "CY2019Q4I", "CY2022Q4I", "CY2025Q2I"]


MIN_ASSETS = 5e7    # $50m: below this a filer is usually a shell or pre-revenue


def select_universe(n_candidates: int = 2500, periods: list[str] | None = None) -> pd.DataFrame:
    """US filers ranked by total assets, with tickers attached where known.

    The XBRL ``frames`` API returns one concept across every filer at once, so
    the whole ranking costs a handful of requests rather than one per firm.

    This is a *candidate* pool, and it deliberately reaches well down the size
    distribution. Ranking by assets and stopping at the top few hundred yields
    a universe of mega-caps, which almost never default -- a default study
    built on it has nothing to predict. Banks and insurers cannot be filtered
    here because SIC codes are not in the frames payload; the caller drops them
    as it walks the pool.
    """
    periods = periods or FRAME_PERIODS
    assets: dict[int, float] = {}
    for period in periods:
        body = sec.session().get(sec.FRAMES_URL.format(tag="Assets", period=period), sec=True)
        if not body:
            log.warning("no frames data for %s", period)
            continue
        for row in body.get("data", []):
            cik, val = int(row["cik"]), float(row.get("val") or 0)
            if val > assets.get(cik, 0):
                assets[cik] = val
    if not assets:
        raise RuntimeError("SEC frames API returned no data; cannot build a universe")

    names = {}
    for period in periods:
        body = sec.session().get(sec.FRAMES_URL.format(tag="Assets", period=period), sec=True)
        for row in (body or {}).get("data", []):
            names.setdefault(int(row["cik"]), row.get("entityName", ""))

    ranked = (pd.Series(assets, name="assets").rename_axis("cik")
              .sort_values(ascending=False).reset_index())
    ranked = ranked[ranked["assets"] >= MIN_ASSETS]
    ranked["frames_name"] = ranked["cik"].map(names)

    # Left join, not inner: a filer missing from SEC's ticker file is usually a
    # delisted one, and those are the observations a default study most needs.
    # They get a ticker later, by name lookup against the price provider.
    tickers = sec.company_tickers().rename(columns={"name": "sec_name"})
    merged = ranked.merge(tickers, on="cik", how="left")
    merged["name"] = merged["sec_name"].fillna(merged["frames_name"])
    return merged.head(n_candidates).reset_index(drop=True)


def firm_profile(cik: int, start: str = DEFAULT_START,
                 skip_financials: bool = True) -> dict | None:
    """SIC code and bankruptcy date for one filer.

    The SIC check runs on the cheap first page and short-circuits before any
    archive page is fetched. Banks and insurers dominate the top of an
    assets-ranked universe and are excluded anyway, so paging through a
    mega-cap bank's twenty years of filings only to discard it is the single
    most expensive mistake available here.
    """
    sub = sec.submissions(cik)
    if not sub:
        return None

    try:
        sic = int(sub.get("sic") or 0)
    except (TypeError, ValueError):
        sic = 0

    profile = {
        "cik": cik,
        "name": sub.get("name", ""),
        "sic": sic,
        "sic_desc": sub.get("sicDescription", ""),
        "default_date": None,
    }
    if skip_financials and sic in sec.FINANCIAL_SIC:
        profile["excluded"] = "financial"
        return profile

    full = sec.submissions(cik, since=start)
    profile["default_date"] = sec.bankruptcy_date(full or sub)
    return profile


def _pit(blob, tags, dates, period="instant"):
    """Point-in-time series for a tag ladder, plus the tag that supplied it."""
    facts, tag = sec.first_available_in(blob, tags, period=period)
    return sec.as_of(facts, dates), tag


def firm_fundamentals(cik: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time balance-sheet and income-statement inputs for one filer.

    The whole fact set is fetched once and every tag ladder is resolved against
    it locally, so a firm costs one request no matter how many tags are tried.
    """
    blob = sec.company_facts(cik)
    out = pd.DataFrame(index=dates)
    tags_used = {}

    short_term, tags_used["debt_st"] = _pit(blob, sec.SHORT_TERM_DEBT_TAGS, dates)
    long_term, tags_used["debt_lt"] = _pit(blob, sec.LONG_TERM_DEBT_TAGS, dates)
    # A firm reporting only long-term debt genuinely has no current portion
    # tagged; treating that as zero is right. A firm reporting neither is
    # unusable, and falls out below when D is required to be positive.
    out["debt_st"] = short_term.fillna(0.0) if long_term.notna().any() else short_term
    out["debt_lt"] = long_term

    for field, tags in sec.BALANCE_SHEET_TAGS.items():
        out[field], tags_used[field] = _pit(blob, tags, dates)
    for field, tags in sec.FLOW_TAGS.items():
        out[field], tags_used[field] = _pit(blob, tags, dates, period="annual")

    shares, tags_used["shares"] = _pit(blob, sec.SHARES_TAGS, dates)
    out["shares"] = shares

    # Total liabilities is frequently untagged even when both sides of the
    # balance sheet are. Falling back to the identity keeps the firm rather
    # than dropping it from every accounting-based comparison.
    derived = out["liabilities_and_equity"].fillna(out["assets"]) - out["equity"]
    out["liabilities"] = out["liabilities"].fillna(derived)

    out.attrs["tags_used"] = tags_used
    return out


def firm_rows(profile: dict, ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Merge market and accounting data into monthly rows for one firm."""
    equity = monthly_equity(ticker)
    if equity.empty:
        return pd.DataFrame()

    equity = equity.reindex(dates)
    fundamentals = firm_fundamentals(profile["cik"], dates)

    frame = pd.concat([equity[["close", "sigma_E"]], fundamentals], axis=1)
    frame["E"] = frame["close"] * frame["shares"]
    frame["D"] = frame["debt_st"].fillna(0.0) + frame["debt_lt"].fillna(0.0)

    frame["cik"] = profile["cik"]
    frame["ticker"] = ticker
    frame["name"] = profile["name"]
    frame["sic"] = profile["sic"]
    frame["industry"] = profile["sic_desc"]
    frame["default_date"] = profile["default_date"]
    frame["tags_used"] = str(fundamentals.attrs.get("tags_used", {}))

    frame = frame[frame["E"].gt(0) & frame["D"].gt(0) & frame["sigma_E"].gt(0)]
    if frame.empty:
        return frame

    # Stop at the bankruptcy filing: equity after that point prices a claim on a
    # reorganization, not a going concern, and the model is not about that.
    default_date = profile["default_date"]
    if default_date is not None and pd.notna(default_date):
        frame = frame[frame.index <= default_date]

    return frame.reset_index(names="date")


def add_labels(panel: pd.DataFrame, horizon_months: int = DEFAULT_HORIZON_MONTHS) -> pd.DataFrame:
    """Flag each row with whether the firm filed for bankruptcy within the horizon."""
    panel = panel.copy()
    horizon = pd.DateOffset(months=horizon_months)
    default_date = pd.to_datetime(panel["default_date"])
    ahead = default_date.notna() & (default_date > panel["date"]) \
        & (default_date <= panel["date"] + horizon)
    panel[f"default_{horizon_months}m"] = ahead.astype(int)
    panel["months_to_default"] = ((default_date - panel["date"]).dt.days / 30.44).round(1)
    return panel


def add_accounting_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Altman Z-score components, the standard accounting-only benchmark."""
    panel = panel.copy()
    assets = panel["assets"].replace(0, np.nan)
    liabilities = panel["liabilities"].replace(0, np.nan)
    panel["wc_ta"] = (panel["current_assets"] - panel["current_liabilities"]) / assets
    panel["re_ta"] = panel["retained_earnings"] / assets
    panel["ebit_ta"] = panel["ebit"] / assets
    panel["mve_tl"] = panel["E"] / liabilities
    panel["sales_ta"] = panel["revenue"] / assets
    panel["altman_z"] = (1.2 * panel["wc_ta"] + 1.4 * panel["re_ta"] + 3.3 * panel["ebit_ta"]
                         + 0.6 * panel["mve_tl"] + 1.0 * panel["sales_ta"])
    panel["leverage_book"] = panel["D"] / assets
    panel["log_assets"] = np.log(assets)
    return panel


def attach_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Merge the 1-year Treasury yield onto each month-end."""
    curve = risk_free_curve().sort_index()
    monthly = curve.resample("ME").last().ffill()
    panel = panel.copy()
    panel["r"] = panel["date"].map(monthly)
    panel["r"] = panel["r"].ffill().bfill()
    return panel
