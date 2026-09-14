"""SEC EDGAR: the universe, point-in-time fundamentals, and default labels.

Three things come from EDGAR, all from primary sources and all free:

1. The registrant universe, with CIK/ticker/SIC.
2. Reported balance-sheet figures via the XBRL ``companyconcept`` API. Every
   fact carries the date it was *filed*, not just the period it covers, which
   is what makes a point-in-time panel possible: a debt number enters the panel
   on the day the market could first have seen it.
3. Default labels. A US issuer entering bankruptcy or receivership must file an
   8-K under Item 1.03 within four business days. Those filings are dated and
   machine-readable in the submissions index, so default dates are read off
   primary filings rather than transcribed from a list.
"""

from __future__ import annotations

import logging

import pandas as pd

from merton.data.http import session

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE = "https://data.sec.gov/submissions/{name}"
CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json"

BANKRUPTCY_ITEM = "1.03"   # 8-K Item 1.03 -- Bankruptcy or Receivership

# Banks and insurers are excluded throughout. Their leverage is a regulatory
# construct rather than a default barrier, and deposits are not debt in the
# sense the model means, so a structural model on a bank measures the wrong
# thing. SIC 6000-6799 is the standard exclusion.
FINANCIAL_SIC = range(6000, 6800)

# Tag ladders. Filers are inconsistent about which us-gaap concept carries a
# given balance-sheet line, so each quantity is resolved by trying tags in
# order of specificity and recording which one supplied the number.
SHORT_TERM_DEBT_TAGS = [
    "DebtCurrent",
    "LongTermDebtCurrent",
    "ShortTermBorrowings",
    "OtherShortTermBorrowings",
    "NotesPayableCurrent",
]
LONG_TERM_DEBT_TAGS = [
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebt",
    "LongTermNotesPayable",
]
BALANCE_SHEET_TAGS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    # Many filers never tag total liabilities directly; these two allow it to
    # be derived from the balance-sheet identity instead of dropping the firm.
    "liabilities_and_equity": ["LiabilitiesAndStockholdersEquity"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}

# Income-statement items, reported over a period rather than at an instant.
FLOW_TAGS = {
    "ebit": ["OperatingIncomeLoss",
             "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
             "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
                "SalesRevenueGoodsNet",
                "SalesRevenueServicesNet"],
}
SHARES_TAGS = [("dei", "EntityCommonStockSharesOutstanding"),
               ("dei", "EntityCommonStockholdersEquitySharesOutstanding")]


def company_tickers() -> pd.DataFrame:
    """Every SEC registrant that currently maps to a listed ticker."""
    body = session().get(TICKERS_URL, sec=True)
    if body is None:
        raise RuntimeError("could not fetch SEC company_tickers.json")
    rows = [{"cik": int(v["cik_str"]), "ticker": v["ticker"].strip().upper(),
             "name": v["title"]} for v in body.values()]
    return pd.DataFrame(rows).drop_duplicates("cik").reset_index(drop=True)


def submissions(cik: int, since: str | None = None) -> dict | None:
    """Filing index for one filer, optionally extended back to a start date.

    ``filings.recent`` holds only the most recent ~1000 filings; for a filer
    that files constantly that can be under three years, so older filings live
    on archive pages. Those pages are large and a mega-cap has many of them, so
    only the pages whose own date range reaches into the window of interest are
    fetched -- the index states each page's span up front, which turns a dozen
    multi-megabyte downloads into none for most filers.
    """
    body = session().get(SUBMISSIONS_URL.format(cik=cik), sec=True)
    if body is None or since is None:
        return body

    recent = body.get("filings", {}).get("recent", {})
    dates = recent.get("filingDate") or []
    if dates and min(dates) <= since:
        return body        # the first page already reaches back far enough

    merged = {k: list(v) for k, v in recent.items()}
    for page in body.get("filings", {}).get("files", []):
        if (page.get("filingTo") or "9999") < since:
            continue       # entirely before the window of interest
        extra = session().get(SUBMISSIONS_PAGE.format(name=page["name"]), sec=True)
        if not extra:
            continue
        for key, values in extra.items():
            merged.setdefault(key, []).extend(values)

    body = dict(body)
    body["filings"] = {"recent": merged}
    return body


def filing_frame(sub: dict) -> pd.DataFrame:
    """Filing index as a frame with form, date and 8-K item codes."""
    recent = sub.get("filings", {}).get("recent", {})
    if not recent.get("form"):
        return pd.DataFrame(columns=["form", "filingDate", "items"])
    n = len(recent["form"])
    frame = pd.DataFrame({
        "form": recent["form"],
        "filingDate": recent["filingDate"],
        "items": recent.get("items", [""] * n),
    })
    frame["filingDate"] = pd.to_datetime(frame["filingDate"], errors="coerce")
    return frame.dropna(subset=["filingDate"])


def bankruptcy_date(sub: dict) -> pd.Timestamp | None:
    """Date of the filer's first Item 1.03 8-K, or None if it never filed one.

    A firm can file more than once (Chapter 11 twice, or an amended 8-K); the
    first is the one that matters for a forward-looking default label.
    """
    frame = filing_frame(sub)
    if frame.empty:
        return None
    is_8k = frame["form"].str.startswith("8-K")
    has_item = frame["items"].fillna("").str.contains(BANKRUPTCY_ITEM, regex=False)
    hits = frame.loc[is_8k & has_item, "filingDate"]
    return None if hits.empty else hits.min()


def company_facts(cik: int) -> dict | None:
    """Every XBRL fact a filer has reported, in one request.

    The per-concept endpoint needs a round trip per tag, and the tag ladders
    here try a dozen or more per firm -- most of which 404, because filers tag
    the same line item differently. Pulling the whole fact set once and
    resolving the ladders locally turns roughly twenty requests per firm into
    one, which is the difference between a build that takes hours and one that
    takes minutes.
    """
    return session().get(FACTS_URL.format(cik=cik), sec=True)


def _facts_to_frame(units: dict, tag: str, period: str) -> pd.DataFrame:
    """Shared normalization for both the per-concept and whole-firm paths."""
    empty = pd.DataFrame(columns=["end", "filed", "val"])
    key = "USD" if "USD" in units else ("shares" if "shares" in units else None)
    if key is None:
        return empty

    frame = pd.DataFrame(units[key])
    if not {"end", "filed", "val"}.issubset(frame.columns):
        return empty

    has_start = "start" in frame.columns and frame["start"].notna().any()
    if period == "instant":
        if has_start:
            frame = frame[frame["start"].isna()]
    else:
        if not has_start:
            return empty
        span = (pd.to_datetime(frame["end"], errors="coerce")
                - pd.to_datetime(frame["start"], errors="coerce")).dt.days
        frame = frame[span.between(330, 400)]   # trailing-year facts only

    frame = frame.copy()
    frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
    frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
    frame = frame.dropna(subset=["end", "filed", "val"])
    if frame.empty:
        return empty

    # One row per period end, keeping the latest filing of it: a restatement
    # supersedes the original, but only from the date it was itself filed.
    frame = frame.sort_values(["end", "filed"]).drop_duplicates("end", keep="last")
    return frame[["end", "filed", "val"]].sort_values("filed").reset_index(drop=True)


def facts_concept(blob: dict, tag: str, taxonomy: str = "us-gaap",
                  period: str = "instant") -> pd.DataFrame:
    """Pull one concept out of an already-fetched companyfacts blob."""
    units = ((blob or {}).get("facts", {}).get(taxonomy, {}).get(tag, {}) or {}).get("units")
    if not units:
        return pd.DataFrame(columns=["end", "filed", "val"])
    return _facts_to_frame(units, tag, period)


def first_available_in(blob: dict, tags: list, taxonomy: str = "us-gaap",
                       period: str = "instant"):
    """Walk a tag ladder against a companyfacts blob, most specific first."""
    for tag in tags:
        tax, name = tag if isinstance(tag, tuple) else (taxonomy, tag)
        frame = facts_concept(blob, name, tax, period)
        if not frame.empty:
            return frame, name
    return pd.DataFrame(columns=["end", "filed", "val"]), None


def concept_facts(cik: int, tag: str, taxonomy: str = "us-gaap",
                  period: str = "instant") -> pd.DataFrame:
    """One XBRL concept for one filer, as (period_end, filed, value).

    ``period`` selects which facts to keep. Balance-sheet items are *instant*
    facts (a stock at a point in time); income-statement items are *duration*
    facts, and only the annual ones are kept so that a quarter's revenue is
    never compared against a year's. Mixing the two would silently corrupt
    every ratio built on them.

    This is the single-concept endpoint, kept for ad-hoc lookups; the panel
    build goes through :func:`company_facts` instead.
    """
    body = session().get(CONCEPT_URL.format(cik=cik, taxonomy=taxonomy, tag=tag), sec=True)
    if not body or "units" not in body:
        return pd.DataFrame(columns=["end", "filed", "val"])
    return _facts_to_frame(body["units"], tag, period)


def first_available(cik: int, tags: list, taxonomy: str = "us-gaap",
                    period: str = "instant"):
    """Walk a tag ladder and return (facts, tag_used) for the first tag that hits.

    Filers are inconsistent about which concept carries a line item, so the
    ladder is ordered most-specific first and the tag that supplied the number
    is reported back for auditability.
    """
    for tag in tags:
        tax, name = tag if isinstance(tag, tuple) else (taxonomy, tag)
        facts = concept_facts(cik, name, tax, period=period)
        if not facts.empty:
            return facts, name
    return pd.DataFrame(columns=["end", "filed", "val"]), None


def as_of(facts: pd.DataFrame, dates: pd.DatetimeIndex, max_staleness_days: int = 450):
    """Point-in-time lookup: the latest value *filed on or before* each date.

    This is the guard against look-ahead. A figure from a 10-K covering December
    is not knowable in January -- it becomes usable only once the 10-K is filed,
    typically in February. Values older than ``max_staleness_days`` are dropped
    so a delinquent filer does not carry a stale barrier forever.
    """
    if facts.empty:
        return pd.Series(float("nan"), index=dates)

    facts = facts.sort_values("filed")
    idx = facts["filed"].searchsorted(dates, side="right") - 1
    values = pd.Series(
        [facts["val"].iloc[i] if i >= 0 else float("nan") for i in idx],
        index=dates, dtype=float,
    )
    filed = pd.Series(
        [facts["filed"].iloc[i] if i >= 0 else pd.NaT for i in idx], index=dates
    )
    stale = (pd.Series(dates, index=dates) - filed).dt.days > max_staleness_days
    return values.mask(stale)
