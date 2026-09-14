"""Risk-free rates and credit-spread benchmarks from FRED.

FRED's fredgraph CSV endpoint needs no API key, which keeps the project
reproducible by anyone who clones it. The series are also cached as CSVs under
``data/`` and read from there by default, so the analysis reproduces offline
and a FRED outage cannot change a published number.

A note on the benchmark choice: the ICE BofA option-adjusted spread indices
are the usual market yardstick, but FRED now serves only a trailing three-year
window of them, which is too short to cover this panel. Moody's seasoned
corporate yields relative to the 10-year Treasury go back decades, are the
long-run credit spread measure used throughout the literature, and cover the
whole sample.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from merton.data.http import session

log = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "fred"
DEFAULT_START = "1990-01-01"

RISK_FREE = "DGS1"       # 1-year constant-maturity Treasury
BAA_SPREAD = "BAA10Y"    # Moody's Baa corporate yield over the 10-year Treasury
AAA_SPREAD = "AAA10Y"    # Moody's Aaa corporate yield over the 10-year Treasury

SERIES = [RISK_FREE, BAA_SPREAD, AAA_SPREAD]


def _cache_path(series: str) -> Path:
    return CACHE_DIR / f"{series}.csv"


def fred_series(series: str, refresh: bool = False) -> pd.Series:
    """Daily FRED series as a float Series indexed by date, missing days dropped.

    Reads the committed CSV unless ``refresh`` is set, in which case it is
    re-downloaded and the cache rewritten.
    """
    path = _cache_path(series)
    if path.exists() and not refresh:
        text = path.read_text()
    else:
        body = session().get(FRED_CSV.format(series=series, start=DEFAULT_START))
        if body is None:
            raise RuntimeError(f"could not fetch FRED series {series}")
        text = body
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    frame = pd.read_csv(io.StringIO(text))
    date_col, value_col = frame.columns[0], frame.columns[1]
    frame[date_col] = pd.to_datetime(frame[date_col])
    values = pd.to_numeric(frame[value_col], errors="coerce")
    return pd.Series(values.values, index=frame[date_col], name=series).dropna()


def risk_free_curve(refresh: bool = False) -> pd.Series:
    """1-year Treasury yield as a decimal (FRED quotes percent)."""
    return fred_series(RISK_FREE, refresh) / 100.0


def credit_spread_benchmarks(refresh: bool = False) -> pd.DataFrame:
    """Corporate credit spreads as decimals, plus the classic Baa-Aaa spread."""
    baa = fred_series(BAA_SPREAD, refresh) / 100.0
    aaa = fred_series(AAA_SPREAD, refresh) / 100.0
    frame = pd.DataFrame({"baa_spread": baa, "aaa_spread": aaa})
    # Baa minus Aaa nets out the Treasury leg and the term premium, leaving a
    # cleaner read on compensation for credit risk alone.
    frame["baa_aaa"] = frame["baa_spread"] - frame["aaa_spread"]
    return frame


def refresh_all() -> None:
    """Re-download every series and rewrite the committed cache."""
    for series in SERIES:
        fred_series(series, refresh=True)
        log.info("refreshed %s", series)
