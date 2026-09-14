"""Risk-free rates and credit-spread benchmarks from FRED.

FRED's fredgraph CSV endpoint needs no API key, which keeps the project
reproducible by anyone who clones it.
"""

from __future__ import annotations

import io

import pandas as pd

from merton.data.http import session

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# 1-year constant-maturity Treasury is the natural discount rate for a 1-year
# Merton horizon. The ICE BofA option-adjusted spreads are the market yardstick
# the model's own spreads get compared against.
RISK_FREE = "DGS1"
HY_OAS = "BAMLH0A0HYM2"      # ICE BofA US High Yield index OAS
BBB_OAS = "BAMLC0A4CBBB"     # ICE BofA BBB corporate index OAS


def fred_series(series: str) -> pd.Series:
    """Daily FRED series as a float Series indexed by date, missing days dropped."""
    body = session().get(FRED_CSV.format(series=series))
    if body is None:
        raise RuntimeError(f"could not fetch FRED series {series}")
    frame = pd.read_csv(io.StringIO(body))
    date_col, value_col = frame.columns[0], frame.columns[1]
    frame[date_col] = pd.to_datetime(frame[date_col])
    values = pd.to_numeric(frame[value_col], errors="coerce")
    out = pd.Series(values.values, index=frame[date_col], name=series).dropna()
    return out


def risk_free_curve() -> pd.Series:
    """1-year Treasury yield as a decimal (FRED quotes percent)."""
    return fred_series(RISK_FREE) / 100.0


def credit_spread_benchmarks() -> pd.DataFrame:
    """HY and BBB option-adjusted spreads as decimals."""
    return pd.DataFrame({
        "hy_oas": fred_series(HY_OAS) / 100.0,
        "bbb_oas": fred_series(BBB_OAS) / 100.0,
    })
