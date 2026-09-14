"""Daily equity prices, and the market cap and realized volatility built on them.

Prices come from stockanalysis.com's public JSON endpoint, which covers
delisted tickers through their final trading day. That coverage is what keeps
the panel honest: a study of credit risk that silently drops firms once they
stop trading has thrown away exactly the observations it is trying to predict.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from merton.data.http import session

log = logging.getLogger(__name__)

HISTORY_URL = "https://stockanalysis.com/api/symbol/s/{symbol}/history?range={range}"

TRADING_DAYS = 252
VOL_WINDOW = TRADING_DAYS       # trailing 12 months of daily returns
MIN_VOL_OBS = 120               # refuse to annualize from a stub of history
VOL_FLOOR = 0.05                # guard against a dead-quote series


def symbol_variants(ticker: str) -> list[str]:
    """Spellings to try for one SEC ticker.

    EDGAR writes share classes with a hyphen (BRK-B) where price sources use a
    dot or nothing at all, so a literal lookup silently loses every dual-class
    firm in the universe.
    """
    base = ticker.strip().lower()
    variants = [base]
    if "-" in base:
        variants += [base.replace("-", "."), base.replace("-", "")]
    seen, out = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def daily_prices(symbol: str, range_: str = "10Y") -> pd.DataFrame:
    """Daily OHLC for one symbol, oldest first.

    Columns: close (raw, for market cap) and adj_close (for returns).
    Returns an empty frame when the symbol is unknown to the source.
    """
    body = None
    for variant in symbol_variants(symbol):
        body = session().get(HISTORY_URL.format(symbol=variant, range=range_))
        if body and body.get("data"):
            break
    if not body or not body.get("data"):
        return pd.DataFrame(columns=["close", "adj_close", "volume"])

    frame = pd.DataFrame(body["data"])
    frame["date"] = pd.to_datetime(frame["t"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    out = pd.DataFrame({
        "close": pd.to_numeric(frame.get("c"), errors="coerce"),
        "adj_close": pd.to_numeric(frame.get("a", frame.get("c")), errors="coerce"),
        "volume": pd.to_numeric(frame.get("v"), errors="coerce"),
    })
    return out.dropna(subset=["close", "adj_close"])


def realized_volatility(adj_close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Trailing annualized volatility of daily log returns.

    Uses the total-return series so a dividend does not register as a price
    drop. The window is backward-looking only -- at month t it uses returns up
    to and including t, never after.
    """
    returns = np.log(adj_close / adj_close.shift(1))
    # A halted or barely-traded name posts long runs of zero returns that would
    # otherwise read as near-zero risk; drop them before estimating.
    returns = returns.replace(0.0, np.nan)
    vol = returns.rolling(window, min_periods=MIN_VOL_OBS).std() * np.sqrt(TRADING_DAYS)
    return vol


def monthly_equity(symbol: str, range_: str = "10Y") -> pd.DataFrame:
    """Month-end close and trailing realized volatility for one symbol."""
    prices = daily_prices(symbol, range_)
    if prices.empty:
        return pd.DataFrame(columns=["close", "sigma_E", "last_trade"])

    vol = realized_volatility(prices["adj_close"])
    frame = pd.DataFrame({"close": prices["close"], "sigma_E": vol})
    monthly = frame.resample("ME").last()
    monthly["last_trade"] = prices.index.max()
    monthly = monthly[monthly["sigma_E"] >= VOL_FLOOR]
    # Month-ends after the final trade are an artifact of resampling a
    # delisted series forward; drop them.
    return monthly[monthly.index <= prices.index.max() + pd.Timedelta(days=31)]
