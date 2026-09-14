"""Does the model move when the credit market moves?

A cross-sectional ranking can look fine while the model is blind to the credit
cycle. Aggregating the panel month by month and setting it against traded
corporate spreads tests the other axis: when the market repriced credit in
March 2020, did model-implied risk move with it, and by how much?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from merton.data.rates import credit_spread_benchmarks

DISTRESS_DD = 2.0      # a conventional "within two standard deviations" cut

# Market yardsticks the model aggregates get compared against.
BENCHMARKS = ("baa_spread", "aaa_spread", "baa_aaa")


def aggregate_by_month(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly cross-sectional summary of the fitted panel."""
    fitted = panel[panel["converged"]].copy()
    grouped = fitted.groupby("date")
    out = pd.DataFrame({
        "n_firms": grouped["cik"].nunique(),
        "median_dd": grouped["dd"].median(),
        "p10_dd": grouped["dd"].quantile(0.10),
        "median_spread": grouped["spread"].median(),
        "mean_asset_vol": grouped["sigma_V"].mean(),
        "median_leverage": grouped["leverage"].median(),
        "share_distressed": grouped["dd"].apply(lambda s: float((s < DISTRESS_DD).mean())),
    })
    return out


def with_market_spreads(monthly: pd.DataFrame) -> pd.DataFrame:
    """Attach month-end Moody's corporate spreads over the 10-year Treasury."""
    benchmarks = credit_spread_benchmarks().resample("ME").last()
    return monthly.join(benchmarks, how="left")


def cycle_correlations(monthly: pd.DataFrame) -> pd.DataFrame:
    """Correlation of model aggregates with market spreads, in levels and changes.

    Levels correlate almost by construction when both series trend; the change
    correlation is the one that says whether the model tracks repricing rather
    than merely sharing a trend.
    """
    cols = [c for c in ("median_dd", "p10_dd", "share_distressed", "median_spread",
                        "mean_asset_vol") if c in monthly]
    rows = []
    for benchmark in BENCHMARKS:
        if benchmark not in monthly:
            continue
        frame = monthly[cols + [benchmark]].dropna()
        if len(frame) < 12:
            continue
        changes = frame.diff().dropna()
        for col in cols:
            rows.append({
                "model_measure": col,
                "benchmark": benchmark,
                "corr_levels": float(frame[col].corr(frame[benchmark])),
                "corr_changes": float(changes[col].corr(changes[benchmark])),
                "n_months": len(frame),
            })
    return pd.DataFrame(rows)


def stress_window(monthly: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice a named episode, e.g. the COVID repricing of early 2020."""
    return monthly.loc[start:end]


def spread_gap(panel: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """How far the model's spreads sit below traded spreads, month by month.

    Reported as a ratio as well as a difference: the model's shortfall on
    investment-grade names is a multiple, not a constant number of basis points.
    """
    frame = monthly.dropna(subset=["median_spread"]).copy()
    for benchmark in BENCHMARKS:
        if benchmark in frame:
            frame[f"gap_{benchmark}_bp"] = 1e4 * (frame[benchmark] - frame["median_spread"])
            frame[f"ratio_{benchmark}"] = frame[benchmark] / frame["median_spread"].replace(0, np.nan)
    return frame
