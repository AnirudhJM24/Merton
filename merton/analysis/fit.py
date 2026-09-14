"""Run the Merton solve across the panel."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from merton.core import solve_panel

log = logging.getLogger(__name__)


def fit_panel(panel: pd.DataFrame, horizon: float = 1.0) -> pd.DataFrame:
    """Solve for (V, sigma_V) on every row and attach the credit measures.

    Rows that fail to converge are kept but carry NaN measures and
    ``converged == False``, so the coverage of the solve is itself reportable
    rather than hidden by a silent dropna.
    """
    started = time.time()
    solved = solve_panel(panel["E"].to_numpy(), panel["sigma_E"].to_numpy(),
                         panel["D"].to_numpy(), panel["r"].to_numpy(), horizon)

    out = panel.copy()
    for key, values in solved.items():
        out[key] = values

    n_ok = int(out["converged"].sum())
    log.info("solved %d/%d rows (%.1f%%) in %.1fs",
             n_ok, len(out), 100 * n_ok / max(len(out), 1), time.time() - started)
    return out


def summarize_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """What the panel actually contains, for the record."""
    defaults = panel.loc[panel["default_date"].notna(), "cik"].nunique()
    rows = [
        ("firm-months", len(panel)),
        ("firms", panel["cik"].nunique()),
        ("months", panel["date"].nunique()),
        ("first month", panel["date"].min().date()),
        ("last month", panel["date"].max().date()),
        ("firms with a bankruptcy filing", defaults),
        ("firm-months labelled default-within-12m", int(panel["default_12m"].sum())),
        ("solver convergence", f"{100 * panel['converged'].mean():.2f}%"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
