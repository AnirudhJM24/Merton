"""The model's own mathematics, checked without touching the network."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from merton.core import (
    asset_value_call,
    credit_metrics,
    d1_d2,
    equity_from_assets,
    solve_assets,
    solve_panel,
)

R = 0.04
T = 1.0


@pytest.mark.parametrize("V,sigma_V,D", [
    (1_000e9, 0.20, 50e9),      # a very safe large-cap
    (100e9, 0.35, 60e9),        # a levered mid-cap
    (12e9, 0.55, 9e9),          # a distressed small-cap
    (5e9, 0.80, 4.9e9),         # barrier almost at asset value
])
def test_solver_inverts_the_forward_map(V, sigma_V, D):
    """Round trip: assets -> equity -> assets must return the inputs.

    This is the single most important test here. The solver is the only place
    where an error would be invisible -- a wrong (V, sigma_V) still produces a
    plausible distance-to-default.
    """
    E, sigma_E = equity_from_assets(V, sigma_V, D, R, T)
    V_hat, sigma_V_hat = solve_assets(E, sigma_E, D, R, T)
    assert V_hat == pytest.approx(V, rel=1e-6)
    assert sigma_V_hat == pytest.approx(sigma_V, rel=1e-6)


def test_balance_sheet_identity():
    """Equity plus the market value of debt must equal asset value."""
    V, sigma_V, D = 100e9, 0.30, 40e9
    metrics = credit_metrics(V, sigma_V, D, R, T)
    equity = asset_value_call(V, sigma_V, D, R, T)
    debt_value = D * math.exp(-(metrics.spread + R) * T)
    assert equity + debt_value == pytest.approx(V, rel=1e-9)


def test_distance_to_default_falls_with_leverage():
    V, sigma_V = 100e9, 0.30
    dds = [credit_metrics(V, sigma_V, D, R, T).dd for D in (10e9, 30e9, 60e9, 90e9)]
    assert all(a > b for a, b in zip(dds, dds[1:]))


def test_distance_to_default_falls_with_volatility():
    V, D = 100e9, 60e9
    dds = [credit_metrics(V, s, D, R, T).dd for s in (0.15, 0.30, 0.50, 0.80)]
    assert all(a > b for a, b in zip(dds, dds[1:]))


def test_pd_and_spread_move_together():
    """Both are monotone functions of d2, so their orderings cannot disagree."""
    V, D = 100e9, 55e9
    results = [credit_metrics(V, s, D, R, T) for s in (0.2, 0.3, 0.4, 0.6)]
    pds = [m.pd for m in results]
    spreads = [m.spread for m in results]
    assert all(a < b for a, b in zip(pds, pds[1:]))
    assert all(a < b for a, b in zip(spreads, spreads[1:]))


def test_spread_is_non_negative():
    """Risky debt cannot be worth more than the risk-free equivalent.

    The tolerance is not slack: for a firm whose debt is a rounding error
    against its assets, the debt value is computed as V minus an equity value
    that is almost exactly V, and the cancellation leaves a few units in the
    last place. The resulting spread is negative at the 1e-15 level, which is
    zero for every purpose the model has.
    """
    for D in (1e9, 20e9, 50e9, 95e9):
        assert credit_metrics(100e9, 0.35, D, R, T).spread >= -1e-12


def test_negligible_debt_implies_no_default_risk():
    metrics = credit_metrics(100e9, 0.25, 1e6, R, T)
    assert metrics.pd < 1e-12
    assert metrics.spread < 1e-8
    assert metrics.dd > 10


def test_d2_matches_closed_form():
    V, sigma_V, D = 80e9, 0.4, 50e9
    _, d2 = d1_d2(V, sigma_V, D, R, T)
    expected = (math.log(V / D) + (R - 0.5 * sigma_V ** 2) * T) / (sigma_V * math.sqrt(T))
    assert d2 == pytest.approx(expected)
    assert credit_metrics(V, sigma_V, D, R, T).pd == pytest.approx(norm.cdf(-expected))


def test_horizon_scaling():
    """Over a longer horizon a levered firm is more likely to default."""
    V, sigma_V, D = 100e9, 0.35, 85e9
    pds = [credit_metrics(V, sigma_V, D, R, t).pd for t in (0.5, 1.0, 3.0, 5.0)]
    assert all(a < b for a, b in zip(pds, pds[1:]))


@pytest.mark.parametrize("bad", [
    {"E": 0, "sigma_E": 0.3, "D": 10e9},
    {"E": -1e9, "sigma_E": 0.3, "D": 10e9},
    {"E": 1e9, "sigma_E": 0, "D": 10e9},
    {"E": 1e9, "sigma_E": 0.3, "D": 0},
])
def test_rejects_degenerate_inputs(bad):
    with pytest.raises(ValueError):
        solve_assets(r=R, T=T, **bad)


def test_panel_isolates_bad_rows():
    """One unusable firm-month must not take the other 100k rows with it."""
    E = np.array([100e9, 0.0, 50e9])
    sigma_E = np.array([0.30, 0.30, 0.45])
    D = np.array([40e9, 10e9, 30e9])
    r = np.array([R, R, R])

    out = solve_panel(E, sigma_E, D, r, T)
    assert out["converged"].tolist() == [True, False, True]
    assert np.isnan(out["dd"][1])
    assert np.isfinite(out["dd"][[0, 2]]).all()


def test_panel_matches_scalar_path():
    E = np.array([100e9, 50e9])
    sigma_E = np.array([0.3, 0.45])
    D = np.array([40e9, 30e9])
    r = np.array([R, R])
    out = solve_panel(E, sigma_E, D, r, T)
    for i in range(2):
        V, sigma_V = solve_assets(E[i], sigma_E[i], D[i], r[i], T)
        assert out["V"][i] == pytest.approx(V, rel=1e-9)
        assert out["dd"][i] == pytest.approx(credit_metrics(V, sigma_V, D[i], r[i], T).dd)
