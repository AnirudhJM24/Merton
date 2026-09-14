"""The Merton (1974) structural model.

Equity is a European call on the firm's assets struck at the face value of debt:

    E         = V N(d1) - D e^{-rT} N(d2)
    sigma_E E = N(d1) sigma_V V

V and sigma_V are unobservable, so the pair is backed out jointly from observed
equity value and equity volatility. The solved pair gives distance-to-default
(d2), the risk-neutral default probability N(-d2), the model-implied credit
spread, and asset leverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

__all__ = [
    "MertonSolution",
    "asset_value_call",
    "credit_metrics",
    "d1_d2",
    "equity_from_assets",
    "solve_assets",
    "solve_panel",
]


@dataclass(frozen=True)
class MertonSolution:
    """Solved asset process plus the credit measures implied by it."""

    V: float          # market value of assets
    sigma_V: float    # annualized asset volatility
    dd: float         # distance to default (d2)
    pd: float         # risk-neutral default probability over the horizon
    spread: float     # model-implied credit spread, continuously compounded
    leverage: float   # D e^{-rT} / V

    def as_dict(self) -> dict:
        return {
            "V": self.V,
            "sigma_V": self.sigma_V,
            "dd": self.dd,
            "pd": self.pd,
            "spread": self.spread,
            "leverage": self.leverage,
        }


def d1_d2(V, sigma_V, D, r, T):
    """Black-Scholes d1 and d2 for a call on assets struck at the debt face value."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(V / D) + (r + 0.5 * np.asarray(sigma_V) ** 2) * T) / (sigma_V * sqrt_T)
    return d1, d1 - sigma_V * sqrt_T


def asset_value_call(V, sigma_V, D, r, T):
    """Equity value implied by an asset value and volatility."""
    d1, d2 = d1_d2(V, sigma_V, D, r, T)
    return V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)


def equity_from_assets(V, sigma_V, D, r, T=1.0):
    """Forward map: the equity value and equity volatility a given asset
    process implies. This is the inverse of :func:`solve_assets`, and having
    both directions available is what makes the solver testable -- a round trip
    through the two must return the inputs.
    """
    d1, _ = d1_d2(V, sigma_V, D, r, T)
    equity = asset_value_call(V, sigma_V, D, r, T)
    sigma_E = norm.cdf(d1) * sigma_V * V / equity
    return float(equity), float(sigma_E)


def _residuals(x, E, sigma_E, D, r, T):
    """Residuals of the two-equation system in log-parameterized unknowns.

    Solving for (log V, log sigma_V) rather than (V, sigma_V) keeps both
    unknowns positive without a penalty branch. Each residual is scaled by its
    own magnitude so a $4T firm and a $90M one converge to the same relative
    tolerance.
    """
    V, sigma_V = math.exp(x[0]), math.exp(x[1])
    d1, d2 = d1_d2(V, sigma_V, D, r, T)
    eq1 = V * norm.cdf(d1) - D * math.exp(-r * T) * norm.cdf(d2) - E
    eq2 = norm.cdf(d1) * sigma_V * V - sigma_E * E
    return [eq1 / E, eq2 / (sigma_E * E)]


def solve_assets(E, sigma_E, D, r, T=1.0, tol=1e-8):
    """Back out (V, sigma_V) from observed equity value and equity volatility.

    Raises RuntimeError if the solver does not converge, rather than returning a
    plausible-looking number that happens to be wrong.
    """
    if not (E > 0 and sigma_E > 0 and D > 0 and T > 0):
        raise ValueError(f"need positive E, sigma_E, D, T (got {E}, {sigma_E}, {D}, {T})")

    V0 = E + D * math.exp(-r * T)   # assets >= equity + discounted debt
    sigma_V0 = sigma_E * E / V0     # de-levered equity volatility
    args = (E, sigma_E, D, r, T)
    x, _, ier, msg = fsolve(
        _residuals, [math.log(V0), math.log(sigma_V0)], args=args, full_output=True
    )
    residual = max(abs(v) for v in _residuals(x, *args))
    if ier != 1 or residual > tol:
        raise RuntimeError(f"solver did not converge (residual={residual:.2e}): {msg}")
    return math.exp(x[0]), math.exp(x[1])


def credit_metrics(V, sigma_V, D, r, T=1.0) -> MertonSolution:
    """Distance-to-default, risk-neutral PD, implied spread and asset leverage."""
    d1, d2 = d1_d2(V, sigma_V, D, r, T)
    equity = V * norm.cdf(d1) - D * math.exp(-r * T) * norm.cdf(d2)
    debt_value = V - equity
    return MertonSolution(
        V=V,
        sigma_V=sigma_V,
        dd=float(d2),
        pd=float(norm.cdf(-d2)),
        spread=float(-math.log(debt_value / D) / T - r),
        leverage=float(D * math.exp(-r * T) / V),
    )


def solve_panel(E, sigma_E, D, r, T=1.0):
    """Solve row-wise over arrays, returning a dict of arrays.

    Rows that fail to converge or carry non-positive inputs come back as NaN and
    are flagged in ``converged`` rather than raising, so one bad firm-month does
    not abort a panel of 100k.
    """
    E, sigma_E, D, r = (np.asarray(a, dtype=float) for a in (E, sigma_E, D, r))
    n = len(E)
    out = {k: np.full(n, np.nan) for k in
           ("V", "sigma_V", "dd", "pd", "spread", "leverage")}
    converged = np.zeros(n, dtype=bool)

    for i in range(n):
        try:
            V, sigma_V = solve_assets(E[i], sigma_E[i], D[i], r[i], T)
            m = credit_metrics(V, sigma_V, D[i], r[i], T)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        for k, v in m.as_dict().items():
            out[k][i] = v
        converged[i] = True

    out["converged"] = converged
    return out
