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


def _newton_panel(E, sigma_E, D, r, T, tol=1e-10, max_iter=60):
    """Vectorized Newton on the two-equation system, with an analytic Jacobian.

    Solving row by row with a generic root finder costs a millisecond or two
    each, which is minutes over a panel of this size. Newton in log space with
    the exact Jacobian converges in a handful of iterations for every row at
    once.

    The Jacobian simplifies sharply because of the Black-Scholes identity
    V*phi(d1) == D*e^{-rT}*phi(d2): the two vega terms in the first equation
    cancel against each other, leaving d(f1)/d(log V) = V*Phi(d1) exactly.
    """
    sqrt_T = math.sqrt(T)
    K = D * np.exp(-r * T)

    u = np.log(E + K)                       # assets >= equity + discounted debt
    w = np.log(np.clip(sigma_E * E / np.exp(u), 1e-8, 5.0))

    active = np.ones(len(E), dtype=bool)
    for _ in range(max_iter):
        V, sigma_V = np.exp(u), np.exp(w)
        A = np.log(V / D) + r * T
        d1 = (A + 0.5 * sigma_V ** 2 * T) / (sigma_V * sqrt_T)
        d2 = d1 - sigma_V * sqrt_T
        Nd1, nd1 = norm.cdf(d1), norm.pdf(d1)

        f1 = (V * Nd1 - K * norm.cdf(d2) - E) / E
        f2 = (Nd1 * sigma_V * V - sigma_E * E) / (sigma_E * E)

        done = np.maximum(np.abs(f1), np.abs(f2)) < tol
        active = active & ~done
        if not active.any():
            break

        dd1_dw = (0.5 * sigma_V ** 2 * T - A) / (sigma_V * sqrt_T)

        j11 = V * Nd1 / E
        j12 = V * nd1 * sigma_V * sqrt_T / E
        j21 = (Nd1 * sigma_V * V + nd1 * V / sqrt_T) / (sigma_E * E)
        j22 = (Nd1 * sigma_V * V + nd1 * dd1_dw * sigma_V * V) / (sigma_E * E)

        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-14, np.nan, det)
        step_u = (j22 * f1 - j12 * f2) / det
        step_w = (-j21 * f1 + j11 * f2) / det

        # Cap the step so a bad iterate cannot throw the solve into a region
        # where the volatility underflows and the residuals stop being finite.
        step_u = np.clip(step_u, -1.0, 1.0)
        step_w = np.clip(step_w, -1.0, 1.0)

        u = np.where(active, u - step_u, u)
        w = np.where(active, w - step_w, w)
        u = np.where(np.isfinite(u), u, np.log(E + K))
        w = np.clip(w, np.log(1e-8), np.log(5.0))

    V, sigma_V = np.exp(u), np.exp(w)
    d1, d2 = d1_d2(V, sigma_V, D, r, T)
    f1 = (V * norm.cdf(d1) - K * norm.cdf(d2) - E) / E
    f2 = (norm.cdf(d1) * sigma_V * V - sigma_E * E) / (sigma_E * E)
    converged = np.maximum(np.abs(f1), np.abs(f2)) < 1e-8
    return V, sigma_V, converged


def _metrics_panel(V, sigma_V, D, r, T):
    """Credit measures for whole arrays at once."""
    d1, d2 = d1_d2(V, sigma_V, D, r, T)
    discounted = D * np.exp(-r * T)
    equity = V * norm.cdf(d1) - discounted * norm.cdf(d2)
    debt_value = V - equity
    return {
        "V": V,
        "sigma_V": sigma_V,
        "dd": d2,
        "pd": norm.cdf(-d2),
        "spread": -np.log(debt_value / D) / T - r,
        "leverage": discounted / V,
    }


def solve_panel(E, sigma_E, D, r, T=1.0):
    """Solve for (V, sigma_V) row-wise over arrays, returning a dict of arrays.

    Runs the vectorized Newton solve first and retries only the rows it could
    not settle with the general-purpose solver, so a difficult handful does not
    set the pace for the whole panel. Rows with unusable inputs, and rows
    neither method can solve, come back as NaN and are flagged in
    ``converged`` rather than silently dropped -- the coverage of the solve is
    itself something worth reporting.
    """
    E, sigma_E, D, r = (np.asarray(a, dtype=float) for a in (E, sigma_E, D, r))
    n = len(E)
    keys = ("V", "sigma_V", "dd", "pd", "spread", "leverage")
    out = {k: np.full(n, np.nan) for k in keys}
    converged = np.zeros(n, dtype=bool)

    usable = (E > 0) & (sigma_E > 0) & (D > 0) & np.isfinite(E) \
        & np.isfinite(sigma_E) & np.isfinite(D) & np.isfinite(r)
    if not usable.any():
        out["converged"] = converged
        return out

    idx = np.flatnonzero(usable)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        V, sigma_V, ok = _newton_panel(E[idx], sigma_E[idx], D[idx], r[idx], T)

    # Rows Newton could not settle get the general-purpose solver, one at a
    # time. There are usually very few of them.
    for local in np.flatnonzero(~ok):
        row = idx[local]
        try:
            V[local], sigma_V[local] = solve_assets(
                E[row], sigma_E[row], D[row], r[row], T)
            ok[local] = True
        except (ValueError, RuntimeError, FloatingPointError):
            continue

    solved = idx[ok]
    if len(solved):
        metrics = _metrics_panel(V[ok], sigma_V[ok], D[solved], r[solved], T)
        for key, values in metrics.items():
            out[key][solved] = values
        converged[solved] = True

    out["converged"] = converged
    return out
