"""Merton (1974) structural credit risk model over a 20-name cross-section.

Treats equity as a European call on the firm's assets struck at the face value
of debt. Equity value and equity volatility are observable; asset value and
asset volatility are not, so the two are backed out jointly from

    E        = V N(d1) - D e^{-rT} N(d2)
    sigma_E E = N(d1) sigma_V V

Solving that system gives distance-to-default, the risk-neutral 1-year default
probability, and the credit spread implied by the model.
"""

import argparse
import csv
import math
import os

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

R_FREE = 0.0403  # 1-year Treasury yield, Aug 2026
T = 1.0

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companies.csv")

# Rating order, best to worst, for grouping the summary table.
RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
                "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-"]


def load_companies(path=DATA_FILE):
    """Read the input file, skipping the leading '#' comment block."""
    with open(path, newline="") as fh:
        rows = [line for line in fh if not line.startswith("#")]
    companies = []
    for row in csv.DictReader(rows):
        companies.append({
            "ticker": row["ticker"],
            "name": row["name"],
            "sector": row["sector"],
            "E": float(row["E_usd_bn"]) * 1e9,
            "sigma_E": float(row["sigma_E"]),
            "D": float(row["D_usd_bn"]) * 1e9,
            "rating": row["rating"],
        })
    return companies


def merton_system(x, E, sigma_E, D, r, T):
    """Residuals of the two-equation system, in log-parameterized unknowns.

    Solving for (log V, log sigma_V) rather than (V, sigma_V) keeps both
    unknowns positive without a penalty branch, and the residuals are scaled by
    their own magnitudes so a $4T firm and a $9B one converge to the same
    relative tolerance.
    """
    V, sigma_V = math.exp(x[0]), math.exp(x[1])
    sqrt_T = math.sqrt(T)
    d1 = (math.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * sqrt_T)
    d2 = d1 - sigma_V * sqrt_T
    eq1 = V * norm.cdf(d1) - D * math.exp(-r * T) * norm.cdf(d2) - E
    eq2 = norm.cdf(d1) * sigma_V * V - sigma_E * E
    return [eq1 / E, eq2 / (sigma_E * E)]


def solve_assets(E, sigma_E, D, r=R_FREE, T=T):
    """Back out (V, sigma_V) from observed equity. Raises if it fails to converge."""
    V0 = E + D * math.exp(-r * T)          # assets >= equity + discounted debt
    sigma_V0 = sigma_E * E / V0            # de-levered equity vol
    args = (E, sigma_E, D, r, T)
    x, _, ier, msg = fsolve(merton_system, [math.log(V0), math.log(sigma_V0)],
                            args=args, full_output=True)
    residual = max(abs(v) for v in merton_system(x, *args))
    if ier != 1 or residual > 1e-8:
        raise RuntimeError(f"solver did not converge (residual={residual:.2e}): {msg}")
    return math.exp(x[0]), math.exp(x[1])


def credit_metrics(V, sigma_V, D, r=R_FREE, T=T):
    """Distance-to-default, risk-neutral PD, implied spread and asset leverage."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * sqrt_T)
    d2 = d1 - sigma_V * sqrt_T
    debt_value = V - (V * norm.cdf(d1) - D * math.exp(-r * T) * norm.cdf(d2))
    return {
        "V": V,
        "sigma_V": sigma_V,
        "dd": d2,                                          # distance to default
        "pd": norm.cdf(-d2),                               # risk-neutral 1yr PD
        "spread": -math.log(debt_value / D) / T - r,       # implied credit spread
        "leverage": D * math.exp(-r * T) / V,
    }


def analyze(companies, r=R_FREE, T=T, equity_shock=0.0):
    """Run the model over every company, optionally shocking equity value down."""
    results = []
    for c in companies:
        E = c["E"] * (1.0 - equity_shock)
        V, sigma_V = solve_assets(E, c["sigma_E"], c["D"], r, T)
        results.append({**c, "E": E, **credit_metrics(V, sigma_V, c["D"], r, T)})
    return sorted(results, key=lambda x: x["dd"], reverse=True)


def fmt_pd(p):
    """PD spans ~30 orders of magnitude across this sample, so switch notation."""
    if p >= 1e-4:
        return f"{p * 100:.2f}%"
    if p < 1e-16:
        return "~0"
    return f"{p * 100:.1e}%"


def fmt_money(v):
    return f"{v / 1e9:,.0f}B" if v >= 1e12 else f"{v / 1e9:,.1f}B"


def fmt_spread(s):
    """Sub-basis-point spreads are numerically indistinguishable from zero here."""
    bp = s * 1e4
    return "<1bp" if abs(bp) < 0.5 else f"{bp:.0f}bp"


COLUMNS = [
    ("Company", 24, "<", lambda x: f"{x['name']} ({x['ticker']})"),
    ("Rating", 7, ">", lambda x: x["rating"]),
    ("Equity", 11, ">", lambda x: fmt_money(x["E"])),
    ("Debt", 10, ">", lambda x: fmt_money(x["D"])),
    ("Asset Val", 11, ">", lambda x: fmt_money(x["V"])),
    ("Asset Vol", 10, ">", lambda x: f"{x['sigma_V'] * 100:.1f}%"),
    ("Lev", 8, ">", lambda x: f"{x['leverage'] * 100:.1f}%"),
    ("DD", 8, ">", lambda x: f"{x['dd']:.2f}"),
    ("PD (1yr)", 10, ">", lambda x: fmt_pd(x["pd"])),
    ("Spread", 10, ">", lambda x: fmt_spread(x["spread"])),
]


def render_table(results, markdown=False):
    if markdown:
        head = "| " + " | ".join(h for h, _, _, _ in COLUMNS) + " |"
        sep = "|" + "|".join("---" if a == "<" else "---:" for _, _, a, _ in COLUMNS) + "|"
        body = ["| " + " | ".join(f(x) for _, _, _, f in COLUMNS) + " |" for x in results]
        return "\n".join([head, sep, *body])
    head = "".join(f"{h:{a}{w}}" for h, w, a, _ in COLUMNS)
    body = ["".join(f"{f(x):{a}{w}}" for _, w, a, f in COLUMNS) for x in results]
    return "\n".join([head, "-" * len(head), *body])


def render_rating_summary(results, markdown=False):
    """Median DD per rating bucket -- does the model rank names the way S&P does?"""
    buckets = {}
    for x in results:
        buckets.setdefault(x["rating"], []).append(x)
    rows = []
    for rating in sorted(buckets, key=lambda x: RATING_ORDER.index(x)):
        group = buckets[rating]
        rows.append((rating, len(group),
                     f"{np.median([g['dd'] for g in group]):.2f}",
                     fmt_spread(float(np.median([g["spread"] for g in group]))),
                     ", ".join(g["ticker"] for g in group)))
    headers = ["Rating", "n", "Median DD", "Median Spread", "Names"]
    if markdown:
        out = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---|"]
        out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
        return "\n".join(out)
    widths = [8, 4, 12, 16, 40]
    out = ["".join(f"{h:<{w}}" for h, w in zip(headers, widths))]
    out.append("-" * sum(widths))
    out += ["".join(f"{str(v):<{w}}" for v, w in zip(row, widths)) for row in rows]
    return "\n".join(out)


def rank_correlation(results):
    """Spearman rank correlation between model DD and agency rating."""
    dd_rank = np.argsort(np.argsort([-x["dd"] for x in results]))
    rating_rank = np.argsort(np.argsort([RATING_ORDER.index(x["rating"]) for x in results]))
    return float(np.corrcoef(dd_rank, rating_rank)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_FILE, help="input CSV of company data")
    parser.add_argument("--rate", type=float, default=R_FREE, help="risk-free rate (decimal)")
    parser.add_argument("--horizon", type=float, default=T, help="horizon in years")
    parser.add_argument("--stress", type=float, default=0.0, metavar="PCT",
                        help="also report results after an equity drawdown, e.g. 30 for -30%%")
    parser.add_argument("--markdown", action="store_true", help="emit markdown tables")
    args = parser.parse_args()

    companies = load_companies(args.data)
    results = analyze(companies, r=args.rate, T=args.horizon)

    print(f"Merton model | {len(results)} companies | r = {args.rate * 100:.2f}% | "
          f"T = {args.horizon:g}y | ranked by distance-to-default\n")
    print(render_table(results, args.markdown))
    print()
    print(render_rating_summary(results, args.markdown))
    print(f"\nSpearman rank correlation, model DD vs S&P rating: {rank_correlation(results):.2f}")

    if args.stress:
        shock = args.stress / 100.0
        stressed = {x["ticker"]: x for x in analyze(companies, args.rate, args.horizon, shock)}
        print(f"\nAfter a {args.stress:g}% equity drawdown (debt and equity vol held fixed):\n")
        rows = [(x["ticker"], f"{x['dd']:.2f}", f"{stressed[x['ticker']]['dd']:.2f}",
                 f"{stressed[x['ticker']]['dd'] - x['dd']:+.2f}",
                 fmt_pd(stressed[x["ticker"]]["pd"])) for x in results]
        headers = ["Ticker", "DD", "Stressed DD", "Change", "Stressed PD"]
        if args.markdown:
            print("| " + " | ".join(headers) + " |")
            print("|---|---:|---:|---:|---:|")
            for row in rows:
                print("| " + " | ".join(row) + " |")
        else:
            widths = [9, 9, 14, 10, 12]
            print("".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
            print("-" * sum(widths))
            for row in rows:
                print("".join(f"{v:<{w}}" for v, w in zip(row, widths)))


if __name__ == "__main__":
    main()
