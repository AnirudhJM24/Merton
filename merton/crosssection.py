"""The hand-built rating cross-section.

The panel has no credit ratings in it -- agency ratings are not available free
at panel scale -- so this small, separately maintained file supplies the one
validation axis the panel cannot: does the model order firms the way a rating
committee does? Twenty names spanning AAA to B-, with inputs quoted in
``companies.csv``.

Everything here is a snapshot, and the figures are rounded. They are good
enough to rank a cross-section and not good enough to quote any single
issuer's default probability; see README.md for provenance.
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np

from merton.core import credit_metrics, solve_assets

R_FREE = 0.0403   # 1-year Treasury yield at the data-pull date
T = 1.0

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "companies.csv")

RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
                "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-"]


def load_companies(path: str = DATA_FILE) -> list[dict]:
    """Read the input file, skipping the leading '#' comment block."""
    with open(path, newline="") as handle:
        rows = [line for line in handle if not line.startswith("#")]
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


def analyze(companies: list[dict], r: float = R_FREE, T: float = T,
            equity_shock: float = 0.0) -> list[dict]:
    """Run the model over every company, optionally shocking equity value down."""
    results = []
    for company in companies:
        equity = company["E"] * (1.0 - equity_shock)
        V, sigma_V = solve_assets(equity, company["sigma_E"], company["D"], r, T)
        metrics = credit_metrics(V, sigma_V, company["D"], r, T)
        results.append({**company, "E": equity, **metrics.as_dict()})
    return sorted(results, key=lambda x: x["dd"], reverse=True)


def rating_rank_correlation(results: list[dict]) -> float:
    """Spearman rank correlation between model distance-to-default and rating."""
    dd_rank = np.argsort(np.argsort([-x["dd"] for x in results]))
    rating_rank = np.argsort(np.argsort([RATING_ORDER.index(x["rating"]) for x in results]))
    return float(np.corrcoef(dd_rank, rating_rank)[0, 1])


def rank_correlation(results: list[dict], key: str, sign: float = 1.0) -> float:
    """Spearman correlation of distance-to-default against another measure."""
    dd_rank = np.argsort(np.argsort([x["dd"] for x in results]))
    other_rank = np.argsort(np.argsort([sign * x[key] for x in results]))
    return float(np.corrcoef(dd_rank, other_rank)[0, 1])


# -- formatting ---------------------------------------------------------------

def fmt_pd(p: float) -> str:
    """PD spans ~30 orders of magnitude across this sample, so switch notation."""
    if p >= 1e-4:
        return f"{p * 100:.2f}%"
    if p < 1e-16:
        return "~0"
    return f"{p * 100:.1e}%"


def fmt_money(v: float) -> str:
    return f"{v / 1e9:,.0f}B" if v >= 1e12 else f"{v / 1e9:,.1f}B"


def fmt_spread(s: float) -> str:
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


def render_table(results: list[dict], markdown: bool = False) -> str:
    if markdown:
        head = "| " + " | ".join(h for h, _, _, _ in COLUMNS) + " |"
        sep = "|" + "|".join("---" if a == "<" else "---:" for _, _, a, _ in COLUMNS) + "|"
        body = ["| " + " | ".join(f(x) for _, _, _, f in COLUMNS) + " |" for x in results]
        return "\n".join([head, sep, *body])
    head = "".join(f"{h:{a}{w}}" for h, w, a, _ in COLUMNS)
    body = ["".join(f"{f(x):{a}{w}}" for _, w, a, f in COLUMNS) for x in results]
    return "\n".join([head, "-" * len(head), *body])


def render_rating_summary(results: list[dict], markdown: bool = False) -> str:
    """Median distance-to-default per rating bucket."""
    buckets: dict[str, list[dict]] = {}
    for row in results:
        buckets.setdefault(row["rating"], []).append(row)

    rows = []
    for rating in sorted(buckets, key=RATING_ORDER.index):
        group = buckets[rating]
        rows.append((rating, len(group),
                     f"{np.median([g['dd'] for g in group]):.2f}",
                     fmt_spread(float(np.median([g['spread'] for g in group]))),
                     ", ".join(g["ticker"] for g in group)))

    headers = ["Rating", "n", "Median DD", "Median Spread", "Names"]
    if markdown:
        out = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---|"]
        out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
        return "\n".join(out)
    widths = [8, 4, 12, 16, 40]
    out = ["".join(f"{h:<{w}}" for h, w in zip(headers, widths)), "-" * sum(widths)]
    out += ["".join(f"{str(v):<{w}}" for v, w in zip(row, widths)) for row in rows]
    return "\n".join(out)
