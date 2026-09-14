"""Command line entry point.

    python -m merton.cli crosssection --markdown
    python -m merton.cli panel --top 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from merton import crosssection as xs

PANEL_PATH = Path(__file__).resolve().parents[1] / "data" / "panel.parquet"


def run_crosssection(args) -> None:
    companies = xs.load_companies(args.data)
    results = xs.analyze(companies, r=args.rate, T=args.horizon)

    print(f"Merton model | {len(results)} companies | r = {args.rate * 100:.2f}% | "
          f"T = {args.horizon:g}y | ranked by distance-to-default\n")
    print(xs.render_table(results, args.markdown))
    print()
    print(xs.render_rating_summary(results, args.markdown))
    print(f"\nSpearman rank correlation, model DD vs S&P rating: "
          f"{xs.rating_rank_correlation(results):.2f}")

    if args.stress:
        shock = args.stress / 100.0
        stressed = {x["ticker"]: x for x in xs.analyze(companies, args.rate, args.horizon, shock)}
        print(f"\nAfter a {args.stress:g}% equity drawdown (debt and equity vol held fixed):\n")
        rows = [(x["ticker"], f"{x['dd']:.2f}", f"{stressed[x['ticker']]['dd']:.2f}",
                 f"{stressed[x['ticker']]['dd'] - x['dd']:+.2f}",
                 xs.fmt_pd(stressed[x["ticker"]]["pd"])) for x in results]
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


def run_panel(args) -> None:
    from merton.analysis.fit import fit_panel, summarize_coverage

    path = Path(args.panel)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/build_panel.py first")

    panel = fit_panel(pd.read_parquet(path))
    print(summarize_coverage(panel).to_string(index=False))

    latest = panel[panel["converged"]].sort_values("date").groupby("cik").tail(1)
    latest = latest.sort_values("dd")
    columns = ["ticker", "name", "date", "dd", "pd", "spread", "leverage", "sigma_V"]

    print(f"\nRiskiest {args.top} firms as of each firm's latest month:\n")
    print(latest[columns].head(args.top).to_string(index=False,
          formatters={"dd": "{:.2f}".format, "pd": "{:.3%}".format,
                      "spread": lambda s: f"{s * 1e4:.0f}bp",
                      "leverage": "{:.1%}".format, "sigma_V": "{:.1%}".format}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cs = sub.add_parser("crosssection", help="the 20-name rating cross-section")
    cs.add_argument("--data", default=xs.DATA_FILE, help="input CSV of company data")
    cs.add_argument("--rate", type=float, default=xs.R_FREE, help="risk-free rate (decimal)")
    cs.add_argument("--horizon", type=float, default=xs.T, help="horizon in years")
    cs.add_argument("--stress", type=float, default=0.0, metavar="PCT",
                    help="also report results after an equity drawdown, e.g. 30 for -30%%")
    cs.add_argument("--markdown", action="store_true", help="emit markdown tables")
    cs.set_defaults(func=run_crosssection)

    pn = sub.add_parser("panel", help="fit the model to the firm-month panel")
    pn.add_argument("--panel", default=str(PANEL_PATH))
    pn.add_argument("--top", type=int, default=20)
    pn.set_defaults(func=run_panel)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
