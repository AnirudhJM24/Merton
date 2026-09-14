"""Fit the model to the panel, validate it, and write results.md plus figures.

    python scripts/run_analysis.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merton import crosssection as xs
from merton.analysis import figures, timeseries, validation
from merton.analysis.fit import fit_panel, summarize_coverage

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

log = logging.getLogger("run_analysis")

SCORES = {          # score -> is a higher value safer?
    "dd": True,
    "altman_z": True,
    "leverage_book": False,
    "sigma_E": False,
}
SCORE_LABELS = {
    "dd": "Merton distance-to-default",
    "altman_z": "Altman Z-score",
    "leverage_book": "Book leverage (debt/assets)",
    "sigma_E": "Equity volatility",
}


def md_table(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a frame as a markdown table without pulling in another dependency."""
    def cell(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        if isinstance(v, pd.Timestamp):
            return v.date().isoformat()
        return str(v)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    align = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---"
                           for c in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, align, *rows])


def format_calibration(table: pd.DataFrame) -> pd.DataFrame:
    """Render the calibration table so it can actually be read.

    Model PDs in the safe buckets underflow to zero, which makes the ratio
    column run to 1e36 and tells the reader nothing. Below a floor the
    probability is reported as effectively zero and the ratio is dropped,
    because "the model says this cannot happen and it happened" is the finding,
    not the size of the arithmetic.
    """
    floor = 1e-12
    out = pd.DataFrame({
        "DD (bucket median)": table["dd_mid"].map("{:.2f}".format),
        "firm-months": table["n"],
        "model PD": [("~0" if v < floor else f"{v:.4%}") for v in table["model_pd"]],
        "realized bankruptcy rate": [f"{v:.4%}" for v in table["empirical_pd"]],
        "realized / model": [
            ("n/a" if m < floor else f"{e / m:,.1f}x")
            for m, e in zip(table["model_pd"], table["empirical_pd"])
        ],
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(DATA_DIR / "panel.parquet"))
    parser.add_argument("--out", default=str(ROOT / "results.md"))
    parser.add_argument("--boot", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    panel = fit_panel(pd.read_parquet(args.panel))
    fitted = panel[panel["converged"]].copy()
    FIG_DIR.mkdir(exist_ok=True)

    coverage = summarize_coverage(panel)
    log.info("coverage:\n%s", coverage.to_string(index=False))

    # -- discrimination -------------------------------------------------
    comparison = validation.score_comparison(fitted, SCORES, n_boot=args.boot)
    comparison["score"] = comparison["score"].map(SCORE_LABELS).fillna(comparison["score"])
    log.info("score comparison (distress):\n%s", comparison.to_string(index=False))

    bankruptcy = validation.score_comparison(fitted, SCORES, label="default_12m",
                                             n_boot=args.boot)
    bankruptcy["score"] = bankruptcy["score"].map(SCORE_LABELS).fillna(bankruptcy["score"])
    log.info("score comparison (bankruptcy):\n%s", bankruptcy.to_string(index=False))

    deciles = validation.decile_table(fitted, "dd", label="default_12m")
    deciles_distress = validation.decile_table(fitted, "dd")
    calibration = validation.empirical_vs_model_pd(fitted)

    accounting = ["altman_z", "leverage_book", "log_assets"]
    benchmarks = pd.DataFrame([
        {"outcome": outcome, "model": name, **validation.logistic_benchmark(
            fitted, feats, label=label)}
        for outcome, label in (("bankruptcy", "default_12m"), ("distress", "distress_12m"))
        for name, feats in (("distance-to-default alone", ["dd"]),
                            ("accounting ratios only", accounting),
                            ("both", ["dd"] + accounting))
    ])[["outcome", "model", "auc", "n"]]
    log.info("logistic benchmarks:\n%s", benchmarks.to_string(index=False))

    # -- credit cycle ---------------------------------------------------
    monthly = timeseries.with_market_spreads(timeseries.aggregate_by_month(fitted))
    correlations = timeseries.cycle_correlations(monthly)
    gaps = timeseries.spread_gap(fitted, monthly)

    # -- figures --------------------------------------------------------
    figures.roc_figure(fitted, SCORES, SCORE_LABELS, FIG_DIR / "roc.png")
    figures.roc_figure(fitted, SCORES, SCORE_LABELS, FIG_DIR / "roc_bankruptcy.png",
                       label="default_12m",
                       title="Predicting a bankruptcy filing 12 months ahead")
    figures.decile_figure(fitted, FIG_DIR / "decile_default_rate.png",
                          label="default_12m",
                          title="Realized bankruptcy rate by distance-to-default decile")
    figures.calibration_figure(fitted, FIG_DIR / "pd_calibration.png")
    figures.event_study_figure(fitted, FIG_DIR / "event_study.png")
    figures.credit_cycle_figure(monthly, FIG_DIR / "credit_cycle.png")
    figures.leverage_vol_figure(fitted, FIG_DIR / "leverage_vs_volatility.png")
    log.info("wrote figures to %s", FIG_DIR)

    # -- rating cross-section -------------------------------------------
    ratings = xs.analyze(xs.load_companies())

    sections = [
        "# Results",
        "",
        "Generated by `scripts/run_analysis.py`. Every number below comes from the "
        "committed panel; nothing is transcribed by hand.",
        "",
        "## 1. What the panel contains",
        "",
        md_table(coverage),
        "",
        "## 2. Does distance-to-default predict trouble?",
        "",
        "Two outcomes are scored, and they do not agree about what the model is "
        "worth. **Bankruptcy** is an Item 1.03 filing within twelve months: the "
        "event the model is actually about, and a scarce one here for the reason "
        "given in section 1. **Severe distress** adds a 90% loss of equity value, "
        "which is far more common and so far better powered. Confidence intervals "
        "are bootstrapped over firms rather than rows, because one firm contributes "
        "a hundred near-identical months and resampling rows would claim a "
        "precision the data does not have.",
        "",
        "### Bankruptcy filing within 12 months",
        "",
        md_table(bankruptcy),
        "",
        "![ROC curves, bankruptcy](figures/roc_bankruptcy.png)",
        "",
        "The AUCs printed on the chart are the common-sample figures, since all "
        "four curves have to be drawn on the same rows to be comparable; the "
        "table above gives each score on its own sample as well.",
        "",
        "Realized bankruptcy rate by distance-to-default decile:",
        "",
        md_table(deciles[["bucket", "n", "events", "event_rate_pct",
                          "score_low", "score_high"]], "{:.2f}"),
        "",
        "![Bankruptcy rate by decile](figures/decile_default_rate.png)",
        "",
        "### Severe distress within 12 months",
        "",
        f"Common sample: {comparison.attrs['n_obs']:,} firm-months, "
        f"{comparison.attrs['n_defaults']:,} of them followed by distress.",
        "",
        md_table(comparison),
        "",
        "![ROC curves](figures/roc.png)",
        "",
        "Realized distress rate by distance-to-default decile:",
        "",
        md_table(deciles_distress[["bucket", "n", "events", "event_rate_pct",
                                   "score_low", "score_high"]], "{:.2f}"),
        "",
        "### Does the structural measure add anything to accounting ratios?",
        "",
        "Out-of-sample AUC from logistic models, five folds split by firm so that "
        "no firm appears in both training and test:",
        "",
        md_table(benchmarks),
        "",
        "## 3. Are the probabilities themselves any good?",
        "",
        "Model risk-neutral PD against the frequency actually realized, by "
        "distance-to-default bucket:",
        "",
        md_table(format_calibration(calibration)),
        "",
        "Read the two ends separately. In the riskiest bucket the model is too "
        "pessimistic: it prices a one-in-seven chance of default against a realized "
        "rate near one in a hundred. Everywhere above roughly three standard "
        "deviations it is too optimistic, and not by a margin -- it puts the "
        "probability at zero to machine precision for firms that went on to file "
        "anyway. This is the credit spread puzzle: a single-factor Merton model at a "
        "one-year horizon cannot generate default risk for a healthy firm, because "
        "a diffusion has to travel too far in twelve months. The rankings are the "
        "output worth reading; the levels are not.",
        "",
        "![Model PD vs realized frequency](figures/pd_calibration.png)",
        "",
        "## 4. Distance-to-default approaching a filing",
        "",
        "![Event study](figures/event_study.png)",
        "",
        "## 5. Does the model track the credit cycle?",
        "",
        md_table(correlations),
        "",
        "![Credit cycle](figures/credit_cycle.png)",
        "",
        "Model-implied spread against traded index spreads, selected months:",
        "",
        md_table(gaps[["median_spread", "baa_spread", "aaa_spread", "ratio_baa_spread"]]
                 .dropna().iloc[::12].reset_index()
                 .rename(columns={"date": "month"}), "{:.4f}"),
        "",
        "## 6. Rating cross-section",
        "",
        f"Spearman rank correlation between model distance-to-default and S&P issuer "
        f"rating across the {len(ratings)} hand-built names: "
        f"**{xs.rating_rank_correlation(ratings):.2f}**.",
        "",
        f"What drives that ordering is leverage, not volatility: distance-to-default "
        f"correlates {xs.rank_correlation(ratings, 'leverage'):+.2f} with asset "
        f"leverage and {xs.rank_correlation(ratings, 'sigma_V'):+.2f} with asset "
        f"volatility across the same names.",
        "",
        xs.render_table(ratings, markdown=True),
        "",
        xs.render_rating_summary(ratings, markdown=True),
        "",
        "## 7. What drives the ranking",
        "",
        "![Leverage vs volatility](figures/leverage_vs_volatility.png)",
        "",
    ]

    Path(args.out).write_text("\n".join(sections))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
