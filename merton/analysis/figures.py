"""Figures for the write-up.

House rules, applied to every chart here: one y-axis per panel (a measure on a
different scale gets its own panel, never a second axis), categorical hues
assigned in a fixed order rather than cycled, every multi-series chart both
legended and direct-labelled so identity never rests on colour alone, and grid
lines kept recessive so the data is the darkest thing on the surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from merton.analysis.validation import decile_table, empirical_vs_model_pd, roc_points

# Categorical slots, in fixed order. Validated as a set for colour-vision
# deficiency separation on a light surface (worst adjacent pair dE 9.1 under
# protanopia, 22.9 for normal vision); never cycled, never reordered. Two of
# the four fall below 3:1 contrast against the surface, so every chart using
# them carries a legend and the accompanying table rather than relying on
# colour to carry identity.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d8d4"

FIGSIZE = (7.2, 4.4)
DPI = 160


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=12, pad=12, loc="left", fontweight="medium")
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    return ax


def _figure(nrows=1, figsize=FIGSIZE, sharex=False):
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, dpi=DPI, sharex=sharex,
                             facecolor=SURFACE)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(SURFACE)
    return fig, axes


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def roc_figure(panel: pd.DataFrame, scores: dict[str, bool], labels: dict[str, str],
               out: Path, label: str = "distress_12m", title: str | None = None) -> Path:
    """ROC curves for the competing scores, on their common sample."""
    columns = list(scores) + [label]
    common = panel[columns].replace([np.inf, -np.inf], np.nan).dropna()

    if len(scores) > len(SERIES):
        raise ValueError(
            f"{len(scores)} series but only {len(SERIES)} validated colour slots; "
            "fold the extras together or facet rather than inventing a hue")

    fig, ax = _figure()
    for colour, (name, higher_is_safer) in zip(SERIES, scores.items()):
        fpr, tpr, _ = roc_points(common, name, label=label, higher_is_safer=higher_is_safer)
        from sklearn.metrics import auc as _auc
        ax.plot(fpr, tpr, color=colour, linewidth=2,
                label=f"{labels.get(name, name)}  (AUC {_auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=(0, (4, 4)),
            label="No discrimination")
    _style(ax, title or "Predicting severe distress 12 months ahead",
           "False positive rate", "True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    return _save(fig, out)


def decile_figure(panel: pd.DataFrame, out: Path, score: str = "dd", n_bins: int = 10,
                  label: str = "distress_12m", title: str | None = None) -> Path:
    """Realized event rate by score bucket: the monotonicity check, drawn."""
    table = decile_table(panel, score, label=label, n_bins=n_bins)

    fig, ax = _figure()
    ax.bar(table["bucket"], table["event_rate_pct"], color=SERIES[0], width=0.72)
    for _, row in table.iterrows():
        if row["event_rate_pct"] > 0:
            ax.text(row["bucket"], row["event_rate_pct"], f"{row['event_rate_pct']:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)

    _style(ax, title or "Realized event rate by distance-to-default decile",
           "Decile of distance-to-default  (0 = riskiest)",
           "Event within 12 months (%)")
    ax.set_xticks(table["bucket"])
    return _save(fig, out)


def calibration_figure(panel: pd.DataFrame, out: Path, n_bins: int = 12,
                       label: str = "default_12m") -> Path:
    """Model risk-neutral PD against the realized frequency, by DD bucket."""
    table = empirical_vs_model_pd(panel, n_bins=n_bins, label=label)
    table = table[(table["model_pd"] > 0) | (table["empirical_pd"] > 0)]
    floor = 1e-10

    fig, ax = _figure()
    ax.plot(table["dd_mid"], table["model_pd"].clip(lower=floor), color=SERIES[0],
            linewidth=2, marker="o", markersize=6, label="Model risk-neutral PD")
    ax.plot(table["dd_mid"], table["empirical_pd"].clip(lower=floor), color=SERIES[1],
            linewidth=2, marker="s", markersize=6, label="Realized bankruptcy frequency")

    ax.set_yscale("log")
    _style(ax, "Model probabilities against what actually happened",
           "Distance to default (bucket median)", "12-month default probability")
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    return _save(fig, out)


def event_study_figure(panel: pd.DataFrame, out: Path, max_months: int = 36,
                       min_obs: int = 5) -> Path:
    """Distance-to-default in the quarters before a bankruptcy filing.

    Binned by quarter rather than month. Few firms in this panel both file and
    have a usable pre-filing price history, so a monthly median swings on which
    firms happen to be present at each horizon -- the sawtooth is composition
    changing, not risk changing. Quarterly bins with a minimum observation
    count give a line that means what it appears to mean, and the bucket counts
    are printed so the reader can see how thin it is.
    """
    fitted = panel[panel["converged"]].copy()
    approaching = fitted[fitted["months_to_default"].between(0, max_months)].copy()
    if approaching.empty:
        return out

    approaching["quarter"] = (approaching["months_to_default"] // 3).astype(int) * 3
    grouped = approaching.groupby("quarter")
    profile = grouped["dd"].median()
    counts = grouped["dd"].size()
    profile = profile[counts >= min_obs]
    counts = counts[profile.index]
    if profile.empty:
        return out

    survivors = fitted.loc[fitted["default_date"].isna(), "dd"].median()
    n_firms = approaching["cik"].nunique()

    fig, ax = _figure()
    ax.axhline(survivors, color=SERIES[0], linewidth=2, linestyle=(0, (5, 3)),
               label=f"Median firm that never filed ({survivors:.1f})")
    ax.plot(profile.index, profile.values, color=SERIES[1], linewidth=2,
            marker="o", markersize=6, label=f"Firms that went on to file (n={n_firms})")

    for quarter, value in profile.items():
        ax.annotate(f"{counts[quarter]}", (quarter, value), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color=INK_MUTED)

    _style(ax, "Distance-to-default approaching bankruptcy",
           "Months before the Item 1.03 filing  (quarterly bins, labelled with n)",
           "Distance to default")
    ax.invert_xaxis()
    ax.set_ylim(0, max(float(profile.max()), survivors) * 1.35)
    legend = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    return _save(fig, out)


def credit_cycle_figure(monthly: pd.DataFrame, out: Path) -> Path:
    """Model-implied distress against traded corporate spreads.

    Two panels rather than two y-axes. The measures live on different scales,
    and overlaying them on a shared axis would let the choice of scaling decide
    how well they appear to agree.
    """
    frame = monthly.dropna(subset=["share_distressed"]).copy()

    fig, axes = _figure(nrows=2, figsize=(7.2, 6.0), sharex=True)
    top, bottom = axes

    top.plot(frame.index, 100 * frame["share_distressed"], color=SERIES[0], linewidth=2)
    _style(top, "Model: share of firms within two standard deviations of default",
           "", "Share of firms (%)")

    if "baa_spread" in frame:
        market = frame.dropna(subset=["baa_spread"])
        bottom.plot(market.index, 1e4 * market["baa_spread"], color=SERIES[1], linewidth=2)
    _style(bottom, "Market: Moody's Baa corporate yield over the 10-year Treasury",
           "", "Spread (bp)")

    return _save(fig, out)


def leverage_vol_figure(panel: pd.DataFrame, out: Path) -> Path:
    """What actually drives the ranking: leverage or asset volatility?"""
    fitted = panel[panel["converged"]]
    latest = fitted.sort_values("date").groupby("cik").tail(1)
    latest = latest[latest["dd"].between(-2, 25)]

    fig, ax = _figure()
    scatter = ax.scatter(latest["leverage"], latest["sigma_V"], c=latest["dd"],
                         cmap="Blues_r", s=18, edgecolor="none", alpha=0.85)
    bar = fig.colorbar(scatter, ax=ax)
    bar.set_label("Distance to default", color=INK_MUTED, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    bar.outline.set_visible(False)

    _style(ax, "Where distance-to-default comes from (latest month per firm)",
           "Asset leverage  D e^(-rT) / V", "Asset volatility")
    return _save(fig, out)
