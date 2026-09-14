"""Validation machinery, checked on synthetic panels with known answers."""

import numpy as np
import pandas as pd
import pytest

from merton.analysis.validation import (
    auc,
    accuracy_ratio,
    bootstrap_auc,
    decile_table,
    logistic_benchmark,
    score_comparison,
)


def synthetic_panel(n_firms=200, n_months=24, signal=1.0, seed=0):
    """A panel where distance-to-default genuinely predicts default.

    Each firm gets a latent quality; defaulters are drawn from the low end, and
    the observed ``dd`` is quality plus noise. Cranking ``signal`` to zero makes
    ``dd`` pure noise, which is the null the tests check against.
    """
    rng = np.random.default_rng(seed)
    quality = rng.normal(0, 1, n_firms)
    defaults = (quality + rng.normal(0, 0.3, n_firms)) < -1.0

    rows = []
    for firm in range(n_firms):
        for month in range(n_months):
            rows.append({
                "cik": firm,
                "date": pd.Timestamp("2020-01-31") + pd.DateOffset(months=month),
                "dd": signal * quality[firm] + rng.normal(0, 0.2),
                "altman_z": signal * 0.5 * quality[firm] + rng.normal(0, 1.0),
                "default_12m": int(defaults[firm]),
                "distress_12m": float(defaults[firm]),
            })
    return pd.DataFrame(rows)


def test_auc_is_one_for_a_perfect_score():
    panel = pd.DataFrame({
        "dd": [5.0, 4.0, 1.0, 0.5],
        "distress_12m": [0, 0, 1, 1],
        "cik": [1, 2, 3, 4],
    })
    assert auc(panel, "dd") == pytest.approx(1.0)
    assert accuracy_ratio(panel, "dd") == pytest.approx(1.0)


def test_auc_respects_score_direction():
    """A measure where high means risky must be declared as such."""
    panel = pd.DataFrame({
        "leverage": [0.1, 0.2, 0.9, 0.95],
        "distress_12m": [0, 0, 1, 1],
        "cik": [1, 2, 3, 4],
    })
    assert auc(panel, "leverage", higher_is_safer=False) == pytest.approx(1.0)
    assert auc(panel, "leverage", higher_is_safer=True) == pytest.approx(0.0)


def test_auc_is_a_half_when_the_score_is_noise():
    panel = synthetic_panel(signal=0.0, seed=3)
    assert auc(panel, "dd") == pytest.approx(0.5, abs=0.08)


def test_auc_is_high_when_the_score_carries_signal():
    assert auc(synthetic_panel(signal=1.0), "dd") > 0.85


def test_auc_is_nan_without_both_classes():
    panel = pd.DataFrame({"dd": [1.0, 2.0], "distress_12m": [0, 0], "cik": [1, 2]})
    assert np.isnan(auc(panel, "dd"))


def test_bootstrap_interval_brackets_the_point_estimate():
    panel = synthetic_panel(signal=1.0, seed=1)
    point = auc(panel, "dd")
    low, high = bootstrap_auc(panel, "dd", n_boot=120, seed=1)
    assert low <= point <= high
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_clusters_on_firms_not_rows():
    """Clustered resampling must not report a tighter interval than firm-level
    variation supports. With few firms the interval has to be wide."""
    narrow = bootstrap_auc(synthetic_panel(n_firms=400, seed=2), "dd", n_boot=120, seed=2)
    wide = bootstrap_auc(synthetic_panel(n_firms=40, seed=2), "dd", n_boot=120, seed=2)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_decile_table_is_monotone_for_a_real_signal():
    table = decile_table(synthetic_panel(signal=1.0), "dd", n_bins=5)
    rates = table["event_rate"].tolist()
    assert rates[0] > rates[-1]
    assert table["n"].sum() == 200 * 24


def test_score_comparison_reports_own_and_common_samples():
    """Scores must also be comparable on identical rows, or the sample does the work."""
    panel = synthetic_panel(signal=1.0)
    panel.loc[panel.index[:500], "altman_z"] = np.nan
    table = score_comparison(panel, {"dd": True, "altman_z": True}, n_boot=40)

    assert table.attrs["n_obs"] == len(panel) - 500
    assert set(table["score"]) == {"dd", "altman_z"}
    assert table["auc"].iloc[0] >= table["auc"].iloc[1]

    # dd exists on every row; altman_z only on the rows that survived.
    by_score = table.set_index("score")
    assert by_score.loc["dd", "n"] == len(panel)
    assert by_score.loc["altman_z", "n"] == len(panel) - 500
    assert by_score["auc_common"].notna().all()


def test_score_comparison_survives_an_empty_common_sample():
    """A score with no overlap must not turn the whole table into NaN."""
    panel = synthetic_panel(signal=1.0)
    panel["sparse"] = np.nan
    table = score_comparison(panel, {"dd": True, "sparse": True}, n_boot=20)
    assert table.loc[table["score"] == "dd", "auc"].iloc[0] > 0.8


def test_logistic_benchmark_splits_by_firm():
    """Grouped folds must not leak a firm between train and test.

    With firm-level labels, a row-wise split would score near 1.0 because the
    same firm's other months sit in training. A grouped split cannot.
    """
    result = logistic_benchmark(synthetic_panel(signal=1.0), ["dd"], n_splits=4)
    assert 0.5 < result["auc"] <= 1.0
    assert result["n"] == 200 * 24


def test_logistic_benchmark_finds_no_signal_in_noise():
    result = logistic_benchmark(synthetic_panel(signal=0.0, seed=7), ["dd"], n_splits=4)
    assert result["auc"] == pytest.approx(0.5, abs=0.12)
