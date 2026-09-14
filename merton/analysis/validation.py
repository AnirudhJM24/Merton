"""Does the model rank credit risk? Discrimination, calibration, benchmarks.

The point of this module is to make the model falsifiable. A structural model
that produces confident-looking default probabilities is worth nothing until
someone checks whether the firms it flagged actually defaulted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL = "default_12m"


def _clean(panel: pd.DataFrame, score: str, label: str = LABEL):
    frame = panel[[score, label]].replace([np.inf, -np.inf], np.nan).dropna()
    return frame[score].to_numpy(), frame[label].to_numpy()


def auc(panel: pd.DataFrame, score: str, label: str = LABEL, higher_is_safer: bool = True) -> float:
    """Area under the ROC curve for a one-dimensional score.

    ``higher_is_safer`` flips the sign for measures like distance-to-default,
    where a large value means low risk; AUC is then comparable across scores
    that point in opposite directions.
    """
    x, y = _clean(panel, score, label)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, -x if higher_is_safer else x))


def accuracy_ratio(panel: pd.DataFrame, score: str, **kwargs) -> float:
    """Gini / accuracy ratio, the convention in credit scoring: 2*AUC - 1."""
    return 2 * auc(panel, score, **kwargs) - 1


def bootstrap_auc(panel: pd.DataFrame, score: str, label: str = LABEL,
                  higher_is_safer: bool = True, n_boot: int = 500,
                  seed: int = 0) -> tuple[float, float]:
    """Confidence interval for AUC, resampling *firms* rather than rows.

    Firm-months are strongly autocorrelated -- one firm contributes a hundred
    near-identical rows -- so resampling rows would report a precision the data
    does not have. Clustering the bootstrap on the firm is the honest version.
    """
    frame = panel[[score, label, "cik"]].replace([np.inf, -np.inf], np.nan).dropna()
    rng = np.random.default_rng(seed)
    firms = frame["cik"].unique()
    by_firm = {cik: part for cik, part in frame.groupby("cik")}

    scores = []
    for _ in range(n_boot):
        drawn = rng.choice(firms, size=len(firms), replace=True)
        sample = pd.concat([by_firm[c] for c in drawn], ignore_index=True)
        if sample[label].nunique() < 2:
            continue
        s = sample[score].to_numpy()
        scores.append(roc_auc_score(sample[label].to_numpy(), -s if higher_is_safer else s))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def roc_points(panel: pd.DataFrame, score: str, label: str = LABEL,
               higher_is_safer: bool = True):
    x, y = _clean(panel, score, label)
    return roc_curve(y, -x if higher_is_safer else x)


def score_comparison(panel: pd.DataFrame, scores: dict[str, bool],
                     label: str = LABEL, n_boot: int = 300) -> pd.DataFrame:
    """AUC table across competing scores, on the rows where all of them exist.

    Restricting to the common sample matters: a score that is only available
    for large, healthy filers would otherwise look good by dint of a friendlier
    sample rather than better discrimination.
    """
    columns = list(scores) + [label, "cik"]
    common = panel[columns].replace([np.inf, -np.inf], np.nan).dropna()

    rows = []
    for name, higher_is_safer in scores.items():
        lo, hi = bootstrap_auc(common, name, label, higher_is_safer, n_boot=n_boot)
        rows.append({
            "score": name,
            "auc": auc(common, name, label, higher_is_safer),
            "ci_low": lo,
            "ci_high": hi,
            "accuracy_ratio": accuracy_ratio(common, name, label=label,
                                             higher_is_safer=higher_is_safer),
        })
    out = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    out.attrs["n_obs"] = len(common)
    out.attrs["n_defaults"] = int(common[label].sum())
    return out


def decile_table(panel: pd.DataFrame, score: str = "dd", label: str = LABEL,
                 n_bins: int = 10) -> pd.DataFrame:
    """Realized default rate by score bucket -- the monotonicity check.

    A usable risk measure does not merely separate defaulters on average; the
    default rate should fall monotonically as the measure improves.
    """
    frame = panel[[score, label]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    frame["bucket"] = pd.qcut(frame[score], n_bins, labels=False, duplicates="drop")
    table = frame.groupby("bucket").agg(
        n=(label, "size"),
        defaults=(label, "sum"),
        default_rate=(label, "mean"),
        score_low=(score, "min"),
        score_high=(score, "max"),
    ).reset_index()
    table["default_rate_pct"] = 100 * table["default_rate"]
    return table


def empirical_vs_model_pd(panel: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    """Compare the model's risk-neutral PD with the realized default frequency.

    This is the credit-spread puzzle in table form. Grouping by distance-to-
    default, each bucket's mean model PD sits next to the fraction of those
    firm-months that actually defaulted within a year.
    """
    cols = ["dd", "pd", LABEL]
    frame = panel[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    frame["bucket"] = pd.qcut(frame["dd"], n_bins, labels=False, duplicates="drop")
    table = frame.groupby("bucket").agg(
        n=("dd", "size"),
        dd_mid=("dd", "median"),
        model_pd=("pd", "mean"),
        empirical_pd=(LABEL, "mean"),
    ).reset_index()
    table["ratio"] = table["empirical_pd"] / table["model_pd"].replace(0, np.nan)
    return table


def logistic_benchmark(panel: pd.DataFrame, features: list[str], label: str = LABEL,
                       n_splits: int = 5, seed: int = 0) -> dict:
    """Out-of-sample AUC for a logistic model, with folds split by firm.

    Splitting by firm rather than by row is the whole point. A random row split
    would put January and February of the same defaulting firm on both sides of
    the fold and report an AUC that could never be achieved on a firm the model
    has not already seen default.
    """
    frame = panel[features + [label, "cik"]].replace([np.inf, -np.inf], np.nan).dropna()
    if frame[label].nunique() < 2:
        return {"auc": float("nan"), "n": len(frame)}

    X, y, groups = frame[features].to_numpy(), frame[label].to_numpy(), frame["cik"].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))
    predictions = np.full(len(frame), np.nan)

    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    mask = ~np.isnan(predictions)
    if len(np.unique(y[mask])) < 2:
        return {"auc": float("nan"), "n": int(mask.sum())}
    return {
        "auc": float(roc_auc_score(y[mask], predictions[mask])),
        "n": int(mask.sum()),
        "features": features,
    }
