"""Measurement suite for the attrition fairness pipeline.

Called unchanged by the baseline, reweighing and equalised odds notebooks so that
the three rounds are directly comparable.

Label convention
----------------
y is encoded 1 = attrition, 0 = retention, which keeps recall and PR-AUC pointed at
the minority class. Retention is the favourable outcome, so the fairness metrics are
computed on the flipped labels (1 = retained) following the AIF360 convention that
the favourable outcome is the positive class.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PERFORMANCE_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "balanced_accuracy",
    "pr_auc",
    "brier",
]

FAIRNESS_METRICS = ["spd", "aod", "ppd"]


def performance(y_true, y_pred, y_prob):
    """The eight predictive performance measures, positive class = attrition."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def _rate(mask, condition):
    """Proportion of `condition` within `mask`; NaN when the group is empty."""
    n = mask.sum()
    if n == 0:
        return np.nan
    return (mask & condition).sum() / n


def fairness(y_true, y_pred, privileged):
    """Statistical parity, average odds and predictive parity differences.

    Each is computed as unprivileged minus privileged, so 0 means parity and the
    sign says which group the model favours. Retention is the favourable outcome.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    privileged = np.asarray(privileged).astype(bool)

    # Flip so the favourable outcome (retention) is the positive class.
    fav_true = y_true == 0
    fav_pred = y_pred == 0

    unpriv = ~privileged

    # Statistical parity: rate of favourable prediction in each group.
    spd = _rate(unpriv, fav_pred) - _rate(privileged, fav_pred)

    # Average odds: mean of the true positive and false positive rate gaps.
    tpr_u = _rate(unpriv & fav_true, fav_pred)
    tpr_p = _rate(privileged & fav_true, fav_pred)
    fpr_u = _rate(unpriv & ~fav_true, fav_pred)
    fpr_p = _rate(privileged & ~fav_true, fav_pred)
    aod = 0.5 * ((fpr_u - fpr_p) + (tpr_u - tpr_p))

    # Predictive parity: precision of the favourable prediction in each group.
    ppv_u = _rate(unpriv & fav_pred, fav_true)
    ppv_p = _rate(privileged & fav_pred, fav_true)
    ppd = ppv_u - ppv_p

    return {"spd": spd, "aod": aod, "ppd": ppd}


def measure(y_true, y_pred, y_prob, protected, n_boot=1000, seed=42):
    """Every metric with a 95% bootstrap confidence interval.

    `protected` is a DataFrame of binary columns, 1 = privileged. Returns a tidy
    frame of metric, attribute, point estimate and interval bounds.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    protected = pd.DataFrame(protected).reset_index(drop=True)

    point = {("performance", k): v for k, v in performance(y_true, y_pred, y_prob).items()}
    for col in protected.columns:
        for k, v in fairness(y_true, y_pred, protected[col]).items():
            point[(col, k)] = v

    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = {key: [] for key in point}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # A resample can miss a class entirely, which makes the AUCs undefined.
        if len(np.unique(y_true[idx])) < 2:
            continue

        for k, v in performance(y_true[idx], y_pred[idx], y_prob[idx]).items():
            draws[("performance", k)].append(v)
        for col in protected.columns:
            group = protected[col].to_numpy()[idx]
            for k, v in fairness(y_true[idx], y_pred[idx], group).items():
                draws[(col, k)].append(v)

    rows = []
    for (attribute, metric), value in point.items():
        sample = np.asarray(draws[(attribute, metric)], dtype=float)
        sample = sample[~np.isnan(sample)]
        low, high = np.percentile(sample, [2.5, 97.5]) if sample.size else (np.nan, np.nan)
        rows.append({
            "attribute": attribute,
            "metric": metric,
            "value": value,
            "ci_low": low,
            "ci_high": high,
            "n_boot": sample.size,
        })

    return pd.DataFrame(rows)


def measure_all(predictions, model_names, protected_cols, n_boot=1000, seed=42):
    """Run `measure` for every model in a predictions frame."""
    out = []
    for name in model_names:
        slug = name.lower().replace(" ", "_")
        result = measure(
            predictions["y_true"],
            predictions[slug + "_pred"],
            predictions[slug + "_prob"],
            predictions[protected_cols],
            n_boot=n_boot,
            seed=seed,
        )
        result.insert(0, "model", name)
        out.append(result)
    return pd.concat(out, ignore_index=True)
