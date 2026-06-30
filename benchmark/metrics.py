"""
benchmark/metrics.py
--------------------
Computes accuracy and regression metrics between ground-truth scores and
system-predicted scores.

Metrics produced
~~~~~~~~~~~~~~~~
Regression (continuous):
  - MAE   (Mean Absolute Error) — average point-% gap
  - RMSE  (Root Mean Squared Error) — penalises large deviations
  - R²    (coefficient of determination) — % variance explained
  - Pearson r — linear correlation between predicted and actual scores

Ranking (ordinal agreement):
  - Spearman ρ — monotonic correlation (are high-scorers ranked highly?)
  - Kendall τ  — pairwise ranking concordance

Tolerance-based accuracy (practical "close enough" measure):
  - Exact match (%)         — predicted == ground-truth (to nearest %)
  - Within ±5 pp  (%)       — |predicted - ground_truth| ≤ 5
  - Within ±10 pp (%)       — |predicted - ground_truth| ≤ 10
  - Within ±15 pp (%)       — |predicted - ground_truth| ≤ 15

Classification accuracy (threshold at 50% to split Fit/Not-Fit):
  - Accuracy, Precision, Recall, F1-score

All results are returned as a flat dictionary and also as a human-readable
formatted string via `format_report()`.
"""

from __future__ import annotations

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / den if den != 0 else 0.0


def _rank(xs: Sequence[float]) -> list[float]:
    """Return fractional ranks (1-based, ties averaged)."""
    sorted_vals = sorted(enumerate(xs), key=lambda t: t[1])
    ranks: list[float] = [0.0] * len(xs)
    i = 0
    while i < len(sorted_vals):
        j = i
        # find the end of this tie group
        while j < len(sorted_vals) - 1 and sorted_vals[j][1] == sorted_vals[j + 1][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def _kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Kendall τ-b (handles ties)."""
    n = len(xs)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                if dx != 0:
                    ties_y += 1
                elif dy != 0:
                    ties_x += 1
    n0 = n * (n - 1) / 2
    den = math.sqrt((n0 - ties_x) * (n0 - ties_y))
    return (concordant - discordant) / den if den != 0 else 0.0


def _r_squared(actual: Sequence[float], predicted: Sequence[float]) -> float:
    mean_actual = _mean(actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def _classification_metrics(
    actual: Sequence[float], predicted: Sequence[float], threshold: float = 50.0
) -> dict[str, float]:
    """
    Binary classification metrics using `threshold` as the cut-off
    (scores >= threshold → positive / "Fit").
    """
    tp = fp = tn = fn = 0
    for a, p in zip(actual, predicted):
        a_pos = a >= threshold
        p_pos = p >= threshold
        if a_pos and p_pos:
            tp += 1
        elif not a_pos and p_pos:
            fp += 1
        elif a_pos and not p_pos:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total          if total  else 0.0
    precision = tp / (tp + fp)             if (tp + fp) else 0.0
    recall    = tp / (tp + fn)             if (tp + fn) else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "classification_threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "classification_accuracy_%": round(accuracy   * 100, 2),
        "precision_%":               round(precision  * 100, 2),
        "recall_%":                  round(recall     * 100, 2),
        "f1_%":                      round(f1         * 100, 2),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    method_name: str = "",
    classification_threshold: float = 50.0,
) -> dict:
    """
    Compute a full suite of accuracy/quality metrics.

    Parameters
    ----------
    actual : sequence of numbers
        Ground-truth scores (e.g., ``Score(%)`` column, 0-100).
    predicted : sequence of numbers
        System-predicted scores (0-100).
    method_name : str
        Label for this method (e.g., "CBOW", "TF-IDF").
    classification_threshold : float
        Score threshold to split Fit (≥) vs Not-Fit (<) for binary metrics.

    Returns
    -------
    dict
        Flat dict with all metric values.
    """
    n = len(actual)
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    signed = [p - a      for a, p in zip(actual, predicted)]

    mae   = _mean(errors)
    rmse  = math.sqrt(_mean([e ** 2 for e in errors]))
    r2    = _r_squared(actual, predicted)
    pear  = _pearson(actual, predicted)
    spear = _spearman(actual, predicted)

    # Tolerance accuracy
    exact       = sum(1 for e in errors if e == 0)  / n * 100
    within_5    = sum(1 for e in errors if e <=  5) / n * 100
    within_10   = sum(1 for e in errors if e <= 10) / n * 100
    within_15   = sum(1 for e in errors if e <= 15) / n * 100

    # Bias
    mean_bias = _mean(signed)   # positive → system over-predicts

    cls = _classification_metrics(actual, predicted, classification_threshold)

    result = {
        "method":       method_name,
        "n_samples":    n,
        # Regression
        "mae":          round(mae,   2),
        "rmse":         round(rmse,  2),
        "r2":           round(r2,    4),
        "pearson_r":    round(pear,  4),
        "spearman_rho": round(spear, 4),
        # Tolerance accuracy
        "exact_match_%":  round(exact,    2),
        "within_5pp_%":   round(within_5, 2),
        "within_10pp_%":  round(within_10, 2),
        "within_15pp_%":  round(within_15, 2),
        # Bias
        "mean_bias_pp":  round(mean_bias, 2),
    }
    result.update(cls)
    return result


def format_report(metrics_list: list[dict]) -> str:
    """
    Format a list of metric dicts (one per method) into a readable report.

    Parameters
    ----------
    metrics_list : list of dicts produced by ``compute_metrics()``.

    Returns
    -------
    str
        Multi-line formatted report string.
    """
    lines = ["=" * 70, "  JD-to-Resume Matching API  —  Benchmark Report", "=" * 70]

    for m in metrics_list:
        lines.append("")
        method_label = m.get("method", "Unknown")
        lines.append(f"  Method : {method_label}")
        lines.append(f"  Samples: {m['n_samples']}")
        lines.append("")
        lines.append("  ── Regression Metrics ──────────────────────────────────────────")
        lines.append(f"    MAE   (Mean Absolute Error)  : {m['mae']:>7.2f} pp")
        lines.append(f"    RMSE  (Root Mean Sq. Error)  : {m['rmse']:>7.2f} pp")
        lines.append(f"    R²    (Variance Explained)   : {m['r2']:>7.4f}")
        lines.append(f"    Pearson r                    : {m['pearson_r']:>7.4f}")
        lines.append(f"    Spearman ρ                   : {m['spearman_rho']:>7.4f}")
        lines.append(f"    Mean Bias (system - GT)      : {m['mean_bias_pp']:>+7.2f} pp")
        lines.append("")
        lines.append("  ── Tolerance-Based Accuracy ────────────────────────────────────")
        lines.append(f"    Exact Match (0 pp tolerance) : {m['exact_match_%']:>6.2f} %")
        lines.append(f"    Within ±5 pp                 : {m['within_5pp_%']:>6.2f} %")
        lines.append(f"    Within ±10 pp                : {m['within_10pp_%']:>6.2f} %")
        lines.append(f"    Within ±15 pp                : {m['within_15pp_%']:>6.2f} %")
        lines.append("")
        thr = m.get("classification_threshold", 50)
        lines.append(f"  ── Binary Classification (threshold={thr}%) ───────────────────")
        lines.append(f"    Accuracy  : {m['classification_accuracy_%']:>6.2f} %   "
                     f"(TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']})")
        lines.append(f"    Precision : {m['precision_%']:>6.2f} %")
        lines.append(f"    Recall    : {m['recall_%']:>6.2f} %")
        lines.append(f"    F1-Score  : {m['f1_%']:>6.2f} %")
        lines.append("")
        lines.append("  " + "-" * 66)

    lines.append("=" * 70)
    return "\n".join(lines)
