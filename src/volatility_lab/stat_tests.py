"""Paired statistical comparisons (MIT 18.6501x: CIs, hypothesis tests; 6.431x: CLT).

Cluster bootstrap
-----------------
Option trades on the same day share the same volatility estimate and the
same market moves, so their errors are NOT independent. We resample whole
clusters (market dates) with replacement, keeping all trades of a chosen
date together. For simulated hedges each path is its own cluster, which
reduces to the ordinary iid bootstrap.

Implementation: for every cluster we pre-compute the sums of the needed
per-observation quantities. A bootstrap replicate is just a vector of
"how many times was each cluster drawn" counts, so all replicates are a
single matrix product (counts @ cluster_sums). This keeps it fast and exact.

Diebold-Mariano
---------------
For a loss differential d_t = L_A,t - L_B,t over time, DM tests E[d] = 0
using a HAC (Newey-West) variance of the mean. We apply it to the
per-date average differential, so that one market date is one time point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cluster_sums(quantities: dict[str, np.ndarray], cluster_ids: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-cluster sums of each quantity, plus the number of observations per cluster."""
    frame = pd.DataFrame(quantities)
    frame["_cluster"] = np.asarray(cluster_ids)
    grouped = frame.groupby("_cluster", sort=True)
    return grouped.sum(), grouped.size().to_numpy()


def bootstrap_count_matrix(num_clusters: int, num_replicates: int, rng: np.random.Generator) -> np.ndarray:
    """(replicates, clusters) matrix: how often each cluster is drawn in each replicate."""
    return rng.multinomial(num_clusters, np.full(num_clusters, 1.0 / num_clusters), size=num_replicates)


def paired_metric_difference(errors_a: np.ndarray, errors_b: np.ndarray, cluster_ids: np.ndarray,
                             rng: np.random.Generator, num_replicates: int = 5000,
                             level: float = 0.95) -> dict:
    """Compare two models on IDENTICAL observations: differences in MAE, RMSE, bias and MSE.

    errors_* are signed errors (model - target) for the same observations.
    Differences are A - B, so a NEGATIVE MAE/RMSE difference means A is more accurate.
    Returns point estimates, percentile CIs from the cluster bootstrap and a
    two-sided bootstrap p-value for "no difference in MSE" and "no difference in MAE".
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if errors_a.shape != errors_b.shape:
        raise ValueError("paired comparison needs errors on the same observations")

    sums, counts = cluster_sums({
        "abs_a": np.abs(errors_a), "abs_b": np.abs(errors_b),
        "sq_a": errors_a**2, "sq_b": errors_b**2,
        "err_a": errors_a, "err_b": errors_b,
    }, cluster_ids)

    def metrics_from(totals: np.ndarray, total_count) -> dict[str, np.ndarray]:
        mean = {name: totals[..., i] / total_count for i, name in enumerate(sums.columns)}
        return {
            "mae_diff": mean["abs_a"] - mean["abs_b"],
            "rmse_diff": np.sqrt(mean["sq_a"]) - np.sqrt(mean["sq_b"]),
            "mse_diff": mean["sq_a"] - mean["sq_b"],
            "bias_diff": mean["err_a"] - mean["err_b"],
            "mae_ratio": mean["abs_a"] / mean["abs_b"],
        }

    point = metrics_from(sums.to_numpy().sum(axis=0), counts.sum())
    draw_counts = bootstrap_count_matrix(len(counts), num_replicates, rng)
    boot = metrics_from(draw_counts @ sums.to_numpy(), draw_counts @ counts)

    tail = (1.0 - level) / 2.0
    result = {"n_observations": int(len(errors_a)), "n_clusters": int(len(counts)),
              "bootstrap_replicates": num_replicates}
    for name, value in point.items():
        result[name] = float(value)
        result[f"{name}_ci_low"] = float(np.quantile(boot[name], tail))
        result[f"{name}_ci_high"] = float(np.quantile(boot[name], 1.0 - tail))
    for name in ["mse_diff", "mae_diff"]:
        # two-sided p-value from the centred bootstrap distribution
        centred = boot[name] - point[name]
        result[f"{name}_bootstrap_p"] = float(np.mean(np.abs(centred) >= abs(point[name])))
    return result


def newey_west_lag(num_points: int) -> int:
    """Common rule of thumb floor(4 (n/100)^(2/9))."""
    return int(np.floor(4.0 * (num_points / 100.0) ** (2.0 / 9.0)))


def diebold_mariano(loss_differential: np.ndarray, max_lag: int | None = None) -> dict:
    """DM test of E[d_t] = 0 with a Bartlett-kernel (Newey-West) long-run variance.

    Assumes d_t is covariance-stationary with weak dependence. Uses the
    Harvey-Leybourne-Newbold small-sample correction and a t(n-1) reference.
    """
    d = np.asarray(loss_differential, dtype=float)
    n = len(d)
    if n < 10:
        raise ValueError("too few time points for a Diebold-Mariano test")
    if max_lag is None:
        max_lag = newey_west_lag(n)
    centred = d - d.mean()
    long_run_variance = np.dot(centred, centred) / n
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * np.dot(centred[lag:], centred[:-lag]) / n
    if long_run_variance <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n": n, "lag": max_lag}
    dm_stat = d.mean() / np.sqrt(long_run_variance / n)
    horizon = max_lag + 1
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    adjusted = dm_stat * correction
    p_value = 2.0 * stats.t.sf(abs(adjusted), df=n - 1)
    return {"dm_stat": float(adjusted), "p_value": float(p_value), "n": n, "lag": max_lag,
            "mean_differential": float(d.mean())}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    names = sorted(p_values, key=lambda name: p_values[name])
    count = len(names)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, name in enumerate(names):
        value = min(1.0, (count - rank) * p_values[name])
        running_max = max(running_max, value)
        adjusted[name] = running_max
    return adjusted


def paired_t_test(values_a: np.ndarray, values_b: np.ndarray, level: float = 0.95) -> dict:
    """Paired t-test and CI for mean(A - B) on independent pairs (e.g. simulated paths)."""
    differences = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    n = len(differences)
    mean = differences.mean()
    standard_error = differences.std(ddof=1) / np.sqrt(n)
    critical = stats.t.ppf(0.5 + level / 2.0, df=n - 1)
    t_stat = mean / standard_error if standard_error > 0 else float("nan")
    return {"n": n, "mean_diff": float(mean), "ci_low": float(mean - critical * standard_error),
            "ci_high": float(mean + critical * standard_error), "t_stat": float(t_stat),
            "p_value": float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))}
