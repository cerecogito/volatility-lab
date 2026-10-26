"""Tests for metrics and the statistical comparison tools."""

import numpy as np
import pytest

from volatility_lab.metrics import expected_shortfall, hedging_error_summary, pricing_error_summary, value_at_risk
from volatility_lab.stat_tests import (
    diebold_mariano, holm_adjust, paired_metric_difference, paired_t_test,
)


def test_pricing_error_summary_by_hand():
    summary = pricing_error_summary(np.array([1.0, 3.0, 2.0]), np.array([2.0, 2.0, 2.0]), relative_floor=1.0)
    assert summary["bias"] == pytest.approx(0.0)
    assert summary["mae"] == pytest.approx(2 / 3)
    assert summary["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert summary["median_ae"] == pytest.approx(1.0)
    assert summary["mean_abs_relative_error"] == pytest.approx(1 / 3)


def test_var_and_expected_shortfall():
    losses = np.arange(1, 101, dtype=float)       # 1..100
    assert value_at_risk(losses, 0.95) == pytest.approx(np.quantile(losses, 0.95))
    assert expected_shortfall(losses, 0.95) == pytest.approx(np.mean(losses[losses >= np.quantile(losses, 0.95)]))
    summary = hedging_error_summary(-losses)
    assert summary["var95_loss"] == pytest.approx(value_at_risk(losses))
    assert summary["mean"] == pytest.approx(-50.5)


def test_paired_difference_point_estimates_and_ci_cover_truth():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(200), 10)
    day_effect = np.repeat(rng.normal(0, 1, 200), 10)    # shared within a cluster
    errors_b = day_effect + rng.normal(0, 1, 2000)
    errors_a = 0.8 * errors_b                            # A is strictly better
    result = paired_metric_difference(errors_a, errors_b, clusters, rng, num_replicates=2000)
    assert result["mae_diff"] == pytest.approx(-0.2 * np.mean(np.abs(errors_b)))
    assert result["mae_diff_ci_high"] < 0
    assert result["n_clusters"] == 200
    assert result["mse_diff_bootstrap_p"] < 0.01


def test_cluster_bootstrap_is_wider_than_naive_when_errors_are_clustered():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(50), 40)
    shared = np.repeat(rng.normal(0, 1, 50), 40)
    errors_a = shared + rng.normal(0, 0.2, 2000)
    errors_b = rng.normal(0, 0.2, 2000)
    clustered = paired_metric_difference(errors_a, errors_b, clusters, rng, 2000)
    naive = paired_metric_difference(errors_a, errors_b, np.arange(2000), rng, 2000)
    width = lambda r: r["bias_diff_ci_high"] - r["bias_diff_ci_low"]  # noqa: E731
    assert width(clustered) > 2 * width(naive)


def test_no_difference_gives_large_p_value_most_of_the_time():
    rng = np.random.default_rng(2)
    rejections = 0
    for _ in range(40):
        a, b = rng.normal(0, 1, 300), rng.normal(0, 1, 300)
        result = paired_metric_difference(a, b, np.arange(300), rng, 1000)
        rejections += result["mse_diff_bootstrap_p"] < 0.05
    assert rejections <= 7     # expected about 2 of 40


def test_diebold_mariano_detects_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(3)
    assert diebold_mariano(rng.normal(0.5, 1, 300))["p_value"] < 0.001
    assert diebold_mariano(rng.normal(0.0, 1, 300))["p_value"] > 0.01


def test_holm_adjustment():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)   # monotone: never below the previous one


def test_paired_t_test():
    result = paired_t_test(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 1.5, 2.5, 3.9]))
    assert result["mean_diff"] == pytest.approx(0.4)
    assert result["ci_low"] < 0.4 < result["ci_high"]
