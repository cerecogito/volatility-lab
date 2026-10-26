"""Tests for Monte Carlo pricing, its standard errors and interval coverage."""

import numpy as np
import pytest

from volatility_lab.black_scholes import bs_price
from volatility_lab.monte_carlo import mc_price

ARGS = dict(spot=100.0, strike=105.0, time_to_expiry=0.5, rate=0.03, volatility=0.25)


@pytest.mark.parametrize("method", ["plain", "antithetic", "control_variate"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_estimate_is_within_a_few_standard_errors(method, option_type):
    exact = float(bs_price(ARGS["spot"], ARGS["strike"], ARGS["time_to_expiry"], ARGS["rate"],
                           ARGS["volatility"], option_type))
    result = mc_price(**ARGS, option_type=option_type, num_paths=200_000,
                      rng=np.random.default_rng(42), method=method)
    assert abs(result.estimate - exact) < 4 * result.standard_error
    assert result.ci_low < result.estimate < result.ci_high


def test_variance_reduction_lowers_the_standard_error():
    errors = {}
    for method in ["plain", "antithetic", "control_variate"]:
        result = mc_price(**ARGS, option_type="call", num_paths=100_000,
                          rng=np.random.default_rng(7), method=method)
        errors[method] = result.standard_error
    assert errors["antithetic"] < errors["plain"]
    assert errors["control_variate"] < errors["plain"]


def test_standard_error_shrinks_like_one_over_root_n():
    small = mc_price(**ARGS, option_type="call", num_paths=10_000, rng=np.random.default_rng(1))
    large = mc_price(**ARGS, option_type="call", num_paths=160_000, rng=np.random.default_rng(1))
    # 16x more paths -> SE about 4x smaller
    assert small.standard_error / large.standard_error == pytest.approx(4.0, rel=0.1)


def test_same_seed_gives_identical_results():
    first = mc_price(**ARGS, option_type="put", num_paths=5000, rng=np.random.default_rng(99))
    second = mc_price(**ARGS, option_type="put", num_paths=5000, rng=np.random.default_rng(99))
    assert first.estimate == second.estimate


@pytest.mark.slow
@pytest.mark.parametrize("method", ["plain", "antithetic", "control_variate"])
def test_95_percent_interval_coverage(method):
    """Over many independent repeats about 95% of intervals should contain the true price."""
    exact = float(bs_price(ARGS["spot"], ARGS["strike"], ARGS["time_to_expiry"], ARGS["rate"],
                           ARGS["volatility"], "call"))
    rng = np.random.default_rng(2024)
    repeats = 400
    covered = 0
    for _ in range(repeats):
        result = mc_price(**ARGS, option_type="call", num_paths=4000, rng=rng, method=method)
        if result.ci_low <= exact <= result.ci_high:
            covered += 1
    coverage = covered / repeats
    # binomial(400, 0.95) has sd ~ 1.1%, so 4 sd either side is a generous band
    assert 0.906 <= coverage <= 0.994
