"""Tests for the Normal-Inverse-Gamma model and Bayesian option pricing."""

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import invgamma

from volatility_lab.bayesian import (
    NIGParameters, bayesian_option_delta, bayesian_option_price, make_prior,
    posterior_mean_variance, posterior_variance_interval, predictive_distribution,
    rolling_posteriors, update_posterior,
)
from volatility_lab.black_scholes import bs_price


def test_conjugate_update_matches_hand_formulas():
    prior = NIGParameters(mean=0.0, kappa=2.0, alpha=3.0, beta=0.002)
    data = np.array([0.01, -0.02, 0.015, 0.005])
    post = update_posterior(prior, data)
    xbar = data.mean()
    ss = np.sum((data - xbar) ** 2)
    assert post.kappa == pytest.approx(6.0)
    assert post.mean == pytest.approx(4 * xbar / 6)
    assert post.alpha == pytest.approx(5.0)
    assert post.beta == pytest.approx(0.002 + ss / 2 + 2 * 4 * xbar**2 / (2 * 6))


def test_sequential_updating_equals_batch_updating():
    # Bayes' rule: updating one observation at a time gives the same posterior
    rng = np.random.default_rng(3)
    data = rng.normal(0.001, 0.02, 50)
    prior = make_prior(10.0, 0.0004)
    batch = update_posterior(prior, data)
    sequential = prior
    for x in data:
        sequential = update_posterior(sequential, np.array([x]))
    for field in ["mean", "kappa", "alpha", "beta"]:
        assert getattr(sequential, field) == pytest.approx(getattr(batch, field), rel=1e-10)


def test_posterior_matches_brute_force_grid_posterior():
    """Check the conjugate posterior for s2 against prior x likelihood on a grid.

    Integrating mu out analytically is part of the derivation, so here we do
    the full 2-d grid (mu, s2) numerically and compare the marginal mean of s2.
    """
    rng = np.random.default_rng(4)
    data = rng.normal(0.0, 0.02, 15)
    prior = NIGParameters(mean=0.0, kappa=1.0, alpha=2.5, beta=0.0006)
    post = update_posterior(prior, data)

    mus = np.linspace(-0.04, 0.04, 401)
    variances = np.linspace(1e-5, 0.003, 1200)
    mu_grid, var_grid = np.meshgrid(mus, variances, indexing="ij")
    log_prior = (invgamma.logpdf(var_grid, a=prior.alpha, scale=prior.beta)
                 - 0.5 * np.log(2 * np.pi * var_grid / prior.kappa)
                 - prior.kappa * (mu_grid - prior.mean) ** 2 / (2 * var_grid))
    log_like = sum(-0.5 * np.log(2 * np.pi * var_grid) - (x - mu_grid) ** 2 / (2 * var_grid) for x in data)
    weights = np.exp(log_prior + log_like - np.max(log_prior + log_like))
    grid_mean_variance = np.sum(weights * var_grid) / np.sum(weights)
    assert grid_mean_variance == pytest.approx(post.beta / (post.alpha - 1), rel=2e-3)


def test_rolling_posteriors_match_single_updates():
    rng = np.random.default_rng(5)
    returns = rng.normal(0, 0.03, 100)
    prior = make_prior(10.0, 0.001)
    rolled = rolling_posteriors(prior, returns, window=30)
    for t in [29, 60, 99]:
        direct = update_posterior(prior, returns[t - 29: t + 1])
        assert rolled["alpha"][t] == pytest.approx(direct.alpha)
        assert rolled["beta"][t] == pytest.approx(direct.beta, rel=1e-9)
        assert rolled["mean"][t] == pytest.approx(direct.mean, abs=1e-12)
    assert np.isnan(rolled["alpha"][28])


def test_posterior_concentrates_on_truth_with_more_data():
    rng = np.random.default_rng(6)
    true_variance = 0.0009
    prior = make_prior(5.0, 0.004)            # deliberately wrong prior guess
    widths = []
    for n in [50, 500, 5000]:
        post = update_posterior(prior, rng.normal(0, np.sqrt(true_variance), n))
        low, high = posterior_variance_interval(post.alpha, post.beta)
        widths.append(high - low)
        if n == 5000:
            assert low < true_variance < high
            assert posterior_mean_variance(post.alpha, post.beta) == pytest.approx(true_variance, rel=0.06)
    assert widths[0] > widths[1] > widths[2]


def test_predictive_is_student_t_with_correct_variance():
    post = NIGParameters(mean=0.001, kappa=40.0, alpha=21.0, beta=0.008)
    predictive = predictive_distribution(post)
    # Var of predictive = E[s2] (1 + 1/kappa)
    expected_variance = post.beta / (post.alpha - 1) * (1 + 1 / post.kappa)
    assert predictive.var() == pytest.approx(expected_variance, rel=1e-10)
    assert predictive.mean() == pytest.approx(0.001)


def test_bayesian_price_matches_direct_integration():
    alpha, beta = 12.0, 12.0 * 0.0008
    spot, strike, time = 100.0, 110.0, 30 / 365

    def integrand(s2):
        return float(bs_price(spot, strike, time, 0.0, np.sqrt(365 * s2), "call")) * invgamma.pdf(s2, a=alpha, scale=beta)

    exact, _ = integrate.quad(integrand, 1e-7, 0.02, limit=400)
    approx = bayesian_option_price(spot, strike, time, 0.0, "call", alpha, beta, num_nodes=256)
    assert approx == pytest.approx(exact, rel=1e-3)


def test_bayesian_price_differs_from_plug_in_price_by_jensen():
    """For an OTM option BS is convex in sigma near here, so averaging over
    uncertainty raises the price above the plug-in price at E[sigma]."""
    alpha, beta = 6.0, 6.0 * 0.0008          # wide posterior (few observations)
    spot, strike, time = 100.0, 125.0, 30 / 365
    nodes = np.sqrt(365 * invgamma.ppf((np.arange(2000) + 0.5) / 2000, a=alpha, scale=beta))
    plug_in = float(bs_price(spot, strike, time, 0.0, nodes.mean(), "call"))
    bayes = float(bayesian_option_price(spot, strike, time, 0.0, "call", alpha, beta))
    assert bayes > plug_in


def test_bayesian_prices_satisfy_put_call_parity_and_vectorise():
    alpha = np.array([10.0, 30.0, 80.0])
    beta = alpha * 0.0005
    strikes = np.array([90.0, 100.0, 120.0])
    calls = bayesian_option_price(100.0, strikes, 0.2, 0.0, "call", alpha, beta)
    puts = bayesian_option_price(100.0, strikes, 0.2, 0.0, "put", alpha, beta)
    np.testing.assert_allclose(calls - puts, 100.0 - strikes, atol=1e-10)


def test_bayesian_delta_matches_finite_difference():
    args = dict(strike=105.0, time_to_expiry=0.1, rate=0.0, option_type="call", alpha=20.0, beta=0.01)
    h = 1e-3
    up = bayesian_option_price(spot=100.0 + h, **args)
    down = bayesian_option_price(spot=100.0 - h, **args)
    assert bayesian_option_delta(spot=100.0, **args) == pytest.approx((up - down) / (2 * h), rel=1e-6)


def test_prior_validation():
    with pytest.raises(ValueError):
        make_prior(0.0, 0.001)
    with pytest.raises(ValueError):
        NIGParameters(0.0, 1.0, -1.0, 1.0)
