"""Tests for the two-state Markov regime model."""

import numpy as np
import pytest

from volatility_lab.regime_model import (
    RegimeModel, fit_regime_model, forward_filter, regime_variance_paths, regime_volatility_forecast,
)
from volatility_lab.simulation import simulate_regime_returns, stationary_distribution

TRANSITION = np.array([[0.98, 0.02], [0.05, 0.95]])


def test_em_recovers_known_parameters_from_simulated_data():
    rng = np.random.default_rng(11)
    returns, _ = simulate_regime_returns(1, 8000, (0.4, 1.2), TRANSITION, rng)
    model = fit_regime_model(returns[0])
    assert model.converged
    daily = np.array([0.4, 1.2]) / np.sqrt(365)
    np.testing.assert_allclose(model.stdevs, daily, rtol=0.08)
    np.testing.assert_allclose(np.diag(model.transition), np.diag(TRANSITION), atol=0.02)


def test_filtered_probabilities_are_valid_and_track_the_true_state():
    rng = np.random.default_rng(12)
    returns, states = simulate_regime_returns(1, 4000, (0.3, 1.5), TRANSITION, rng)
    model = fit_regime_model(returns[0])
    filtered, _ = forward_filter(returns[0], model)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0)
    assert np.all((filtered >= 0) & (filtered <= 1))
    accuracy = np.mean((filtered[:, 1] > 0.5) == (states[0] == 1))
    assert accuracy > 0.85


def test_em_log_likelihood_never_decreases():
    rng = np.random.default_rng(13)
    returns, _ = simulate_regime_returns(1, 1500, (0.5, 1.0), TRANSITION, rng)
    log_likelihoods = [fit_regime_model(returns[0], max_iterations=k).log_likelihood for k in [1, 3, 10, 50]]
    assert all(b >= a - 1e-6 for a, b in zip(log_likelihoods, log_likelihoods[1:]))


def test_long_horizon_forecast_tends_to_stationary_variance():
    model = RegimeModel(means=np.zeros(2), stdevs=np.array([0.01, 0.04]), transition=TRANSITION,
                        initial_probs=np.array([0.5, 0.5]), log_likelihood=0.0, iterations=0, converged=True)
    stationary = stationary_distribution(TRANSITION)
    expected = np.sqrt(stationary @ model.stdevs**2)
    calm_now = np.array([[1.0, 0.0]])
    forecast = regime_volatility_forecast(calm_now, model, horizon_days=5000, days_per_year=1)
    assert forecast[0] == pytest.approx(expected, rel=0.01)
    # short horizon from the calm state should be close to the calm volatility
    short = regime_volatility_forecast(calm_now, model, horizon_days=1, days_per_year=1)
    assert short[0] < 0.5 * (model.stdevs[0] + model.stdevs[1])


def test_variance_path_simulation_matches_analytic_forecast():
    model = RegimeModel(means=np.zeros(2), stdevs=np.array([0.01, 0.04]), transition=TRANSITION,
                        initial_probs=np.array([0.5, 0.5]), log_likelihood=0.0, iterations=0, converged=True)
    now = np.array([0.3, 0.7])
    simulated = regime_variance_paths(now, model, 30, 200_000, np.random.default_rng(1))
    analytic = regime_volatility_forecast(now[None, :], model, 30, days_per_year=1)[0] ** 2
    assert simulated.mean() == pytest.approx(analytic, rel=0.01)
