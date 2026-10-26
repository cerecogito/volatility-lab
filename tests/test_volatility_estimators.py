"""Tests for rolling and EWMA volatility estimators."""

import numpy as np
import pytest

from volatility_lab.volatility_estimators import (
    ewma_volatility, realised_volatility_ahead, rolling_volatility,
)


def test_rolling_matches_numpy_std_on_each_window():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.03, 300)
    window = 30
    estimates = rolling_volatility(returns, window, days_per_year=365)
    assert np.all(np.isnan(estimates[: window - 1]))
    for t in [29, 100, 299]:
        expected = np.std(returns[t - window + 1: t + 1], ddof=1) * np.sqrt(365)
        assert estimates[t] == pytest.approx(expected, rel=1e-10)


def test_course_annualisation_example():
    # course replication example: daily st dev 0.019978 x sqrt(252) = 0.317139.
    # 0.019978 is itself rounded to 6 dp (+-5e-7), which is +-8e-6 after x sqrt(252)
    assert 0.019978 * np.sqrt(252) == pytest.approx(0.317139, abs=1e-5)


def test_rolling_works_on_2d_arrays_row_by_row():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.02, (5, 120))
    together = rolling_volatility(returns, 20)
    one_by_one = np.vstack([rolling_volatility(row, 20) for row in returns])
    np.testing.assert_allclose(together, one_by_one)


def test_rolling_estimate_is_consistent_for_constant_volatility():
    # law of large numbers: with a long window the estimate approaches the truth
    rng = np.random.default_rng(2)
    true_daily_vol = 0.02
    returns = rng.normal(0, true_daily_vol, 200_000)
    estimate = rolling_volatility(returns, 100_000, days_per_year=1)[-1]
    assert estimate == pytest.approx(true_daily_vol, rel=0.01)


def test_ewma_recursion_by_hand():
    returns = np.array([0.01, -0.02, 0.03, 0.00, -0.01])
    decay = 0.9
    estimates = ewma_volatility(returns, decay, days_per_year=1, seed_window=2)
    variance = (0.01**2 + 0.02**2) / 2
    expected = [np.nan, variance]
    for r in returns[2:]:
        variance = decay * variance + (1 - decay) * r**2
        expected.append(variance)
    np.testing.assert_allclose(estimates[1:], np.sqrt(expected[1:]))
    assert np.isnan(estimates[0])


def test_ewma_reacts_faster_than_long_rolling_window():
    calm = np.full(200, 0.01) * np.where(np.arange(200) % 2, 1, -1)
    storm = np.full(20, 0.05) * np.where(np.arange(20) % 2, 1, -1)
    returns = np.concatenate([calm, storm])
    ewma = ewma_volatility(returns, 0.94, days_per_year=1)
    rolling = rolling_volatility(returns, 180, days_per_year=1)
    assert ewma[-1] > rolling[-1]


def test_invalid_parameters():
    with pytest.raises(ValueError):
        rolling_volatility(np.zeros(10), 1)
    with pytest.raises(ValueError):
        ewma_volatility(np.zeros(10), 1.0)


def test_realised_volatility_ahead_uses_the_next_h_returns():
    returns = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    realised = realised_volatility_ahead(returns, horizon=2, days_per_year=1)
    assert realised[0] == pytest.approx(np.sqrt((0.1**2 + 0.2**2) / 2))
    assert np.isnan(realised[-1]) and np.isnan(realised[-2])
