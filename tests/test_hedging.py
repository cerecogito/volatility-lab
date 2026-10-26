"""Tests for the delta-hedging simulator.

The first test reproduces the Caltech course's replication experiment
(section 27): a call on Microsoft stock over 20 trading days in 2000, with
sigma = 0.317139, K = 55, r = 0. The stock prices and the call price /
delta / wealth columns below are the values printed in the course table.
"""

import numpy as np
import pytest

from volatility_lab.black_scholes import bs_delta, bs_price
from volatility_lab.hedging import run_delta_hedge
from volatility_lab.simulation import prices_from_returns, simulate_gbm_returns

COURSE_STOCK = [55.5625, 55.3750, 55.4375, 56.5625, 59.1250, 60.3125, 61.3125, 60.6250, 62.6875, 61.2500,
                63.2500, 64.1875, 64.2500, 65.0000, 63.0000, 64.1875, 65.8125, 68.2500, 68.1250, 68.8125]
COURSE_CALL = [2.2715, 2.1176, 2.1009, 2.7256, 4.5831, 5.5702, 6.4562, 5.7970, 7.7346, 6.3360,
               8.2681, 9.1933, 9.2531, 10.0007, 8.0025, 9.1877, 10.8125, 13.2500, 13.1250, 13.8125]
COURSE_DELTA = [0.5629, 0.5483, 0.5539, 0.6481, 0.8268, 0.8899, 0.9313, 0.9166, 0.9724, 0.9507,
                0.9873, 0.9953, 0.9972, 0.9992, 0.9974, 0.9997, 1.0000, 1.0000, 1.0000, 1.0000]
COURSE_WEALTH = [2.2715, 2.1660, 2.2003, 2.8235, 4.4841, 5.4659, 6.3558, 5.7155, 7.6060, 6.2082,
                 8.1095, 9.0351, 9.0973, 9.8452, 7.8467, 9.0310, 10.6556, 13.0931, 12.9681, 13.6556]


def test_reproduces_course_replication_experiment():
    stock = np.array(COURSE_STOCK)
    sigma, strike = 0.317139, 55.0
    # The slide says T = 20/252 = 0.08 years, and the course spreadsheet uses the
    # ROUNDED 0.08: with 0.08 * (20 - k)/20 years left at row k every printed
    # price matches to 4 dp (with the unrounded 20/252 they are ~0.008 lower).
    times_left = 0.08 * (20 - np.arange(19)) / 20.0
    calls = bs_price(stock[:19], strike, times_left, 0.0, sigma, "call")
    deltas = bs_delta(stock[:19], strike, times_left, 0.0, sigma, "call")
    np.testing.assert_allclose(calls, COURSE_CALL[:19], atol=6e-4)
    np.testing.assert_allclose(deltas, COURSE_DELTA[:19], atol=6e-4)

    result = run_delta_hedge(stock[None, :], deltas[None, :], strike, "call",
                             premium=calls[0], rate=0.0, step_years=1 / 252)
    np.testing.assert_allclose(result.wealth_path[0], COURSE_WEALTH, atol=1e-3)
    # course: loss with hedging = 13.8125 - 13.6556 = 0.1569
    assert -result.hedge_error[0] == pytest.approx(0.1569, abs=1e-3)
    assert result.terminal_payoff[0] == pytest.approx(13.8125)


def test_cash_account_grows_at_the_risk_free_rate():
    # zero hedge: the premium just sits in the bank
    prices = np.array([[100.0, 105.0, 95.0, 120.0]])
    result = run_delta_hedge(prices, np.zeros((1, 3)), 200.0, "call", premium=10.0, rate=0.05, step_years=0.25)
    assert result.hedge_error[0] == pytest.approx(10.0 * np.exp(0.05 * 0.75))


def test_constant_hedge_is_buy_and_hold():
    prices = np.array([[100.0, 110.0, 90.0, 130.0]])
    result = run_delta_hedge(prices, np.ones((1, 3)), 100.0, "call", premium=0.0, rate=0.0, step_years=1.0)
    # long one share bought at 100 -> worth 130; short call pays 30
    assert result.hedge_error[0] == pytest.approx(0.0)


def test_transaction_costs_and_turnover_by_hand():
    prices = np.array([[100.0, 110.0, 120.0]])
    ratios = np.array([[0.5, 0.8]])
    cost_rate = 0.01
    result = run_delta_hedge(prices, ratios, 100.0, "call", premium=5.0, rate=0.0, step_years=1.0,
                             cost_rate=cost_rate)
    expected_costs = cost_rate * (0.5 * 100 + 0.3 * 110 + 0.8 * 120)
    assert result.transaction_costs[0] == pytest.approx(expected_costs)
    assert result.share_turnover[0] == pytest.approx(0.5 + 0.3 + 0.8)
    cash = 5.0 - 50.0 - 0.5 - 33.0 - 0.33
    expected_error = cash + 0.8 * 120 - 0.96 - 20.0
    assert result.hedge_error[0] == pytest.approx(expected_error)


def test_put_payoff_handling():
    prices = np.array([[100.0, 80.0]])
    result = run_delta_hedge(prices, np.zeros((1, 1)), 90.0, "put", premium=0.0, rate=0.0, step_years=1.0)
    assert result.terminal_payoff[0] == pytest.approx(10.0)
    assert result.hedge_error[0] == pytest.approx(-10.0)


def test_hedge_error_shrinks_with_more_frequent_rebalancing_in_bs_world():
    """With the true sigma, the hedging error std falls roughly like 1/sqrt(rebalances)."""
    rng = np.random.default_rng(21)
    sigma, strike, spot, maturity_days = 0.5, 100.0, 100.0, 64
    returns = simulate_gbm_returns(4000, maturity_days, 0.1, sigma, rng)
    daily_prices = prices_from_returns(spot, returns)
    stds = []
    for every in [16, 4, 1]:
        prices = daily_prices[:, ::every]
        times_left = (maturity_days - np.arange(0, maturity_days, every)) / 365.0
        deltas = bs_delta(prices[:, :-1], strike, times_left[None, :], 0.0, sigma, "call")
        premium = float(bs_price(spot, strike, maturity_days / 365.0, 0.0, sigma, "call"))
        result = run_delta_hedge(prices, deltas, strike, "call", premium, 0.0, every / 365.0)
        stds.append(result.hedge_error.std())
        assert abs(result.hedge_error.mean()) < 4 * result.hedge_error.std() / np.sqrt(4000) + 0.05
    assert stds[0] > stds[1] > stds[2]
    # 16x more rebalances -> std about 4x smaller (allow slack)
    assert 2.5 < stds[0] / stds[2] < 5.5


def test_shape_and_nan_validation():
    with pytest.raises(ValueError):
        run_delta_hedge(np.ones((2, 4)), np.ones((2, 4)), 1.0, "call", 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        run_delta_hedge(np.ones((1, 3)), np.array([[0.5, np.nan]]), 1.0, "call", 0.0, 0.0, 1.0)
