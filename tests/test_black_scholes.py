"""Tests for Black-Scholes prices, Greeks and no-arbitrage relations.

The reference numbers are NOT produced by our own code:
* Hull, "Options, Futures and Other Derivatives": S=42, K=40, r=10%,
  sigma=20%, T=0.5 -> call 4.76, put 0.81 (rounded to cents);
* Hull's delta-hedging example: S=49, K=50, r=5%, sigma=20%, T=20/52 ->
  call 2.40, delta 0.522, gamma 0.066, vega 12.1, theta -4.31/yr, rho 8.91;
* an independent price from numerically integrating the payoff against the
  lognormal density (scipy.integrate.quad), and finite-difference Greeks.
"""

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import lognorm

from volatility_lab.black_scholes import (
    bs_delta, bs_gamma, bs_price, bs_rho, bs_theta, bs_vega, d1_d2, no_arbitrage_bounds,
)


def test_hull_textbook_example_prices():
    d1, d2 = d1_d2(42.0, 40.0, 0.5, 0.10, 0.20)
    assert d1 == pytest.approx(0.7693, abs=1e-4)
    assert d2 == pytest.approx(0.6278, abs=1e-4)
    assert bs_price(42.0, 40.0, 0.5, 0.10, 0.20, "call") == pytest.approx(4.76, abs=0.005)
    assert bs_price(42.0, 40.0, 0.5, 0.10, 0.20, "put") == pytest.approx(0.81, abs=0.005)


def test_hull_delta_hedging_example_greeks():
    args = (49.0, 50.0, 20.0 / 52.0, 0.05, 0.20)
    assert bs_price(*args, "call") == pytest.approx(2.40, abs=0.005)
    assert bs_delta(*args, "call") == pytest.approx(0.522, abs=0.0005)
    assert bs_gamma(*args) == pytest.approx(0.066, abs=0.0005)
    assert bs_vega(*args) == pytest.approx(12.1, abs=0.05)
    assert bs_theta(*args, "call") == pytest.approx(-4.31, abs=0.005)
    assert bs_rho(*args, "call") == pytest.approx(8.91, abs=0.005)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("strike", [70.0, 100.0, 130.0])
def test_price_matches_numerical_integration_of_payoff(option_type, strike):
    """Price = e^{-rT} * integral payoff(s) f_Q(s) ds, with f_Q the lognormal density."""
    spot, time, rate, vol, div = 100.0, 0.75, 0.03, 0.35, 0.01
    mean_log = np.log(spot) + (rate - div - 0.5 * vol**2) * time
    density = lognorm(s=vol * np.sqrt(time), scale=np.exp(mean_log))
    if option_type == "call":
        value, _ = integrate.quad(lambda s: (s - strike) * density.pdf(s), strike, np.inf, limit=200)
    else:
        value, _ = integrate.quad(lambda s: (strike - s) * density.pdf(s), 0.0, strike, limit=200)
    expected = np.exp(-rate * time) * value
    assert bs_price(spot, strike, time, rate, vol, option_type, div) == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_match_central_finite_differences(option_type):
    spot, strike, time, rate, vol, div = 100.0, 105.0, 0.4, 0.02, 0.3, 0.01
    h_spot, h_vol, h_time, h_rate = 1e-3, 1e-5, 1e-6, 1e-6

    def price(s=spot, t=time, r=rate, v=vol):
        return float(bs_price(s, strike, t, r, v, option_type, div))

    fd_delta = (price(s=spot + h_spot) - price(s=spot - h_spot)) / (2 * h_spot)
    fd_gamma = (price(s=spot + h_spot) - 2 * price() + price(s=spot - h_spot)) / h_spot**2
    fd_vega = (price(v=vol + h_vol) - price(v=vol - h_vol)) / (2 * h_vol)
    fd_theta = -(price(t=time + h_time) - price(t=time - h_time)) / (2 * h_time)
    fd_rho = (price(r=rate + h_rate) - price(r=rate - h_rate)) / (2 * h_rate)

    assert bs_delta(spot, strike, time, rate, vol, option_type, div) == pytest.approx(fd_delta, rel=1e-6)
    assert bs_gamma(spot, strike, time, rate, vol, div) == pytest.approx(fd_gamma, rel=1e-4)
    assert bs_vega(spot, strike, time, rate, vol, div) == pytest.approx(fd_vega, rel=1e-6)
    assert bs_theta(spot, strike, time, rate, vol, option_type, div) == pytest.approx(fd_theta, rel=1e-5)
    assert bs_rho(spot, strike, time, rate, vol, option_type, div) == pytest.approx(fd_rho, rel=1e-5)


def test_put_call_parity_over_a_grid():
    spots = np.linspace(50, 150, 11)[:, None]
    strikes = np.linspace(60, 140, 9)[None, :]
    time, rate, vol, div = 0.6, 0.04, 0.25, 0.015
    calls = bs_price(spots, strikes, time, rate, vol, "call", div)
    puts = bs_price(spots, strikes, time, rate, vol, "put", div)
    parity = spots * np.exp(-div * time) - strikes * np.exp(-rate * time)
    np.testing.assert_allclose(calls - puts, parity, atol=1e-10)


def test_prices_respect_no_arbitrage_bounds():
    rng = np.random.default_rng(1)
    n = 2000
    spot = rng.uniform(20, 200, n)
    strike = rng.uniform(20, 200, n)
    time = rng.uniform(0.01, 3, n)
    rate = rng.uniform(0, 0.08, n)
    vol = rng.uniform(0.05, 1.5, n)
    for option_type in ["call", "put"]:
        prices = bs_price(spot, strike, time, rate, vol, option_type)
        lower, upper = no_arbitrage_bounds(spot, strike, time, rate, option_type)
        assert np.all(prices >= lower - 1e-9)
        assert np.all(prices <= upper + 1e-9)


def test_monotonicity_in_spot_strike_volatility_and_time():
    strikes = np.linspace(50, 150, 101)
    calls = bs_price(100.0, strikes, 1.0, 0.03, 0.2, "call")
    puts = bs_price(100.0, strikes, 1.0, 0.03, 0.2, "put")
    assert np.all(np.diff(calls) < 0)            # calls fall as strike rises
    assert np.all(np.diff(puts) > 0)             # puts rise as strike rises

    vols = np.linspace(0.05, 1.0, 50)
    assert np.all(np.diff(bs_price(100.0, 100.0, 1.0, 0.03, vols, "call")) > 0)
    assert np.all(np.diff(bs_price(100.0, 100.0, 1.0, 0.03, vols, "put")) > 0)

    spots = np.linspace(50, 150, 101)
    assert np.all(np.diff(bs_price(spots, 100.0, 1.0, 0.03, 0.2, "call")) > 0)

    # with no dividends, a longer-dated call is worth more
    times = np.linspace(0.05, 3, 60)
    assert np.all(np.diff(bs_price(100.0, 100.0, times, 0.03, 0.2, "call")) > 0)


def test_convexity_in_strike_butterflies_are_non_negative():
    strikes = np.linspace(50, 150, 201)
    for option_type in ["call", "put"]:
        prices = bs_price(100.0, strikes, 0.5, 0.02, 0.3, option_type)
        butterflies = prices[:-2] - 2 * prices[1:-1] + prices[2:]
        assert np.all(butterflies >= -1e-12)


def test_limits_small_volatility_and_near_expiry():
    # vol -> 0: price -> discounted intrinsic value of the forward
    spot, strike, time, rate = 100.0, 90.0, 1.0, 0.05
    tiny_vol_price = bs_price(spot, strike, time, rate, 1e-6, "call")
    assert tiny_vol_price == pytest.approx(spot - strike * np.exp(-rate * time), abs=1e-8)
    # T -> 0: price -> payoff
    assert bs_price(100.0, 90.0, 1e-10, 0.05, 0.2, "call") == pytest.approx(10.0, abs=1e-6)
    assert bs_price(100.0, 110.0, 1e-10, 0.05, 0.2, "call") == pytest.approx(0.0, abs=1e-12)
    # extreme inputs must not produce NaN or inf
    assert np.isfinite(bs_price(1e5, 1.0, 10.0, 0.0, 5.0, "put"))
    assert np.isfinite(bs_price(1.0, 1e5, 1e-4, 0.0, 0.01, "call"))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bs_price(-1.0, 100.0, 1.0, 0.0, 0.2)
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 0.0, 0.0, 0.2)
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.0, -0.2)
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.0, 0.2, "straddle")
