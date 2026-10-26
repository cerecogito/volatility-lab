"""Tests for the implied-volatility and implied-forward solvers."""

import numpy as np
import pytest

from volatility_lab.black_scholes import bs_price
from volatility_lab.implied_vol import (
    STATUS_ABOVE_BOUND, STATUS_BELOW_BOUND, STATUS_LOW_VEGA, STATUS_MISSING, STATUS_OK,
    STATUS_TOO_SHORT, implied_forward_from_vol, implied_volatility, implied_volatility_many,
)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_recovers_known_volatility_over_a_grid(option_type):
    for strike in [70.0, 90.0, 100.0, 110.0, 140.0]:
        for time in [0.02, 0.25, 1.0, 3.0]:
            for true_vol in [0.08, 0.3, 0.8, 2.0]:
                price = float(bs_price(100.0, strike, time, 0.02, true_vol, option_type))
                vol, status = implied_volatility(price, 100.0, strike, time, 0.02, option_type)
                if status == STATUS_OK:
                    assert vol == pytest.approx(true_vol, abs=1e-6)
                else:
                    # the only allowed failure is the deep-ITM/OTM, low vega case
                    assert status in (STATUS_LOW_VEGA, STATUS_BELOW_BOUND)


def test_ordinary_cases_always_invert():
    vols, statuses = implied_volatility_many(
        prices=[float(bs_price(100.0, k, 0.5, 0.0, 0.6, "call")) for k in [80, 100, 120]],
        spots=[100.0] * 3, strikes=[80.0, 100.0, 120.0], times=[0.5] * 3,
        rates=[0.0] * 3, option_types=["call"] * 3)
    assert list(statuses) == [STATUS_OK] * 3
    np.testing.assert_allclose(vols, 0.6, atol=1e-7)


def test_prices_outside_arbitrage_bounds_are_reported():
    # call worth less than S - K e^{-rT}: impossible
    vol, status = implied_volatility(5.0, 100.0, 90.0, 1.0, 0.0, "call")
    assert np.isnan(vol) and status == STATUS_BELOW_BOUND
    # call worth more than the stock
    vol, status = implied_volatility(101.0, 100.0, 90.0, 1.0, 0.0, "call")
    assert np.isnan(vol) and status == STATUS_ABOVE_BOUND
    # put worth more than the discounted strike
    vol, status = implied_volatility(95.0, 100.0, 90.0, 1.0, 0.05, "put")
    assert status == STATUS_ABOVE_BOUND


def test_missing_and_too_short_inputs():
    assert implied_volatility(np.nan, 100.0, 100.0, 0.5, 0.0, "call")[1] == STATUS_MISSING
    assert implied_volatility(-1.0, 100.0, 100.0, 0.5, 0.0, "call")[1] == STATUS_MISSING
    assert implied_volatility(1.0, 100.0, 100.0, 1e-6, 0.0, "call")[1] == STATUS_TOO_SHORT


def test_very_short_expiry_atm_still_inverts():
    time = 2.0 / 365.0
    price = float(bs_price(100.0, 100.0, time, 0.0, 0.5, "put"))
    vol, status = implied_volatility(price, 100.0, 100.0, time, 0.0, "put")
    assert status == STATUS_OK and vol == pytest.approx(0.5, abs=1e-7)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_forward_recovery(option_type):
    true_forward, strike, time, vol = 61_234.0, 60_000.0, 30 / 365, 0.55
    price = float(bs_price(true_forward, strike, time, 0.0, vol, option_type))
    forward = implied_forward_from_vol(price, strike, time, vol, option_type, spot_guess=60_900.0)
    assert forward == pytest.approx(true_forward, rel=1e-8)
