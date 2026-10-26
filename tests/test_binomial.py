"""Tests for the CRR binomial tree.

The first three tests use the worked numerical examples from the Caltech
course slides (section 11 and 13): they were calculated by hand/spreadsheet
in the course, not by this code.
"""

import numpy as np
import pytest

from volatility_lab.binomial import binomial_price_from_up_down, crr_price
from volatility_lab.black_scholes import bs_price


def test_course_single_period_example():
    # S=100, u=1.01, d=0.99, simple rate 0.5% per period -> call 0.746, q = 0.75
    continuous_rate = np.log(1.005)
    price = binomial_price_from_up_down(100.0, 100.0, 1.01, 0.99, continuous_rate, 1.0, 1, "call")
    assert price == pytest.approx(0.746, abs=0.0005)


def test_course_two_step_european_call():
    # S=100, u=1.1, d=0.9, K=100, r=5% continuous, dt=1 -> C(0) = 10.8703
    price = binomial_price_from_up_down(100.0, 100.0, 1.1, 0.9, 0.05, 1.0, 2, "call")
    assert price == pytest.approx(10.8703, abs=1e-4)


def test_course_two_step_american_put():
    # same tree, American put -> A(0) = 2.4844 (early exercise at the down node)
    price = binomial_price_from_up_down(100.0, 100.0, 1.1, 0.9, 0.05, 1.0, 2, "put", american=True)
    assert price == pytest.approx(2.4844, abs=1e-4)
    european = binomial_price_from_up_down(100.0, 100.0, 1.1, 0.9, 0.05, 1.0, 2, "put")
    assert european < price  # early exercise is worth something here


def test_no_arbitrage_condition_is_enforced():
    with pytest.raises(ValueError):
        binomial_price_from_up_down(100.0, 100.0, 1.01, 0.99, 0.05, 1.0, 2)  # e^0.05 > u


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_crr_converges_to_black_scholes(option_type):
    args = dict(spot=100.0, strike=95.0, time_to_expiry=0.75, rate=0.03, volatility=0.25)
    exact = float(bs_price(args["spot"], args["strike"], args["time_to_expiry"], args["rate"],
                           args["volatility"], option_type))
    errors = [abs(crr_price(**args, steps=n, option_type=option_type) - exact)
              for n in [50, 200, 800, 3200]]
    assert errors[-1] < 2e-3
    # CRR error oscillates but its envelope shrinks roughly like 1/n
    assert errors[-1] < errors[0] / 8


def test_american_call_without_dividends_equals_european():
    # course section 9: never optimal to exercise an American call early if no dividends
    european = crr_price(100.0, 100.0, 1.0, 0.05, 0.3, 500, "call")
    american = crr_price(100.0, 100.0, 1.0, 0.05, 0.3, 500, "call", american=True)
    assert american == pytest.approx(european, abs=1e-10)


def test_american_put_is_worth_at_least_european_and_intrinsic():
    for strike in [80.0, 100.0, 120.0, 150.0]:
        european = crr_price(100.0, strike, 1.0, 0.05, 0.3, 400, "put")
        american = crr_price(100.0, strike, 1.0, 0.05, 0.3, 400, "put", american=True)
        assert american >= european - 1e-12
        assert american >= max(strike - 100.0, 0.0) - 1e-12


def test_binomial_put_call_parity_for_european_options():
    call = crr_price(100.0, 110.0, 0.5, 0.04, 0.2, 300, "call", dividend_yield=0.01)
    put = crr_price(100.0, 110.0, 0.5, 0.04, 0.2, 300, "put", dividend_yield=0.01)
    parity = 100.0 * np.exp(-0.01 * 0.5) - 110.0 * np.exp(-0.04 * 0.5)
    assert call - put == pytest.approx(parity, abs=1e-10)
