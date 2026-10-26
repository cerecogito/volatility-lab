"""Black-Scholes-Merton prices and Greeks for European calls and puts.

Model (Caltech course, sections 14-18): under the risk-neutral measure Q

    dS_t = (r - q) S_t dt + sigma S_t dW_t

so log S_T is normal and the price is the discounted Q-expectation of the
payoff. With d1, d2 defined as usual:

    call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

`q` is a continuous dividend yield (or any continuous "carry"). Setting
spot = F (a forward price) and rate = dividend_yield = 0 gives Black's 1976
formula for an undiscounted option on a forward, which is what we use for the
Deribit data.

All functions accept numpy arrays and broadcast, so a whole option chain can
be priced in one call. Time is in years and volatility is annualised.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

ArrayLike = float | np.ndarray


def _check_inputs(spot, strike, time_to_expiry, volatility) -> None:
    """Raise ValueError for inputs where the formula is not defined."""
    if np.any(np.asarray(spot) <= 0) or np.any(np.asarray(strike) <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(np.asarray(time_to_expiry) <= 0):
        raise ValueError("time_to_expiry must be positive (use the payoff at expiry)")
    if np.any(np.asarray(volatility) <= 0):
        raise ValueError("volatility must be positive")


def _is_call(option_type) -> np.ndarray:
    """Turn 'call'/'put' (or an array of them) into a boolean array."""
    option_type = np.asarray(option_type)
    valid = np.isin(option_type, ["call", "put"])
    if not np.all(valid):
        raise ValueError("option_type must be 'call' or 'put'")
    return option_type == "call"


def d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield=0.0):
    """The two standardised log-moneyness terms of the BSM formula."""
    _check_inputs(spot, strike, time_to_expiry, volatility)
    spot = np.asarray(spot, dtype=float)
    sqrt_time = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (
        volatility * sqrt_time
    )
    d2 = d1 - volatility * sqrt_time
    return d1, d2


def bs_price(spot, strike, time_to_expiry, rate, volatility, option_type="call",
             dividend_yield=0.0) -> ArrayLike:
    """Black-Scholes-Merton price of a European call or put."""
    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * np.exp(-rate * time_to_expiry)

    call_price = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    put_price = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
    return np.where(_is_call(option_type), call_price, put_price)


def bs_delta(spot, strike, time_to_expiry, rate, volatility, option_type="call",
             dividend_yield=0.0) -> ArrayLike:
    """dV/dS: number of shares in the replicating portfolio."""
    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    carry = np.exp(-dividend_yield * time_to_expiry)
    return np.where(_is_call(option_type), carry * norm.cdf(d1), carry * (norm.cdf(d1) - 1.0))


def bs_gamma(spot, strike, time_to_expiry, rate, volatility, dividend_yield=0.0) -> ArrayLike:
    """d^2V/dS^2 (the same for calls and puts)."""
    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    carry = np.exp(-dividend_yield * time_to_expiry)
    return carry * norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))


def bs_vega(spot, strike, time_to_expiry, rate, volatility, dividend_yield=0.0) -> ArrayLike:
    """dV/dsigma per 1.00 change in volatility (divide by 100 for 'per vol point')."""
    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    carry = np.exp(-dividend_yield * time_to_expiry)
    return spot * carry * norm.pdf(d1) * np.sqrt(time_to_expiry)


def bs_theta(spot, strike, time_to_expiry, rate, volatility, option_type="call",
             dividend_yield=0.0) -> ArrayLike:
    """dV/dt per year of calendar time (= -dV/dT). Usually negative for long options."""
    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * np.exp(-rate * time_to_expiry)
    time_decay = -discounted_spot * norm.pdf(d1) * volatility / (2.0 * np.sqrt(time_to_expiry))

    call_theta = (time_decay - rate * discounted_strike * norm.cdf(d2)
                  + dividend_yield * discounted_spot * norm.cdf(d1))
    put_theta = (time_decay + rate * discounted_strike * norm.cdf(-d2)
                 - dividend_yield * discounted_spot * norm.cdf(-d1))
    return np.where(_is_call(option_type), call_theta, put_theta)


def bs_rho(spot, strike, time_to_expiry, rate, volatility, option_type="call",
           dividend_yield=0.0) -> ArrayLike:
    """dV/dr per 1.00 change in the interest rate."""
    _, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    discounted_strike = strike * np.exp(-rate * time_to_expiry)
    call_rho = time_to_expiry * discounted_strike * norm.cdf(d2)
    put_rho = -time_to_expiry * discounted_strike * norm.cdf(-d2)
    return np.where(_is_call(option_type), call_rho, put_rho)


def no_arbitrage_bounds(spot, strike, time_to_expiry, rate, option_type="call",
                        dividend_yield=0.0) -> tuple[ArrayLike, ArrayLike]:
    """Model-free lower/upper bounds for a European option (course section 9).

    call: max(S e^{-qT} - K e^{-rT}, 0) <= c <= S e^{-qT}
    put:  max(K e^{-rT} - S e^{-qT}, 0) <= p <= K e^{-rT}
    """
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * np.exp(-rate * time_to_expiry)
    is_call = _is_call(option_type)
    lower = np.where(is_call,
                     np.maximum(discounted_spot - discounted_strike, 0.0),
                     np.maximum(discounted_strike - discounted_spot, 0.0))
    upper = np.where(is_call, discounted_spot, discounted_strike)
    return lower, upper
