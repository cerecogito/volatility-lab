"""Implied volatility by Brent's root-finding method.

Given an observed option price V_obs we look for sigma with
    BS(sigma) - V_obs = 0.
The BS price is strictly increasing in sigma (vega > 0), so there is at most
one root. A root exists only if V_obs is strictly inside the no-arbitrage
bounds, because BS(sigma) -> lower bound as sigma -> 0 and -> upper bound as
sigma -> infinity.

Instead of raising errors, each observation gets a status string, so we can
report exactly how many observations could not be inverted and why.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from volatility_lab.black_scholes import bs_price, bs_vega, no_arbitrage_bounds

MIN_VOL = 1e-4
MAX_VOL = 10.0     # 1000% annualised - anything above this is not believable
MIN_TIME = 1.0 / (365.0 * 24.0)   # one hour
VEGA_FLOOR = 1e-8

STATUS_OK = "ok"
STATUS_MISSING = "missing_or_nonpositive_input"
STATUS_TOO_SHORT = "time_to_expiry_too_small"
STATUS_BELOW_BOUND = "price_at_or_below_lower_bound"
STATUS_ABOVE_BOUND = "price_at_or_above_upper_bound"
STATUS_LOW_VEGA = "vega_too_small"
STATUS_NO_ROOT = "no_root_in_vol_bracket"


def implied_volatility(price: float, spot: float, strike: float, time_to_expiry: float,
                       rate: float, option_type: str, dividend_yield: float = 0.0,
                       tolerance: float = 1e-10) -> tuple[float, str]:
    """Return (implied_vol, status). implied_vol is NaN unless status == 'ok'."""
    inputs = [price, spot, strike, time_to_expiry]
    if any(x is None or not np.isfinite(x) for x in inputs) or spot <= 0 or strike <= 0 or price <= 0:
        return np.nan, STATUS_MISSING
    if time_to_expiry < MIN_TIME:
        return np.nan, STATUS_TOO_SHORT

    lower, upper = no_arbitrage_bounds(spot, strike, time_to_expiry, rate, option_type, dividend_yield)
    # relative tolerance so that tiny rounding at the bound does not count
    if price <= float(lower) * (1 + 1e-12):
        return np.nan, STATUS_BELOW_BOUND
    if price >= float(upper) * (1 - 1e-12):
        return np.nan, STATUS_ABOVE_BOUND

    def pricing_error(sigma: float) -> float:
        return float(bs_price(spot, strike, time_to_expiry, rate, sigma, option_type, dividend_yield)) - price

    error_low, error_high = pricing_error(MIN_VOL), pricing_error(MAX_VOL)
    if error_low * error_high > 0:
        return np.nan, STATUS_NO_ROOT

    sigma = brentq(pricing_error, MIN_VOL, MAX_VOL, xtol=tolerance, maxiter=200)

    # deep ITM/OTM: price barely moves with sigma, so the inverted vol is
    # extremely sensitive to rounding in the price -> flag it
    vega = float(bs_vega(spot, strike, time_to_expiry, rate, sigma, dividend_yield))
    if vega < VEGA_FLOOR * spot:
        return np.nan, STATUS_LOW_VEGA
    return sigma, STATUS_OK


def implied_volatility_many(prices, spots, strikes, times, rates, option_types,
                            dividend_yields=None) -> tuple[np.ndarray, np.ndarray]:
    """Loop version for arrays. Returns (vols, statuses)."""
    count = len(prices)
    if dividend_yields is None:
        dividend_yields = np.zeros(count)
    vols = np.full(count, np.nan)
    statuses = np.empty(count, dtype=object)
    for i in range(count):
        vols[i], statuses[i] = implied_volatility(
            float(prices[i]), float(spots[i]), float(strikes[i]), float(times[i]),
            float(rates[i]), str(option_types[i]), float(dividend_yields[i]),
        )
    return vols, statuses


def implied_forward_from_vol(price: float, strike: float, time_to_expiry: float,
                             volatility: float, option_type: str,
                             spot_guess: float) -> float:
    """Solve Black-76 (r = 0) for the forward F that reproduces `price` at `volatility`.

    Used for the Deribit data: Deribit reports each trade's IV computed against
    its own forward price, but the forward itself is not in the trade record.
    Calls are increasing in F and puts are decreasing in F, so the root is
    unique. Returns NaN if no root exists within +-50% of spot_guess.
    """
    def pricing_error(forward: float) -> float:
        return float(bs_price(forward, strike, time_to_expiry, 0.0, volatility, option_type)) - price

    low, high = 0.5 * spot_guess, 1.5 * spot_guess
    if pricing_error(low) * pricing_error(high) > 0:
        return np.nan
    return brentq(pricing_error, low, high, xtol=1e-8 * spot_guess, maxiter=200)
