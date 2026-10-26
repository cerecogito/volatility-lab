"""Classical (non-Bayesian) volatility estimators.

Timing convention used everywhere in the project:
    returns[t] = log(S_t / S_{t-1}) is the return that ENDS at fixing t.
    estimate[t] uses returns[0..t] only, so it is known at fixing t
    and can be used for any decision made at or after time t.
Nothing here ever looks at returns[t+1:] - see tests/test_leakage.py.
"""

from __future__ import annotations

import numpy as np


def running_window_sums(values: np.ndarray, window: int) -> np.ndarray:
    """Sum of each block of `window` consecutive values along the last axis.

    Output position i holds sum(values[..., i : i + window]).
    """
    zeros = np.zeros(values.shape[:-1] + (1,))
    cumulative = np.concatenate([zeros, np.cumsum(values, axis=-1)], axis=-1)
    return cumulative[..., window:] - cumulative[..., :-window]


def rolling_volatility(returns: np.ndarray, window: int, days_per_year: float = 365.0) -> np.ndarray:
    """Annualised sample standard deviation of the last `window` daily returns.

    sigma_hat[t] = sqrt(days_per_year) * std(returns[t-window+1 .. t], ddof=1).
    The first window-1 entries are NaN (not enough history yet). Works along
    the last axis, so `returns` can be one series or a (paths, days) array.
    """
    returns = np.asarray(returns, dtype=float)
    if window < 2:
        raise ValueError("window must be at least 2")
    estimates = np.full(returns.shape, np.nan)
    if returns.shape[-1] < window:
        return estimates

    # running sums make this O(n) instead of O(n * window)
    window_sum = running_window_sums(returns, window)
    window_sum_sq = running_window_sums(returns**2, window)
    sample_variance = (window_sum_sq - window_sum**2 / window) / (window - 1)
    sample_variance = np.maximum(sample_variance, 0.0)   # guard against tiny negative rounding

    estimates[..., window - 1:] = np.sqrt(sample_variance * days_per_year)
    return estimates


def ewma_volatility(returns: np.ndarray, decay: float, days_per_year: float = 365.0,
                    seed_window: int = 30) -> np.ndarray:
    """Exponentially weighted (RiskMetrics-style) volatility.

    variance[t] = decay * variance[t-1] + (1 - decay) * returns[t]^2

    It uses a zero mean, which is standard for daily returns. The recursion
    is started with the mean squared return of the first `seed_window`
    returns, and earlier entries are NaN. The effective memory is about
    1 / (1 - decay) days.
    """
    returns = np.asarray(returns, dtype=float)
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be strictly between 0 and 1")
    estimates = np.full(returns.shape, np.nan)
    if returns.shape[-1] < seed_window:
        return estimates

    variance = np.mean(returns[..., :seed_window] ** 2, axis=-1)
    estimates[..., seed_window - 1] = variance
    for t in range(seed_window, returns.shape[-1]):
        variance = decay * variance + (1.0 - decay) * returns[..., t] ** 2
        estimates[..., t] = variance
    return np.sqrt(estimates * days_per_year)


def realised_volatility_ahead(returns: np.ndarray, horizon: int, days_per_year: float = 365.0) -> np.ndarray:
    """What actually happened: annualised RMS of returns[t+1 .. t+horizon].

    This is the *target* in forecast evaluations. It uses future data on
    purpose and must never be fed into a model.
    """
    returns = np.asarray(returns, dtype=float)
    realised = np.full(len(returns), np.nan)
    cumulative_sum_sq = np.concatenate([[0.0], np.cumsum(returns**2)])
    for t in range(len(returns) - horizon):
        realised[t] = (cumulative_sum_sq[t + 1 + horizon] - cumulative_sum_sq[t + 1]) / horizon
    return np.sqrt(realised * days_per_year)
