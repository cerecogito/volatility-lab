"""Monte Carlo pricing of European options under geometric Brownian motion.

Under Q, S_T = S_0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z) with Z ~ N(0,1),
so we can sample S_T exactly (no time stepping needed for a European payoff).

    price ~= e^{-rT} * (1/n) sum_i payoff(S_T^(i))

By the CLT the estimator is approximately normal with standard error
s / sqrt(n), where s is the sample std of the discounted payoffs. This
gives a 95% confidence interval of estimate +- 1.96 * SE.

Variance reduction:
* antithetic variates: use Z and -Z and average each pair. This helps when
  the payoff is monotone in Z, which vanilla calls/puts are;
* control variate: the discounted terminal price Y = e^{-rT} S_T has known
  mean S_0 e^{-qT}. We use X - b (Y - E[Y]), with b estimated from the sample
  (the optimal b = Cov(X,Y)/Var(Y)).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from volatility_lab.binomial import payoff

Z_95 = 1.959963984540054


@dataclass
class MonteCarloResult:
    method: str
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    sample_size: int        # number of independent samples behind the SE
    runtime_seconds: float


def _summarise(method: str, samples: np.ndarray, started: float) -> MonteCarloResult:
    estimate = float(samples.mean())
    standard_error = float(samples.std(ddof=1) / np.sqrt(len(samples)))
    return MonteCarloResult(method, estimate, standard_error,
                            estimate - Z_95 * standard_error, estimate + Z_95 * standard_error,
                            len(samples), time.perf_counter() - started)


def terminal_prices(spot, time_to_expiry, rate, volatility, normals, dividend_yield=0.0):
    """Exact GBM terminal prices from standard normal draws."""
    drift = (rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
    return spot * np.exp(drift + volatility * np.sqrt(time_to_expiry) * normals)


def mc_price(spot: float, strike: float, time_to_expiry: float, rate: float, volatility: float,
             option_type: str, num_paths: int, rng: np.random.Generator,
             method: str = "plain", dividend_yield: float = 0.0) -> MonteCarloResult:
    """Monte Carlo price with method = 'plain', 'antithetic' or 'control_variate'.

    `num_paths` is the number of normal draws used in every method, so the
    comparison between methods is at equal simulation cost.
    """
    started = time.perf_counter()
    discount = np.exp(-rate * time_to_expiry)

    if method == "plain":
        normals = rng.standard_normal(num_paths)
        prices = terminal_prices(spot, time_to_expiry, rate, volatility, normals, dividend_yield)
        return _summarise(method, discount * payoff(prices, strike, option_type), started)

    if method == "antithetic":
        if num_paths % 2:
            raise ValueError("antithetic method needs an even number of paths")
        normals = rng.standard_normal(num_paths // 2)
        prices_plus = terminal_prices(spot, time_to_expiry, rate, volatility, normals, dividend_yield)
        prices_minus = terminal_prices(spot, time_to_expiry, rate, volatility, -normals, dividend_yield)
        # each pair average is one independent sample
        pair_means = 0.5 * discount * (payoff(prices_plus, strike, option_type)
                                       + payoff(prices_minus, strike, option_type))
        return _summarise(method, pair_means, started)

    if method == "control_variate":
        normals = rng.standard_normal(num_paths)
        prices = terminal_prices(spot, time_to_expiry, rate, volatility, normals, dividend_yield)
        discounted_payoffs = discount * payoff(prices, strike, option_type)
        control = discount * prices
        control_mean = spot * np.exp(-dividend_yield * time_to_expiry)   # known exactly
        covariance = np.cov(discounted_payoffs, control, ddof=1)
        best_coefficient = covariance[0, 1] / covariance[1, 1]
        adjusted = discounted_payoffs - best_coefficient * (control - control_mean)
        # note: estimating b from the same sample adds a tiny O(1/n) bias - fine for large n
        return _summarise(method, adjusted, started)

    raise ValueError(f"unknown method {method!r}")
