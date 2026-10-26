"""Cox-Ross-Rubinstein binomial tree (Caltech course, sections 10-13).

In each step of length dt the price goes up by u or down by d. No arbitrage
needs d < e^{(r-q) dt} < u, and then the unique risk-neutral up-probability is

    p* = (e^{(r-q) dt} - d) / (u - d).

The option value is found by backward induction:
    V_N(j) = payoff(S_N(j))
    V_n(j) = e^{-r dt} [p* V_{n+1}(j+1) + (1 - p*) V_{n+1}(j)]
and for an American option we also take max(continuation, exercise value)
at every node.

CRR choice:  u = e^{sigma sqrt(dt)},  d = 1/u. As the number of steps grows,
the tree price converges to the Black-Scholes price.
"""

from __future__ import annotations

import numpy as np


def payoff(prices: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    """Vanilla payoff max(S-K,0) for a call or max(K-S,0) for a put."""
    if option_type == "call":
        return np.maximum(prices - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - prices, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def binomial_price_from_up_down(spot: float, strike: float, up: float, down: float,
                                rate: float, step_length: float, steps: int,
                                option_type: str = "call", american: bool = False,
                                dividend_yield: float = 0.0) -> float:
    """Price on a general recombining tree with given up/down factors.

    `rate` is continuously compounded, so one step grows money by e^{r dt}
    (the course examples use exactly this convention).
    """
    if steps < 1:
        raise ValueError("need at least one step")
    growth = np.exp((rate - dividend_yield) * step_length)
    if not (down < growth < up):
        raise ValueError(f"no-arbitrage condition d < e^((r-q)dt) < u fails: d={down}, u={up}, growth={growth}")

    up_probability = (growth - down) / (up - down)
    discount = np.exp(-rate * step_length)

    # prices at maturity: node j has j up-moves and (steps - j) down-moves
    up_moves = np.arange(steps + 1)
    terminal_prices = spot * up**up_moves * down ** (steps - up_moves)
    values = payoff(terminal_prices, strike, option_type)

    # step backwards through the tree one layer at a time
    for step in range(steps - 1, -1, -1):
        values = discount * (up_probability * values[1:] + (1.0 - up_probability) * values[:-1])
        if american:
            up_moves = np.arange(step + 1)
            node_prices = spot * up**up_moves * down ** (step - up_moves)
            values = np.maximum(values, payoff(node_prices, strike, option_type))

    return float(values[0])


def crr_price(spot: float, strike: float, time_to_expiry: float, rate: float,
              volatility: float, steps: int, option_type: str = "call",
              american: bool = False, dividend_yield: float = 0.0) -> float:
    """Cox-Ross-Rubinstein price with u = exp(sigma sqrt(dt)), d = 1/u."""
    if time_to_expiry <= 0 or volatility <= 0:
        raise ValueError("time_to_expiry and volatility must be positive")
    step_length = time_to_expiry / steps
    up = np.exp(volatility * np.sqrt(step_length))
    return binomial_price_from_up_down(spot, strike, up, 1.0 / up, rate, step_length, steps,
                                       option_type, american, dividend_yield)
