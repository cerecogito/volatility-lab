"""Discrete delta hedging of a short European option (Caltech course, section 27).

Set-up (the hedger SELLS one option and hedges it):
    time 0     : receive the premium, buy delta_0 shares, the rest goes in the cash account
    time k     : cash grows by e^{r dt}; rebalance to delta_k shares (pay transaction costs)
    time N (=T): cash grows once more; sell the remaining shares; pay the option payoff

Without costs and with r = 0 this is the course's wealth recursion
    X(k+1) = X(k) + delta(k) [S(k+1) - S(k)],   X(0) = premium.

SIGN CONVENTION (used in every report):
    hedge_error = X(T) - payoff
    hedge_error > 0  -> the hedger has money left over (gain)
    hedge_error < 0  -> the hedge did not cover the payoff (loss)
    loss = -hedge_error  (used for VaR and Expected Shortfall)

Transaction costs: proportional, cost_rate * |shares traded| * price, on
every trade including the first purchase and the final unwind.

This function does accounting only. The hedge ratios are computed
elsewhere by each volatility model, using only information available at
each rebalance time (see tests/test_leakage.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volatility_lab.binomial import payoff


@dataclass
class HedgeResult:
    hedge_error: np.ndarray        # (paths,) X(T) - payoff
    terminal_payoff: np.ndarray    # (paths,)
    transaction_costs: np.ndarray  # (paths,) total costs paid
    share_turnover: np.ndarray     # (paths,) sum |shares traded| (incl. first buy and final unwind)
    wealth_path: np.ndarray        # (paths, N+1) portfolio value before closing costs


def run_delta_hedge(prices: np.ndarray, hedge_ratios: np.ndarray, strike: float, option_type: str,
                    premium, rate: float, step_years: float, cost_rate: float = 0.0) -> HedgeResult:
    """Simulate the hedge on given price paths.

    prices:       (paths, N+1) underlying prices at rebalance times t_0 .. t_N (t_N = expiry)
    hedge_ratios: (paths, N) shares held over (t_k, t_{k+1}], chosen at t_k
    premium:      scalar or (paths,) option premium received at t_0
    """
    prices = np.atleast_2d(np.asarray(prices, dtype=float))
    hedge_ratios = np.atleast_2d(np.asarray(hedge_ratios, dtype=float))
    num_paths, num_points = prices.shape
    num_steps = num_points - 1
    if hedge_ratios.shape != (num_paths, num_steps):
        raise ValueError(f"hedge_ratios must have shape {(num_paths, num_steps)}, got {hedge_ratios.shape}")
    if np.any(~np.isfinite(hedge_ratios)):
        raise ValueError("hedge ratios contain NaN/inf - the model had no estimate at some step")

    growth = np.exp(rate * step_years)
    premium = np.broadcast_to(np.asarray(premium, dtype=float), (num_paths,))

    # time 0: buy the first hedge
    shares = hedge_ratios[:, 0]
    costs = cost_rate * np.abs(shares) * prices[:, 0]
    turnover = np.abs(shares).copy()
    cash = premium - shares * prices[:, 0] - costs

    wealth_path = np.zeros((num_paths, num_points))
    wealth_path[:, 0] = cash + shares * prices[:, 0]

    for k in range(1, num_steps + 1):
        cash = cash * growth
        wealth_path[:, k] = cash + shares * prices[:, k]
        if k == num_steps:
            break
        # rebalance to the new hedge ratio
        new_shares = hedge_ratios[:, k]
        traded = new_shares - shares
        trade_cost = cost_rate * np.abs(traded) * prices[:, k]
        cash = cash - traded * prices[:, k] - trade_cost
        costs = costs + trade_cost
        turnover = turnover + np.abs(traded)
        shares = new_shares

    # expiry: unwind the shares and pay the payoff
    final_price = prices[:, -1]
    unwind_cost = cost_rate * np.abs(shares) * final_price
    costs = costs + unwind_cost
    turnover = turnover + np.abs(shares)
    terminal_value = cash + shares * final_price - unwind_cost
    terminal_payoff = payoff(final_price, strike, option_type)

    return HedgeResult(terminal_value - terminal_payoff, terminal_payoff, costs, turnover, wealth_path)
