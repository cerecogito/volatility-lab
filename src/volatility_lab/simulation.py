"""Simulated daily log-return paths with KNOWN data-generating processes (Track A).

Two DGPs:
1. Constant-volatility GBM: daily log returns are iid N(mu_d, sigma_d^2).
   This is exactly the Black-Scholes world (discretely observed).
2. Two-state regime switching: a hidden Markov chain picks a calm or a
   turbulent daily volatility each day, and returns are normal given the
   state. Volatility is random but independent of the return shocks.

All functions take an explicit numpy Generator so every experiment is
reproducible from its seed.
"""

from __future__ import annotations

import numpy as np


def simulate_gbm_returns(num_paths: int, num_days: int, annual_drift: float, annual_vol: float,
                         rng: np.random.Generator, days_per_year: float = 365.0) -> np.ndarray:
    """Daily log returns of GBM: (mu - sigma^2/2) dt + sigma sqrt(dt) Z. Shape (paths, days)."""
    step = 1.0 / days_per_year
    shocks = rng.standard_normal((num_paths, num_days))
    return (annual_drift - 0.5 * annual_vol**2) * step + annual_vol * np.sqrt(step) * shocks


def simulate_regime_returns(num_paths: int, num_days: int, annual_vols: tuple[float, float],
                            transition: np.ndarray, rng: np.random.Generator,
                            annual_drift: float = 0.0, days_per_year: float = 365.0,
                            initial_state_probs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Regime-switching returns. Returns (returns, states), both shaped (paths, days).

    If initial_state_probs is None the chain starts from its stationary distribution.
    """
    transition = np.asarray(transition, dtype=float)
    if initial_state_probs is None:
        initial_state_probs = stationary_distribution(transition)
    step = 1.0 / days_per_year
    vols = np.asarray(annual_vols, dtype=float)

    states = np.zeros((num_paths, num_days), dtype=int)
    current = (rng.random(num_paths) < initial_state_probs[1]).astype(int)
    for day in range(num_days):
        if day > 0:
            stay = rng.random(num_paths) < transition[current, current]
            current = np.where(stay, current, 1 - current)
        states[:, day] = current

    daily_vol = vols[states]
    shocks = rng.standard_normal((num_paths, num_days))
    returns = (annual_drift - 0.5 * daily_vol**2) * step + daily_vol * np.sqrt(step) * shocks
    return returns, states


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """pi with pi P = pi for a 2-state chain: pi_1 = p01 / (p01 + p10)."""
    p01, p10 = transition[0, 1], transition[1, 0]
    pi_high = p01 / (p01 + p10)
    return np.array([1.0 - pi_high, pi_high])


def prices_from_returns(start_price: float | np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Price paths including the starting price: shape (paths, days + 1)."""
    start = np.broadcast_to(np.asarray(start_price, dtype=float), returns.shape[:-1])[..., None]
    return start * np.exp(np.concatenate([np.zeros(returns.shape[:-1] + (1,)),
                                          np.cumsum(returns, axis=-1)], axis=-1))
