"""Two-state Markov regime model for daily returns (a Gaussian hidden Markov model).

Model (MIT 6.431x: Markov chains; conditional distributions):
    hidden state s_t in {0, 1} follows a Markov chain with transition matrix P,
        P[i, j] = Pr(s_{t+1} = j | s_t = i)
    given the state, the return is normal:
        x_t | s_t = k  ~  N(mu_k, sigma_k^2)

Parameters are fitted by maximum likelihood with the EM (Baum-Welch)
algorithm on the TRAINING returns only. After that they are frozen, and we
run the forward filter through time, which only uses past returns:

    filtered[t, k] = Pr(s_t = k | x_1..x_t)

Volatility forecast for the next h days, made at time t:
    Pr(s_{t+j} = k | x_1..x_t) = (filtered[t] @ P^j)[k]
    expected average variance = (1/h) sum_{j=1..h} sum_k Pr(s_{t+j}=k) * (sigma_k^2 + mu_k^2)
(the mu_k^2 term is tiny for daily returns but keeps the formula exact for
E[x^2]). We annualise and take a square root to get a volatility.

State 0 is always the low-volatility state (we sort after fitting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class RegimeModel:
    means: np.ndarray            # (2,)
    stdevs: np.ndarray           # (2,)
    transition: np.ndarray       # (2, 2), rows sum to 1
    initial_probs: np.ndarray    # (2,)
    log_likelihood: float
    iterations: int
    converged: bool


def _state_densities(returns: np.ndarray, means: np.ndarray, stdevs: np.ndarray) -> np.ndarray:
    """f(x_t | s_t = k) for every t and k, shape (n, 2)."""
    return norm.pdf(returns[:, None], loc=means[None, :], scale=stdevs[None, :])


def forward_filter(returns: np.ndarray, model: RegimeModel) -> tuple[np.ndarray, float]:
    """Scaled forward recursion. Returns (filtered_probs (n,2), log_likelihood).

    filtered[t] = Pr(s_t | x_1..x_t). Each step only uses x_t and the previous
    filtered probabilities, so it is a genuine real-time filter.
    """
    densities = _state_densities(returns, model.means, model.stdevs)
    n = len(returns)
    filtered = np.zeros((n, 2))
    log_likelihood = 0.0
    predicted = model.initial_probs
    for t in range(n):
        joint = predicted * densities[t]
        normaliser = joint.sum()
        if normaliser <= 0 or not np.isfinite(normaliser):
            raise FloatingPointError(f"filter broke down at t={t}")
        filtered[t] = joint / normaliser
        log_likelihood += np.log(normaliser)
        predicted = filtered[t] @ model.transition   # Pr(s_{t+1} | x_1..x_t)
    return filtered, log_likelihood


def _backward_smoother(returns, model, filtered):
    """Smoothed probabilities Pr(s_t | all data) and expected transitions (for EM only).

    Uses the standard backward recursion written in terms of filtered and
    one-step predicted probabilities (no densities needed at this point).
    """
    n = len(returns)
    smoothed = np.zeros((n, 2))
    smoothed[-1] = filtered[-1]
    expected_transitions = np.zeros((2, 2))
    for t in range(n - 2, -1, -1):
        predicted_next = filtered[t] @ model.transition
        ratio = smoothed[t + 1] / np.maximum(predicted_next, 1e-300)
        smoothed[t] = filtered[t] * (model.transition @ ratio)
        # Pr(s_t = i, s_{t+1} = j | all data)
        pair = filtered[t][:, None] * model.transition * ratio[None, :]
        expected_transitions += pair
    return smoothed, expected_transitions


def fit_regime_model(returns: np.ndarray, max_iterations: int = 500, tolerance: float = 1e-8) -> RegimeModel:
    """Fit the 2-state Gaussian HMM by EM. Starting values come from the data quantiles."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 50:
        raise ValueError("need at least 50 returns to fit a regime model")

    overall_std = returns.std()
    model = RegimeModel(
        means=np.array([0.0, 0.0]),
        stdevs=np.array([0.6 * overall_std, 1.6 * overall_std]),
        transition=np.array([[0.97, 0.03], [0.05, 0.95]]),
        initial_probs=np.array([0.5, 0.5]),
        log_likelihood=-np.inf, iterations=0, converged=False,
    )

    previous_log_likelihood = -np.inf
    for iteration in range(1, max_iterations + 1):
        # E-step
        filtered, log_likelihood = forward_filter(returns, model)
        smoothed, expected_transitions = _backward_smoother(returns, model, filtered)

        # M-step: weighted means/variances and normalised transition counts
        weights = smoothed.sum(axis=0)
        means = (smoothed * returns[:, None]).sum(axis=0) / weights
        variances = (smoothed * (returns[:, None] - means[None, :]) ** 2).sum(axis=0) / weights
        stdevs = np.sqrt(np.maximum(variances, 1e-12))
        transition = expected_transitions / expected_transitions.sum(axis=1, keepdims=True)

        model = RegimeModel(means, stdevs, transition, smoothed[0], log_likelihood, iteration, False)
        if abs(log_likelihood - previous_log_likelihood) < tolerance * abs(log_likelihood):
            model.converged = True
            break
        previous_log_likelihood = log_likelihood

    # label switching: make state 0 the calm state
    if model.stdevs[0] > model.stdevs[1]:
        order = np.array([1, 0])
        model = RegimeModel(model.means[order], model.stdevs[order],
                            model.transition[np.ix_(order, order)], model.initial_probs[order],
                            model.log_likelihood, model.iterations, model.converged)
    _, model.log_likelihood = forward_filter(returns, model)
    return model


def regime_volatility_forecast(filtered: np.ndarray, model: RegimeModel, horizon_days: int,
                               days_per_year: float = 365.0) -> np.ndarray:
    """Annualised forecast of average volatility over the next `horizon_days`, for every t."""
    second_moments = model.stdevs**2 + model.means**2
    state_probs = filtered.copy()
    total_variance = np.zeros(len(filtered))
    for _ in range(horizon_days):
        state_probs = state_probs @ model.transition
        total_variance += state_probs @ second_moments
    return np.sqrt(days_per_year * total_variance / horizon_days)


def forward_filter_paths(returns: np.ndarray, model: RegimeModel) -> np.ndarray:
    """Same filter as forward_filter, run on many paths at once. returns: (paths, days).

    Output (paths, days, 2) with filtered[p, t] = Pr(s_t | that path's returns up to t).
    """
    num_paths, num_days = returns.shape
    filtered = np.zeros((num_paths, num_days, 2))
    predicted = np.tile(model.initial_probs, (num_paths, 1))
    for t in range(num_days):
        densities = norm.pdf(returns[:, t, None], loc=model.means[None, :], scale=model.stdevs[None, :])
        joint = predicted * densities
        filtered[:, t] = joint / joint.sum(axis=1, keepdims=True)
        predicted = filtered[:, t] @ model.transition
    return filtered


def average_second_moment_by_horizon(model: RegimeModel, max_horizon_days: int) -> np.ndarray:
    """(max_horizon, 2) table c[h-1, i] = E[(1/h) sum_{j=1..h} x_{t+j}^2 | s_t = i].

    A forecast from filtered probabilities pi_t is then pi_t @ c[h-1].
    """
    second_moments = model.stdevs**2 + model.means**2
    table = np.zeros((max_horizon_days, 2))
    power = np.eye(2)
    running_total = np.zeros(2)
    for h in range(max_horizon_days):
        power = power @ model.transition
        running_total = running_total + power @ second_moments
        table[h] = running_total / (h + 1)
    return table


def regime_variance_paths(filtered_now: np.ndarray, model: RegimeModel, horizon_days: int,
                          num_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate future regime paths and return each path's average daily variance.

    Used to price options under the regime model by mixing Black-Scholes prices
    over the distribution of average variance (see pricing_models.py).
    """
    states = (rng.random(num_paths) < filtered_now[1]).astype(int)
    total_variance = np.zeros(num_paths)
    for _ in range(horizon_days):
        stay_probability = model.transition[states, states]
        switch = rng.random(num_paths) >= stay_probability
        states = np.where(switch, 1 - states, states)
        total_variance += model.stdevs[states] ** 2
    return total_variance / horizon_days
