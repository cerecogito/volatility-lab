"""Bayesian volatility: the conjugate Normal-Inverse-Gamma (NIG) model.

Model for the daily log returns x_1..x_n in a window (MIT 6.431x: Bayesian
inference; MIT 18.6501x: likelihood):

    x_i | mu, s2  ~  N(mu, s2)        independently
    mu  | s2      ~  N(m0, s2 / k0)
    s2            ~  InvGamma(a0, b0)

Because the prior is conjugate, the posterior has the same form, with

    kn = k0 + n
    mn = (k0 m0 + n xbar) / kn
    an = a0 + n / 2
    bn = b0 + 0.5 * sum (x_i - xbar)^2 + k0 n (xbar - m0)^2 / (2 kn)

so  s2 | data ~ InvGamma(an, bn), and the posterior predictive for one new
return is a Student-t with 2 an degrees of freedom, location mn and
scale^2 = bn (kn + 1) / (an kn). The derivation is in
docs/mathematical_methodology.md.

The prior is written in the "scaled inverse chi-square" way, which is
easier to explain:
    a0 = nu0 / 2,   b0 = nu0 * s0_sq / 2
where nu0 is the prior strength (worth nu0 pseudo-observations) and s0_sq
is a prior guess of the daily variance. We also use k0 = nu0 and m0 = 0.

Option pricing with an uncertain variance
-----------------------------------------
For a fixed s2 the risk-neutral price is BS(sigma = sqrt(365 * s2)).
Averaging over the posterior gives the Bayesian expected price

    V_bayes = E_post[ BS(sigma) ]

which is NOT the same as BS(E_post[sigma]) or BS(sqrt(E_post[s2])),
because BS is non-linear in sigma (Jensen's inequality). We compute the
posterior expectation with a deterministic "quantile midpoint" rule:

    E[g(s2)] ~= (1/N) sum_j g(F^{-1}((j + 0.5) / N))

where F^{-1} is the inverse-gamma quantile function. Each node has
probability mass 1/N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import invgamma, t as student_t

from volatility_lab.black_scholes import bs_delta, bs_price
from volatility_lab.volatility_estimators import running_window_sums


@dataclass(frozen=True)
class NIGParameters:
    """Parameters of a Normal-Inverse-Gamma distribution (prior or posterior)."""
    mean: float       # m
    kappa: float      # k: confidence in the mean, in pseudo-observations
    alpha: float      # a: shape of the inverse gamma on the variance
    beta: float       # b: scale of the inverse gamma on the variance

    def __post_init__(self):
        if self.kappa <= 0 or self.alpha <= 0 or self.beta <= 0:
            raise ValueError("kappa, alpha and beta must be positive")


def make_prior(prior_strength: float, prior_daily_variance: float, prior_mean: float = 0.0) -> NIGParameters:
    """Prior worth `prior_strength` pseudo-observations, centred on `prior_daily_variance`."""
    if prior_strength <= 0 or prior_daily_variance <= 0:
        raise ValueError("prior_strength and prior_daily_variance must be positive")
    return NIGParameters(mean=prior_mean, kappa=prior_strength, alpha=prior_strength / 2.0,
                         beta=prior_strength * prior_daily_variance / 2.0)


def update_posterior(prior: NIGParameters, returns: np.ndarray) -> NIGParameters:
    """Conjugate update of the NIG prior with a window of returns."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n == 0:
        return prior
    sample_mean = returns.mean()
    sum_sq_deviations = np.sum((returns - sample_mean) ** 2)

    kappa_n = prior.kappa + n
    mean_n = (prior.kappa * prior.mean + n * sample_mean) / kappa_n
    alpha_n = prior.alpha + n / 2.0
    beta_n = (prior.beta + 0.5 * sum_sq_deviations
              + prior.kappa * n * (sample_mean - prior.mean) ** 2 / (2.0 * kappa_n))
    return NIGParameters(mean_n, kappa_n, alpha_n, beta_n)


def rolling_posteriors(prior: NIGParameters, returns: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Posterior parameters for every rolling window ending at t (vectorised).

    Returns arrays (alpha, beta, kappa, mean) aligned with `returns` (works
    along the last axis, so a (paths, days) array is fine).
    Entries before the first full window are NaN.
    """
    returns = np.asarray(returns, dtype=float)
    output = {key: np.full(returns.shape, np.nan) for key in ["alpha", "beta", "kappa", "mean"]}
    if returns.shape[-1] < window:
        return output

    window_sum = running_window_sums(returns, window)
    window_sum_sq = running_window_sums(returns**2, window)
    sample_mean = window_sum / window
    sum_sq_deviations = np.maximum(window_sum_sq - window * sample_mean**2, 0.0)

    kappa_n = prior.kappa + window
    output["kappa"][..., window - 1:] = kappa_n
    output["mean"][..., window - 1:] = (prior.kappa * prior.mean + window * sample_mean) / kappa_n
    output["alpha"][..., window - 1:] = prior.alpha + window / 2.0
    output["beta"][..., window - 1:] = (prior.beta + 0.5 * sum_sq_deviations
                                        + prior.kappa * window * (sample_mean - prior.mean) ** 2
                                        / (2.0 * kappa_n))
    return output


def posterior_mean_variance(alpha, beta):
    """E[s2 | data] = b / (a - 1), defined for a > 1."""
    alpha = np.asarray(alpha, dtype=float)
    return np.where(alpha > 1.0, beta / np.maximum(alpha - 1.0, 1e-12), np.nan)


def posterior_variance_interval(alpha: float, beta: float, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed credible interval for the daily variance s2."""
    tail = (1.0 - level) / 2.0
    distribution = invgamma(a=alpha, scale=beta)
    return float(distribution.ppf(tail)), float(distribution.ppf(1.0 - tail))


def predictive_distribution(posterior: NIGParameters):
    """Posterior predictive of ONE new daily return: a scipy Student-t object."""
    degrees_of_freedom = 2.0 * posterior.alpha
    scale = np.sqrt(posterior.beta * (posterior.kappa + 1.0) / (posterior.alpha * posterior.kappa))
    return student_t(df=degrees_of_freedom, loc=posterior.mean, scale=scale)


def variance_quadrature_nodes(alpha, beta, num_nodes: int = 64) -> np.ndarray:
    """Quantile-midpoint nodes of InvGamma(alpha, beta), each with weight 1/num_nodes.

    alpha and beta can be arrays of shape (m,), giving nodes of shape (m, num_nodes).
    """
    probabilities = (np.arange(num_nodes) + 0.5) / num_nodes
    alpha = np.asarray(alpha, dtype=float)[..., None]
    beta = np.asarray(beta, dtype=float)[..., None]
    # InvGamma(a, b) quantile = b / Gamma(a, 1) upper quantile
    return invgamma.ppf(probabilities, a=alpha, scale=beta)


def bayesian_option_price(spot, strike, time_to_expiry, rate, option_type, alpha, beta,
                          days_per_year: float = 365.0, num_nodes: int = 64,
                          dividend_yield: float = 0.0) -> np.ndarray:
    """Posterior-expected Black-Scholes price  E_post[ BS(sqrt(days_per_year * s2)) ].

    Every argument can be an array of the same shape (one entry per option).
    """
    daily_variance_nodes = variance_quadrature_nodes(alpha, beta, num_nodes)
    vol_nodes = np.sqrt(days_per_year * daily_variance_nodes)
    expand = lambda value: np.asarray(value)[..., None] if np.ndim(value) else value  # noqa: E731
    prices = bs_price(expand(spot), expand(strike), expand(time_to_expiry), expand(rate), vol_nodes,
                      expand(option_type), expand(dividend_yield))
    return prices.mean(axis=-1)


def bayesian_option_delta(spot, strike, time_to_expiry, rate, option_type, alpha, beta,
                          days_per_year: float = 365.0, num_nodes: int = 64,
                          dividend_yield: float = 0.0) -> np.ndarray:
    """Delta of the Bayesian price = E_post[ BS delta(sigma) ] (differentiate under the integral)."""
    daily_variance_nodes = variance_quadrature_nodes(alpha, beta, num_nodes)
    vol_nodes = np.sqrt(days_per_year * daily_variance_nodes)
    expand = lambda value: np.asarray(value)[..., None] if np.ndim(value) else value  # noqa: E731
    deltas = bs_delta(expand(spot), expand(strike), expand(time_to_expiry), expand(rate), vol_nodes,
                      expand(option_type), expand(dividend_yield))
    return deltas.mean(axis=-1)
