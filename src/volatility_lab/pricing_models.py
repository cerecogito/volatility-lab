"""The competing pricing models, all built on the same Black-Scholes core.

Every model turns the daily index-return history into a volatility input
for each date, using only returns up to and including that date's 08:00
UTC fixing. The options are then priced with Black-76 on the forward
(spot = F, r = q = 0), which is Deribit's convention.

Models
------
rolling_{w}       BS with the w-day sample volatility
ewma_{lambda}     BS with the EWMA volatility
bayes_plugin      BS with sigma = sqrt(365 * E_post[s2])     (point estimate)
bayes_integrated  E_post[ BS(sigma) ]                         (integrates over uncertainty)
regime            BS with the regime model's forecast of average volatility over
                  the option's remaining life (a plug-in variance forecast)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volatility_lab.bayesian import (
    NIGParameters, bayesian_option_delta, bayesian_option_price, posterior_mean_variance, rolling_posteriors,
)
from volatility_lab.black_scholes import bs_delta, bs_price
from volatility_lab.regime_model import RegimeModel, forward_filter, regime_variance_paths
from volatility_lab.volatility_estimators import ewma_volatility, rolling_volatility


def regime_mixture_price(current_state_probs: np.ndarray, model: RegimeModel, spot: float, strikes: np.ndarray,
                         maturity_days: int, option_type: str, num_paths: int, rng: np.random.Generator,
                         days_per_year: float = 365.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact-in-distribution price under the regime model (Hull-White mixing argument).

    If volatility moves independently of the price shocks and volatility risk
    is not priced, then conditional on the variance path the log price is
    normal, so
        price = E[ BS(sigma = sqrt(average variance over the option's life)) ].
    We estimate that expectation by simulating regime paths. Returns
    (prices, monte_carlo_standard_errors), one per strike.
    """
    average_daily_variance = regime_variance_paths(current_state_probs, model, maturity_days, num_paths, rng)
    vols = np.sqrt(days_per_year * average_daily_variance)[:, None]
    values = bs_price(spot, np.asarray(strikes)[None, :], maturity_days / days_per_year, 0.0, vols, option_type)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(num_paths)


def regime_forecast_table(returns: np.ndarray, model: RegimeModel, max_horizon_days: int,
                          days_per_year: float = 365.0) -> np.ndarray:
    """(dates, horizons) table: forecast annualised vol over the next h days, h = 1..max."""
    filtered, _ = forward_filter(returns, model)
    second_moments = model.stdevs**2 + model.means**2
    state_probs = filtered.copy()
    cumulative_variance = np.zeros((len(returns), max_horizon_days))
    running_total = np.zeros(len(returns))
    for h in range(max_horizon_days):
        state_probs = state_probs @ model.transition
        running_total = running_total + state_probs @ second_moments
        cumulative_variance[:, h] = running_total / (h + 1)
    return np.sqrt(days_per_year * cumulative_variance)


def volatility_inputs_by_date(index_returns: pd.DataFrame, rolling_windows: list[int],
                              ewma_lambdas: list[float], bayes_window: int, prior: NIGParameters,
                              days_per_year: float = 365.0) -> pd.DataFrame:
    """One row per index fixing date with every model's volatility input.

    Columns: date, rolling_{w}, ewma_{lam}, bayes_alpha, bayes_beta, bayes_plugin.
    """
    returns = index_returns["log_return"].to_numpy()
    table = pd.DataFrame({"date": index_returns["date"].to_numpy()})
    for window in rolling_windows:
        table[f"rolling_{window}"] = rolling_volatility(returns, window, days_per_year)
    for decay in ewma_lambdas:
        table[f"ewma_{decay:.2f}"] = ewma_volatility(returns, decay, days_per_year)
    posterior = rolling_posteriors(prior, returns, bayes_window)
    table["bayes_alpha"] = posterior["alpha"]
    table["bayes_beta"] = posterior["beta"]
    table["bayes_plugin"] = np.sqrt(days_per_year * posterior_mean_variance(posterior["alpha"], posterior["beta"]))
    return table


def attach_latest_fixing(trades: pd.DataFrame, vol_table: pd.DataFrame,
                         max_staleness: pd.Timedelta = pd.Timedelta(days=1)) -> pd.DataFrame:
    """Give every trade the volatility inputs of the latest fixing AT OR BEFORE its trade time.

    This is the no-look-ahead rule written as code: merge_asof with
    direction='backward' can never pick a fixing that happens after the trade.
    If the latest fixing is older than `max_staleness` the inputs are left as
    NaN (the trade is then excluded as having no up-to-date history).
    The output keeps the original row order of `trades`.
    """
    table = vol_table.rename(columns={"date": "fixing_time"})
    table["fixing_time"] = table["fixing_time"].astype("datetime64[ns, UTC]")
    table = table.sort_values("fixing_time")
    ordered = trades.reset_index(drop=True).assign(_original_row=lambda frame: np.arange(len(frame)))
    ordered["trade_time"] = ordered["trade_time"].astype("datetime64[ns, UTC]")
    ordered = ordered.sort_values("trade_time")
    merged = pd.merge_asof(ordered, table, left_on="trade_time", right_on="fixing_time",
                           direction="backward", tolerance=max_staleness)
    return merged.sort_values("_original_row").drop(columns="_original_row").reset_index(drop=True)


def price_trades(trades: pd.DataFrame, vol_column: str) -> np.ndarray:
    """Black-76 prices (USD) of trades given a column of annualised volatilities."""
    return bs_price(trades["forward"].to_numpy(), trades["strike"].to_numpy(),
                    trades["time_to_expiry"].to_numpy(), 0.0, trades[vol_column].to_numpy(),
                    trades["option_type"].to_numpy())


def bayes_integrated_prices(trades: pd.DataFrame, num_nodes: int, days_per_year: float = 365.0) -> np.ndarray:
    return bayesian_option_price(trades["forward"].to_numpy(), trades["strike"].to_numpy(),
                                 trades["time_to_expiry"].to_numpy(), 0.0, trades["option_type"].to_numpy(),
                                 trades["bayes_alpha"].to_numpy(), trades["bayes_beta"].to_numpy(),
                                 days_per_year, num_nodes)


def model_delta(model_name: str, spot, strike, time_to_expiry, option_type, vol=None,
                alpha=None, beta=None, num_nodes: int = 32, days_per_year: float = 365.0):
    """Hedge ratio for any model (r = 0). Plug-in models use `vol`; bayes_integrated uses alpha/beta."""
    if model_name == "bayes_integrated":
        return bayesian_option_delta(spot, strike, time_to_expiry, 0.0, option_type, alpha, beta,
                                     days_per_year, num_nodes)
    return bs_delta(spot, strike, time_to_expiry, 0.0, vol, option_type)


def model_price(model_name: str, spot, strike, time_to_expiry, option_type, vol=None,
                alpha=None, beta=None, num_nodes: int = 32, days_per_year: float = 365.0):
    """Price for any model (r = 0)."""
    if model_name == "bayes_integrated":
        return bayesian_option_price(spot, strike, time_to_expiry, 0.0, option_type, alpha, beta,
                                     days_per_year, num_nodes)
    return bs_price(spot, strike, time_to_expiry, 0.0, vol, option_type)
