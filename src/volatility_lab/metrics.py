"""Error and risk metrics. Every function states the unit of its output = unit of its input."""

from __future__ import annotations

import numpy as np


def pricing_error_summary(model_prices: np.ndarray, target_prices: np.ndarray,
                          relative_floor: float | None = None) -> dict:
    """Signed bias, MAE, RMSE, median AE (all in the price unit) and relative error.

    error = model - target (positive = model over-prices).
    Relative error |error| / target is only averaged over targets >= relative_floor,
    because dividing by a near-zero price makes the ratio meaningless.
    """
    model_prices = np.asarray(model_prices, dtype=float)
    target_prices = np.asarray(target_prices, dtype=float)
    errors = model_prices - target_prices
    summary = {
        "n": int(len(errors)),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_ae": float(np.median(np.abs(errors))),
    }
    if relative_floor is not None:
        usable = target_prices >= relative_floor
        summary["relative_n"] = int(usable.sum())
        summary["mean_abs_relative_error"] = (
            float(np.mean(np.abs(errors[usable]) / target_prices[usable])) if usable.any() else float("nan")
        )
    return summary


def value_at_risk(losses: np.ndarray, level: float = 0.95) -> float:
    """Empirical VaR: the `level` quantile of the loss distribution (loss = -hedge_error)."""
    return float(np.quantile(np.asarray(losses, dtype=float), level))


def expected_shortfall(losses: np.ndarray, level: float = 0.95) -> float:
    """Average loss in the worst (1 - level) fraction of outcomes (losses >= VaR)."""
    losses = np.asarray(losses, dtype=float)
    threshold = value_at_risk(losses, level)
    return float(losses[losses >= threshold].mean())


def hedging_error_summary(hedge_errors: np.ndarray, transaction_costs: np.ndarray | None = None,
                          turnover: np.ndarray | None = None) -> dict:
    """Summary of signed hedge errors (X(T) - payoff). Same unit as the option price."""
    errors = np.asarray(hedge_errors, dtype=float)
    losses = -errors
    summary = {
        "n": int(len(errors)),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "std": float(errors.std(ddof=1)),
        "q01": float(np.quantile(errors, 0.01)),
        "q05": float(np.quantile(errors, 0.05)),
        "q95": float(np.quantile(errors, 0.95)),
        "q99": float(np.quantile(errors, 0.99)),
        "var95_loss": value_at_risk(losses, 0.95),
        "es95_loss": expected_shortfall(losses, 0.95),
        "var99_loss": value_at_risk(losses, 0.99),
        "es99_loss": expected_shortfall(losses, 0.99),
    }
    if transaction_costs is not None:
        summary["mean_transaction_cost"] = float(np.mean(transaction_costs))
    if turnover is not None:
        summary["mean_share_turnover"] = float(np.mean(turnover))
    return summary
