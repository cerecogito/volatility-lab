"""Step 5: Track B - pricing real Deribit BTC option trades (HISTORICAL evidence).

For every clean trade we price the option with Black-76 on the trade's
forward, using volatility estimated ONLY from daily index returns up to and
including that day's 08:00 UTC fixing (trades happen 08:00-09:00).

Protocol (docs/evaluation_protocol.md):
  1. Bayesian prior and regime model fitted on index returns up to the
     prior_training_end date (end of the development period).
  2. VALIDATION period: choose the rolling window, the EWMA lambda and the
     Bayesian prior by validation RMSE. The Bayesian model uses the same
     window as the chosen rolling estimator, so that the comparison isolates
     the Bayesian treatment.
  3. TEST period: evaluated once with the frozen choices. Three primary
     comparisons (Holm-adjusted); everything else is exploratory.

Usage:  uv run python scripts/05_track_b_market_pricing.py
Output: reports/experiments/track_b_pricing.json
"""

import json

import numpy as np
import pandas as pd

from volatility_lab.bayesian import make_prior
from volatility_lab.config import EXPERIMENTS_DIR, PROCESSED_DIR, load_config
from volatility_lab.implied_vol import implied_volatility
from volatility_lab.metrics import pricing_error_summary
from volatility_lab.pricing_models import (
    attach_latest_fixing, bayes_integrated_prices, price_trades, regime_forecast_table,
    volatility_inputs_by_date,
)
from volatility_lab.regime_model import fit_regime_model, forward_filter
from volatility_lab.stat_tests import diebold_mariano, holm_adjust, paired_metric_difference
from volatility_lab.volatility_estimators import realised_volatility_ahead

PRIMARY_PAIRS = [("bayes_integrated", "rolling"), ("bayes_integrated", "ewma"), ("bayes_integrated", "bayes_plugin")]
EXPLORATORY_PAIRS = [("ewma", "rolling"), ("regime", "rolling"), ("regime", "ewma"), ("bayes_plugin", "rolling")]
MODELS = ["rolling", "ewma", "bayes_plugin", "bayes_integrated", "regime"]
BOOTSTRAP_REPLICATES = 5000
MAX_HORIZON_DAYS = 181


def load_inputs():
    index_daily = pd.read_csv(PROCESSED_DIR / "index_daily.csv", parse_dates=["date"])
    returns = index_daily.dropna(subset=["log_return"]).reset_index(drop=True)[["date", "log_return"]]
    trades = pd.read_csv(PROCESSED_DIR / "option_trades_clean.csv.gz",
                         parse_dates=["trade_time", "market_date", "expiry"])
    return returns, trades


def percent_of_spot(prices_usd: np.ndarray, trades: pd.DataFrame) -> np.ndarray:
    return 100.0 * prices_usd / trades["index_price"].to_numpy()


def evaluate_rmse(trades: pd.DataFrame, model_prices: np.ndarray) -> float:
    errors = percent_of_spot(model_prices, trades) - percent_of_spot(trades["price_usd"].to_numpy(), trades)
    return float(np.sqrt(np.mean(errors**2)))


def main() -> None:
    config = load_config()
    vol_config = config["volatility"]
    days_per_year = vol_config["days_per_year"]
    nodes = vol_config["posterior_quadrature_nodes"]
    returns, trades = load_inputs()

    # ------------------------------------------------ 1. prior + regime model from training returns only
    training_end = pd.Timestamp(config["splits"]["prior_training_end"], tz="UTC") + pd.Timedelta(hours=8)
    training_returns = returns[returns["date"] <= training_end]["log_return"].to_numpy()
    training_variance = float(np.var(training_returns, ddof=1))
    regime_model = fit_regime_model(training_returns)
    filtered, _ = forward_filter(returns["log_return"].to_numpy(), regime_model)
    regime_table = regime_forecast_table(returns["log_return"].to_numpy(), regime_model, MAX_HORIZON_DAYS,
                                         days_per_year)

    # ------------------------------------------------ 2. all candidate vol inputs by date
    def build_table(bayes_window: int, strength: float, multiplier: float) -> pd.DataFrame:
        prior = make_prior(strength, multiplier * training_variance)
        table = volatility_inputs_by_date(returns, vol_config["rolling_windows"], vol_config["ewma_lambdas"],
                                          bayes_window, prior, days_per_year)
        table["turbulent_probability"] = filtered[:, 1]
        table["row_number"] = np.arange(len(table))
        return table

    longest = max(vol_config["rolling_windows"])
    base = attach_latest_fixing(trades, build_table(longest, 10.0, 1.0))
    usable = base[f"rolling_{longest}"].notna() & base["split"].isin(["validation", "test"])
    history_exclusions = {
        "missing_index_fixing_or_insufficient_history": int((~base[f"rolling_{longest}"].notna()).sum()),
        "development_period_not_evaluated": int((base["split"] == "development").sum()),
    }

    def split_frame(table: pd.DataFrame, split: str) -> pd.DataFrame:
        frame = attach_latest_fixing(trades, table)
        frame = frame[usable.to_numpy() & (frame["split"] == split).to_numpy()].reset_index(drop=True)
        return frame

    # ------------------------------------------------ 3. validation: choose hyper-parameters
    validation = split_frame(build_table(longest, 10.0, 1.0), "validation")
    rolling_scores = {w: evaluate_rmse(validation, price_trades(validation, f"rolling_{w}"))
                      for w in vol_config["rolling_windows"]}
    ewma_scores = {lam: evaluate_rmse(validation, price_trades(validation, f"ewma_{lam:.2f}"))
                   for lam in vol_config["ewma_lambdas"]}
    best_window = min(rolling_scores, key=rolling_scores.get)
    best_lambda = min(ewma_scores, key=ewma_scores.get)

    prior_scores = []
    for strength in vol_config["prior_strengths"]:
        for multiplier in vol_config["prior_scale_multipliers"]:
            frame = split_frame(build_table(best_window, strength, multiplier), "validation")
            prior_scores.append({
                "prior_strength": strength, "prior_variance_multiplier": multiplier,
                "validation_rmse_integrated": evaluate_rmse(frame, bayes_integrated_prices(frame, nodes, days_per_year)),
                "validation_rmse_plugin": evaluate_rmse(frame, price_trades(frame, "bayes_plugin")),
            })
    best_prior = min(prior_scores, key=lambda row: row["validation_rmse_integrated"])
    print(f"validation choices: window={best_window}, lambda={best_lambda}, prior={best_prior}")

    # ------------------------------------------------ 4. test: evaluate once with frozen choices
    final_table = build_table(best_window, best_prior["prior_strength"], best_prior["prior_variance_multiplier"])
    results = {}
    for split in ["validation", "test"]:
        frame = split_frame(final_table, split)
        horizon = np.clip(np.ceil(frame["time_to_expiry"] * days_per_year).astype(int), 1, MAX_HORIZON_DAYS)
        # (the left merge turns row_number into a float column, so cast it back)
        frame["regime_vol"] = regime_table[frame["row_number"].to_numpy().astype(int), horizon.to_numpy() - 1]
        frame["rolling_vol"] = frame[f"rolling_{best_window}"]
        frame["ewma_vol"] = frame[f"ewma_{best_lambda:.2f}"]

        prices = {
            "rolling": price_trades(frame, "rolling_vol"),
            "ewma": price_trades(frame, "ewma_vol"),
            "bayes_plugin": price_trades(frame, "bayes_plugin"),
            "bayes_integrated": bayes_integrated_prices(frame, nodes, days_per_year),
            "regime": price_trades(frame, "regime_vol"),
        }
        market_pct = percent_of_spot(frame["price_usd"].to_numpy(), frame)
        errors_pct = {name: percent_of_spot(p, frame) - market_pct for name, p in prices.items()}

        # implied vol of each model price (vol-space error vs the trade's reported IV)
        vol_points = {}
        for name in MODELS:
            if name == "bayes_integrated":
                model_iv = np.array([implied_volatility(p, f, k, t, 0.0, kind)[0] for p, f, k, t, kind in zip(
                    prices[name], frame["forward"], frame["strike"], frame["time_to_expiry"], frame["option_type"])])
            else:
                column = {"rolling": "rolling_vol", "ewma": "ewma_vol", "bayes_plugin": "bayes_plugin",
                          "regime": "regime_vol"}[name]
                model_iv = frame[column].to_numpy()
            vol_points[name] = 100.0 * (model_iv - frame["reported_iv"].to_numpy())

        summary = {}
        for name in MODELS:
            summary[name] = {
                "pct_of_spot": pricing_error_summary(percent_of_spot(prices[name], frame), market_pct),
                # every clean trade costs >= 0.0005 BTC, so dividing by the trade price is meaningful
                "usd": pricing_error_summary(prices[name], frame["price_usd"].to_numpy(), relative_floor=0.0),
                "vol_points_mean": float(np.nanmean(vol_points[name])),
                "vol_points_mae": float(np.nanmean(np.abs(vol_points[name]))),
                "vol_points_uninvertible": int(np.isnan(vol_points[name]).sum()),
            }

        # grouped errors (percent of spot)
        frame["moneyness_bucket"] = pd.cut(frame["log_moneyness"], [-0.51, -0.2, -0.05, 0.05, 0.2, 0.51],
                                           labels=["K<<F", "K<F", "ATM", "K>F", "K>>F"])
        frame["maturity_bucket"] = pd.cut(frame["time_to_expiry"] * 365, [2.9, 14, 45, 90, 180.1],
                                          labels=["3-14d", "14-45d", "45-90d", "90-180d"])
        frame["regime_bucket"] = np.where(frame["turbulent_probability"] > 0.5, "turbulent", "calm")
        grouped = []
        for group_column in ["moneyness_bucket", "maturity_bucket", "regime_bucket", "option_type", "direction"]:
            for group_value, rows in frame.groupby(group_column, observed=True):
                mask = frame.index.isin(rows.index)
                for name in MODELS:
                    grouped.append({"group": group_column, "value": str(group_value), "model": name,
                                    **pricing_error_summary(percent_of_spot(prices[name], frame)[mask],
                                                            market_pct[mask])})

        # paired comparisons, clustered by market date
        clusters = frame["market_date"].dt.strftime("%Y-%m-%d").to_numpy()
        comparisons = []
        for family, pairs in [("primary", PRIMARY_PAIRS), ("exploratory", EXPLORATORY_PAIRS)]:
            for model_a, model_b in pairs:
                result = paired_metric_difference(errors_pct[model_a], errors_pct[model_b], clusters,
                                                  np.random.default_rng(config["simulation"]["master_seed"]),
                                                  BOOTSTRAP_REPLICATES)
                daily = pd.DataFrame({"date": clusters, "d": errors_pct[model_a] ** 2 - errors_pct[model_b] ** 2})
                dm = diebold_mariano(daily.groupby("date")["d"].mean().to_numpy())
                comparisons.append({"family": family, "model_a": model_a, "model_b": model_b,
                                    "dm_squared_error": dm, **result})
        primary_p = {f"{c['model_a']} vs {c['model_b']}": c["mse_diff_bootstrap_p"]
                     for c in comparisons if c["family"] == "primary"}
        for name, adjusted in holm_adjust(primary_p).items():
            for c in comparisons:
                if f"{c['model_a']} vs {c['model_b']}" == name:
                    c["mse_diff_holm_p"] = adjusted

        results[split] = {
            "n_trades": int(len(frame)), "n_market_dates": int(frame["market_date"].nunique()),
            "n_instruments": int(frame["instrument_name"].nunique()),
            "first_trade": str(frame["trade_time"].min()), "last_trade": str(frame["trade_time"].max()),
            "mean_reported_iv": float(frame["reported_iv"].mean()),
            "mean_model_vol": {"rolling": float(frame["rolling_vol"].mean()), "ewma": float(frame["ewma_vol"].mean()),
                               "bayes_plugin": float(frame["bayes_plugin"].mean()),
                               "regime": float(frame["regime_vol"].mean())},
            "summary": summary, "grouped": grouped, "comparisons": comparisons,
        }
        print(f"{split}: {len(frame)} trades on {frame['market_date'].nunique()} dates")
        for name in MODELS:
            print(f"   {name:17s} RMSE {summary[name]['pct_of_spot']['rmse']:.4f}%  "
                  f"bias {summary[name]['pct_of_spot']['bias']:+.4f}%  vol-pts bias {summary[name]['vol_points_mean']:+.2f}")

    # ------------------------------------------------ 5. does the regime model forecast realised vol? (test dates)
    test_start = pd.Timestamp(config["splits"]["test_start"], tz="UTC")
    test_end = pd.Timestamp(config["splits"]["test_end"], tz="UTC") + pd.Timedelta(days=1)
    horizon = 30
    realised = realised_volatility_ahead(returns["log_return"].to_numpy(), horizon, days_per_year)
    forecasts = {"rolling": final_table[f"rolling_{best_window}"].to_numpy(),
                 "ewma": final_table[f"ewma_{best_lambda:.2f}"].to_numpy(),
                 "bayes_plugin": final_table["bayes_plugin"].to_numpy(),
                 "regime": regime_table[:, horizon - 1]}
    in_test = ((returns["date"] >= test_start) & (returns["date"] < test_end)).to_numpy() & np.isfinite(realised)
    qlike = {}
    for name, forecast in forecasts.items():
        ratio = realised[in_test] ** 2 / forecast[in_test] ** 2
        qlike[name] = ratio - np.log(ratio) - 1.0           # QLIKE loss, 0 = perfect
    forecast_eval = {"horizon_days": horizon, "n_forecast_dates": int(in_test.sum()),
                     "note": "overlapping 30-day targets -> DM uses Newey-West lag 30",
                     "mean_qlike": {name: float(loss.mean()) for name, loss in qlike.items()},
                     "comparisons": {}}
    for model_a, model_b in [("regime", "rolling"), ("regime", "ewma"), ("ewma", "rolling"), ("bayes_plugin", "rolling")]:
        forecast_eval["comparisons"][f"{model_a} vs {model_b}"] = diebold_mariano(qlike[model_a] - qlike[model_b],
                                                                                   max_lag=horizon)

    output = {
        "experiment_id": config["experiment_id"], "evidence_type": "historical (Deribit BTC option trades)",
        "target": "observed trade price (not a quote midpoint)",
        "error_definition": "model price - trade price; primary unit = % of index price at the trade",
        "training": {"returns_used": int(len(training_returns)), "end": str(training_end),
                     "training_daily_variance": training_variance,
                     "training_annual_vol": float(np.sqrt(days_per_year * training_variance)),
                     "regime_model": {"daily_stdevs": regime_model.stdevs.tolist(),
                                      "annual_vols": (regime_model.stdevs * np.sqrt(days_per_year)).tolist(),
                                      "means": regime_model.means.tolist(),
                                      "transition": regime_model.transition.tolist(),
                                      "converged": regime_model.converged,
                                      "iterations": regime_model.iterations}},
        "validation_selection": {"rolling_rmse_by_window": {str(k): v for k, v in rolling_scores.items()},
                                 "ewma_rmse_by_lambda": {f"{k:.2f}": v for k, v in ewma_scores.items()},
                                 "prior_grid": prior_scores, "chosen_window": best_window,
                                 "chosen_lambda": best_lambda, "chosen_prior": best_prior},
        "exclusions_before_evaluation": history_exclusions,
        "results": results, "volatility_forecast_evaluation": forecast_eval,
        "bootstrap": f"cluster (market date) bootstrap, {BOOTSTRAP_REPLICATES} replicates",
    }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "track_b_pricing.json").write_text(json.dumps(output, indent=1, default=str))
    print("QLIKE:", forecast_eval["mean_qlike"])


if __name__ == "__main__":
    main()
