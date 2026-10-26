"""Step 6: Track B - HYPOTHETICAL delta hedges on the OBSERVED BTC index path.

This is not a trading strategy and nothing was executed. We ask: if someone
had sold a 30-day at-the-money call at each model's own price and delta
hedged it at the daily 08:00 UTC fixings, how big would the hedge error
have been under each volatility model?

Assumptions: r = 0, forward = spot (no basis), European cash-settled call,
K = index at the sale, 5 bp proportional cost, hedge in the index itself.
Model hyper-parameters are the ones frozen on the validation period in
step 5 (read from track_b_pricing.json). Only TEST-period start dates are used.

Usage:  uv run python scripts/06_track_b_historical_hedging.py
Output: reports/experiments/track_b_hedging.json
"""

import json

import numpy as np
import pandas as pd

from volatility_lab.bayesian import make_prior
from volatility_lab.config import EXPERIMENTS_DIR, PROCESSED_DIR, load_config
from volatility_lab.hedging import run_delta_hedge
from volatility_lab.metrics import hedging_error_summary
from volatility_lab.pricing_models import model_delta, model_price, regime_forecast_table, volatility_inputs_by_date
from volatility_lab.regime_model import fit_regime_model
from volatility_lab.stat_tests import holm_adjust, paired_metric_difference

MODELS = ["rolling", "ewma", "bayes_plugin", "bayes_integrated", "regime"]
PAIRS = [("bayes_integrated", "rolling"), ("bayes_integrated", "ewma"), ("bayes_integrated", "bayes_plugin"),
         ("ewma", "rolling"), ("regime", "rolling")]
BLOCK_SIZE = 5            # consecutive weekly starts per bootstrap block (hedges overlap ~4 weeks)
BOOTSTRAP_REPLICATES = 5000


def main() -> None:
    config = load_config()
    hedge_config = config["hedging_historical"]
    days_per_year = config["volatility"]["days_per_year"]
    pricing = json.loads((EXPERIMENTS_DIR / "track_b_pricing.json").read_text())
    choice = pricing["validation_selection"]
    window, decay = choice["chosen_window"], choice["chosen_lambda"]
    prior_choice = choice["chosen_prior"]

    index_daily = pd.read_csv(PROCESSED_DIR / "index_daily.csv", parse_dates=["date"])
    returns = index_daily.dropna(subset=["log_return"]).reset_index(drop=True)
    prices = returns["index_price"].to_numpy()

    # same frozen ingredients as the pricing study
    training_end = pd.Timestamp(config["splits"]["prior_training_end"], tz="UTC") + pd.Timedelta(hours=8)
    training_returns = returns[returns["date"] <= training_end]["log_return"].to_numpy()
    prior = make_prior(prior_choice["prior_strength"],
                       prior_choice["prior_variance_multiplier"] * np.var(training_returns, ddof=1))
    table = volatility_inputs_by_date(returns[["date", "log_return"]], [window], [decay], window, prior, days_per_year)
    regime_model = fit_regime_model(training_returns)
    maturity = hedge_config["maturity_days"]
    regime_table = regime_forecast_table(returns["log_return"].to_numpy(), regime_model, maturity, days_per_year)

    # start dates: every 7th day inside the test period with a full 30-day path afterwards
    test_start = pd.Timestamp(config["splits"]["test_start"], tz="UTC")
    test_end = pd.Timestamp(config["splits"]["test_end"], tz="UTC") + pd.Timedelta(days=1)
    candidates = np.where((returns["date"] >= test_start) & (returns["date"] < test_end))[0]
    candidates = candidates[candidates + maturity < len(returns)][:: hedge_config["start_every_days"]]
    # every day between start and expiry must be a consecutive daily fixing
    gaps = returns["date"].diff().dt.total_seconds().to_numpy() / 86400.0
    starts = [i for i in candidates if np.allclose(gaps[i + 1: i + maturity + 1], 1.0)]

    paths = np.array([prices[i: i + maturity + 1] for i in starts])        # (hedges, 31)
    day_index = np.array([np.arange(i, i + maturity + 1) for i in starts])  # row in the vol table
    strikes = paths[:, 0]

    results = {}
    for every in sorted({1, hedge_config["rebalance_every_days"], 7}):
        schedule = np.unique(np.append(np.arange(0, maturity, every), maturity))
        rows_now = day_index[:, schedule[:-1]]
        spot_now = paths[:, schedule[:-1]]
        time_left = ((maturity - schedule[:-1]) / days_per_year)[None, :]
        inputs = {
            "rolling": {"vol": table[f"rolling_{window}"].to_numpy()[rows_now]},
            "ewma": {"vol": table[f"ewma_{decay:.2f}"].to_numpy()[rows_now]},
            "bayes_plugin": {"vol": table["bayes_plugin"].to_numpy()[rows_now]},
            "bayes_integrated": {"alpha": table["bayes_alpha"].to_numpy()[rows_now],
                                 "beta": table["bayes_beta"].to_numpy()[rows_now]},
            "regime": {"vol": regime_table[rows_now, (maturity - schedule[:-1])[None, :] - 1]},
        }
        for cost_rate in sorted({0.0, hedge_config["transaction_cost_rate"]}):
            errors, summaries = {}, {}
            for name in MODELS:
                kwargs = inputs[name]
                deltas = model_delta(name, spot_now, strikes[:, None], time_left, "call", **kwargs)
                first = {key: value[:, 0] for key, value in kwargs.items()}
                premium = model_price(name, spot_now[:, 0], strikes, time_left[0, 0], "call", **first)
                # each hedge has its own strike, but prices, payoffs and costs are all
                # homogeneous of degree 1 in (S, K): divide by K and hedge with K = 1
                scaled = run_delta_hedge(paths[:, schedule] / strikes[:, None], deltas, 1.0, "call",
                                         premium / strikes, 0.0, every / days_per_year, cost_rate)
                errors[name] = 100.0 * scaled.hedge_error          # % of the strike (= spot at sale)
                summaries[name] = hedging_error_summary(errors[name], 100.0 * scaled.transaction_costs,
                                                        scaled.share_turnover)
                summaries[name]["mean_premium_pct"] = float(np.mean(100.0 * premium / strikes))

            clusters = np.arange(len(starts)) // BLOCK_SIZE
            comparisons = []
            for model_a, model_b in PAIRS:
                comparison = paired_metric_difference(errors[model_a], errors[model_b], clusters,
                                                      np.random.default_rng(config["simulation"]["master_seed"]),
                                                      BOOTSTRAP_REPLICATES)
                comparison["std_diff"] = float(errors[model_a].std(ddof=1) - errors[model_b].std(ddof=1))
                comparisons.append({"model_a": model_a, "model_b": model_b, **comparison})
            primary = {f"{c['model_a']} vs {c['model_b']}": c["mse_diff_bootstrap_p"] for c in comparisons[:3]}
            adjusted = holm_adjust(primary)
            for c in comparisons[:3]:
                c["mse_diff_holm_p"] = adjusted[f"{c['model_a']} vs {c['model_b']}"]
            results[f"every{every}_cost{cost_rate}"] = {"rebalance_every_days": every, "cost_rate": cost_rate,
                                                       "summaries": summaries, "comparisons": comparisons}
            print(f"every {every}d cost {cost_rate}: " + ", ".join(
                f"{m} rmse {summaries[m]['rmse']:.3f}" for m in MODELS))

    output = {
        "experiment_id": config["experiment_id"],
        "evidence_type": "hypothetical hedge simulation on observed historical BTC index paths (not executed trades)",
        "design": {"contract": f"short {maturity}-day ATM European call, K = index at sale, r = 0, forward = spot",
                   "premium": "each model's own price at the sale", "hedge_instrument": "the index itself",
                   "unit": "% of the index level at the sale",
                   "sign_convention": "hedge_error = hedge value at expiry - payoff; loss = -hedge_error",
                   "chosen_parameters": {"window": window, "lambda": decay, "prior": prior_choice},
                   "bootstrap": f"blocks of {BLOCK_SIZE} consecutive weekly starts, {BOOTSTRAP_REPLICATES} replicates"},
        "n_hedges": len(starts), "first_start": str(returns["date"].iloc[starts[0]]),
        "last_start": str(returns["date"].iloc[starts[-1]]),
        "last_expiry": str(returns["date"].iloc[starts[-1] + maturity]),
        "results": results,
    }
    (EXPERIMENTS_DIR / "track_b_hedging.json").write_text(json.dumps(output, indent=1))


if __name__ == "__main__":
    main()
