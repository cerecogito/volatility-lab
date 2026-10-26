"""Step 4: Track A - controlled simulation (ALL results here are SIMULATED).

Part 1 - pricing under a known data-generating process (DGP)
    Simulate a return history, estimate volatility from it, price options,
    and compare with the TRUE price implied by the known DGP.

Part 2 - delta hedging under a known DGP
    Simulate history + future path, sell an ATM call at the model price,
    delta hedge it with the model's own hedge ratio (re-estimated from data
    available at each rebalance), and measure the hedge error at expiry.

Usage:  uv run python scripts/04_track_a_simulation.py
Output: reports/experiments/track_a_pricing.json, track_a_hedging.json (+ csv tables)
"""

import json
import time

import numpy as np
import pandas as pd

from volatility_lab.bayesian import (
    bayesian_option_delta, bayesian_option_price, make_prior, posterior_mean_variance, rolling_posteriors,
)
from volatility_lab.black_scholes import bs_delta, bs_price
from volatility_lab.config import EXPERIMENTS_DIR, load_config
from volatility_lab.hedging import run_delta_hedge
from volatility_lab.metrics import hedging_error_summary, pricing_error_summary
from volatility_lab.pricing_models import regime_mixture_price
from volatility_lab.regime_model import (
    RegimeModel, average_second_moment_by_horizon, forward_filter_paths,
)
from volatility_lab.simulation import (
    prices_from_returns, simulate_gbm_returns, simulate_regime_returns, stationary_distribution,
)
from volatility_lab.stat_tests import holm_adjust, paired_metric_difference
from volatility_lab.volatility_estimators import ewma_volatility, rolling_volatility

DAYS_PER_YEAR = 365.0
SPOT = 100.0      # prices are therefore "per 100 of spot" = % of spot
REGIME_VOLS = (0.4, 1.0)
REGIME_TRANSITION = np.array([[0.98, 0.02], [0.05, 0.95]])

DGPS = {
    "gbm_vol20": {"kind": "gbm", "vol": 0.20},
    "gbm_vol60": {"kind": "gbm", "vol": 0.60},
    "regime_40_100": {"kind": "regime", "vols": REGIME_VOLS},
}
PRICING_MATURITIES = [7, 30, 90]
PRICING_MONEYNESS = [0.8, 0.9, 1.0, 1.1, 1.2]
PRICING_HISTORIES = [30, 90]
PRICING_NUM_HISTORIES = 20_000
TRUE_REGIME_PRICE_PATHS = 400_000

HEDGE_NUM_PATHS = 10_000
HEDGE_HISTORY_DAYS = 90
HEDGE_MATURITIES = [30, 90]
HEDGE_REBALANCE_DAYS = [1, 7]
HEDGE_COST_RATES = [0.0, 0.001]
WEAK_PRIOR_STRENGTH = 2.0
BOOTSTRAP_REPLICATES = 2000


def true_regime_model() -> RegimeModel:
    daily = np.array(REGIME_VOLS) / np.sqrt(DAYS_PER_YEAR)
    return RegimeModel(means=np.zeros(2), stdevs=daily, transition=REGIME_TRANSITION,
                       initial_probs=stationary_distribution(REGIME_TRANSITION),
                       log_likelihood=float("nan"), iterations=0, converged=True)


def unconditional_annual_variance(dgp: dict) -> float:
    if dgp["kind"] == "gbm":
        return dgp["vol"] ** 2
    return float(stationary_distribution(REGIME_TRANSITION) @ np.array(dgp["vols"]) ** 2)


def simulate_returns(dgp: dict, num_paths: int, num_days: int, rng: np.random.Generator):
    """Returns (returns, states) - states is None for GBM."""
    if dgp["kind"] == "gbm":
        return simulate_gbm_returns(num_paths, num_days, 0.05, dgp["vol"], rng), None
    return simulate_regime_returns(num_paths, num_days, dgp["vols"], REGIME_TRANSITION, rng, annual_drift=0.05)


# ----------------------------------------------------------------------------- part 1: pricing

def pricing_experiment(seed: int) -> dict:
    strikes = SPOT * np.array(PRICING_MONEYNESS)
    rows, comparisons, sensitivity = [], [], []
    scenario_meta = []
    scenario_number = 0

    for dgp_name, dgp in DGPS.items():
        for history_days in PRICING_HISTORIES:
            scenario_number += 1
            scenario_seed = seed + 1000 * scenario_number
            rng = np.random.default_rng(scenario_seed)
            returns, states = simulate_returns(dgp, PRICING_NUM_HISTORIES, history_days, rng)
            final_state = None if states is None else states[:, -1]

            # --- volatility estimates at the end of each history
            prior_variance_daily = unconditional_annual_variance(dgp) / DAYS_PER_YEAR
            prior = make_prior(WEAK_PRIOR_STRENGTH, prior_variance_daily)
            posterior = rolling_posteriors(prior, returns, history_days)
            alpha, beta = posterior["alpha"][:, -1], posterior["beta"][:, -1]
            estimates = {
                "rolling": rolling_volatility(returns, history_days, DAYS_PER_YEAR)[:, -1],
                "ewma_0.94": ewma_volatility(returns, 0.94, DAYS_PER_YEAR, seed_window=min(30, history_days))[:, -1],
                "bayes_plugin": np.sqrt(DAYS_PER_YEAR * posterior_mean_variance(alpha, beta)),
            }

            for maturity in PRICING_MATURITIES:
                tau = maturity / DAYS_PER_YEAR
                # --- the true price under the DGP
                if dgp["kind"] == "gbm":
                    true_by_strike = bs_price(SPOT, strikes, tau, 0.0, dgp["vol"], "call")
                    true_prices = np.tile(true_by_strike, (PRICING_NUM_HISTORIES, 1))
                    true_mc_se = np.zeros(len(strikes))
                else:
                    # conditional on the true current regime (known to the DGP, not to the estimators)
                    model = true_regime_model()
                    true_rng = np.random.default_rng(scenario_seed + maturity)
                    by_state = [regime_mixture_price(np.eye(2)[state], model, SPOT, strikes, maturity, "call",
                                                     TRUE_REGIME_PRICE_PATHS, true_rng) for state in [0, 1]]
                    true_prices = np.where(final_state[:, None] == 0, by_state[0][0][None, :], by_state[1][0][None, :])
                    true_mc_se = np.maximum(by_state[0][1], by_state[1][1])

                model_prices = {name: bs_price(SPOT, strikes[None, :], tau, 0.0, vol[:, None], "call")
                                for name, vol in estimates.items()}
                model_prices["bayes_integrated"] = bayesian_option_price(
                    SPOT, strikes[None, :], tau, 0.0, "call", alpha[:, None], beta[:, None], DAYS_PER_YEAR, 64)

                for model_name, prices in model_prices.items():
                    for k, moneyness in enumerate(PRICING_MONEYNESS):
                        groups = {"all": np.ones(PRICING_NUM_HISTORIES, bool)}
                        if final_state is not None:
                            groups.update({"calm_state": final_state == 0, "turbulent_state": final_state == 1})
                        for group_name, mask in groups.items():
                            summary = pricing_error_summary(prices[mask, k], true_prices[mask, k],
                                                            relative_floor=0.05)
                            rows.append({"dgp": dgp_name, "history_days": history_days, "maturity_days": maturity,
                                         "moneyness": moneyness, "model": model_name, "regime_group": group_name,
                                         "true_price_mean": float(true_prices[mask, k].mean()),
                                         "true_price_mc_se": float(true_mc_se[k]), **summary})

                # --- paired comparisons over all strikes of this maturity (cluster = history)
                clusters = np.repeat(np.arange(PRICING_NUM_HISTORIES), len(strikes))
                target = true_prices.ravel()
                for model_a, model_b in [("bayes_integrated", "rolling"), ("bayes_integrated", "bayes_plugin"),
                                         ("bayes_integrated", "ewma_0.94"), ("ewma_0.94", "rolling")]:
                    result = paired_metric_difference(model_prices[model_a].ravel() - target,
                                                      model_prices[model_b].ravel() - target, clusters,
                                                      np.random.default_rng(scenario_seed + 7), BOOTSTRAP_REPLICATES)
                    comparisons.append({"dgp": dgp_name, "history_days": history_days, "maturity_days": maturity,
                                        "model_a": model_a, "model_b": model_b, **result})

            # --- prior sensitivity (strength x centre), 30-day options
            tau = 30 / DAYS_PER_YEAR
            true_30 = (bs_price(SPOT, strikes, tau, 0.0, dgp["vol"], "call") if dgp["kind"] == "gbm" else None)
            if true_30 is not None:
                for strength in [2.0, 10.0, 30.0, 100.0]:
                    for multiplier in [0.5, 1.0, 2.0]:
                        prior_k = make_prior(strength, multiplier * prior_variance_daily)
                        post_k = rolling_posteriors(prior_k, returns, history_days)
                        prices_k = bayesian_option_price(SPOT, strikes[None, :], tau, 0.0, "call",
                                                         post_k["alpha"][:, -1, None], post_k["beta"][:, -1, None],
                                                         DAYS_PER_YEAR, 64)
                        summary = pricing_error_summary(prices_k.ravel(), np.tile(true_30, PRICING_NUM_HISTORIES))
                        sensitivity.append({"dgp": dgp_name, "history_days": history_days, "maturity_days": 30,
                                            "prior_strength": strength, "prior_variance_multiplier": multiplier,
                                            **summary})

            scenario_meta.append({"dgp": dgp_name, "history_days": history_days, "seed": scenario_seed,
                                  "simulated_histories": PRICING_NUM_HISTORIES})
            print(f"  pricing scenario {dgp_name} / history {history_days} done")

    # Holm correction across the primary family: bayes_integrated vs rolling, MSE difference
    primary = {f"{c['dgp']}|h{c['history_days']}|T{c['maturity_days']}": c["mse_diff_bootstrap_p"]
               for c in comparisons if c["model_a"] == "bayes_integrated" and c["model_b"] == "rolling"}
    adjusted = holm_adjust(primary)
    for c in comparisons:
        key = f"{c['dgp']}|h{c['history_days']}|T{c['maturity_days']}"
        if c["model_a"] == "bayes_integrated" and c["model_b"] == "rolling":
            c["mse_diff_holm_p"] = adjusted[key]

    return {
        "evidence_type": "simulated",
        "design": {
            "dgps": DGPS, "regime_transition": REGIME_TRANSITION.tolist(), "spot": SPOT, "rate": 0.0,
            "maturities_days": PRICING_MATURITIES, "moneyness_K_over_S": PRICING_MONEYNESS,
            "option_type": "call (r = 0, so put errors equal call errors by put-call parity)",
            "history_days": PRICING_HISTORIES, "simulated_histories_per_scenario": PRICING_NUM_HISTORIES,
            "contracts_per_history": len(PRICING_MATURITIES) * len(PRICING_MONEYNESS),
            "true_price": "GBM: Black-Scholes at the true sigma. Regime: Hull-White mixture conditional on the "
                          f"true current state, {TRUE_REGIME_PRICE_PATHS} simulated regime paths",
            "estimators": {"rolling": "sample std over the whole history", "ewma_0.94": "lambda 0.94",
                           "bayes": f"NIG, prior strength {WEAK_PRIOR_STRENGTH} centred on the unconditional variance",
                           "bayes_integrated_nodes": 64},
            "error_unit": "price units per 100 of spot (= % of spot); error = model - true",
            "relative_error_floor": 0.05,
            "bootstrap": f"cluster = simulated history, {BOOTSTRAP_REPLICATES} replicates",
            "scenarios": scenario_meta,
        },
        "rows": rows, "comparisons": comparisons, "prior_sensitivity": sensitivity,
    }


# ----------------------------------------------------------------------------- part 2: hedging

def hedge_ratios_for_models(dgp: dict, returns: np.ndarray, daily_prices: np.ndarray, schedule: np.ndarray,
                            maturity: int, strike: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For every model: (premium (paths,), hedge ratios (paths, steps)).

    The estimate used at rebalance day k is computed from returns up to day k
    only: it sits at column HEDGE_HISTORY_DAYS - 1 + k of the combined array.
    """
    columns = HEDGE_HISTORY_DAYS - 1 + schedule[:-1]
    prices_now = daily_prices[:, schedule[:-1]]
    times_left = (maturity - schedule[:-1]) / DAYS_PER_YEAR

    prior = make_prior(WEAK_PRIOR_STRENGTH, unconditional_annual_variance(dgp) / DAYS_PER_YEAR)
    posterior = rolling_posteriors(prior, returns, HEDGE_HISTORY_DAYS)
    vols = {
        "rolling_90": rolling_volatility(returns, HEDGE_HISTORY_DAYS, DAYS_PER_YEAR)[:, columns],
        "ewma_0.94": ewma_volatility(returns, 0.94, DAYS_PER_YEAR)[:, columns],
        "bayes_plugin": np.sqrt(DAYS_PER_YEAR * posterior_mean_variance(posterior["alpha"], posterior["beta"]))[:, columns],
    }
    if dgp["kind"] == "gbm":
        vols["true_vol"] = np.full(prices_now.shape, dgp["vol"])
    else:
        # infeasible benchmark: regime filter that knows the TRUE regime parameters
        model = true_regime_model()
        filtered = forward_filter_paths(returns, model)[:, columns]
        table = average_second_moment_by_horizon(model, maturity)
        horizon_index = (maturity - schedule[:-1]) - 1
        forecast_variance = np.einsum("psk,sk->ps", filtered, table[horizon_index])
        vols["regime_true_params"] = np.sqrt(DAYS_PER_YEAR * forecast_variance)

    output = {}
    for name, vol in vols.items():
        premium = bs_price(prices_now[:, 0], strike, times_left[0], 0.0, vol[:, 0], "call")
        output[name] = (premium, bs_delta(prices_now, strike, times_left[None, :], 0.0, vol, "call"))

    alpha, beta = posterior["alpha"][:, columns], posterior["beta"][:, columns]
    premium = bayesian_option_price(prices_now[:, 0], strike, times_left[0], 0.0, "call", alpha[:, 0], beta[:, 0],
                                    DAYS_PER_YEAR, 32)
    deltas = bayesian_option_delta(prices_now, strike, times_left[None, :], 0.0, "call", alpha, beta,
                                   DAYS_PER_YEAR, 32)
    output["bayes_integrated"] = (premium, deltas)
    return output


def hedging_experiment(seed: int) -> dict:
    rows, comparisons, scenario_meta = [], [], []
    scenario_number = 0
    for dgp_name, dgp in DGPS.items():
        for maturity in HEDGE_MATURITIES:
            scenario_number += 1
            scenario_seed = seed + 50_000 + 1000 * scenario_number
            rng = np.random.default_rng(scenario_seed)
            # one set of paths per (dgp, maturity): reused for every model, schedule and cost -> paired
            returns, _ = simulate_returns(dgp, HEDGE_NUM_PATHS, HEDGE_HISTORY_DAYS + maturity, rng)
            all_prices = prices_from_returns(SPOT, returns)
            # rescale so that every path is at 100 when the option is sold (day 0)
            future_prices = all_prices[:, HEDGE_HISTORY_DAYS:] / all_prices[:, [HEDGE_HISTORY_DAYS]] * SPOT
            strike = SPOT

            for every in HEDGE_REBALANCE_DAYS:
                schedule = np.unique(np.append(np.arange(0, maturity, every), maturity))
                ratios = hedge_ratios_for_models(dgp, returns, future_prices, schedule, maturity, strike)
                rebalance_prices = future_prices[:, schedule]
                for cost_rate in HEDGE_COST_RATES:
                    errors = {}
                    for model_name, (premium, deltas) in ratios.items():
                        result = run_delta_hedge(rebalance_prices, deltas, strike, "call", premium, 0.0,
                                                 every / DAYS_PER_YEAR, cost_rate)
                        errors[model_name] = result.hedge_error
                        rows.append({"dgp": dgp_name, "maturity_days": maturity, "rebalance_every_days": every,
                                     "rebalances": len(schedule) - 1, "cost_rate": cost_rate, "model": model_name,
                                     "mean_premium": float(np.mean(premium)),
                                     **hedging_error_summary(result.hedge_error, result.transaction_costs,
                                                             result.share_turnover)})
                    clusters = np.arange(HEDGE_NUM_PATHS)
                    pairs = [("bayes_integrated", "rolling_90"), ("bayes_integrated", "bayes_plugin"),
                             ("ewma_0.94", "rolling_90")]
                    pairs.append(("rolling_90", "true_vol") if dgp["kind"] == "gbm"
                                 else ("rolling_90", "regime_true_params"))
                    for model_a, model_b in pairs:
                        result = paired_metric_difference(errors[model_a], errors[model_b], clusters,
                                                          np.random.default_rng(scenario_seed + every),
                                                          BOOTSTRAP_REPLICATES)
                        std_diff = float(errors[model_a].std(ddof=1) - errors[model_b].std(ddof=1))
                        comparisons.append({"dgp": dgp_name, "maturity_days": maturity, "rebalance_every_days": every,
                                            "cost_rate": cost_rate, "model_a": model_a, "model_b": model_b,
                                            "std_diff": std_diff, **result})
            scenario_meta.append({"dgp": dgp_name, "maturity_days": maturity, "seed": scenario_seed,
                                  "paths": HEDGE_NUM_PATHS, "history_days": HEDGE_HISTORY_DAYS})
            print(f"  hedging scenario {dgp_name} / T={maturity} done")

    primary = {f"{c['dgp']}|T{c['maturity_days']}|every{c['rebalance_every_days']}|cost{c['cost_rate']}":
               c["mse_diff_bootstrap_p"] for c in comparisons
               if c["model_a"] == "bayes_integrated" and c["model_b"] == "rolling_90"}
    adjusted = holm_adjust(primary)
    for c in comparisons:
        if c["model_a"] == "bayes_integrated" and c["model_b"] == "rolling_90":
            c["mse_diff_holm_p"] = adjusted[f"{c['dgp']}|T{c['maturity_days']}|every{c['rebalance_every_days']}|cost{c['cost_rate']}"]

    return {
        "evidence_type": "simulated",
        "design": {
            "contract": "short 1 ATM European call (K = S0 = 100), r = 0, premium = each model's own price",
            "sign_convention": "hedge_error = hedge portfolio value at expiry - payoff; loss = -hedge_error",
            "paths_per_scenario": HEDGE_NUM_PATHS, "history_days_before_sale": HEDGE_HISTORY_DAYS,
            "maturities_days": HEDGE_MATURITIES, "rebalance_every_days": HEDGE_REBALANCE_DAYS,
            "transaction_cost_rates": HEDGE_COST_RATES,
            "cost_model": "cost_rate * |shares traded| * price, incl. first purchase and final unwind",
            "models": "rolling_90, ewma_0.94, bayes_plugin, bayes_integrated (32 nodes), plus true_vol (GBM) or "
                      "regime_true_params (regime DGP, infeasible benchmark)",
            "information": "vol re-estimated at every rebalance from returns up to that day only",
            "physical_drift": 0.05, "dgps": DGPS, "unit": "per 100 of initial spot (= % of spot)",
            "bootstrap": f"iid over paths, {BOOTSTRAP_REPLICATES} replicates", "scenarios": scenario_meta,
        },
        "rows": rows, "comparisons": comparisons,
    }


def main() -> None:
    config = load_config()
    seed = config["simulation"]["master_seed"]
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("Track A part 1: pricing")
    pricing = pricing_experiment(seed)
    pricing["experiment_id"] = config["experiment_id"]
    pricing["master_seed"] = seed
    (EXPERIMENTS_DIR / "track_a_pricing.json").write_text(json.dumps(pricing, indent=1))
    pd.DataFrame(pricing["rows"]).to_csv(EXPERIMENTS_DIR / "track_a_pricing_rows.csv", index=False)

    print("Track A part 2: hedging")
    hedging = hedging_experiment(seed)
    hedging["experiment_id"] = config["experiment_id"]
    hedging["master_seed"] = seed
    (EXPERIMENTS_DIR / "track_a_hedging.json").write_text(json.dumps(hedging, indent=1))
    pd.DataFrame(hedging["rows"]).to_csv(EXPERIMENTS_DIR / "track_a_hedging_rows.csv", index=False)
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
