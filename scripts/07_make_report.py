"""Step 7: build the charts, results_summary.md, final_research_report.md and cv_evidence.json.

Every number written here is read from the saved experiment outputs in
reports/experiments/ and data/processed/data_manifest.json. The only new
calculations are (a) the per-date series used for one chart and (b) a timing
benchmark of the pricing methods (computational cost), both rebuilt with the
frozen validation choices.

Usage:  uv run python scripts/07_make_report.py
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from volatility_lab.bayesian import make_prior
from volatility_lab.config import EXPERIMENTS_DIR, FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR, load_config
from volatility_lab.pricing_models import (
    attach_latest_fixing, bayes_integrated_prices, price_trades, regime_forecast_table, volatility_inputs_by_date,
)
from volatility_lab.regime_model import fit_regime_model
from volatility_lab.reporting import (
    MODEL_COLOURS, TEXT_SECONDARY, fmt, fmt_ci, fmt_p, load_json, markdown_table, new_figure, save_figure,
    style_axes,
)

MODEL_LABELS = {"rolling": "Rolling", "ewma": "EWMA", "bayes_plugin": "Bayes plug-in",
                "bayes_integrated": "Bayes integrated", "regime": "Regime (2-state)"}
TRACK_A_NAMES = {"rolling": "rolling", "rolling_90": "rolling", "ewma_0.94": "ewma",
                 "bayes_plugin": "bayes_plugin", "bayes_integrated": "bayes_integrated"}


# ============================================================================ loading

def load_everything():
    return {
        "manifest": load_json(PROCESSED_DIR / "data_manifest.json"),
        "math": load_json(EXPERIMENTS_DIR / "math_validation.json"),
        "a_pricing": load_json(EXPERIMENTS_DIR / "track_a_pricing.json"),
        "a_hedging": load_json(EXPERIMENTS_DIR / "track_a_hedging.json"),
        "b_pricing": load_json(EXPERIMENTS_DIR / "track_b_pricing.json"),
        "b_hedging": load_json(EXPERIMENTS_DIR / "track_b_hedging.json"),
    }


def track_a_pricing_overview(a_pricing: dict) -> pd.DataFrame:
    """RMSE / bias / MAE per (dgp, history, model), pooled over all 15 contracts (equal n per cell)."""
    rows = pd.DataFrame(a_pricing["rows"])
    rows = rows[rows["regime_group"] == "all"]
    pooled = rows.groupby(["dgp", "history_days", "model"]).apply(
        lambda g: pd.Series({"rmse": float(np.sqrt((g["rmse"] ** 2).mean())), "bias": float(g["bias"].mean()),
                             "mae": float(g["mae"].mean()), "n": int(g["n"].sum())}), include_groups=False)
    return pooled.reset_index()


def rebuild_track_b_frames(config: dict, b_pricing: dict):
    """Re-create the frozen Track B inputs (for the chart and the timing benchmark)."""
    choice = b_pricing["validation_selection"]
    days_per_year = config["volatility"]["days_per_year"]
    index_daily = pd.read_csv(PROCESSED_DIR / "index_daily.csv", parse_dates=["date"])
    returns = index_daily.dropna(subset=["log_return"]).reset_index(drop=True)[["date", "log_return"]]
    trades = pd.read_csv(PROCESSED_DIR / "option_trades_clean.csv.gz", parse_dates=["trade_time", "market_date", "expiry"])

    training_end = pd.Timestamp(config["splits"]["prior_training_end"], tz="UTC") + pd.Timedelta(hours=8)
    training_returns = returns[returns["date"] <= training_end]["log_return"].to_numpy()
    prior = make_prior(choice["chosen_prior"]["prior_strength"],
                       choice["chosen_prior"]["prior_variance_multiplier"] * np.var(training_returns, ddof=1))
    window, decay = choice["chosen_window"], choice["chosen_lambda"]
    table = volatility_inputs_by_date(returns, [window], [decay], window, prior, days_per_year)
    regime = fit_regime_model(training_returns)
    regime_table = regime_forecast_table(returns["log_return"].to_numpy(), regime, 181, days_per_year)
    table["regime_30d"] = regime_table[:, 29]
    table["row_number"] = np.arange(len(table))

    frame = attach_latest_fixing(trades, table)
    frame = frame[frame["split"].isin(["validation", "test"]) & frame[f"rolling_{window}"].notna()].reset_index(drop=True)
    horizon = np.clip(np.ceil(frame["time_to_expiry"] * days_per_year).astype(int), 1, 181)
    frame["regime_vol"] = regime_table[frame["row_number"].to_numpy().astype(int), horizon.to_numpy() - 1]
    frame["rolling_vol"] = frame[f"rolling_{window}"]
    frame["ewma_vol"] = frame[f"ewma_{decay:.2f}"]
    return frame, table, window, decay


# ============================================================================ charts

def chart_binomial(math: dict) -> str:
    rows = pd.DataFrame(math["binomial_convergence"])
    calls = rows[rows["option_type"] == "call"]
    puts = rows[rows["option_type"] == "put"]
    # put-call parity holds exactly in the tree AND in Black-Scholes, so the put error equals the call error;
    # plotting both would hide one line under the other, so we plot one and say so
    assert np.allclose(calls["abs_error"].to_numpy(), puts["abs_error"].to_numpy(), rtol=1e-6)
    figure, axes = new_figure(7.5, 4)
    axes.loglog(calls["steps"], calls["abs_error"], marker="o", markersize=5, linewidth=2,
                color=MODEL_COLOURS["rolling"], label="CRR error (call = put, by put-call parity)")
    steps = np.array([10, 5000])
    axes.loglog(steps, calls["abs_error"].iloc[0] * 10 / steps, linestyle="--", linewidth=1, color=TEXT_SECONDARY,
                label="reference slope 1/n")
    style_axes(axes, "CRR binomial price error vs Black-Scholes\n(S=K=100, T=0.5, r=3%, sigma=30%)",
               "tree steps n (log scale)", "absolute price error (log scale)")
    axes.legend(frameon=False)
    path = FIGURES_DIR / "binomial_convergence.png"
    save_figure(figure, path)
    return path.name


def chart_vol_vs_market(frame: pd.DataFrame, window: int, decay: float) -> str:
    """Per-date median market IV of near-ATM 14-45 day trades vs each model's volatility input."""
    near = frame[(frame["log_moneyness"].abs() <= 0.05) & (frame["time_to_expiry"] * 365).between(14, 45)]
    daily = near.groupby("market_date").agg(market_iv=("reported_iv", "median"), rolling=("rolling_vol", "first"),
                                            ewma=("ewma_vol", "first"), bayes_plugin=("bayes_plugin", "first"),
                                            regime=("regime_30d", "first"))
    figure, axes = new_figure(9, 5.0)
    axes.plot(daily.index, 100 * daily["market_iv"], color="#0b0b0b", linewidth=2.2,
              label="market: median reported IV (|ln K/F|<=0.05, 14-45d)")
    for name in ["rolling", "ewma", "bayes_plugin", "regime"]:
        axes.plot(daily.index, 100 * daily[name], color=MODEL_COLOURS[name], linewidth=1.6,
                  label=f"{MODEL_LABELS[name]}" + (f" ({window}d)" if name == "rolling" else
                                                   f" (lambda={decay})" if name == "ewma" else ""))
    split_line = pd.Timestamp("2024-01-01", tz="UTC")
    axes.axvline(split_line, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    axes.text(split_line, 20, "  validation | test ->", color=TEXT_SECONDARY, va="bottom")
    axes.set_ylim(bottom=18)
    style_axes(axes, "Historical-volatility inputs vs market implied volatility (Deribit BTC, sampled Wednesdays)",
               "market date", "annualised volatility (%)")
    axes.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    path = FIGURES_DIR / "track_b_vol_vs_market_iv.png"
    save_figure(figure, path)
    return path.name


def chart_track_b_rmse_by_maturity(b_pricing: dict) -> str:
    grouped = pd.DataFrame(b_pricing["results"]["test"]["grouped"])
    grouped = grouped[grouped["group"] == "maturity_bucket"]
    order = ["3-14d", "14-45d", "45-90d", "90-180d"]
    models = ["rolling", "ewma", "bayes_plugin", "bayes_integrated", "regime"]
    figure, axes = new_figure(9, 4.2)
    width = 0.16
    for i, name in enumerate(models):
        values = [grouped[(grouped["model"] == name) & (grouped["value"] == bucket)]["rmse"].iloc[0] for bucket in order]
        axes.bar(np.arange(len(order)) + (i - 2) * width, values, width=width - 0.02, color=MODEL_COLOURS[name],
                 label=MODEL_LABELS[name])
    axes.set_xticks(np.arange(len(order)), order)
    style_axes(axes, "Track B test period: pricing RMSE by maturity (historical trades, 2024-01 to 2026-09)",
               "days to expiry", "RMSE (% of index price)")
    axes.legend(frameon=False, fontsize=8, ncol=3)
    path = FIGURES_DIR / "track_b_rmse_by_maturity.png"
    save_figure(figure, path)
    return path.name


def chart_track_a_hedging(a_hedging: dict) -> str:
    rows = pd.DataFrame(a_hedging["rows"])
    rows = rows[(rows["maturity_days"] == 30) & (rows["rebalance_every_days"] == 1) & (rows["cost_rate"] == 0.0)]
    figure, axes = new_figure(9, 4.8)
    dgps = ["gbm_vol20", "gbm_vol60", "regime_40_100"]
    models = ["rolling_90", "ewma_0.94", "bayes_plugin", "bayes_integrated"]
    width = 0.17
    for i, name in enumerate(models):
        values = [rows[(rows["dgp"] == dgp) & (rows["model"] == name)]["rmse"].iloc[0] for dgp in dgps]
        key = TRACK_A_NAMES[name]
        axes.bar(np.arange(3) + (i - 1.5) * width, values, width=width - 0.02, color=MODEL_COLOURS[key],
                 label=MODEL_LABELS[key])
    # benchmarks that know the truth: drawn as hollow markers so they never look like a 5th series colour
    benchmark = [rows[(rows["dgp"] == dgp) & (rows["model"].isin(["true_vol", "regime_true_params"]))]["rmse"].iloc[0]
                 for dgp in dgps]
    axes.scatter(np.arange(3), benchmark, marker="D", s=60, facecolor="none", edgecolor="#0b0b0b", linewidth=1.5,
                 zorder=5, label="benchmark with true parameters")
    axes.set_xticks(np.arange(3), ["GBM sigma=20%", "GBM sigma=60%", "Regime 40%/100%"])
    style_axes(axes, "Track A (simulated): RMSE of hedge error, 30-day ATM call, daily rebalancing, no costs",
               "data-generating process", "RMSE of hedge error (% of spot)")
    axes.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    path = FIGURES_DIR / "track_a_hedging_rmse.png"
    save_figure(figure, path)
    return path.name


# ============================================================================ timing

def timing_benchmark(frame: pd.DataFrame, nodes: int) -> dict:
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    timings = {}
    for name, function in [("plug-in Black-76 (any plug-in vol)", lambda: price_trades(test, "rolling_vol")),
                           (f"Bayes integrated ({nodes} nodes)", lambda: bayes_integrated_prices(test, nodes))]:
        best = np.inf
        for _ in range(3):
            started = time.perf_counter()
            function()
            best = min(best, time.perf_counter() - started)
        timings[name] = {"seconds": best, "trades": int(len(test)), "microseconds_per_trade": 1e6 * best / len(test)}
    return timings


# ============================================================================ markdown sections

def sample_size_section(data: dict) -> str:
    manifest, b, bh = data["manifest"], data["b_pricing"], data["b_hedging"]
    trades = manifest["option_trades"]
    lines = ["## 1. Historical-data sample sizes (real Deribit data)", ""]
    lines.append(markdown_table(["Quantity", "Value"], [
        ["Option-trade windows downloaded (08:00-09:00 UTC, Wednesdays)", trades["sample_windows"]],
        ["Raw trade rows (incl. page overlaps)", f"{trades['raw_rows']:,}"],
        ["Unique raw trade ids", f"{trades['raw_unique_trade_ids']:,}"],
        ["Distinct instruments in raw data", f"{trades['raw_instruments']:,}"],
        ["Valid trades after cleaning", f"{trades['clean_rows']:,}"],
        ["Distinct instruments after cleaning", f"{trades['clean_instruments']:,}"],
        ["Market dates after cleaning", trades["clean_market_dates"]],
        ["Clean-trade coverage", f"{trades['first_trade'][:10]} to {trades['last_trade'][:10]}"],
        ["Calls / puts after cleaning", f"{trades['calls']:,} / {trades['puts']:,}"],
        ["Index fixings (btc_usd)", f"{manifest['index']['rows']:,} ({manifest['index']['first_fixing'][:10]} to "
                                    f"{manifest['index']['last_fixing'][:10]})"],
        ["Daily index returns (daily history from " + manifest["index"]["daily_history_starts"][:10] + ")",
         f"{manifest['index']['daily_returns']:,}"],
        ["Returns used to fit prior and regime model (to 2021-12-31)", b["training"]["returns_used"]],
    ]))
    lines += ["", "**Cleaning exclusions** (in order):", ""]
    lines.append(markdown_table(["Rule", "Removed", "Remaining"],
                                [[s["step"], f"{s['removed']:,}", f"{s['remaining']:,}"] for s in trades["exclusions"]]))
    excl = b["exclusions_before_evaluation"]
    lines += ["", f"Before scoring, a further {excl['missing_index_fixing_or_insufficient_history']:,} trades had no "
              f"fixing within a day or too little history for the 180-day window, and "
              f"{excl['development_period_not_evaluated']:,} development-period trades were not scored "
              "(the two groups overlap).", ""]
    split_rows = []
    for name in ["validation", "test"]:
        r = b["results"][name]
        split_rows.append([name, f"{r['n_trades']:,}", r["n_market_dates"], f"{r['n_instruments']:,}",
                           f"{r['first_trade'][:10]} to {r['last_trade'][:10]}"])
    dev = trades["splits"]["development"]
    split_rows.insert(0, ["development (not scored)", f"{dev['trades']:,}", dev["market_dates"],
                          f"{dev['instruments']:,}", f"{dev['first'][:10]} to {dev['last'][:10]}"])
    lines.append(markdown_table(["Split", "Trades", "Market dates", "Instruments", "Coverage"], split_rows))
    lines += ["", f"Every primary pricing comparison is paired on the same **{b['results']['test']['n_trades']:,} test "
              f"trades on {b['results']['test']['n_market_dates']} market dates**.",
              f"Hypothetical historical hedges: **{bh['n_hedges']}** 30-day hedges started weekly from "
              f"{bh['first_start'][:10]} to {bh['last_start'][:10]} (last expiry {bh['last_expiry'][:10]}), "
              "each with 30 daily rebalances, on the observed index path.", ""]
    return "\n".join(lines)


def simulation_size_section(data: dict) -> str:
    math, ap, ah = data["math"], data["a_pricing"], data["a_hedging"]
    design = ap["design"]
    lines = ["## 2. Simulation sample sizes (Track A - simulated, NOT market data)", ""]
    coverage = math["monte_carlo_coverage"][0]
    lines.append(markdown_table(["Experiment", "Size", "Seeds"], [
        ["MC method comparison", f"{math['monte_carlo_comparison'][0]['normal_draws']:,} normal draws per method",
         math["seeds"]["monte_carlo_comparison"]],
        ["MC 95% CI coverage", f"{coverage['repeats']:,} repeats x {coverage['normal_draws_per_estimate']:,} draws, "
                               "3 methods", math["seeds"]["monte_carlo_coverage"]],
        ["Implied-vol recovery grid", f"{math['implied_vol_recovery']['cases']:,} priced contracts", "deterministic"],
        ["Greek check", f"{math['greek_check']['random_cases']} random contracts", math["seeds"]["greek_check"]],
        ["Pricing under known DGP", f"{len(design['scenarios'])} scenarios (3 DGPs x 2 history lengths) x "
                                    f"{design['simulated_histories_per_scenario']:,} simulated histories x "
                                    f"{design['contracts_per_history']} contracts (3 maturities x 5 strikes) = "
                                    f"{len(design['scenarios']) * design['simulated_histories_per_scenario'] * design['contracts_per_history']:,} "
                                    "priced contracts per model",
         ", ".join(str(s["seed"]) for s in design["scenarios"])],
        ["True regime-DGP prices", "400,000 regime paths per (maturity, state)", "scenario seed + maturity"],
        ["Hedging under known DGP", f"{len(ah['design']['scenarios'])} path sets (3 DGPs x 2 maturities) x "
                                    f"{ah['design']['paths_per_scenario']:,} paths (each with "
                                    f"{ah['design']['history_days_before_sale']} days of history); "
                                    "rebalancing every 1 day (30 or 90 rebalances) or 7 days (5 or 13); "
                                    "costs 0 or 10 bp = 24 paired scenarios per model",
         ", ".join(str(s["seed"]) for s in ah["design"]["scenarios"])],
    ]))
    lines += ["", f"Master seed: `{ap['master_seed']}`. Scenario parameters: GBM sigma = 20% and 60%; regime "
              "vols 40%/100% with transition matrix [[0.98, 0.02], [0.05, 0.95]]; physical drift 5%; r = 0.", ""]
    return "\n".join(lines)


def math_section(math: dict) -> str:
    lines = ["## 3. Numerical validation of the pricing engine (analytic / simulated)", "",
             f"Test gate: `{math['tests']['summary']}`.", ""]
    conv = pd.DataFrame(math["binomial_convergence"])
    lines.append(markdown_table(["CRR steps", "call abs error", "put abs error"],
                                [[s, f"{conv[(conv.steps == s) & (conv.option_type == 'call')].abs_error.iloc[0]:.2e}",
                                  f"{conv[(conv.steps == s) & (conv.option_type == 'put')].abs_error.iloc[0]:.2e}"]
                                 for s in [10, 100, 1000, 5000]]))
    lines += ["", markdown_table(["MC method", "estimate", "SE", "95% CI", "error vs BS", "variance reduction",
                                  "runtime (s)"],
                                 [[r["method"], fmt(r["estimate"], 4), fmt(r["standard_error"], 4),
                                   fmt_ci(r["ci_low"], r["ci_high"], 4), fmt(r["error"], 4),
                                   f"{r['variance_reduction_factor']:.2f}x", fmt(r["runtime_seconds"], 4)]
                                  for r in math["monte_carlo_comparison"]]), ""]
    lines.append(markdown_table(["MC method", "95% CI coverage", "binomial SE", "sd of estimates", "mean reported SE"],
                                [[r["method"], fmt(r["coverage_95"]), fmt(r["coverage_binomial_se"], 4),
                                  fmt(r["empirical_sd_of_estimates"], 4), fmt(r["mean_reported_se"], 4)]
                                 for r in math["monte_carlo_coverage"]]))
    iv = math["implied_vol_recovery"]
    lines += ["", f"Implied-vol recovery: {iv['status_counts'].get('ok', 0):,} of {iv['cases']:,} cases inverted, max "
              f"absolute vol error {iv['max_abs_vol_error_when_ok']:.1e}. Non-invertible cases by reason: "
              + ", ".join(f"{k} {v}" for k, v in iv["status_counts"].items() if k != "ok") + ".",
              f"Greeks, analytic vs finite difference over {math['greek_check']['random_cases']} random contracts, max "
              "absolute difference: " + ", ".join(f"{k} {v:.1e}" for k, v in math["greek_check"]["max_abs_difference"].items())
              + ".", ""]
    return "\n".join(lines)


def track_a_section(data: dict) -> str:
    ap, ah = data["a_pricing"], data["a_hedging"]
    overview = track_a_pricing_overview(ap)
    lines = ["## 4. Track A results (SIMULATED)", "",
             "### 4a. Pricing error vs the true price (unit: % of spot; error = model - true; pooled over 3 "
             "maturities x 5 strikes)", ""]
    table_rows = []
    for (dgp, history), part in overview.groupby(["dgp", "history_days"]):
        cells = [dgp, history]
        for model in ["rolling", "ewma_0.94", "bayes_plugin", "bayes_integrated"]:
            row = part[part["model"] == model].iloc[0]
            cells.append(f"{fmt(row['rmse'], 4)} ({row['bias']:+.4f})")
        table_rows.append(cells)
    lines.append(markdown_table(["DGP", "history (days)", "rolling RMSE (bias)", "EWMA 0.94", "Bayes plug-in",
                                 "Bayes integrated"], table_rows))
    lines += ["", "**Primary comparison: Bayes integrated - rolling** (paired on identical histories; cluster "
              "bootstrap over histories, 2,000 replicates; Holm across all 18 cells):", ""]
    comparisons = pd.DataFrame(ap["comparisons"])
    primary = comparisons[(comparisons.model_a == "bayes_integrated") & (comparisons.model_b == "rolling")]
    rolling_rows = pd.DataFrame(ap["rows"])
    rolling_rows = rolling_rows[(rolling_rows.regime_group == "all") & (rolling_rows.model == "rolling")]
    rows = []
    for _, c in primary.iterrows():
        base = rolling_rows[(rolling_rows.dgp == c.dgp) & (rolling_rows.history_days == c.history_days)
                            & (rolling_rows.maturity_days == c.maturity_days)]
        base_rmse = float(np.sqrt((base.rmse ** 2).mean()))
        rows.append([c.dgp, c.history_days, c.maturity_days, fmt(c.rmse_diff, 4),
                     fmt_ci(c.rmse_diff_ci_low, c.rmse_diff_ci_high, 4), f"{100 * c.rmse_diff / base_rmse:+.1f}%",
                     fmt(c.mae_diff, 4), fmt_p(c.mse_diff_holm_p, c.bootstrap_replicates)])
    lines.append(markdown_table(["DGP", "history", "maturity", "RMSE diff", "95% CI", "relative to rolling RMSE",
                                 "MAE diff", "Holm p (MSE)"], rows))

    integration = comparisons[(comparisons.model_a == "bayes_integrated") & (comparisons.model_b == "bayes_plugin")]
    lines += ["", f"Integration effect (Bayes integrated - Bayes plug-in, same posterior): RMSE difference between "
              f"{integration.rmse_diff.min():.4f} and {integration.rmse_diff.max():.4f} across the 18 cells; "
              f"negative (integration better) in {(integration.rmse_diff_ci_high < 0).sum()} of 18 cells with the "
              "95% CI excluding zero.", ""]

    # moneyness detail for one scenario
    detail = pd.DataFrame(ap["rows"])
    detail = detail[(detail.regime_group == "all") & (detail.dgp == "gbm_vol60") & (detail.history_days == 30)
                    & (detail.maturity_days == 30)]
    lines += ["Bias by strike, GBM sigma = 60%, 30-day history, 30-day options (% of spot):", ""]
    lines.append(markdown_table(["K/S", "rolling", "EWMA", "Bayes plug-in", "Bayes integrated"],
                                [[m] + [f"{detail[(detail.moneyness == m) & (detail.model == mod)].bias.iloc[0]:+.4f}"
                                        for mod in ["rolling", "ewma_0.94", "bayes_plugin", "bayes_integrated"]]
                                 for m in sorted(detail.moneyness.unique())]))

    sens = pd.DataFrame(ap["prior_sensitivity"])
    lines += ["", "**Prior sensitivity** (Bayes integrated, 30-day options, RMSE % of spot; centre = multiple of the "
              "TRUE variance, so 0.5 and 2 are deliberately wrong priors):", ""]
    sens_rows = []
    for (dgp, history), part in sens.groupby(["dgp", "history_days"]):
        base = rolling_rows[(rolling_rows.dgp == dgp) & (rolling_rows.history_days == history)
                            & (rolling_rows.maturity_days == 30)]
        base_rmse = float(np.sqrt((base.rmse ** 2).mean()))
        for strength, s_part in part.groupby("prior_strength"):
            sens_rows.append([dgp, history, strength] + [fmt(s_part[s_part.prior_variance_multiplier == m].rmse.iloc[0], 4)
                                                         for m in [0.5, 1.0, 2.0]] + [fmt(base_rmse, 4)])
    lines.append(markdown_table(["DGP", "history", "prior strength", "centre 0.5x", "centre 1x", "centre 2x",
                                 "rolling (reference)"], sens_rows))

    lines += ["", "### 4b. Hedging (short ATM call at the model's own price; hedge error = hedge value - payoff; "
              "% of spot)", ""]
    hedge_rows = pd.DataFrame(ah["rows"])
    rows = []
    for (dgp, maturity, every, cost), part in hedge_rows.groupby(["dgp", "maturity_days", "rebalance_every_days", "cost_rate"]):
        get = lambda model, col: part[part.model == model][col].iloc[0] if (part.model == model).any() else np.nan  # noqa: E731
        bench = "true_vol" if (part.model == "true_vol").any() else "regime_true_params"
        rows.append([dgp, maturity, every, f"{cost * 1e4:.0f} bp"]
                    + [fmt(get(m, "rmse")) for m in ["rolling_90", "ewma_0.94", "bayes_plugin", "bayes_integrated", bench]]
                    + [fmt(get("bayes_integrated", "es95_loss")), fmt(get("rolling_90", "es95_loss"))])
    lines.append(markdown_table(["DGP", "T (d)", "rebalance (d)", "cost", "rolling RMSE", "EWMA", "Bayes plug-in",
                                 "Bayes integrated", "true-parameter benchmark", "ES95 loss Bayes int.",
                                 "ES95 loss rolling"], rows))
    hedge_comparisons = pd.DataFrame(ah["comparisons"])
    primary = hedge_comparisons[(hedge_comparisons.model_a == "bayes_integrated") & (hedge_comparisons.model_b == "rolling_90")]
    lines += ["", "**Primary comparison: Bayes integrated - rolling (RMSE of hedge error, % of spot)**, paired on "
              "identical paths, 10,000 paths, iid bootstrap 2,000 replicates, Holm across 24 scenarios:", ""]
    lines.append(markdown_table(["DGP", "T", "rebalance", "cost", "RMSE diff", "95% CI", "sd diff", "Holm p"],
                                [[c.dgp, c.maturity_days, c.rebalance_every_days, f"{c.cost_rate * 1e4:.0f} bp",
                                  fmt(c.rmse_diff, 4), fmt_ci(c.rmse_diff_ci_low, c.rmse_diff_ci_high, 4),
                                  fmt(c.std_diff, 4), fmt_p(c.mse_diff_holm_p, c.bootstrap_replicates)]
                                 for _, c in primary.iterrows()]))
    lines.append("")
    return "\n".join(lines)


def track_b_section(data: dict, timings: dict, figures: dict) -> str:
    b, bh = data["b_pricing"], data["b_hedging"]
    test = b["results"]["test"]
    choice = b["validation_selection"]
    lines = ["## 5. Track B results (HISTORICAL Deribit BTC option trades)", "",
             f"Target: **observed trade price** (not a quote midpoint, not fair value). Error = model - trade price. "
             f"Choices frozen on validation (2022-2023): rolling window **{choice['chosen_window']} days**, EWMA "
             f"lambda **{choice['chosen_lambda']}**, prior strength **{choice['chosen_prior']['prior_strength']:g}** "
             f"pseudo-observations centred on **{choice['chosen_prior']['prior_variance_multiplier']:g}x** the "
             f"training variance (training annualised vol {100 * b['training']['training_annual_vol']:.1f}%).", ""]
    lines.append(markdown_table(["Model (test period)", "RMSE (% spot)", "MAE (% spot)", "median AE", "bias (% spot)",
                                 "MAE (USD)", "mean |rel. error|", "vol-point bias", "vol-point MAE"],
                                [[MODEL_LABELS[m], fmt(s["pct_of_spot"]["rmse"], 4), fmt(s["pct_of_spot"]["mae"], 4),
                                  fmt(s["pct_of_spot"]["median_ae"], 4), f"{s['pct_of_spot']['bias']:+.4f}",
                                  f"{s['usd']['mae']:,.0f}", fmt(s["usd"]["mean_abs_relative_error"]),
                                  f"{s['vol_points_mean']:+.2f}", fmt(s["vol_points_mae"], 2)]
                                 for m, s in test["summary"].items()]))
    lines += ["", f"n = {test['n_trades']:,} trades, {test['n_market_dates']} market dates, "
              f"{test['n_instruments']:,} instruments. Mean reported IV {100 * test['mean_reported_iv']:.1f}% vs mean "
              "model vol: " + ", ".join(f"{MODEL_LABELS[k]} {100 * v:.1f}%" for k, v in test["mean_model_vol"].items()) + ".",
              "", "**Paired comparisons on the test period** (difference = A - B; negative = A more accurate; "
              "date-clustered bootstrap, 5,000 replicates; Diebold-Mariano on per-date mean squared-error "
              "differences as a cross-check):", ""]
    lines.append(markdown_table(["Family", "A", "B", "RMSE diff", "95% CI", "MAE diff", "95% CI", "bootstrap p (MSE)",
                                 "Holm p", "DM stat", "DM p"],
                                [[c["family"], MODEL_LABELS[c["model_a"]], MODEL_LABELS[c["model_b"]],
                                  fmt(c["rmse_diff"], 4), fmt_ci(c["rmse_diff_ci_low"], c["rmse_diff_ci_high"], 4),
                                  fmt(c["mae_diff"], 4), fmt_ci(c["mae_diff_ci_low"], c["mae_diff_ci_high"], 4),
                                  fmt_p(c["mse_diff_bootstrap_p"], c["bootstrap_replicates"]),
                                  fmt_p(c["mse_diff_holm_p"]) if "mse_diff_holm_p" in c else "-",
                                  fmt(c["dm_squared_error"]["dm_stat"], 2), fmt_p(c["dm_squared_error"]["p_value"])]
                                 for c in test["comparisons"]]))
    grouped = pd.DataFrame(test["grouped"])
    lines += ["", "RMSE by group (% of spot, test period):", ""]
    group_rows = []
    for (group, value), part in grouped.groupby(["group", "value"], sort=False):
        group_rows.append([group, value, int(part.n.iloc[0])] + [fmt(part[part.model == m].rmse.iloc[0])
                                                                 for m in MODEL_LABELS])
    lines.append(markdown_table(["group", "value", "n"] + list(MODEL_LABELS.values()), group_rows))

    prior_grid = pd.DataFrame(choice["prior_grid"])
    lines += ["", "**Prior sensitivity on the validation period** (RMSE % of spot, Bayes integrated / plug-in):", ""]
    lines.append(markdown_table(["prior strength", "centre 0.5x", "centre 1x", "centre 2x"],
                                [[s] + [f"{p[p.prior_variance_multiplier == m].validation_rmse_integrated.iloc[0]:.4f} / "
                                        f"{p[p.prior_variance_multiplier == m].validation_rmse_plugin.iloc[0]:.4f}"
                                        for m in [0.5, 1.0, 2.0]]
                                 for s, p in prior_grid.groupby("prior_strength")]))
    lines += ["", "Validation RMSE by rolling window: " + ", ".join(f"{k}d {v:.4f}" for k, v in
                                                                   choice["rolling_rmse_by_window"].items())
              + "; by EWMA lambda: " + ", ".join(f"{k} {v:.4f}" for k, v in choice["ewma_rmse_by_lambda"].items()) + ".", ""]

    forecast = b["volatility_forecast_evaluation"]
    lines += [f"**Does the regime model forecast realised volatility?** Test-period 30-day realised-vol forecasts, "
              f"{forecast['n_forecast_dates']:,} daily forecast dates, QLIKE loss (lower is better): "
              + ", ".join(f"{MODEL_LABELS[k]} {v:.4f}" for k, v in forecast["mean_qlike"].items()) + ".", ""]
    lines.append(markdown_table(["comparison", "mean QLIKE diff", "DM stat (NW lag 30)", "p"],
                                [[k, fmt(v["mean_differential"], 4), fmt(v["dm_stat"], 2), fmt_p(v["p_value"])]
                                 for k, v in forecast["comparisons"].items()]))
    regime = b["training"]["regime_model"]
    lines += ["", f"Fitted regime model (training data 2020-02 to 2021-12): calm {100 * regime['annual_vols'][0]:.0f}% / "
              f"turbulent {100 * regime['annual_vols'][1]:.0f}% annualised; P(stay calm) "
              f"{regime['transition'][0][0]:.3f}, P(stay turbulent) {regime['transition'][1][1]:.3f}.", ""]

    lines += ["### 5b. Hypothetical delta hedges on the observed BTC index path (NOT executed trades)", "",
              f"{bh['n_hedges']} hedges; {bh['design']['contract']}; premium = each model's own price; unit = % of the "
              "index at the sale; bootstrap over blocks of 5 consecutive starts (29 blocks).", ""]
    hedge_rows = []
    for result in bh["results"].values():
        for model, s in result["summaries"].items():
            hedge_rows.append([result["rebalance_every_days"], f"{result['cost_rate'] * 1e4:.0f} bp", MODEL_LABELS[model],
                               fmt(s["mean_premium_pct"]), f"{s['mean']:+.3f}", fmt(s["median"]), fmt(s["std"]),
                               fmt(s["rmse"]), fmt(s["q05"]), fmt(s["var95_loss"]), fmt(s["es95_loss"]),
                               fmt(s["mean_transaction_cost"]), fmt(s["mean_share_turnover"], 2)])
    lines.append(markdown_table(["rebalance (d)", "cost", "model", "premium", "mean error", "median", "sd", "RMSE",
                                 "5% quantile", "VaR95 loss", "ES95 loss", "mean cost", "share turnover"], hedge_rows))
    lines += ["", "Paired comparisons (A - B, RMSE of hedge error):", ""]
    comp_rows = []
    for result in bh["results"].values():
        for c in result["comparisons"]:
            comp_rows.append([result["rebalance_every_days"], f"{result['cost_rate'] * 1e4:.0f} bp",
                              MODEL_LABELS[c["model_a"]], MODEL_LABELS[c["model_b"]], fmt(c["rmse_diff"]),
                              fmt_ci(c["rmse_diff_ci_low"], c["rmse_diff_ci_high"]), f"{c['bias_diff']:+.3f}",
                              f"{c['std_diff']:+.3f}", fmt_p(c["mse_diff_bootstrap_p"], c["bootstrap_replicates"]),
                              fmt_p(c["mse_diff_holm_p"]) if "mse_diff_holm_p" in c else "-"])
    lines.append(markdown_table(["rebalance", "cost", "A", "B", "RMSE diff", "95% CI", "mean diff", "sd diff",
                                 "bootstrap p", "Holm p"], comp_rows))

    lines += ["", "### 5c. Computational cost (test-period trades, best of 3 runs on this machine)", ""]
    lines.append(markdown_table(["method", "trades", "seconds", "microseconds per trade"],
                                [[k, f"{v['trades']:,}", fmt(v["seconds"], 3), fmt(v["microseconds_per_trade"], 2)]
                                 for k, v in timings.items()]))
    lines += ["", f"![vol vs market]({'figures/' + figures['vol']})", "",
              f"![rmse by maturity]({'figures/' + figures['maturity']})", ""]
    return "\n".join(lines)


def best_methods_section(data: dict, timings: dict) -> tuple[str, dict]:
    b, bh, ah = data["b_pricing"], data["b_hedging"], data["a_hedging"]
    test = b["results"]["test"]["summary"]
    best_pricing = min(test, key=lambda m: test[m]["pct_of_spot"]["rmse"])
    daily = bh["results"]["every1_cost0.0005"]["summaries"]
    best_hedge_rmse = min(daily, key=lambda m: daily[m]["rmse"])
    best_hedge_sd = min(daily, key=lambda m: daily[m]["std"])
    best_hedge_es = min(daily, key=lambda m: daily[m]["es95_loss"])
    fastest = min(timings, key=lambda k: timings[k]["seconds"])
    a_rows = pd.DataFrame(ah["rows"])
    a_rows = a_rows[~a_rows.model.isin(["true_vol", "regime_true_params"])]
    a_wins = a_rows.loc[a_rows.groupby(["dgp", "maturity_days", "rebalance_every_days", "cost_rate"]).rmse.idxmin()]
    bayes = a_rows[a_rows.model == "bayes_integrated"].set_index(["dgp", "maturity_days", "rebalance_every_days", "cost_rate"])
    rolling = a_rows[a_rows.model == "rolling_90"].set_index(["dgp", "maturity_days", "rebalance_every_days", "cost_rate"])
    max_gap = float(((rolling.rmse - bayes.rmse).abs() / rolling.rmse).max())
    best = {"max_relative_bayes_vs_rolling_hedge_gap": max_gap, "historical_pricing_rmse": best_pricing, "historical_hedge_rmse_daily_5bp": best_hedge_rmse,
            "historical_hedge_sd_daily_5bp": best_hedge_sd, "historical_hedge_es95_daily_5bp": best_hedge_es,
            "fastest": fastest, "simulated_hedge_rmse_wins": a_wins.model.value_counts().to_dict()}
    lines = ["## 6. Best method per objective", "", markdown_table(["Objective", "Best (point estimate)", "Caveat"], [
        ["Historical pricing RMSE (test)", MODEL_LABELS[best_pricing],
         "not distinguishable from EWMA (see primary comparison 2)"],
        ["Historical hedge RMSE (daily, 5 bp)", MODEL_LABELS[best_hedge_rmse], "143 overlapping hedges, wide intervals"],
        ["Historical hedge error sd (daily, 5 bp)", MODEL_LABELS[best_hedge_sd], "sd isolates hedge-ratio quality"],
        ["Historical ES95 of hedge loss (daily, 5 bp)", MODEL_LABELS[best_hedge_es],
         "driven by the higher premium the model charges, not better hedging"],
        ["Simulated hedge RMSE (24 scenarios)", ", ".join(f"{k}: {v}" for k, v in best["simulated_hedge_rmse_wins"].items()),
         f"excluding true-parameter benchmarks; Bayes-vs-rolling gaps are at most "
         f"{100 * best['max_relative_bayes_vs_rolling_hedge_gap']:.1f}% of rolling RMSE"],
        ["Computation", fastest, "Bayes integrated costs ~" +
         f"{timings[[k for k in timings if k != fastest][0]]['seconds'] / timings[fastest]['seconds']:.0f}x more"],
    ]), ""]
    return "\n".join(lines), best


# ============================================================================ CV evidence

def cv_evidence(data: dict, best: dict, timings: dict, config: dict) -> dict:
    b, ap, ah, math, manifest = data["b_pricing"], data["a_pricing"], data["a_hedging"], data["math"], data["manifest"]
    test = b["results"]["test"]
    comparisons = {f"{c['model_a']} vs {c['model_b']}": c for c in test["comparisons"]}
    vs_rolling = comparisons["bayes_integrated vs rolling"]
    vs_ewma = comparisons["bayes_integrated vs ewma"]
    vs_plugin = comparisons["bayes_integrated vs bayes_plugin"]
    plugin_vs_rolling = comparisons["bayes_plugin vs rolling"]
    a_comp = pd.DataFrame(ap["comparisons"])
    a_primary = a_comp[(a_comp.model_a == "bayes_integrated") & (a_comp.model_b == "rolling")]
    h_comp = pd.DataFrame(ah["comparisons"])
    h_primary = h_comp[(h_comp.model_a == "bayes_integrated") & (h_comp.model_b == "rolling_90")]
    coverage = {r["method"]: r["coverage_95"] for r in math["monte_carlo_coverage"]}
    vrf = {r["method"]: r["variance_reduction_factor"] for r in math["monte_carlo_comparison"]}
    rmse = {m: s["pct_of_spot"]["rmse"] for m, s in test["summary"].items()}

    # simulated pricing: does the gain survive a WRONG prior centre? compare with rolling at the same cell
    rows = pd.DataFrame(ap["rows"])
    rolling_30 = rows[(rows.regime_group == "all") & (rows.model == "rolling") & (rows.maturity_days == 30)]
    wrong_prior = []
    for item in ap["prior_sensitivity"]:
        if item["prior_variance_multiplier"] == 1.0:
            continue
        base = rolling_30[(rolling_30.dgp == item["dgp"]) & (rolling_30.history_days == item["history_days"])]
        base_rmse = float(np.sqrt((base.rmse ** 2).mean()))
        wrong_prior.append({"dgp": item["dgp"], "history_days": item["history_days"],
                            "prior_strength": item["prior_strength"],
                            "prior_variance_multiplier": item["prior_variance_multiplier"],
                            "bayes_rmse": item["rmse"], "rolling_rmse": base_rmse,
                            "bayes_better": bool(item["rmse"] < base_rmse)})
    weak_wrong = [w for w in wrong_prior if w["prior_strength"] == 2.0]
    true_gap = h_comp[(h_comp.model_a == "rolling_90") & (h_comp.model_b == "true_vol")]
    regime_gap = h_comp[(h_comp.model_a == "rolling_90") & (h_comp.model_b == "regime_true_params")]
    hedge_daily = data["b_hedging"]["results"]["every1_cost0.0005"]["summaries"]

    claims = [
        {"claim_id": "H1_test_rmse_by_model", "evidence_type": "historical",
         "description": "Test-period pricing RMSE (% of index price) of BTC option trade prices",
         "values": rmse, "unit": "% of index price", "n_trades": test["n_trades"], "n_market_dates": test["n_market_dates"],
         "vol_point_bias": {m: s["vol_points_mean"] for m, s in test["summary"].items()},
         "vol_point_mae": {m: s["vol_points_mae"] for m, s in test["summary"].items()},
         "n_instruments": test["n_instruments"], "test_period": [test["first_trade"], test["last_trade"]],
         "source": "reports/experiments/track_b_pricing.json:results.test.summary"},
        {"claim_id": "H2_bayes_vs_rolling", "evidence_type": "historical",
         "description": "Bayes integrated minus rolling, RMSE difference, date-clustered bootstrap",
         "rmse_diff": vs_rolling["rmse_diff"], "ci95": [vs_rolling["rmse_diff_ci_low"], vs_rolling["rmse_diff_ci_high"]],
         "relative_change": vs_rolling["rmse_diff"] / rmse["rolling"],
         "holm_p": vs_rolling["mse_diff_holm_p"], "dm_p": vs_rolling["dm_squared_error"]["p_value"],
         "n_trades": vs_rolling["n_observations"], "n_clusters": vs_rolling["n_clusters"],
         "share_explained_by_prior_shrinkage": plugin_vs_rolling["rmse_diff"] / vs_rolling["rmse_diff"],
         "source": "reports/experiments/track_b_pricing.json:results.test.comparisons[0]"},
        {"claim_id": "H3_bayes_vs_ewma", "evidence_type": "historical",
         "description": "Bayes integrated minus EWMA, RMSE difference (null result)",
         "rmse_diff": vs_ewma["rmse_diff"], "ci95": [vs_ewma["rmse_diff_ci_low"], vs_ewma["rmse_diff_ci_high"]],
         "holm_p": vs_ewma["mse_diff_holm_p"], "n_trades": vs_ewma["n_observations"],
         "source": "reports/experiments/track_b_pricing.json:results.test.comparisons[1]"},
        {"claim_id": "H4_integration_effect", "evidence_type": "historical",
         "description": "Bayes integrated minus Bayes plug-in (effect of integrating over the posterior)",
         "rmse_diff": vs_plugin["rmse_diff"], "ci95": [vs_plugin["rmse_diff_ci_low"], vs_plugin["rmse_diff_ci_high"]],
         "holm_p": vs_plugin["mse_diff_holm_p"], "dm_p": vs_plugin["dm_squared_error"]["p_value"],
         "n_trades": vs_plugin["n_observations"],
         "source": "reports/experiments/track_b_pricing.json:results.test.comparisons[2]"},
        {"claim_id": "H5_historical_hedging", "evidence_type": "hypothetical hedge on historical paths",
         "description": "Daily-rebalanced 30-day ATM call hedges, 5 bp costs: RMSE of hedge error by model",
         "values": {m: s["rmse"] for m, s in hedge_daily.items()},
         "mean_error": {m: s["mean"] for m, s in hedge_daily.items()},
         "sd_error": {m: s["std"] for m, s in hedge_daily.items()},
         "es95_loss": {m: s["es95_loss"] for m, s in hedge_daily.items()},
         "mean_premium": {m: s["mean_premium_pct"] for m, s in hedge_daily.items()},
         "unit": "% of index at sale", "n_hedges": data["b_hedging"]["n_hedges"],
         "source": "reports/experiments/track_b_hedging.json:results.every1_cost0.0005"},
        {"claim_id": "S1_simulated_pricing", "evidence_type": "simulated",
         "description": "Bayes integrated minus rolling RMSE vs the true price, 18 scenario-maturity cells",
         "cells": int(len(a_primary)), "cells_with_ci_below_zero": int((a_primary.rmse_diff_ci_high < 0).sum()),
         "rmse_diff_range": [float(a_primary.rmse_diff.min()), float(a_primary.rmse_diff.max())],
         "max_holm_p": float(a_primary.mse_diff_holm_p.max()),
         "bootstrap_replicates": int(a_primary.bootstrap_replicates.iloc[0]),
         "mean_rmse_diff_by_history_days": {str(k): float(v) for k, v in
                                            a_primary.groupby("history_days").rmse_diff.mean().items()},
         "main_prior": "strength 2 centred on the TRUE unconditional variance",
         "wrong_prior_check_30d": wrong_prior,
         "weak_wrong_prior_cells_better_than_rolling": [sum(w["bayes_better"] for w in weak_wrong), len(weak_wrong)],
         "all_wrong_prior_cells_better_than_rolling": [sum(w["bayes_better"] for w in wrong_prior), len(wrong_prior)],
         "histories_per_scenario": ap["design"]["simulated_histories_per_scenario"],
         "contracts_per_history": ap["design"]["contracts_per_history"], "master_seed": ap["master_seed"],
         "source": "reports/experiments/track_a_pricing.json:comparisons"},
        {"claim_id": "S2_simulated_hedging", "evidence_type": "simulated",
         "description": "Bayes integrated minus rolling RMSE of hedge error, 24 paired scenarios",
         "scenarios": int(len(h_primary)), "scenarios_with_ci_below_zero": int((h_primary.rmse_diff_ci_high < 0).sum()),
         "rmse_diff_range": [float(h_primary.rmse_diff.min()), float(h_primary.rmse_diff.max())],
         "paths_per_scenario": ah["design"]["paths_per_scenario"], "master_seed": ah["master_seed"],
         "rolling_minus_true_vol_rmse_range_gbm": [float(true_gap.rmse_diff.min()), float(true_gap.rmse_diff.max())],
         "rolling_minus_true_regime_rmse_range": [float(regime_gap.rmse_diff.min()), float(regime_gap.rmse_diff.max())],
         "source": "reports/experiments/track_a_hedging.json:comparisons"},
        {"claim_id": "S3_engine_validation", "evidence_type": "simulated/analytic",
         "description": "Pricing-engine validation", "test_summary": math["tests"]["summary"],
         "mc_coverage_95": coverage, "mc_variance_reduction_factor": vrf,
         "mc_coverage_repeats": math["monte_carlo_coverage"][0]["repeats"],
         "source": "reports/experiments/math_validation.json"},
    ]
    return {
        "experiment_id": config["experiment_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_checksums": {"raw_manifest_sha256": manifest["raw_manifest_sha256"],
                             "raw_files_verified": manifest["raw_inputs"]},
        "data_coverage": {"clean_trades": manifest["option_trades"]["clean_rows"],
                          "first_trade": manifest["option_trades"]["first_trade"],
                          "last_trade": manifest["option_trades"]["last_trade"],
                          "split_boundaries": manifest["split_boundaries"]},
        "claims": claims, "best_methods": best, "timings": timings,
    }


# ============================================================================ final report

def final_report(data: dict, evidence: dict, figures: dict) -> str:
    claims = {c["claim_id"]: c for c in evidence["claims"]}
    h1, h2, h3, h4, h5 = (claims[k] for k in ["H1_test_rmse_by_model", "H2_bayes_vs_rolling", "H3_bayes_vs_ewma",
                                                "H4_integration_effect", "H5_historical_hedging"])
    s1, s2, s3 = claims["S1_simulated_pricing"], claims["S2_simulated_hedging"], claims["S3_engine_validation"]
    b = data["b_pricing"]
    test = b["results"]["test"]
    reg = test["summary"]["regime"]
    qlike = b["volatility_forecast_evaluation"]["mean_qlike"]
    return f"""# Volatility Lab - final research report

*Experiment `{evidence['experiment_id']}`. Generated by `scripts/07_make_report.py` from saved outputs; every number
below is in `reports/cv_evidence.json` or `reports/results_summary.md`.*

## Question

Does explicitly modelling uncertainty in volatility (a conjugate Normal-Inverse-Gamma posterior, with the option price
averaged over it) improve option-price estimates or reduce delta-hedging risk compared with rolling and EWMA
volatility, when evaluated out of sample?

## Data and design

* **Track A (simulated):** GBM (20%, 60% vol) and a two-state regime-switching DGP with known true prices;
  {s1['histories_per_scenario']:,} simulated histories per scenario for pricing and {s2['paths_per_scenario']:,}
  paths per scenario for hedging; master seed {s1['master_seed']}.
* **Track B (historical):** {data['manifest']['option_trades']['clean_rows']:,} clean Deribit BTC option trades
  (European, cash-settled) from 353 one-hour windows, with volatility estimated only from daily index returns up to
  the same day's 08:00 UTC fixing. Hyper-parameters chosen on 2022-2023; **test = {test['n_trades']:,} trades on
  {test['n_market_dates']} dates, {test['first_trade'][:10]} to {test['last_trade'][:10]}**. The target is the observed
  trade price (no bid/ask available).

## Findings

1. **Historical pricing.** Test RMSE (% of index price): rolling {h1['values']['rolling']:.3f}, EWMA
   {h1['values']['ewma']:.3f}, Bayes plug-in {h1['values']['bayes_plugin']:.3f}, Bayes integrated
   {h1['values']['bayes_integrated']:.3f}, regime {h1['values']['regime']:.3f}.
2. **Bayes integrated vs rolling:** RMSE {h2['rmse_diff']:+.3f} pp ({100 * h2['relative_change']:+.0f}%), 95% CI
   [{h2['ci95'][0]:.3f}, {h2['ci95'][1]:.3f}], Holm-adjusted bootstrap p = {h2['holm_p']:.3f}. The date-level
   Diebold-Mariano test does **not** confirm it (p = {h2['dm_p']:.2f}), so the evidence is moderate. About
   {100 * h2['share_explained_by_prior_shrinkage']:.0f}% of the improvement is already obtained by the Bayesian
   *point* estimate. It comes from shrinkage towards the high-volatility 2020-21 training variance, which happens to
   offset the market's volatility premium (the rolling estimator sits {-test['summary']['rolling']['vol_points_mean']:.1f}
   vol points below traded IV on average). It is not evidence that uncertainty modelling captures market pricing.
3. **Bayes integrated vs EWMA: no detectable difference** (RMSE {h3['rmse_diff']:+.3f}, CI
   [{h3['ci95'][0]:.3f}, {h3['ci95'][1]:.3f}], Holm p = {h3['holm_p']:.2f}).
4. **Integrating over the posterior** (vs the plug-in from the same posterior) is statistically detectable but tiny
   in the market data: RMSE {h4['rmse_diff']:+.4f} pp, CI [{h4['ci95'][0]:.4f}, {h4['ci95'][1]:.4f}], Holm p =
   {h4['holm_p']:.3f}.
5. **Simulation.** When the DGP is known, the posterior-integrated price beats the rolling plug-in in
   {s1['cells_with_ci_below_zero']} of {s1['cells']} cells (every Holm p {fmt_p(s1['max_holm_p'], s1['bootstrap_replicates'])}).
   The improvement is {-s1['rmse_diff_range'][1]:.4f} to {-s1['rmse_diff_range'][0]:.4f} percentage points of spot.
   On average it is {-s1['mean_rmse_diff_by_history_days']['30']:.4f} pp with 30 days of history and
   {-s1['mean_rmse_diff_by_history_days']['90']:.4f} pp with 90 days. Its source is understood: a plug-in sigma-hat is
   biased low in short samples, which makes ATM options too cheap, and averaging over the posterior removes most of
   that ATM bias. *Caveat:* the main simulated prior (strength 2) is centred on the true variance. With a deliberately
   wrong centre (0.5x or 2x), the weak prior still beats rolling in {s1['weak_wrong_prior_cells_better_than_rolling'][0]}
   of {s1['weak_wrong_prior_cells_better_than_rolling'][1]} GBM cells, but stronger wrong priors lose badly (all
   strengths: {s1['all_wrong_prior_cells_better_than_rolling'][0]} of {s1['all_wrong_prior_cells_better_than_rolling'][1]}).
6. **Hedging.** In simulation the Bayesian hedge has lower RMSE than the rolling hedge in
   {s2['scenarios_with_ci_below_zero']} of {s2['scenarios']} scenarios, but by only
   {-s2['rmse_diff_range'][1]:.4f} to {-s2['rmse_diff_range'][0]:.4f} pp of spot. Knowing the true sigma instead would
   cut RMSE by {s2['rolling_minus_true_vol_rmse_range_gbm'][0]:.3f} to {s2['rolling_minus_true_vol_rmse_range_gbm'][1]:.3f}
   pp (GBM), and knowing the true regime parameters by {s2['rolling_minus_true_regime_rmse_range'][0]:.3f} to
   {s2['rolling_minus_true_regime_rmse_range'][1]:.3f} pp. So parameter uncertainty that the posterior can represent is a
   small part of hedging risk. On observed BTC paths (daily, 5 bp) the Bayesian hedge has a **higher** RMSE
   ({h5['values']['bayes_integrated']:.2f} vs rolling {h5['values']['rolling']:.2f}, EWMA {h5['values']['ewma']:.2f}).
   That is because it charges a higher premium ({h5['mean_premium']['bayes_integrated']:.2f}% vs
   {h5['mean_premium']['rolling']:.2f}% of spot) than realised volatility justified (mean error
   {h5['mean_error']['bayes_integrated']:+.2f} vs {h5['mean_error']['rolling']:+.2f}). Its hedge-error sd is similar
   ({h5['sd_error']['bayes_integrated']:.2f} vs {h5['sd_error']['rolling']:.2f}; EWMA {h5['sd_error']['ewma']:.2f}).
7. **Null/negative result: the regime model.** Fitted on 2020-21, it over-predicts later volatility. Pricing bias is
   {reg['vol_points_mean']:+.1f} vol points, test RMSE {reg['pct_of_spot']['rmse']:.3f} (worst), 30-day realised-vol
   QLIKE {qlike['regime']:.3f} vs {qlike['rolling']:.3f} for rolling (worst), and hedge RMSE
   {h5['values']['regime']:.2f} (worst). Its ES95 hedge loss ({h5['es95_loss']['regime']:.2f}) looks best only because it
   charges a {h5['mean_premium']['regime']:.1f}% premium. That is a pricing bias, not better risk control.
8. **Engine validation:** {s3['test_summary']}; Monte Carlo 95% CI coverage over {s3['mc_coverage_repeats']:,} repeats:
   {', '.join(f'{k} {v:.3f}' for k, v in s3['mc_coverage_95'].items())}; control-variate variance reduction
   {s3['mc_variance_reduction_factor']['control_variate']:.1f}x.

## Answer to the research question

*Under a correctly specified model, with short histories and a weak prior* (simulation), integrating over
volatility uncertainty gives a small, reliable improvement in pricing and an even smaller one in hedging. A strong
prior with the wrong centre removes the benefit. *On real BTC option trades*, the
Bayesian model beat a rolling estimator, but:

* the gain came mostly from prior shrinkage rather than from uncertainty integration;
* it was not confirmed by the Diebold-Mariano test;
* it was not better than a well-tuned EWMA.

Rolling and EWMA volatility sit {-h1['vol_point_bias']['rolling']:.1f} and
{-h1['vol_point_bias']['ewma']:.1f} vol points below traded IV on average, consistent with a volatility risk
premium. The Bayesian model's near-zero average gap ({h1['vol_point_bias']['bayes_integrated']:+.1f}) is inherited
from its high-volatility prior, and per-trade errors stay large for every model (vol-point MAE
{min(h1['vol_point_mae'].values()):.1f} to {max(h1['vol_point_mae'].values()):.1f}).

## Main limitations

One asset (BTC); trade prices without bid/ask; forward recovered from Deribit's reported IV; r = 0; Gaussian
likelihood for fat-tailed returns; 143 overlapping historical hedges; hypothetical hedges ignore funding, margin and
market impact. See `docs/limitations.md`.

![simulated hedging]({'figures/' + figures['hedge']})
"""


# ============================================================================ main

def main() -> None:
    config = load_config()
    data = load_everything()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    frame, _, window, decay = rebuild_track_b_frames(config, data["b_pricing"])
    # consistency check: the rebuilt test frame must match the saved sample size
    assert int((frame["split"] == "test").sum()) == data["b_pricing"]["results"]["test"]["n_trades"]
    timings = timing_benchmark(frame, config["volatility"]["posterior_quadrature_nodes"])

    figures = {"binomial": chart_binomial(data["math"]), "vol": chart_vol_vs_market(frame, window, decay),
               "maturity": chart_track_b_rmse_by_maturity(data["b_pricing"]),
               "hedge": chart_track_a_hedging(data["a_hedging"])}

    best_text, best = best_methods_section(data, timings)
    summary = "\n".join([
        "# Results summary", "",
        f"Experiment `{config['experiment_id']}` - generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        "by `scripts/07_make_report.py` from `reports/experiments/*.json`. **Simulated and historical results are kept "
        "in separate sections and must not be mixed.**", "",
        "Raw-input manifest checksums: " + ", ".join(f"`{k}` sha256 `{v[:16]}...`"
                                                       for k, v in data["manifest"]["raw_manifest_sha256"].items()), "",
        sample_size_section(data), simulation_size_section(data), math_section(data["math"]),
        f"![binomial convergence](figures/{figures['binomial']})", "",
        track_a_section(data), f"![track A hedging](figures/{figures['hedge']})", "",
        track_b_section(data, timings, figures), best_text,
    ])
    (REPORTS_DIR / "results_summary.md").write_text(summary, encoding="utf-8")

    evidence = cv_evidence(data, best, timings, config)
    (REPORTS_DIR / "cv_evidence.json").write_text(json.dumps(evidence, indent=1, default=str))
    (REPORTS_DIR / "final_research_report.md").write_text(final_report(data, evidence, figures), encoding="utf-8")
    print("wrote results_summary.md, cv_evidence.json, final_research_report.md and", len(figures), "figures")


if __name__ == "__main__":
    main()
