# Evaluation protocol

This protocol was written down (in code and here) before the test period was evaluated.

## Two evidence tracks, never mixed

| Track | Data | What it can show | What it cannot show |
|---|---|---|---|
| **A – controlled simulation** | Simulated paths from known DGPs (GBM at 20% and 60% vol; two-state regime switching 40%/100%) | Whether a method recovers the true price or hedges well **when its assumptions (or a stated misspecification) hold** | Anything about real market prices |
| **B – historical market** | Real Deribit BTC option trades plus the BTC index | How close each model's price is to **observed BTC option trade prices**, and the hypothetical hedge error on **observed BTC paths** | Accuracy on other assets, fair value (trades ≠ fair value), executed trading performance |

## Chronological splits (Track B)

Chosen after looking only at the date coverage (daily index history starts 2020-02-04):

| Split | Option-trade dates | Use |
|---|---|---|
| Training (returns only) | index returns 2020-02-05 → 2021-12-31 | Bayesian prior centre $s_0^2$; regime-model parameters (EM) |
| Development | 2020-01-01 → 2021-12-31 | Pipeline building and debugging only; **not scored** |
| Validation | 2022-01-01 → 2023-12-31 | Choose rolling window $w\in\{30,60,90,180\}$, EWMA $\lambda\in\{0.90,0.94,0.97,0.99\}$, prior $(\nu_0,m)\in\{2,10,30,100\}\times\{0.5,1,2\}$, all by validation RMSE |
| Test | 2024-01-01 → 2026-09-30 | Final evaluation with frozen choices |

The boundaries are in `configs/experiment.toml` and are copied into
`data/processed/data_manifest.json` and the experiment outputs.

**Information rule.** A trade at time $t$ is priced with the volatility inputs of the latest
index fixing at or before $t$, which is the same day's 08:00 fixing because trades are sampled
08:00–09:00. These inputs use only returns up to that fixing. This is enforced by
`attach_latest_fixing` (an as-of join that cannot look forward) and tested in
`tests/test_leakage.py`. The forward $F$ is market information at time $t$ (it is the
futures/synthetic forward the trade was struck against), not a forecast.

The Bayesian model uses the **same window as the chosen rolling estimator**. So
"bayes_plugin vs rolling" isolates the prior shrinkage, and "bayes_integrated vs
bayes_plugin" isolates integrating over posterior uncertainty.

## Primary hypotheses (fixed in advance)

Track B pricing. The loss is the squared pricing error in % of spot, on the test period:

1. `bayes_integrated` vs `rolling`
2. `bayes_integrated` vs `ewma`
3. `bayes_integrated` vs `bayes_plugin`

For each: paired differences in MAE, RMSE, MSE and bias; a 95% **date-clustered bootstrap**
interval (5,000 replicates); a bootstrap p-value for the MSE difference;
**Holm-adjusted** across the three; and a Diebold–Mariano test on per-date mean squared-error
differences as a cross-check. Everything else is labelled **exploratory**: the regime model,
EWMA vs rolling, plug-in vs rolling, grouped results and the hedging sensitivities.

Track A uses the same three-way structure. The primary family is `bayes_integrated` vs
`rolling`, Holm-adjusted across all scenarios.

## Hedging protocol

- **Track A:** for each DGP × maturity there is one set of 10,000 paths (90 days of history
  plus the option's life), reused for every model, rebalancing frequency (1 or 7 days) and cost
  level (0 or 10 bp), so all comparisons are paired. The hedger sells an ATM call at its own
  model price and re-estimates volatility at every rebalance from data up to that day.
  Benchmarks: the true σ (GBM), or a regime filter that knows the true parameters (regime DGP;
  infeasible in practice).
- **Track B:** a 30-day ATM call is sold every 7th day of the test period on the observed
  index, hedged at the daily fixings (also every 7 days as a sensitivity check), with 5 bp
  costs (and 0 bp). Hedges overlap, so the bootstrap resamples blocks of 5 consecutive starts.

## Dependence and multiple testing

- Option trades on the same date share the same σ estimate and the same market state, so
  dates are the resampling unit, never individual trades.
- Only the pre-specified primary families get Holm adjustment and "significance" language.
  The other comparisons are reported with intervals but described as exploratory.
- A p-value is never reported without the effect size and its interval.

## Re-runs (honest log)

- `05_track_b_market_pricing.py` crashed once at an indexing bug while scoring the
  **validation** split, before any test-period numbers were computed. A second run was
  stopped deliberately (again before test numbers were printed) so the trade-to-fixing join
  could be replaced by the leak-proof as-of join. The test period was first scored in the
  third run. That is the run reported. No test-period result was used to change any setting.
- Afterwards the whole pipeline (`scripts/run_all.py`, steps 00 and 02–07) was re-run from the
  raw files as a **reproducibility check**. This re-scored the test period. All five experiment
  outputs and the cleaned dataset were byte-identical (SHA-256, ignoring wall-clock runtime and
  timestamp fields) to the first complete run.
