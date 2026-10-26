# Volatility Lab

**Does explicitly modelling uncertainty in volatility improve option-price estimates or reduce
hedging risk compared with conventional volatility estimates, when evaluated out of sample?**

This is an educational, offline research project:

- a tested option-pricing engine (CRR binomial, Black–Scholes–Merton and Greeks, implied vol,
  Monte Carlo with variance reduction);
- three volatility estimators: rolling, EWMA and a conjugate Normal–Inverse-Gamma Bayesian model
  whose option price is averaged over the posterior;
- an optional two-state Markov regime model;
- a delta-hedging simulator;
- paired, chronologically split statistical evaluation.

It has two clearly separated evidence tracks:

| Track | Data | Label used everywhere |
|---|---|---|
| A | Simulated paths from known models (true prices known) | **simulated** |
| B | Real Deribit BTC option trades and the BTC index (free public API) | **historical** / **hypothetical hedge on historical paths** |

No live trading, broker connections or real transactions. Nothing here is a trading strategy.

**Results are generated locally and not published here.** They are aggregate statistics
derived from Deribit market data, and Deribit's terms restrict that to personal use. Running the
pipeline (below) produces `reports/results_summary.md`, `reports/final_research_report.md`,
`reports/cv_evidence.json`, the data-feasibility audit and the charts. See
[`reports/README.md`](reports/README.md).

## Install

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the pinned
dependencies from `uv.lock`: numpy, scipy, pandas, matplotlib, and pytest for tests.

```bash
cd volatility-lab
uv sync
```

## Reproduce

```bash
uv run python scripts/00_check_sources.py         # are the free sources reachable?
uv run python scripts/01_download_data.py --sample-only   # bounded pilot (4 hourly windows)
uv run python scripts/01_download_data.py         # full bounded sample: 353 windows (~5 min, resumable)
uv run python scripts/02_build_dataset.py         # verify checksums, clean, split -> data/processed/
uv run python scripts/03_validate_math.py         # runs math tests first, then numerical validation
uv run python scripts/04_track_a_simulation.py    # simulated pricing + hedging (~6 min)
uv run python scripts/05_track_b_market_pricing.py  # validation selection, then test evaluation
uv run python scripts/06_track_b_historical_hedging.py
uv run python scripts/07_make_report.py           # charts, results_summary.md, cv_evidence.json, report
uv run pytest                                     # 101 tests
```

Or run every step after the download with `uv run python scripts/run_all.py` (add `--download` to
fetch the data too).

All settings (sample days, filters, split dates, candidate hyper-parameters, seeds) are in
[`configs/experiment.toml`](configs/experiment.toml).

## Repository layout

```
src/volatility_lab/     reusable modules
  black_scholes.py      BSM prices, Greeks, no-arbitrage bounds
  binomial.py           CRR tree (European/American)
  implied_vol.py        Brent implied vol with reason codes; implied forward
  monte_carlo.py        plain / antithetic / control-variate MC with CIs
  volatility_estimators.py  rolling, EWMA, realised vol
  bayesian.py           Normal-Inverse-Gamma posterior, predictive, posterior-averaged price and delta
  regime_model.py       2-state Gaussian HMM (EM, filter, forecasts)
  pricing_models.py     the competing models + leak-proof trade/fixing join
  hedging.py            discrete delta-hedge accounting with costs
  simulation.py         GBM and regime-switching path simulators
  metrics.py, stat_tests.py   error metrics, VaR/ES, cluster bootstrap, Diebold-Mariano, Holm
  data_download.py, data_processing.py, splits.py, config.py, reporting.py
scripts/                numbered, reproducible steps (00-07) + run_all.py
tests/                  101 tests (formulas, invariants, leakage, data, reports)
docs/                   methodology, data dictionary, evaluation protocol, limitations, interview prep
references/             course concept map, Caltech notes README (notes not included)
reports/                generated locally, git-ignored (see reports/README.md)
data/raw, data/processed   git-ignored (Deribit terms: personal use only)
```

## Data and licensing

Market data come from Deribit's public endpoints (`history.deribit.com` trades and
`get_delivery_prices`), with no account required. Deribit's terms say market data and derived
data are for **personal use only**. So neither the data nor the generated reports (which
contain aggregate statistics derived from it) are committed.

Free sources investigated during the data audit and not used:

| Source | Reason not used |
|---|---|
| Tardis.dev free first-of-month Deribit files | Blocked by a browser challenge from this environment (not bypassed) |
| Cboe DataShop option EOD data | Paid; no free sample found |
| OptionMetrics IvyDB (WRDS) | Needs an institutional login |
| Yahoo Finance option chains | Current snapshot only, no history |
| Kaggle community option datasets | Need a login; unclear provenance and licence |
| Stooq daily equity prices | Behind a JavaScript proof-of-work check (not bypassed) |
| FRED (VIX, S&P 500, T-bills) | Available, but not needed for the final BTC design |

Course materials are not redistributed. See `references/course_concept_map.md` for how the
Caltech and MIT courses informed each method, and which materials were actually accessible.

## AI assistance

This project was implemented with help from an AI coding assistant (Claude), which wrote much
of the code, tests and documentation under the author's direction. AI assistance does not
replace understanding. Every result should be re-derived, re-run and explained by the author.
`docs/interview_preparation.md` lists questions for checking that understanding.
