# Volatility Lab

Volatility is the one Black-Scholes input you can't observe. You estimate it from past returns,
and that estimate is noisy. With 60 days of data, a "60% vol" could easily be 50% or 70%. So I
wanted to answer a simple question properly:

**If I carry that uncertainty into the option price instead of plugging in a single number, do I
get better prices or safer hedges?**

To find out, I built an option-pricing engine from scratch, three ways of estimating volatility
(plus an optional regime model), and a delta-hedging simulator. I tested all of them out of
sample with proper statistics.

## What's inside

- **Pricing engine:** CRR binomial trees (European and American), Black-Scholes-Merton with all
  the Greeks, an implied-vol solver using Brent's method, and Monte Carlo with antithetic and
  control-variate variance reduction.
- **Volatility models:**
  - rolling historical vol;
  - EWMA;
  - a conjugate Normal-Inverse-Gamma Bayesian model. Its option price is averaged over the whole
    posterior instead of using a point estimate.
  - a two-state Markov regime model, fitted with EM.
- **Hedging simulator:** sells an option, delta hedges it with transaction costs, and measures
  what's left at expiry.
- **Statistics:** chronological train, validation and test splits; leak-proof joins between
  trades and data; date-clustered bootstrap; Diebold-Mariano tests; Holm corrections.

I tested everything in two separate tracks and never mixed them:

| Track | What it uses | What it can tell me |
|---|---|---|
| A: simulation | paths from models where I know the true price | whether a method works when its assumptions hold |
| B: real market | Deribit BTC option trades and the BTC index (free public API) | how close each model gets to actual traded prices |

No live trading, no broker connections, no real money. This is a research project, not a
trading strategy.

## What I found

- **In simulation, modelling uncertainty helps, but only a little.** The posterior-averaged price
  beat the rolling estimate in all 18 scenario cells I tested. The gain was 0.4% to 6.4% of
  pricing RMSE, and biggest with short histories. The reason is neat: a plug-in volatility
  estimate is biased low in small samples, so at-the-money options come out too cheap.
  Averaging over the posterior removes most of that bias.
- **For hedging, the effect is tiny.** It was at most about 1.6% of hedging RMSE. Most hedging
  risk comes from rebalancing in discrete steps and from volatility actually changing, not from
  parameter uncertainty.
- **On real BTC trades the story is more humbling.** The Bayesian model beat a rolling estimator,
  but almost all of that gain came from the prior pulling volatility upwards, not from the
  uncertainty modelling. It also didn't beat a well-tuned EWMA. The gap that matters is between
  historical and implied volatility, and that's a risk premium. No amount of statistics on past
  returns explains it.
- **The regime model was a clear miss out of sample.** I kept that result in on purpose.

The engine itself is validated against textbook values, worked examples from my Caltech course
notes, numerical integration and finite-difference Greeks, with 101 tests in total. The control
variate cut Monte Carlo variance about 4x, and the 95% confidence intervals hit 94% to 96%
coverage over 1,000 repeats.

## Running it

You need [uv](https://docs.astral.sh/uv/). It sets up Python 3.12 and the pinned dependencies
from `uv.lock`.

```bash
cd volatility-lab
uv sync

uv run python scripts/00_check_sources.py              # check the free data sources are up
uv run python scripts/01_download_data.py --sample-only # small pilot download
uv run python scripts/01_download_data.py              # full sample, 353 one-hour windows (resumable)
uv run python scripts/02_build_dataset.py              # verify checksums, clean, split
uv run python scripts/03_validate_math.py              # math tests first, then numerical checks
uv run python scripts/04_track_a_simulation.py         # simulated pricing and hedging
uv run python scripts/05_track_b_market_pricing.py     # validation selection, then the test set
uv run python scripts/06_track_b_historical_hedging.py
uv run python scripts/07_make_report.py                # charts, summary, CV evidence
uv run pytest
```

Once the data is downloaded, `uv run python scripts/run_all.py` runs everything in one go. Every
setting (sample days, filters, split dates, candidate parameters, seeds) lives in
[`configs/experiment.toml`](configs/experiment.toml).

## Why the results aren't in this repo

The market data comes from Deribit's public API, and their terms say market data and anything
derived from it is for personal use only. So the data and the generated reports stay on my
machine. Run the pipeline and you'll get the full reports locally; see
[`reports/README.md`](reports/README.md).

Before settling on Deribit I checked a bunch of other free sources:

| Source | Why I didn't use it |
|---|---|
| Tardis.dev free monthly Deribit files | blocked by a browser challenge, and I wasn't going to bypass it |
| Cboe DataShop | paid, with no free sample |
| OptionMetrics (WRDS) | needs an institutional login |
| Yahoo Finance option chains | current snapshot only, no history |
| Kaggle option datasets | need a login, and the licensing is unclear |
| Stooq daily prices | behind a JavaScript proof-of-work check |
| FRED (VIX, S&P 500, T-bills) | worked fine, but my final design didn't need it |

## Layout

```
src/volatility_lab/       the library
  black_scholes.py        prices, Greeks, no-arbitrage bounds
  binomial.py             CRR trees
  implied_vol.py          Brent solver with reason codes, implied forward
  monte_carlo.py          plain, antithetic and control-variate MC
  volatility_estimators.py  rolling, EWMA, realised vol
  bayesian.py             NIG posterior, predictive, posterior-averaged price and delta
  regime_model.py         two-state Gaussian HMM
  pricing_models.py       the competing models and the leak-proof join
  hedging.py              delta-hedge accounting with costs
  simulation.py           GBM and regime-switching paths
  metrics.py, stat_tests.py  errors, VaR/ES, cluster bootstrap, Diebold-Mariano, Holm
scripts/                  numbered pipeline steps 00 to 07, plus run_all.py
tests/                    101 tests
docs/                     methodology, data dictionary, evaluation protocol, limitations
references/               how my course material maps to each method
```

The theory side draws on Caltech's *Pricing Options with Mathematical Models* and MIT's
probability and statistics courses. `references/course_concept_map.md` shows which idea went
where.
