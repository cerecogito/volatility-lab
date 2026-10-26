# Limitations

Ordered from most to least important for interpreting the results.

## What the historical evidence can and cannot support

1. **One asset, one venue.** All market evidence is Deribit BTC options. BTC volatility is
   high, returns are fat-tailed and the market trades 24/7. None of
   this shows anything about equity-index or single-stock options.
2. **Trade prices, not fair values.** The free data have no bid/ask quotes, so the target is
   the observed trade price. A trade can be at the bid or the ask (results are reported
   separately for buyer- and seller-initiated trades), stale relative to the index, or a small
   lot. A model error of "x% of spot" is therefore partly microstructure noise.
3. **Forward recovered from Deribit's reported IV.** The trade record has no forward price.
   We solve for the forward that matches (trade price, reported IV). This depends on Deribit's
   own IV calculation; trades that fail the recovery check are dropped and counted in
   `data/processed/exclusions.csv`. All
   models share the same forward, so comparisons *between* models are unaffected, but absolute
   errors include any forward error.
4. **Implied vs historical volatility is a risk premium, not an estimation error.** All
   estimators use physical-measure returns, while option prices embed a volatility risk premium
   (the market price of volatility risk $\kappa$ in Caltech §23). The Bayesian model's
   near-zero average gap to market IV comes from a prior centred on the high-volatility 2020–21
   training period. If volatility had risen instead of fallen after training, that same prior
   could have hurt. The "win" over rolling is therefore partly luck of the regime.
5. **Mixed significance for the main historical result.** Bayes-integrated vs rolling is
   significant under the date-clustered bootstrap but not under the date-level
   Diebold–Mariano test (see the locally generated `reports/results_summary.md`). Both use the
   same test dates. The bootstrap weights dates by
   their trade counts, while DM weights every date equally. We report the result as moderate
   evidence, not as established.
6. **Sampling design.** Trades come from one hour (08:00–09:00 UTC) on Wednesdays only. That
   gives a clean information set (the daily fixing is known) but excludes intraday and weekday
   effects and other hours' liquidity.
7. **Historical hedging sample is small and overlapping.** 143 hedges of 30 days started
   weekly overlap by about 4 weeks, so there are only ≈29 independent blocks. Intervals are
   wide and most hedging differences are not significant. The hedges are hypothetical: forward
   = spot, $r=0$, 5 bp costs, no funding, margin, market impact or liquidity constraints.
   Nothing was traded.

## Modelling assumptions

8. **Gaussian likelihood** in the Bayesian and regime models, which is misspecified for BTC
   (jumps, fat tails). The posterior for $\sigma^2$ is too confident when kurtosis is high.
9. **Parameter uncertainty is priced at zero premium.** $E_{post}[BS(\sigma)]$ treats the
   posterior as if it were risk-neutral uncertainty. A risk-averse dealer would charge more.
10. **The regime model is frozen after training (2020–21).** It was never re-fitted, which is
    why it over-predicts later volatility. A rolling re-fit would be fairer, but it was not part
    of the pre-specified design, so it is not reported as a result.
11. **Plug-in regime pricing in Track B.** The regime model prices with its forecast average
    volatility rather than the full mixture over regime paths (the mixture is only used for the
    true price in Track A).
12. **Constant interest rate of zero** (Deribit convention). This is irrelevant to the
    comparisons but would matter for absolute pricing of long-dated options in USD.

## Simulation (Track A)

13. **The Track A prior is centred on the true variance** (with weight 2 pseudo-observations),
    which favours the Bayesian method. The prior-sensitivity grid shows that wrongly-centred
    *strong* priors lose to rolling: Bayes is better in only 12 of 32 wrong-prior cells.
14. **The DGPs are simple** (constant-vol GBM and a two-state chain, with vol independent of
    returns). There is no leverage effect, jumps or vol-of-vol, so the simulation shows how the
    methods behave under these assumptions, not in reality.
15. **Statistical vs practical significance.** With 20,000 histories or 10,000 paths, tiny
    differences (≈ 0.1–1.6% of RMSE in hedging) are "highly significant". The reports always
    give the effect size next to the p-value.

## Engineering

16. Bootstrap p-values have a resolution of 1/replicates (reported as "< 0.0005" or "< 0.0002").
17. The posterior expectation uses a 64-node quantile rule (32 inside hedging loops). It is
    checked to $10^{-3}$ relative error against adaptive quadrature, not exact.
18. The Track B pricing script was run three times before its test scores were first computed
    (see `docs/evaluation_protocol.md`, "Re-runs"); no setting was changed after seeing test
    results.
