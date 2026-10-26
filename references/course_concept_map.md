# Course concept map

How each implemented method relates to the three courses. **Implemented** means it is
in the code with tests. **Background** means it informed the design but is not
implemented. **Future** means it is a possible extension.

This map shows which course ideas the project *uses*. It does not mean the author has
completed these courses. See "Access notes" at the end for what was actually read.

## Sources

| Course | What was consulted | Access |
|---|---|---|
| Caltech, *Pricing Options with Mathematical Models* (edX) | The author's local copy of the course slides (`All slides.pdf`, 285 pages, kept **outside** the repository) plus the public edX course page | Slides read in full where the text was extractable; equation slides that are images were viewed as images (§§17, 23, 27). The edX page body did not render, so no syllabus text was obtained from it. |
| MIT 6.431x, *Probability: The Science of Uncertainty and Data* | Official MIT Open Learning course page (topic list) | Course materials are on edX behind enrolment/login, which was **not available**. Lecture content was not read. |
| MIT 18.6501x, *Fundamentals of Statistics* | Official MIT Open Learning course page (description) | Same: edX materials need a login that was **not available**. |

Course sections below refer to the slide-deck section numbers (e.g. "Caltech §13 Binomial
tree pricing"). No slide content is reproduced beyond short factual references and the
worked numerical examples used as test cases.

## Implemented methods

| Method | Course concepts | Key assumptions | Module | Validating tests |
|---|---|---|---|---|
| No-arbitrage bounds and put–call parity | Caltech §9 (model-independent relations 1–8, put–call parity) | Frictionless market, European exercise, no arbitrage | `black_scholes.no_arbitrage_bounds` | `test_black_scholes.py::test_put_call_parity_over_a_grid`, `test_prices_respect_no_arbitrage_bounds`, `test_convexity_in_strike_*` |
| Binomial (CRR) tree, risk-neutral probability, backward induction, American exercise | Caltech §10–13 (multi-period model, self-financing, risk-neutral pricing, fundamental theorems, binomial pricing, American options); MIT 6.431x (discrete random variables, expectation) | $d<e^{(r-q)\Delta t}<u$; recombining tree; constant parameters | `binomial.py` | `test_binomial.py` (course examples 0.746, 10.8703, 2.4844; convergence to BS; American ≥ European) |
| CRR → Black–Scholes convergence | MIT 6.431x (central limit theorem); Caltech §14 (discretised Brownian motion) | iid steps with matching mean/variance | `binomial.crr_price` | `test_crr_converges_to_black_scholes`; `scripts/03_validate_math.py` convergence table |
| Black–Scholes–Merton prices | Caltech §14–18 (Brownian motion, Itô's rule, BSM PDE, risk-neutral pricing, Girsanov); §19 (continuous dividends) | GBM with constant $\sigma$, $r$, $q$; continuous frictionless trading | `black_scholes.py` | Hull textbook values; numerical integration against the lognormal density |
| Greeks (delta, gamma, vega, theta, rho) | Caltech §27–28 (replication delta, portfolio sensitivities / Taylor expansion) | Same as BSM | `black_scholes.py` | `test_greeks_match_central_finite_differences`; Hull Greek values |
| Black-76 on a forward | Caltech §19 (options on futures), §8 (forward pricing $F=S\,B(t,T)$) | Forward is the correct underlying for a cash-settled European option; $r=0$ convention | `bs_price(spot=F, r=0, q=0)` | `test_implied_vol.py::test_implied_forward_recovery` |
| Implied volatility (Brent) | Caltech §18 (implied volatility, smile); MIT 18.6501x (estimation by solving an equation) | Monotone price in $\sigma$; price inside the arbitrage bounds | `implied_vol.py` | `test_implied_vol.py` (grid recovery, bounds, short expiry, missing data) |
| Monte Carlo pricing with CIs, antithetic and control variates | MIT 6.431x (LLN, CLT, variance of sums, covariance); MIT 18.6501x (confidence intervals) | iid draws; finite variance; control variate with known mean | `monte_carlo.py` | `test_monte_carlo.py` (accuracy, SE ∝ $1/\sqrt n$, variance reduction, 95% coverage) |
| Rolling and EWMA volatility | MIT 18.6501x (estimators, consistency, bias); Caltech §27 (sample st. dev. × √252) | Stationary returns inside the window (rolling); geometric down-weighting (EWMA) | `volatility_estimators.py` | `test_volatility_estimators.py` (hand calculations, course annualisation example, LLN consistency) |
| Normal–Inverse-Gamma posterior and Student-t predictive | MIT 6.431x (Bayesian inference, conditional distributions, conjugate priors); MIT 18.6501x (likelihood, Bayesian statistics) | iid normal returns in the window; conjugate prior | `bayesian.py` | `test_bayesian.py` (hand formulas, sequential = batch, brute-force grid posterior, predictive variance, concentration) |
| Posterior-expected option price $E_{post}[BS(\sigma)]$ | MIT 6.431x (expectation of a function of a random variable, total expectation, Jensen); Caltech §18 (price = expectation under Q) | Parameter uncertainty priced at zero premium | `bayesian.bayesian_option_price`, `bayesian_option_delta` | `test_bayes_price_matches_direct_integration`, `test_..._put_call_parity...`, `test_bayesian_delta_matches_finite_difference` |
| Two-state hidden Markov regime model (EM, forward filter, multi-step forecast) | MIT 6.431x (Markov chains, transition matrices, steady state, Bayes' rule for filtering); MIT 18.6501x (maximum likelihood) | Two Gaussian regimes; time-homogeneous chain | `regime_model.py` | `test_regime_model.py` (parameter recovery, monotone EM likelihood, filter accuracy, stationary forecast) |
| Hull–White mixing price under random volatility | Caltech §23 (stochastic volatility, incomplete markets, choice of $Q_\kappa$) | Vol independent of price shocks; zero volatility risk premium | `pricing_models.regime_mixture_price` | `test_variance_path_simulation_matches_analytic_forecast` (variance paths); used as the Track A "true" price |
| Discrete delta hedging, self-financing wealth, transaction costs | Caltech §10 (self-financing condition, gains process), §27 (replication in BSM, the Microsoft example), §28 (portfolio sensitivities) | Rebalancing at discrete times; proportional costs; cash at $r$ | `hedging.py` | `test_hedging.py` (reproduces the course's 20-day wealth column; cash account; costs; $\sqrt{\Delta t}$ error scaling) |
| VaR and Expected Shortfall of hedge losses | MIT 6.431x (quantiles, distributions); Caltech §27–28 (hedging risk) | Empirical distribution is representative | `metrics.py` | `test_stats_and_metrics.py::test_var_and_expected_shortfall` |
| Paired cluster bootstrap, Diebold–Mariano, Holm | MIT 18.6501x (hypothesis tests, p-values, confidence intervals, multiple testing); MIT 6.431x (CLT, dependence) | Clusters (dates) are independent of each other; DM needs a stationary loss differential | `stat_tests.py` | `test_stats_and_metrics.py` (CI covers truth, clustered CI wider, size under the null, DM power, Holm) |
| Chronological splits and no look-ahead | MIT 18.6501x (model selection, train/validation/test) | Time order respected | `splits.py`, `pricing_models.attach_latest_fixing` | `test_leakage.py`, `test_data_processing.py` |

## Background only (not implemented)

- **Itô integral and Itô's lemma (Caltech §15–16):** used to justify the BSM PDE and the
  GBM solution. The code uses the closed forms; no SDE is discretised.
- **Girsanov's theorem (Caltech §18):** the reason the price does not depend on $\mu$ (in
  the tests, the drift of the simulated paths does not enter the true price).
- **Fundamental theorems of asset pricing (Caltech §12):** completeness of the binomial/BSM
  models versus incompleteness with stochastic volatility. They explain why the Track B
  implied-minus-historical gap can persist.
- **Model selection and goodness of fit (MIT 18.6501x):** informs the validation design. No
  formal information criteria are used.

## Future extensions

- **Heston / SABR (Caltech §23):** need calibration to a cross-section of quotes (bid/ask),
  which our trade data cannot support robustly.
- **Merton jump-diffusion (Caltech §24):** BTC returns have jumps, so this is a natural next
  likelihood.
- **Fat-tailed Bayesian likelihood** (e.g. Student-t returns via MCMC); **Poisson processes**
  (MIT 6.431x) for jump arrivals.
