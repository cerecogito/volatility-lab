# Mathematical methodology

Notation: $S_t$ is the underlying price, $K$ the strike, $\tau = T - t$ the time to expiry
in years, $r$ the risk-free rate, $q$ a continuous dividend yield/carry and $\sigma$ the
annualised volatility. $N(\cdot)$ is the standard normal CDF and $\varphi$ its density.
Every result below names the module that implements it and the test that checks it.

---

## 1. Binomial (Cox–Ross–Rubinstein) tree — `binomial.py`

**Model.** Over each step $\Delta t = \tau/n$ the price moves to $uS$ or $dS$.
A portfolio of $\delta_1$ shares and $\delta_0$ in the bank replicates any payoff in one
step. Solving the two replication equations gives the price as a discounted expectation under

$$p^* = \frac{e^{(r-q)\Delta t}-d}{u-d}.$$

$0<p^*<1$ exactly when $d<e^{(r-q)\Delta t}<u$. This is the no-arbitrage condition
(first fundamental theorem in the binomial model). The code raises an error if it fails.

**Backward induction.** $V_n(j) = \text{payoff}(S_0u^jd^{n-j})$ and
$V_k(j) = e^{-r\Delta t}[p^*V_{k+1}(j+1)+(1-p^*)V_{k+1}(j)]$. For American options we take
$\max(\text{continuation}, \text{exercise})$ at every node.

**CRR choice.** $u=e^{\sigma\sqrt{\Delta t}}$, $d=1/u$. Then the log-price increment has
risk-neutral mean $(r-q-\sigma^2/2)\Delta t + O(\Delta t^{3/2})$ and variance
$\sigma^2\Delta t+O(\Delta t^2)$. By the CLT the terminal log price converges in distribution
to the Black–Scholes normal, and the tree price converges to the BS price. The error is
$O(1/n)$ with an oscillating constant.

*Assumptions:* frictionless market, constant $r$, $q$, $\sigma$, recombining tree, exercise only at nodes.

*Worked example (course §13):* $S=K=100$, $u=1.1$, $d=0.9$, $r=5\%$, $\Delta t=1$, two steps:
$p^*=(e^{0.05}-0.9)/0.2=0.7564$. Then $C_u=e^{-0.05}(0.7564\cdot 21)=15.1088$,
$C_0=e^{-0.05}(0.7564\cdot15.1088)=10.8703$. The American put on the same tree is
$2.4844$, with early exercise at the down node.
Tests: `tests/test_binomial.py::test_course_two_step_*`, `test_crr_converges_to_black_scholes`.

---

## 2. Black–Scholes–Merton — `black_scholes.py`

Under $Q$, $dS_t=(r-q)S_tdt+\sigma S_tdW_t^Q$, so
$\ln S_T\sim N(\ln S_t+(r-q-\tfrac12\sigma^2)\tau,\ \sigma^2\tau)$. Computing
$e^{-r\tau}E^Q[(S_T-K)^+]$ with the lognormal density gives

$$c=Se^{-q\tau}N(d_1)-Ke^{-r\tau}N(d_2),\qquad p=Ke^{-r\tau}N(-d_2)-Se^{-q\tau}N(-d_1),$$
$$d_{1}=\frac{\ln(S/K)+(r-q+\tfrac12\sigma^2)\tau}{\sigma\sqrt\tau},\qquad d_2=d_1-\sigma\sqrt\tau.$$

**Greeks** (implemented analytically, checked against finite differences):
$\Delta_c=e^{-q\tau}N(d_1)$, $\Delta_p=e^{-q\tau}(N(d_1)-1)$,
$\Gamma=e^{-q\tau}\varphi(d_1)/(S\sigma\sqrt\tau)$, $\mathcal V=Se^{-q\tau}\varphi(d_1)\sqrt\tau$,
$\Theta_c=-\frac{Se^{-q\tau}\varphi(d_1)\sigma}{2\sqrt\tau}-rKe^{-r\tau}N(d_2)+qSe^{-q\tau}N(d_1)$,
$\rho_c=K\tau e^{-r\tau}N(d_2)$.

**Black-76.** Setting $S=F$ and $r=q=0$ gives the undiscounted price of an option on a
forward $F$. This is what we use for Deribit options, where Deribit's convention is $r=0$
with the carry inside $F$.

**Invariants tested** (`tests/test_black_scholes.py`):
- put–call parity $c-p=Se^{-q\tau}-Ke^{-r\tau}$;
- the bounds $\max(Se^{-q\tau}-Ke^{-r\tau},0)\le c\le Se^{-q\tau}$;
- monotonicity in $S,K,\sigma,\tau$;
- convexity in $K$ (non-negative butterflies);
- the limits $\sigma\to0$ and $\tau\to0$;
- agreement with direct numerical integration of the payoff against the lognormal density;
- Hull's textbook values.

---

## 3. Implied volatility — `implied_vol.py`

$\sigma\mapsto BS(\sigma)$ is continuous and strictly increasing ($\mathcal V>0$). It tends to
the lower no-arbitrage bound as $\sigma\to0$ and to the upper bound as $\sigma\to\infty$. So
a root of $BS(\sigma)=V_{obs}$ exists **iff** $V_{obs}$ lies strictly inside the bounds, and
it is unique. We bracket $[10^{-4},10]$ and use **Brent's method**, which combines bisection,
secant and inverse quadratic steps and is guaranteed to converge for a bracketed root.

Observations that cannot be inverted get a reason code instead of an error:
missing input, $\tau<1$ hour, price at/below the lower bound, price at/above the upper
bound, no root in the bracket, or vega below $10^{-8}S$. In the last case (deep ITM/OTM),
tiny price rounding moves the IV a lot.

**Recovering the forward.** A Deribit trade record has the trade price and Deribit's
reported IV but not the forward Deribit used. Calls increase and puts decrease strictly in
$F$, so $\text{Black76}(F;K,\tau,\sigma_{rep})=V_{obs}$ has a unique root, which we find with
Brent's method on $[0.5S, 1.5S]$.

---

## 4. Monte Carlo — `monte_carlo.py`

$S_T$ is sampled exactly: $S_T=S_0\exp((r-q-\frac12\sigma^2)T+\sigma\sqrt TZ)$. The
estimator $\hat V=\frac1n\sum e^{-rT}g(S_T^{(i)})$ is unbiased. By the CLT,
$\hat V\approx N(V, s^2/n)$, giving the 95% interval $\hat V\pm1.96\,s/\sqrt n$.

- **Antithetic variates.** Average $g(Z)$ and $g(-Z)$. For a monotone payoff,
  $\mathrm{Cov}(g(Z),g(-Z))\le0$, so the variance of each pair mean is at most half the
  variance of a single draw.
- **Control variate.** $Y=e^{-rT}S_T$ has known mean $S_0e^{-qT}$. We use
  $X-b(Y-E[Y])$ with $b^*=\mathrm{Cov}(X,Y)/\mathrm{Var}(Y)$, which cuts the variance by the
  factor $1-\rho_{XY}^2$. Estimating $b$ from the same sample adds an $O(1/n)$ bias.

The coverage of the 95% intervals is measured over 1,000 independent repeats in
`scripts/03_validate_math.py`.

---

## 5. Volatility estimators — `volatility_estimators.py`

With daily log returns $x_t=\ln(S_t/S_{t-1})$ (365 days per year, because BTC trades every day):

- **Rolling:** $\hat\sigma_t=\sqrt{365}\cdot\mathrm{sd}(x_{t-w+1},\dots,x_t)$ (sample sd, $n-1$ divisor).
- **EWMA:** $v_t=\lambda v_{t-1}+(1-\lambda)x_t^2$, $\hat\sigma_t=\sqrt{365v_t}$
  (zero mean, effective memory $\approx 1/(1-\lambda)$ days).

Both use only $x_{\le t}$. The estimate dated $t$ is known at the 08:00 UTC fixing on day $t$.

---

## 6. Bayesian volatility: Normal–Inverse-Gamma — `bayesian.py`

### Model

$$x_i\mid\mu,s^2\overset{iid}{\sim}N(\mu,s^2),\qquad \mu\mid s^2\sim N(m_0,s^2/\kappa_0),\qquad s^2\sim\mathrm{IG}(a_0,b_0).$$

### Posterior derivation

The likelihood is
$\propto (s^2)^{-n/2}\exp\{-\frac{1}{2s^2}[\sum(x_i-\bar x)^2+n(\bar x-\mu)^2]\}$.
Multiply it by the prior density
$\propto (s^2)^{-1/2}e^{-\kappa_0(\mu-m_0)^2/2s^2}\,(s^2)^{-a_0-1}e^{-b_0/s^2}$.
Then complete the square in $\mu$:

$$\kappa_0(\mu-m_0)^2+n(\bar x-\mu)^2=\kappa_n(\mu-m_n)^2+\frac{\kappa_0n}{\kappa_n}(\bar x-m_0)^2,$$

with $\kappa_n=\kappa_0+n$ and $m_n=(\kappa_0m_0+n\bar x)/\kappa_n$. The joint posterior is
therefore NIG with

$$a_n=a_0+\tfrac n2,\qquad b_n=b_0+\tfrac12\sum(x_i-\bar x)^2+\frac{\kappa_0n(\bar x-m_0)^2}{2\kappa_n}.$$

Integrating out $\mu$ leaves $s^2\mid x\sim\mathrm{IG}(a_n,b_n)$, with
$E[s^2\mid x]=b_n/(a_n-1)$. Equal-tailed credible intervals come from the IG quantiles.

### Predictive distribution

For a new return, $x^*\mid\mu,s^2\sim N(\mu,s^2)$ and $\mu\mid s^2\sim N(m_n,s^2/\kappa_n)$, so
$x^*\mid s^2\sim N(m_n,s^2(1+1/\kappa_n))$. Mixing a normal over an inverse-gamma variance
gives a Student-t:

$$x^*\mid x\sim t_{2a_n}\!\left(m_n,\ \frac{b_n(\kappa_n+1)}{a_n\kappa_n}\right).$$

### Prior used

We parameterise the prior as a scaled inverse-$\chi^2$: $a_0=\nu_0/2$, $b_0=\nu_0s_0^2/2$,
$\kappa_0=\nu_0$, $m_0=0$. Here $\nu_0$ is the prior's weight in pseudo-observations and
$s_0^2$ is a prior guess of the daily variance. In Track B, $s_0^2$ is a multiple $m$ of the
**training-period** sample variance (returns up to 2021-12-31 only). The pair $(\nu_0,m)$ is
chosen on the validation period from $\{2,10,30,100\}\times\{0.5,1,2\}$, and the whole grid is
reported as a prior-sensitivity analysis.

### From posterior to option price

For a fixed $s^2$ the risk-neutral price is $BS(\sigma=\sqrt{365s^2})$. The **Bayesian
expected price** is

$$V_{\text{Bayes}}=E_{\text{post}}\big[BS(\sqrt{365\,s^2})\big]=\int BS(\sqrt{365v})\,f_{IG(a_n,b_n)}(v)\,dv,$$

and the **Bayesian delta** is $E_{\text{post}}[\Delta_{BS}]$ (differentiate under the integral).

This is **not** $BS(\sqrt{365\,E[s^2]})$ (the "plug-in"), because $BS$ is non-linear in
$\sigma$. ATM prices are almost linear in $\sigma$ but concave in $v=\sigma^2$. OTM prices are
convex in $\sigma$ (volga $>0$), so integrating over uncertainty raises OTM prices relative to
the plug-in. Both are reported separately (`bayes_plugin` vs `bayes_integrated`).

Each node of the integral is a lognormal risk-neutral model, so $V_{\text{Bayes}}$ is a mixture of
arbitrage-free prices. Put–call parity and convexity in $K$ still hold (tested). Note that
pricing directly with a Student-t for $\ln S_T$ would fail: $E[e^{X}]=\infty$ for a $t$
variable, so the discounted price could not be a martingale. Mixing *conditional* normals,
each with its own martingale drift, avoids this.

**Quadrature.** $E[g(v)]\approx\frac1N\sum_jg(F^{-1}_{IG}((j+\frac12)/N))$ (quantile
midpoints, each with mass $1/N$; $N=64$ for pricing, 32 inside the hedging loops). It is checked
against `scipy.integrate.quad` to a relative error of $10^{-3}$ in `tests/test_bayesian.py`.

*Assumptions:* iid normal returns inside the window (BTC returns are fat-tailed and
volatility clusters, so this is misspecified). Parameter uncertainty is treated as if it were
risk-neutral uncertainty, i.e. no premium for volatility-estimation risk.

---

## 7. Two-state regime model — `regime_model.py`

The hidden chain $s_t\in\{0,1\}$ has transition matrix $P$, and $x_t\mid s_t=k\sim N(\mu_k,\sigma_k^2)$.

- **Filtering** (forward algorithm): $\pi_{t\mid t}\propto(\pi_{t-1\mid t-1}P)\odot f(x_t\mid\cdot)$, normalised at each step.
- **Fitting:** maximum likelihood by EM (Baum–Welch). The E-step uses the forward filter and
  the backward smoother. The M-step uses weighted means and variances and normalised expected
  transition counts. The log-likelihood is non-decreasing (tested). It is fitted **once** on
  training returns, then frozen.
- **Forecast:** $\Pr(s_{t+j}\mid x_{\le t})=\pi_{t\mid t}P^j$, and the average variance over
  $h$ days is $\frac1h\sum_{j=1}^h\pi_{t\mid t}P^j(\sigma^2+\mu^2)$.
- **Pricing:** Track B uses the plug-in $BS(\hat\sigma_{\text{regime}}(h=\tau))$. In Track A the
  *true* price under the regime DGP uses the **Hull–White mixing** result. If volatility
  moves independently of the price shocks and volatility risk carries zero premium, then given
  the variance path $\ln S_T$ is normal with variance $\int\sigma_t^2dt$, so
  $V=E[BS(\bar\sigma)]$, where $\bar\sigma^2$ is the average variance over the option's life.
  This is computed by simulating 400,000 regime paths.

The course's stochastic-volatility section (§23) explains why the zero-premium assumption is
a *choice*. With a second Brownian motion the market is incomplete, and every
market price of volatility risk $\kappa$ gives a different risk-neutral measure $Q_\kappa$.

### Why no Heston model

Heston (§23) needs calibration to a cross-section of option quotes at one point in time.
Our real data are scattered trades (no bid/ask, one-hour windows), so calibrating five
parameters with $\rho$ and vol-of-vol would be poorly identified. The question here is about
*estimation* uncertainty in $\sigma$, which Heston does not address. It is listed as a
future extension.

---

## 8. Delta hedging — `hedging.py`

The hedger sells one option at the model's own price $V_0$ and holds $\delta_k$ shares over
$(t_k,t_{k+1}]$. Cash grows at $e^{r\Delta t}$. Proportional costs of $c|\Delta\delta|S$ are
paid on every trade, including the first purchase and the final unwind. The self-financing
wealth (course §10, §27) is

$$X_{k+1}=X_k+\delta_k(S_{k+1}-S_k)+r(X_k-\delta_kS_k)\Delta t\quad(\text{exactly: cash}\times e^{r\Delta t}).$$

**Hedge error** $=X_T-\text{payoff}$. Positive means a surplus; **loss** $=-$hedge error.
VaR$_{95}$ is the 95% empirical quantile of the loss, and ES$_{95}$ is the mean loss at or
beyond it.

Why the error is not zero: in the BS world, with the true $\sigma$ and continuous trading,
$X_T=$ payoff exactly. With discrete rebalancing the error has mean $\approx0$ and standard
deviation $\propto\sqrt{\Delta t}$ (tested: $16\times$ more rebalances gives an sd about
$4\times$ smaller). With a wrong $\hat\sigma$, the P&L accumulates
$\frac12\int\Gamma_tS_t^2(\sigma^2_{\text{realised}}-\hat\sigma^2)dt$. The mean of the error
therefore reflects pricing bias, and its sd reflects hedge-ratio quality plus gamma risk.

Test: the course's Microsoft replication table (§27) is reproduced exactly. The table
uses $T=0.08$ years; its 20 printed wealth values match to $10^{-3}$. Loss with hedging is
$0.1569$, versus $11.5410$ unhedged.

---

## 9. Statistics — `stat_tests.py`, `metrics.py`

- **Paired comparisons:** both models are evaluated on identical observations. Differences
  are A − B, so a negative MAE/RMSE difference means A is better.
- **Cluster bootstrap:** whole clusters are resampled with replacement (a market date in
  Track B pricing, a block of 5 consecutive weekly starts in Track B hedging, a path or
  history in Track A). Percentile 95% intervals. The two-sided p-value for "no difference in
  MSE" is $\Pr(|\theta^*-\hat\theta|\ge|\hat\theta|)$.
- **Diebold–Mariano:** $d_t=L_{A,t}-L_{B,t}$ on the per-date average squared error, with a
  Bartlett/Newey–West long-run variance, the Harvey–Leybourne–Newbold correction and a
  $t_{n-1}$ reference. It assumes covariance-stationary, weakly dependent $d_t$.
- **Holm–Bonferroni** adjustment within each pre-specified primary family.
- **QLIKE** volatility-forecast loss: $L=\frac{RV}{\hat\sigma^2}-\ln\frac{RV}{\hat\sigma^2}-1$.
  It is robust to noise in the realised-variance proxy.
