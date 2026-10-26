"""No-look-ahead tests: future observations must not change earlier decisions.

The idea of every test: compute something, then change all the data AFTER a
cut-off time, compute again, and check that nothing at or before the cut-off
moved. If a function secretly used future data, these tests would fail.
"""

import numpy as np
import pandas as pd

from volatility_lab.bayesian import make_prior, rolling_posteriors
from volatility_lab.pricing_models import (
    attach_latest_fixing, model_delta, regime_forecast_table, volatility_inputs_by_date,
)
from volatility_lab.regime_model import RegimeModel, forward_filter, forward_filter_paths
from volatility_lab.volatility_estimators import ewma_volatility, rolling_volatility

CUT = 150


def _returns_and_tampered(seed=0, length=300):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.03, length)
    tampered = returns.copy()
    tampered[CUT + 1:] = rng.normal(0, 0.3, length - CUT - 1)   # wildly different future
    return returns, tampered


def _toy_regime_model():
    return RegimeModel(means=np.zeros(2), stdevs=np.array([0.02, 0.06]),
                       transition=np.array([[0.97, 0.03], [0.06, 0.94]]),
                       initial_probs=np.array([0.6, 0.4]), log_likelihood=0.0, iterations=0, converged=True)


def test_estimators_ignore_the_future():
    original, tampered = _returns_and_tampered()
    for estimator in [lambda r: rolling_volatility(r, 30), lambda r: ewma_volatility(r, 0.94)]:
        np.testing.assert_array_equal(estimator(original)[: CUT + 1], estimator(tampered)[: CUT + 1])
    prior = make_prior(10.0, 0.0009)
    for key in ["alpha", "beta", "mean"]:
        np.testing.assert_array_equal(rolling_posteriors(prior, original, 60)[key][: CUT + 1],
                                      rolling_posteriors(prior, tampered, 60)[key][: CUT + 1])


def test_regime_filter_and_forecasts_ignore_the_future():
    original, tampered = _returns_and_tampered(1)
    model = _toy_regime_model()
    np.testing.assert_array_equal(forward_filter(original, model)[0][: CUT + 1],
                                  forward_filter(tampered, model)[0][: CUT + 1])
    np.testing.assert_array_equal(regime_forecast_table(original, model, 30)[: CUT + 1],
                                  regime_forecast_table(tampered, model, 30)[: CUT + 1])
    paths = np.vstack([original, tampered])
    filtered = forward_filter_paths(paths, model)
    np.testing.assert_array_equal(filtered[0, : CUT + 1], filtered[1, : CUT + 1])
    # the vectorised filter must agree with the single-path filter
    np.testing.assert_allclose(filtered[0], forward_filter(original, model)[0])


def test_volatility_table_rows_ignore_later_returns():
    original, tampered = _returns_and_tampered(2)
    dates = pd.date_range("2024-01-01 08:00", periods=len(original), freq="D", tz="UTC")
    prior = make_prior(10.0, 0.0009)
    table_a = volatility_inputs_by_date(pd.DataFrame({"date": dates, "log_return": original}), [30], [0.94], 30, prior)
    table_b = volatility_inputs_by_date(pd.DataFrame({"date": dates, "log_return": tampered}), [30], [0.94], 30, prior)
    pd.testing.assert_frame_equal(table_a.iloc[: CUT + 1], table_b.iloc[: CUT + 1])


def test_trades_only_see_fixings_at_or_before_the_trade():
    fixings = pd.DataFrame({"date": pd.date_range("2024-03-01 08:00", periods=4, freq="D", tz="UTC"),
                            "vol": [0.1, 0.2, 0.3, 0.4]})
    trades = pd.DataFrame({"trade_time": pd.to_datetime([
        "2024-03-02 08:30",   # after the 2 Mar fixing -> 0.2
        "2024-03-02 07:59",   # BEFORE the 2 Mar fixing -> must use 1 Mar (0.1), never 0.2
        "2024-03-04 08:00",   # exactly at the fixing -> allowed (known at that instant)
        "2024-03-09 08:30",   # last fixing is 5 days old -> too stale -> NaN
    ], utc=True)})
    merged = attach_latest_fixing(trades, fixings)
    assert merged["vol"].iloc[0] == 0.2
    assert merged["vol"].iloc[1] == 0.1
    assert merged["vol"].iloc[2] == 0.4
    assert np.isnan(merged["vol"].iloc[3])
    assert (merged["fixing_time"].dropna() <= merged["trade_time"][merged["fixing_time"].notna()]).all()
    # adding a FUTURE fixing must not change any existing match
    later = pd.concat([fixings, pd.DataFrame({"date": [pd.Timestamp("2024-03-10 08:00", tz="UTC")], "vol": [9.9]})])
    merged_later = attach_latest_fixing(trades.iloc[:3], later)
    np.testing.assert_array_equal(merged_later["vol"].to_numpy(), merged["vol"].iloc[:3].to_numpy())


def test_hedge_ratio_at_step_k_depends_only_on_data_up_to_k():
    """Simulate the Track A hedge-ratio pipeline on two paths that agree up to day k."""
    original, tampered = _returns_and_tampered(3, length=250)
    returns = np.vstack([original, tampered])
    prices = 100 * np.exp(np.cumsum(returns, axis=1))
    steps = np.arange(40, 250)                    # rebalance days
    vol = rolling_volatility(returns, 30)[:, steps]
    prior = make_prior(10.0, 0.0009)
    posterior = rolling_posteriors(prior, returns, 30)
    for name, kwargs in [("rolling", {"vol": vol}),
                         ("bayes_integrated", {"alpha": posterior["alpha"][:, steps],
                                               "beta": posterior["beta"][:, steps]})]:
        deltas = model_delta(name, prices[:, steps], 100.0, 0.1, "call", **kwargs)
        known_by_cut = steps <= CUT
        # the two paths agree up to CUT, so every hedge ratio chosen at or before CUT agrees...
        np.testing.assert_array_equal(deltas[0, known_by_cut], deltas[1, known_by_cut])
        # ...and the test is not vacuous: after CUT they really do differ
        assert np.any(deltas[0, ~known_by_cut] != deltas[1, ~known_by_cut])
