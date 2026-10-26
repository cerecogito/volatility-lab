"""Consistency tests: reported numbers must equal the saved metrics and the processed inputs.

These run only after the pipeline has produced its outputs (they are skipped
on a fresh clone without data).
"""

import json

import numpy as np
import pandas as pd
import pytest

from volatility_lab.config import EXPERIMENTS_DIR, PROCESSED_DIR, REPORTS_DIR
from volatility_lab.simulation import simulate_gbm_returns, simulate_regime_returns

needs_outputs = pytest.mark.skipif(
    not (REPORTS_DIR / "cv_evidence.json").exists() or not (PROCESSED_DIR / "data_manifest.json").exists(),
    reason="pipeline outputs not generated yet (run scripts/run_all.py)",
)


def _load(path):
    return json.loads(path.read_text())


@needs_outputs
def test_manifest_counts_match_processed_files():
    manifest = _load(PROCESSED_DIR / "data_manifest.json")
    trades = pd.read_csv(PROCESSED_DIR / "option_trades_clean.csv.gz", usecols=["trade_id", "split", "market_date"])
    assert len(trades) == manifest["option_trades"]["clean_rows"]
    assert trades["trade_id"].is_unique
    for split, counts in manifest["option_trades"]["splits"].items():
        assert int((trades["split"] == split).sum()) == counts["trades"]
    exclusions = pd.read_csv(PROCESSED_DIR / "exclusions.csv")
    assert exclusions["remaining"].iloc[-1] == len(trades)
    assert exclusions["remaining"].iloc[0] - exclusions["removed"].sum() == len(trades)


@needs_outputs
def test_reported_test_sample_size_matches_processed_inputs():
    b = _load(EXPERIMENTS_DIR / "track_b_pricing.json")
    trades = pd.read_csv(PROCESSED_DIR / "option_trades_clean.csv.gz", usecols=["split"])
    excluded = b["exclusions_before_evaluation"]["missing_index_fixing_or_insufficient_history"]
    scored = b["results"]["validation"]["n_trades"] + b["results"]["test"]["n_trades"]
    # every validation/test trade is either scored or counted as excluded for missing history
    in_scope = int(trades["split"].isin(["validation", "test"]).sum())
    assert scored <= in_scope
    assert in_scope - scored <= excluded
    # paired comparisons use exactly the scored test trades
    for comparison in b["results"]["test"]["comparisons"]:
        assert comparison["n_observations"] == b["results"]["test"]["n_trades"]
        assert comparison["n_clusters"] == b["results"]["test"]["n_market_dates"]


@needs_outputs
def test_cv_evidence_matches_saved_metrics():
    evidence = _load(REPORTS_DIR / "cv_evidence.json")
    b = _load(EXPERIMENTS_DIR / "track_b_pricing.json")
    claims = {c["claim_id"]: c for c in evidence["claims"]}
    for model, summary in b["results"]["test"]["summary"].items():
        assert claims["H1_test_rmse_by_model"]["values"][model] == summary["pct_of_spot"]["rmse"]
    first = b["results"]["test"]["comparisons"][0]
    assert (first["model_a"], first["model_b"]) == ("bayes_integrated", "rolling")
    assert claims["H2_bayes_vs_rolling"]["rmse_diff"] == first["rmse_diff"]
    assert claims["H2_bayes_vs_rolling"]["ci95"] == [first["rmse_diff_ci_low"], first["rmse_diff_ci_high"]]
    assert claims["H1_test_rmse_by_model"]["n_trades"] == b["results"]["test"]["n_trades"]
    # every claim says whether it is simulated or historical
    assert all(c["evidence_type"] for c in evidence["claims"])
    # checksums are carried through from the data manifest
    manifest = _load(PROCESSED_DIR / "data_manifest.json")
    assert evidence["source_checksums"]["raw_manifest_sha256"] == manifest["raw_manifest_sha256"]


@needs_outputs
def test_markdown_reports_quote_the_saved_numbers():
    summary = (REPORTS_DIR / "results_summary.md").read_text(encoding="utf-8")
    report = (REPORTS_DIR / "final_research_report.md").read_text(encoding="utf-8")
    b = _load(EXPERIMENTS_DIR / "track_b_pricing.json")
    test = b["results"]["test"]
    assert f"{test['n_trades']:,}" in summary and f"{test['n_trades']:,}" in report
    for values in test["summary"].values():
        assert f"{values['pct_of_spot']['rmse']:.4f}" in summary
        assert f"{values['pct_of_spot']['rmse']:.3f}" in report
    hedging = _load(EXPERIMENTS_DIR / "track_b_hedging.json")
    assert f"**{hedging['n_hedges']}**" in summary


def test_simulations_are_reproducible_from_the_seed():
    first = simulate_gbm_returns(50, 20, 0.05, 0.6, np.random.default_rng(123))
    second = simulate_gbm_returns(50, 20, 0.05, 0.6, np.random.default_rng(123))
    other = simulate_gbm_returns(50, 20, 0.05, 0.6, np.random.default_rng(124))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    transition = np.array([[0.98, 0.02], [0.05, 0.95]])
    regime_a = simulate_regime_returns(20, 30, (0.4, 1.0), transition, np.random.default_rng(5))
    regime_b = simulate_regime_returns(20, 30, (0.4, 1.0), transition, np.random.default_rng(5))
    np.testing.assert_array_equal(regime_a[0], regime_b[0])
    np.testing.assert_array_equal(regime_a[1], regime_b[1])
