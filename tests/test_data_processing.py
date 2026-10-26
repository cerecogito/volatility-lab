"""Tests for instrument parsing, cleaning rules, splits and download bookkeeping."""

import json

import numpy as np
import pandas as pd
import pytest

from volatility_lab.black_scholes import bs_price
from volatility_lab.data_download import DownloadManifest, sample_days, write_json_atomically
from volatility_lab.data_processing import (
    clean_option_trades, daily_log_returns, first_date_of_unbroken_daily_history, parse_instrument_name,
)
from volatility_lab.splits import assign_split, check_boundaries, split_boundaries

FILTERS = {"min_days_to_expiry": 3.0, "max_days_to_expiry": 180.0, "min_price_btc": 0.0005,
           "max_abs_log_moneyness": 0.5, "min_reported_iv": 5.0, "max_reported_iv": 300.0,
           "max_abs_annual_basis": 0.5}


def test_parse_instrument_names():
    parsed = parse_instrument_name("BTC-8JAN21-24000-P")
    assert parsed["expiry"] == pd.Timestamp("2021-01-08 08:00", tz="UTC")
    assert parsed["strike"] == 24000.0 and parsed["option_type"] == "put"
    assert parse_instrument_name("BTC-31MAR23-24000-C")["option_type"] == "call"
    assert parse_instrument_name("BTC-PERPETUAL") is None
    assert parse_instrument_name("BTC-31MAR23") is None


def _fake_trade(trade_id, instrument, price_btc, iv, index_price, timestamp, **extra):
    return {"trade_id": trade_id, "instrument_name": instrument, "price": price_btc, "iv": iv,
            "index_price": index_price, "timestamp": timestamp, "mark_price": price_btc,
            "direction": "buy", "amount": 1.0, "source_file": "test.json", **extra}


def test_cleaning_rules_and_forward_recovery():
    trade_time = pd.Timestamp("2023-03-01 08:30", tz="UTC")
    expiry = pd.Timestamp("2023-03-31 08:00", tz="UTC")
    tau = (expiry - trade_time).total_seconds() / (365 * 24 * 3600)
    true_forward, index_price, iv = 23_200.0, 23_000.0, 0.5
    price_usd = float(bs_price(true_forward, 24_000.0, tau, 0.0, iv, "call"))
    good = _fake_trade("1", "BTC-31MAR23-24000-C", price_usd / index_price, 50.0, index_price,
                       int(trade_time.timestamp() * 1000))
    rows = [
        good,
        dict(good),                                                  # duplicate id
        {**good, "trade_id": "2", "price": 0.0001},                  # too cheap
        {**good, "trade_id": "3", "instrument_name": "BTC-2MAR23-24000-C"},   # < 3 days to expiry
        {**good, "trade_id": "4", "block_trade_id": "B1"},           # block trade
        {**good, "trade_id": "5", "iv": np.nan},                     # missing iv
        {**good, "trade_id": "6", "instrument_name": "BTC-31MAR23-60000-C"},  # far OTM
    ]
    raw = pd.DataFrame(rows)
    raw["block_trade_id"] = raw.get("block_trade_id")
    clean, exclusions = clean_option_trades(raw, FILTERS)
    assert list(clean["trade_id"]) == ["1"]
    assert clean["forward"].iloc[0] == pytest.approx(true_forward, rel=1e-6)
    assert exclusions["remaining"].iloc[-1] == 1
    # every removed row is accounted for by exactly one rule
    assert exclusions["removed"].sum() == len(raw) - len(clean)


def test_daily_returns_skip_weekly_gaps():
    dates = pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-09", "2020-01-10"]).tz_localize("UTC")
    prices = pd.DataFrame({"date": dates, "index_price": [100.0, 110.0, 121.0, 108.9]})
    returns = daily_log_returns(prices)
    assert len(returns) == 2                       # the 7-day gap is not a daily return
    assert returns["log_return"].iloc[0] == pytest.approx(np.log(1.1))
    assert first_date_of_unbroken_daily_history(prices) == dates[1]


SPLITS = {"development_start": "2020-01-01", "development_end": "2021-12-31",
          "validation_start": "2022-01-01", "validation_end": "2023-12-31",
          "test_start": "2024-01-01", "test_end": "2026-09-30"}


def test_splits_are_chronological_and_disjoint():
    boundaries = split_boundaries(SPLITS)
    times = pd.Series(pd.to_datetime(["2021-12-31 23:59", "2022-01-01 00:00", "2023-12-31 12:00",
                                      "2024-01-01 08:00", "2030-01-01 00:00"], utc=True))
    labels = assign_split(times, boundaries)
    assert list(labels) == ["development", "validation", "validation", "test", ""]
    # every validation time is after every development time, every test time after validation
    assert boundaries["development"][1] < boundaries["validation"][0]
    assert boundaries["validation"][1] < boundaries["test"][0]


def test_overlapping_splits_are_rejected():
    overlapping = dict(SPLITS, validation_start="2021-06-01")     # starts before development ends
    with pytest.raises(ValueError):
        split_boundaries(overlapping)
    backwards = {"development": (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC")),
                 "validation": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-06-01", tz="UTC")),
                 "test": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-06-01", tz="UTC"))}
    with pytest.raises(ValueError):
        check_boundaries(backwards)


def test_sample_days_are_wednesdays():
    days = sample_days(pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-31").date(), weekday=2)
    assert [d.day for d in days] == [3, 10, 17, 24, 31]


def test_manifest_detects_changed_files(tmp_path):
    file_path = tmp_path / "window.json"
    write_json_atomically(file_path, {"pages": []})
    manifest = DownloadManifest(tmp_path / "manifest.json")
    manifest.record(file_path, "https://example.org", {}, record_count=0)
    manifest.save()
    reloaded = DownloadManifest(tmp_path / "manifest.json")
    assert reloaded.is_complete(file_path)
    file_path.write_text(json.dumps({"pages": ["tampered"]}))
    assert not reloaded.is_complete(file_path)       # a corrupted/partial file would be re-downloaded
