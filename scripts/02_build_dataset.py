"""Step 2: validate the raw files and rebuild data/processed/ from them.

Usage:  uv run python scripts/02_build_dataset.py

Writes
    data/processed/index_daily.csv          daily 08:00 UTC index fixings + log returns
    data/processed/option_trades_clean.csv.gz
    data/processed/exclusions.csv            rows removed by each cleaning rule
    data/processed/data_manifest.json        checksums, coverage and counts
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from volatility_lab.config import PROCESSED_DIR, RAW_DIR, load_config
from volatility_lab.data_download import DownloadManifest, sha256_of_file
from volatility_lab.data_processing import (
    clean_option_trades, daily_log_returns, first_date_of_unbroken_daily_history,
    load_delivery_prices, load_raw_option_trades,
)
from volatility_lab.splits import assign_split, split_boundaries


def verify_raw_checksums() -> dict:
    """Re-hash every raw file and compare with the download manifests."""
    report = {}
    for manifest_name, folder in [("manifest_delivery_prices.json", "delivery_prices"),
                                  ("manifest_option_trades.json", "option_trades")]:
        manifest = DownloadManifest(RAW_DIR / "deribit" / manifest_name)
        files = sorted((RAW_DIR / "deribit" / folder).glob("*.json"))
        mismatches = [f.name for f in files if not manifest.is_complete(f)]
        if mismatches:
            raise RuntimeError(f"{len(mismatches)} raw files do not match {manifest_name}: {mismatches[:3]}")
        retrieved = sorted(entry["retrieved_utc"] for entry in manifest.entries.values())
        report[folder] = {"files": len(files), "checksum_mismatches": 0,
                          "first_retrieved_utc": retrieved[0], "last_retrieved_utc": retrieved[-1],
                          "source_url": next(iter(manifest.entries.values()))["url"].split("?")[0]}
    return report


def main() -> None:
    config = load_config()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Verifying raw checksums...")
    raw_report = verify_raw_checksums()

    # ---------------- underlying index
    prices = load_delivery_prices(RAW_DIR, config["data"]["index_name"])
    daily_start = first_date_of_unbroken_daily_history(prices)
    returns = daily_log_returns(prices, first_date=str(daily_start.date()))
    assert returns["date"].is_monotonic_increasing and returns["date"].is_unique
    assert np.all(np.isfinite(returns["log_return"]))
    index_daily = prices.merge(returns, on="date", how="left")
    index_daily.to_csv(PROCESSED_DIR / "index_daily.csv", index=False)

    # ---------------- option trades
    raw_trades = load_raw_option_trades(RAW_DIR)
    clean, exclusions = clean_option_trades(raw_trades, config["filters"])
    boundaries = split_boundaries(config["splits"])
    clean["split"] = assign_split(clean["trade_time"], boundaries)
    clean.to_csv(PROCESSED_DIR / "option_trades_clean.csv.gz", index=False)
    exclusions.to_csv(PROCESSED_DIR / "exclusions.csv", index=False)

    # sanity checks on the cleaned data
    assert clean["trade_id"].is_unique
    assert (clean["time_to_expiry"] > 0).all() and (clean["forward"] > 0).all()
    assert (clean["trade_time"] < clean["expiry"]).all()

    split_counts = {}
    for name in ["development", "validation", "test"]:
        part = clean[clean["split"] == name]
        split_counts[name] = {"trades": int(len(part)),
                              "market_dates": int(part["market_date"].nunique()),
                              "instruments": int(part["instrument_name"].nunique()),
                              "first": str(part["trade_time"].min()), "last": str(part["trade_time"].max())}

    manifest = {
        "experiment_id": config["experiment_id"],
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_inputs": raw_report,
        "raw_manifest_sha256": {
            name: sha256_of_file(RAW_DIR / "deribit" / name)
            for name in ["manifest_delivery_prices.json", "manifest_option_trades.json"]
        },
        "index": {
            "rows": int(len(prices)), "first_fixing": str(prices["date"].min()),
            "last_fixing": str(prices["date"].max()), "daily_history_starts": str(daily_start),
            "daily_returns": int(len(returns)),
            "duplicate_dates": int(prices["date"].duplicated().sum()),
        },
        "option_trades": {
            "raw_rows": int(len(raw_trades)),
            "raw_unique_trade_ids": int(raw_trades["trade_id"].nunique()),
            "raw_instruments": int(raw_trades["instrument_name"].nunique()),
            "sample_windows": int(raw_trades["source_file"].nunique()),
            "clean_rows": int(len(clean)),
            "clean_instruments": int(clean["instrument_name"].nunique()),
            "clean_market_dates": int(clean["market_date"].nunique()),
            "first_trade": str(clean["trade_time"].min()), "last_trade": str(clean["trade_time"].max()),
            "calls": int((clean["option_type"] == "call").sum()),
            "puts": int((clean["option_type"] == "put").sum()),
            "outside_all_splits": int((clean["split"] == "").sum()),
            "exclusions": exclusions.to_dict(orient="records"),
            "splits": split_counts,
        },
        "conventions": {
            "timestamps": "UTC", "expiry": "08:00 UTC on expiry date",
            "option_price_unit": "BTC per 1 BTC notional; price_usd = price * index_price",
            "time_unit": "years of 365 calendar days", "interest_rate": 0.0,
            "forward": "recovered per trade from (price, Deribit reported iv) with Black-76",
        },
        "split_boundaries": {name: [str(a), str(b)] for name, (a, b) in boundaries.items()},
    }
    (PROCESSED_DIR / "data_manifest.json").write_text(json.dumps(manifest, indent=1))

    print(exclusions.to_string(index=False))
    print(pd.DataFrame(split_counts).T.to_string())
    print(f"index: {len(prices)} fixings, daily returns from {daily_start.date()} ({len(returns)} returns)")


if __name__ == "__main__":
    main()
