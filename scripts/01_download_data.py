"""Step 1: download raw data from Deribit's public API (resumable).

Usage:
    uv run python scripts/01_download_data.py --sample-only   # small pilot sample
    uv run python scripts/01_download_data.py                 # full bounded sample
"""

import argparse
from datetime import date

from volatility_lab.config import RAW_DIR, load_config
from volatility_lab.data_download import (
    download_delivery_prices,
    download_option_trade_sample,
    sample_days,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-only", action="store_true",
                        help="only download the few pilot days listed in the config")
    args = parser.parse_args()

    config = load_config()
    data_config = config["data"]

    print("Downloading daily index delivery prices...")
    download_delivery_prices(RAW_DIR, data_config["index_name"])

    if args.sample_only:
        days = [date.fromisoformat(day) for day in data_config["pilot_days"]]
    else:
        days = sample_days(
            date.fromisoformat(data_config["first_sample_day"]),
            date.fromisoformat(data_config["last_sample_day"]),
            weekday=data_config["sample_weekday"],
        )

    print(f"Downloading option trades for {len(days)} sample days...")
    summary = download_option_trade_sample(
        RAW_DIR, days,
        data_config["window_start_hour_utc"], data_config["window_end_hour_utc"],
    )
    print("Done:", summary)


if __name__ == "__main__":
    main()
