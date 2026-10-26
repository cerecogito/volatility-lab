"""Step 0: check that the free data sources are reachable (one tiny request each).

Usage:  uv run python scripts/00_check_sources.py
"""

from volatility_lab.data_download import DELIVERY_PRICES_URL, OPTION_TRADES_URL, fetch_json


def main() -> None:
    checks = {
        "Deribit delivery prices (btc_usd)": (DELIVERY_PRICES_URL,
                                              {"index_name": "btc_usd", "offset": 0, "count": 1}),
        "Deribit historical option trades": (OPTION_TRADES_URL,
                                             {"currency": "BTC", "kind": "option",
                                              "start_timestamp": 1717574400000,
                                              "end_timestamp": 1717578000000, "count": 1}),
    }
    for name, (url, params) in checks.items():
        try:
            payload, _ = fetch_json(url, params)
            print(f"OK    {name}: keys {sorted(payload['result'].keys())}")
        except RuntimeError as error:
            print(f"FAIL  {name}: {error}")


if __name__ == "__main__":
    main()
