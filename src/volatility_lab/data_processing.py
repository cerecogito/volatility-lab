"""Turning the raw Deribit JSON files into clean pandas tables.

The raw files are never modified - everything here can be re-run from
data/raw/ to rebuild data/processed/.

Conventions (see docs/data_dictionary.md):
* all timestamps are UTC;
* Deribit options expire at 08:00 UTC on the expiry date;
* option prices are quoted in BTC. USD price = BTC price * index price;
* time to expiry is in years of 365 calendar days (crypto trades 24/7);
* interest rate is set to 0 and the carry is absorbed in the forward price
  (this is Deribit's own convention for its implied vols).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from volatility_lab.implied_vol import implied_forward_from_vol

INSTRUMENT_PATTERN = re.compile(r"^(?P<currency>[A-Z]+)-(?P<day>\d{1,2})(?P<month>[A-Z]{3})(?P<year>\d{2})"
                                r"-(?P<strike>\d+(?:d\d+)?)-(?P<kind>[CP])$")
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
EXPIRY_HOUR_UTC = 8
SECONDS_PER_YEAR = 365.0 * 24 * 3600


# ---------------------------------------------------------------- index prices

def load_delivery_prices(raw_dir: Path, index_name: str = "btc_usd") -> pd.DataFrame:
    """Read all delivery-price pages into one table sorted by date.

    Returns columns: date (Timestamp, 08:00 UTC fixing), index_price.
    """
    page_files = sorted((raw_dir / "deribit" / "delivery_prices").glob(f"{index_name}_page_*.json"))
    if not page_files:
        raise FileNotFoundError("no delivery-price files - run scripts/01_download_data.py first")

    rows = []
    for page_file in page_files:
        payload = json.loads(page_file.read_text())
        rows.extend(payload["result"]["data"])

    prices = pd.DataFrame(rows).rename(columns={"delivery_price": "index_price"})
    prices["date"] = pd.to_datetime(prices["date"]) + pd.Timedelta(hours=EXPIRY_HOUR_UTC)
    prices["date"] = prices["date"].dt.tz_localize("UTC")
    prices = prices.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return prices


def daily_log_returns(prices: pd.DataFrame, first_date: str | None = None) -> pd.DataFrame:
    """Log returns between consecutive *calendar days* of the index.

    Early history is weekly, not daily, so a return is only kept when the two
    fixings are exactly one day apart. Returns columns: date, log_return,
    where `date` is the END of the return interval (the return is known at
    that 08:00 UTC fixing, never earlier).
    """
    data = prices.copy()
    if first_date is not None:
        data = data[data["date"] >= pd.Timestamp(first_date, tz="UTC")]
    data = data.sort_values("date")
    gap_days = data["date"].diff().dt.total_seconds() / 86400.0
    data["log_return"] = np.log(data["index_price"]).diff()
    daily = data[np.isclose(gap_days, 1.0)][["date", "log_return"]]
    return daily.reset_index(drop=True)


def first_date_of_unbroken_daily_history(prices: pd.DataFrame) -> pd.Timestamp:
    """The first date after which every calendar day has a fixing."""
    gaps = prices["date"].diff().dt.total_seconds() / 86400.0
    non_daily = np.where(~np.isclose(gaps, 1.0))[0]
    # index 0 is always NaN (no previous row), so the last bad gap tells us where daily data starts
    return prices["date"].iloc[non_daily[-1]] if len(non_daily) else prices["date"].iloc[0]


# --------------------------------------------------------------- option trades

def parse_instrument_name(name: str) -> dict | None:
    """'BTC-31MAR23-24000-C' -> currency, expiry (08:00 UTC), strike, option_type.

    Returns None if the name does not look like a standard option.
    """
    match = INSTRUMENT_PATTERN.match(name)
    if match is None:
        return None
    parts = match.groupdict()
    expiry = pd.Timestamp(year=2000 + int(parts["year"]), month=MONTHS[parts["month"]],
                          day=int(parts["day"]), hour=EXPIRY_HOUR_UTC, tz="UTC")
    strike = float(parts["strike"].replace("d", "."))
    return {"currency": parts["currency"], "expiry": expiry, "strike": strike,
            "option_type": "call" if parts["kind"] == "C" else "put"}


def load_raw_option_trades(raw_dir: Path) -> pd.DataFrame:
    """Read every saved trade window (all pages) into a single raw table, unchanged."""
    window_files = sorted((raw_dir / "deribit" / "option_trades").glob("*.json"))
    rows = []
    for window_file in window_files:
        payload = json.loads(window_file.read_text())
        for page in payload["pages"]:
            for trade in page["result"]["trades"]:
                trade = dict(trade)
                trade["source_file"] = window_file.name
                rows.append(trade)
    return pd.DataFrame(rows)


class ExclusionLog:
    """Records how many rows each cleaning rule removed, in order."""

    def __init__(self, starting_rows: int):
        self.steps = [{"step": "raw rows", "removed": 0, "remaining": starting_rows}]

    def apply(self, table: pd.DataFrame, keep_mask, step_name: str) -> pd.DataFrame:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        kept = table[keep_mask]
        self.steps.append({"step": step_name, "removed": int((~keep_mask).sum()),
                           "remaining": int(len(kept))})
        return kept

    def as_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def clean_option_trades(raw_trades: pd.DataFrame, filters: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the documented filtering rules. Returns (clean_trades, exclusion_table)."""
    log = ExclusionLog(len(raw_trades))
    trades = raw_trades.copy()

    # 1. pages can overlap at the same millisecond -> drop repeated trade ids
    trades = log.apply(trades, ~trades.duplicated(subset="trade_id"), "duplicate trade_id")

    # 2. instrument names must parse as standard options
    parsed = trades["instrument_name"].map(parse_instrument_name)
    trades = log.apply(trades, parsed.notna().to_numpy(), "unparseable instrument name")
    parsed = parsed[parsed.notna()]
    trades = pd.concat([trades.reset_index(drop=True),
                        pd.DataFrame(list(parsed)).reset_index(drop=True)], axis=1)

    # 3. missing or non-positive core fields
    for column in ["price", "index_price", "iv", "timestamp"]:
        if column not in trades:
            trades[column] = np.nan
    good_fields = (trades["price"] > 0) & (trades["index_price"] > 0) & (trades["iv"] > 0)
    trades = log.apply(trades, good_fields.fillna(False), "missing/non-positive price, index or iv")

    # 4. liquidations, block trades and combo legs are not normal screen trades
    #    (their prices can be negotiated or allocated across legs)
    for flag_column in ["liquidation", "block_trade_id", "combo_id"]:
        if flag_column in trades:
            trades = log.apply(trades, trades[flag_column].isna(), f"{flag_column} present")

    # 5. time to expiry
    trades["trade_time"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    trades["time_to_expiry"] = (trades["expiry"] - trades["trade_time"]).dt.total_seconds() / SECONDS_PER_YEAR
    days_left = trades["time_to_expiry"] * 365.0
    trades = log.apply(trades, (days_left >= filters["min_days_to_expiry"])
                       & (days_left <= filters["max_days_to_expiry"]), "days to expiry outside range")

    # 6. very cheap options: a one-tick move is a huge relative error
    trades = log.apply(trades, trades["price"] >= filters["min_price_btc"], "price below minimum")

    # 7. reported IV sanity range
    trades = log.apply(trades, (trades["iv"] >= filters["min_reported_iv"])
                       & (trades["iv"] <= filters["max_reported_iv"]), "reported iv outside range")

    # 8. recover Deribit's forward price from (price, reported iv); check it is sane
    trades = trades.copy()
    trades["price_usd"] = trades["price"] * trades["index_price"]
    trades["reported_iv"] = trades["iv"] / 100.0
    trades["forward"] = [
        implied_forward_from_vol(price, strike, tau, vol, kind, spot)
        for price, strike, tau, vol, kind, spot in zip(
            trades["price_usd"], trades["strike"], trades["time_to_expiry"],
            trades["reported_iv"], trades["option_type"], trades["index_price"])
    ]
    trades["annual_basis"] = np.log(trades["forward"] / trades["index_price"]) / trades["time_to_expiry"]
    basis_ok = trades["annual_basis"].abs() <= filters["max_abs_annual_basis"]
    trades = log.apply(trades, basis_ok.fillna(False), "forward recovery failed or basis implausible")

    # 9. moneyness
    trades["log_moneyness"] = np.log(trades["strike"] / trades["forward"])
    trades = log.apply(trades, trades["log_moneyness"].abs() <= filters["max_abs_log_moneyness"],
                       "|log moneyness| too large")

    trades["market_date"] = trades["trade_time"].dt.floor("D")
    keep_columns = ["trade_id", "trade_time", "market_date", "instrument_name", "option_type",
                    "strike", "expiry", "time_to_expiry", "price", "price_usd", "mark_price",
                    "index_price", "forward", "annual_basis", "log_moneyness", "reported_iv",
                    "direction", "amount", "source_file"]
    clean = trades[keep_columns].sort_values("trade_time").reset_index(drop=True)
    return clean, log.as_table()
