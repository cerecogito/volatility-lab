"""Downloading raw data from Deribit's public (no login) API.

Two things are downloaded:

1. The daily BTC index "delivery price" (the 08:00 UTC settlement value of
   the btc_usd index). This is our underlying-price history.
2. Historical BTC option *trades* in a fixed one-hour window on sampled
   days. Each trade has the instrument name (strike/expiry/call-put), trade
   price in BTC, Deribit's mark price, the IV Deribit reports and the index
   price at the time of the trade.

Every raw response is saved unchanged as a JSON file and recorded in a
manifest with its URL, retrieval time and SHA-256 checksum. If a download is
interrupted we can just run it again: files that already exist *and* match
their checksum in the manifest are skipped.

Deribit's terms say market data is for personal use only, so the raw files
stay in data/raw/ and are git-ignored (never committed or redistributed).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DELIVERY_PRICES_URL = "https://www.deribit.com/api/v2/public/get_delivery_prices"
OPTION_TRADES_URL = (
    "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
)

# be polite to a free public API: wait between requests
SECONDS_BETWEEN_REQUESTS = 0.6
MAX_RETRIES = 4
USER_AGENT = "volatility-lab-educational-research/0.1"


def sha256_of_file(file_path: Path) -> str:
    """Return the SHA-256 hex digest of a file (read in chunks)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_json(base_url: str, params: dict) -> tuple[dict, str]:
    """GET a JSON endpoint with simple retry + exponential backoff.

    Returns (parsed_json, full_url). Raises RuntimeError if every retry fails
    or the API reports an error.
    """
    full_url = base_url + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            request = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "error" in payload:
                raise RuntimeError(f"API error for {full_url}: {payload['error']}")
            return payload, full_url
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            wait_seconds = 2 ** (attempt + 1)
            print(f"  request failed ({error}); retrying in {wait_seconds}s")
            time.sleep(wait_seconds)

    raise RuntimeError(f"giving up on {full_url}: {last_error}")


class DownloadManifest:
    """Keeps track of every raw file: where it came from and its checksum."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        if manifest_path.exists():
            self.entries: dict[str, dict] = json.loads(manifest_path.read_text())
        else:
            self.entries = {}

    def is_complete(self, file_path: Path) -> bool:
        """True if the file exists and still matches the checksum we recorded."""
        key = file_path.name
        if key not in self.entries or not file_path.exists():
            return False
        return sha256_of_file(file_path) == self.entries[key]["sha256"]

    def record(self, file_path: Path, url: str, params: dict, record_count: int) -> None:
        self.entries[file_path.name] = {
            "url": url,
            "params": params,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": sha256_of_file(file_path),
            "bytes": file_path.stat().st_size,
            "record_count": record_count,
        }

    def save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.entries, indent=1, sort_keys=True))


def write_json_atomically(file_path: Path, payload: dict) -> None:
    """Write to a temp file first then rename, so a crash never leaves half a file."""
    temp_path = file_path.with_suffix(".partial")
    temp_path.write_text(json.dumps(payload))
    temp_path.replace(file_path)


def download_delivery_prices(raw_dir: Path, index_name: str = "btc_usd") -> Path:
    """Download the full daily delivery-price history for an index.

    The API is paged with offset/count, newest first. We store every page.
    Delivery prices are only ever appended, so re-downloading always fetches
    fresh pages (the history is small: a few requests).
    """
    out_dir = raw_dir / "deribit" / "delivery_prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = DownloadManifest(raw_dir / "deribit" / "manifest_delivery_prices.json")

    page_size = 1000
    offset = 0
    page_number = 0
    while True:
        params = {"index_name": index_name, "offset": offset, "count": page_size}
        payload, url = fetch_json(DELIVERY_PRICES_URL, params)
        rows = payload["result"]["data"]
        total_records = payload["result"]["records_total"]

        page_file = out_dir / f"{index_name}_page_{page_number:03d}.json"
        write_json_atomically(page_file, payload)
        manifest.record(page_file, url, params, record_count=len(rows))
        print(f"  delivery prices page {page_number}: {len(rows)} rows")

        offset += len(rows)
        page_number += 1
        if len(rows) == 0 or offset >= total_records:
            break
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    manifest.save()
    return out_dir


def to_milliseconds(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def download_option_trades_for_window(
    raw_dir: Path,
    sample_day: date,
    start_hour_utc: int,
    end_hour_utc: int,
    manifest: DownloadManifest,
    currency: str = "BTC",
) -> tuple[Path, int, bool]:
    """Download all option trades in [start_hour, end_hour) UTC on one day.

    Pages through the API (1000 trades per request) by moving the start
    timestamp forward past the last trade we received. All pages are saved
    together in one JSON file for the window.

    Returns (file_path, number_of_trades, was_skipped_because_already_done).
    """
    out_dir = raw_dir / "deribit" / "option_trades"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{currency}_{sample_day.isoformat()}_{start_hour_utc:02d}-{end_hour_utc:02d}UTC.json"
    window_file = out_dir / file_name

    if manifest.is_complete(window_file):
        return window_file, manifest.entries[file_name]["record_count"], True

    window_start = datetime(sample_day.year, sample_day.month, sample_day.day,
                            start_hour_utc, tzinfo=timezone.utc)
    window_end = window_start + timedelta(hours=end_hour_utc - start_hour_utc)

    all_pages = []
    all_trade_ids: set[str] = set()
    page_start_ms = to_milliseconds(window_start)
    end_ms = to_milliseconds(window_end) - 1  # end is exclusive
    first_url = ""
    first_params: dict = {}

    while True:
        params = {
            "currency": currency,
            "kind": "option",
            "start_timestamp": page_start_ms,
            "end_timestamp": end_ms,
            "count": 1000,
            "sorting": "asc",
        }
        payload, url = fetch_json(OPTION_TRADES_URL, params)
        if not first_url:
            first_url, first_params = url, params
        all_pages.append(payload)

        trades = payload["result"]["trades"]
        for trade in trades:
            all_trade_ids.add(trade["trade_id"])

        if not payload["result"]["has_more"] or len(trades) == 0:
            break
        # next page starts at the last timestamp we saw. Trades sharing that
        # millisecond may come back twice - we de-duplicate by trade_id later.
        page_start_ms = trades[-1]["timestamp"]
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    write_json_atomically(window_file, {"pages": all_pages})
    manifest.record(window_file, first_url, first_params, record_count=len(all_trade_ids))
    return window_file, len(all_trade_ids), False


def sample_days(first_day: date, last_day: date, weekday: int) -> list[date]:
    """Every date between first_day and last_day (inclusive) that falls on `weekday` (Mon=0)."""
    days = []
    current = first_day
    while current.weekday() != weekday:
        current += timedelta(days=1)
    while current <= last_day:
        days.append(current)
        current += timedelta(days=7)
    return days


def download_option_trade_sample(
    raw_dir: Path,
    days_to_download: list[date],
    start_hour_utc: int,
    end_hour_utc: int,
) -> dict:
    """Download the trade windows for a list of days (resumable). Returns a summary."""
    manifest = DownloadManifest(raw_dir / "deribit" / "manifest_option_trades.json")
    downloaded, skipped, total_trades = 0, 0, 0

    for day_number, sample_day in enumerate(days_to_download):
        _, trade_count, was_skipped = download_option_trades_for_window(
            raw_dir, sample_day, start_hour_utc, end_hour_utc, manifest
        )
        total_trades += trade_count
        if was_skipped:
            skipped += 1
            continue

        downloaded += 1
        print(f"  [{day_number + 1}/{len(days_to_download)}] {sample_day}: {trade_count} trades")
        # save the manifest as we go so an interruption loses at most one window
        manifest.save()
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    manifest.save()
    return {"windows_downloaded": downloaded, "windows_skipped": skipped,
            "total_trades": total_trades}
