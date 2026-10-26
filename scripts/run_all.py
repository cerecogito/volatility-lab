"""Run the whole study in order (each step can also be run on its own).

Usage:
    uv run python scripts/run_all.py              # everything, using already-downloaded raw data
    uv run python scripts/run_all.py --download   # also (re)download raw data first (resumable)
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STEPS = [
    ("check sources", "00_check_sources.py"),
    ("build + validate dataset", "02_build_dataset.py"),
    ("mathematical validation", "03_validate_math.py"),
    ("Track A simulation", "04_track_a_simulation.py"),
    ("Track B market pricing", "05_track_b_market_pricing.py"),
    ("Track B historical hedging", "06_track_b_historical_hedging.py"),
    ("charts + reports", "07_make_report.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="run 01_download_data.py first")
    args = parser.parse_args()
    steps = list(STEPS)
    if args.download:
        steps.insert(1, ("download raw data", "01_download_data.py"))

    for name, script in steps:
        print(f"\n=== {name} ({script}) ===", flush=True)
        completed = subprocess.run([sys.executable, str(SCRIPTS / script)])
        if completed.returncode != 0:
            raise SystemExit(f"step '{name}' failed - stopping")
    print("\nall steps finished; now run: uv run pytest")


if __name__ == "__main__":
    main()
