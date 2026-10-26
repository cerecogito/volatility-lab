"""Loading the experiment configuration and project paths."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.toml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
FIGURES_DIR = REPORTS_DIR / "figures"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Read the TOML config into a plain dictionary."""
    with open(config_path, "rb") as config_file:
        return tomllib.load(config_file)
