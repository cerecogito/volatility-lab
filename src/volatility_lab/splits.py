"""Chronological train / validation / test splits (never random shuffles)."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_NAMES = ["development", "validation", "test"]


def split_boundaries(split_config: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Read the [splits] config into {name: (start, end_inclusive)} in UTC days."""
    boundaries = {}
    for name in SPLIT_NAMES:
        start = pd.Timestamp(split_config[f"{name}_start"], tz="UTC")
        end = pd.Timestamp(split_config[f"{name}_end"], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        boundaries[name] = (start, end)
    check_boundaries(boundaries)
    return boundaries


def check_boundaries(boundaries: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    """Splits must be in time order and must not overlap."""
    previous_end = None
    for name in SPLIT_NAMES:
        start, end = boundaries[name]
        if start > end:
            raise ValueError(f"split {name} starts after it ends")
        if previous_end is not None and start <= previous_end:
            raise ValueError(f"split {name} overlaps the previous split")
        previous_end = end


def assign_split(times: pd.Series, boundaries: dict) -> np.ndarray:
    """Label every timestamp with its split name ('' if it is outside all splits)."""
    labels = np.full(len(times), "", dtype=object)
    for name, (start, end) in boundaries.items():
        labels[((times >= start) & (times <= end)).to_numpy()] = name
    return labels
