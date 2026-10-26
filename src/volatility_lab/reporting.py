"""Small helpers for turning saved experiment outputs into markdown and charts.

Nothing in here computes a result. It only formats numbers that were already
saved by the experiment scripts, so the reports cannot drift from the metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt  # noqa: E402

# reference categorical palette (fixed order, never cycled), light mode
MODEL_COLOURS = {
    "rolling": "#2a78d6",
    "ewma": "#eb6834",
    "bayes_plugin": "#1baf7a",
    "bayes_integrated": "#eda100",
    "regime": "#e87ba4",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def fmt(value: float, digits: int = 3) -> str:
    """Format a number with a fixed number of decimals (keeps tables tidy)."""
    return f"{value:.{digits}f}"


def fmt_p(p_value: float, replicates: int | None = None) -> str:
    """Bootstrap p-values of exactly 0 are reported as '< 1/replicates'."""
    if replicates and p_value == 0.0:
        return f"< {1 / replicates:.4f}"
    if p_value < 0.0001:
        return "< 0.0001"
    return f"{p_value:.4f}"


def fmt_ci(low: float, high: float, digits: int = 3) -> str:
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def markdown_table(headers: list[str], rows: list[list]) -> str:
    """Plain GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def style_axes(axes, title: str, x_label: str, y_label: str) -> None:
    """Recessive grid/axes, text in text colours (never series colours)."""
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=11)
    axes.set_xlabel(x_label, color=TEXT_SECONDARY)
    axes.set_ylabel(y_label, color=TEXT_SECONDARY)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY)


def new_figure(width: float = 8.0, height: float = 4.2):
    figure, axes = plt.subplots(figsize=(width, height), dpi=130)
    figure.patch.set_facecolor(SURFACE)
    return figure, axes


def save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)
