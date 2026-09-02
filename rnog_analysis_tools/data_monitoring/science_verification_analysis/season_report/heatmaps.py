"""Categorical channel-health heatmaps built from per-run SVA summaries."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


STATUS_VALUE = {"-": -2, "?": -1, "": -1, "OK": 0, "!!": 1, "X": 2}
STATUS_COLORS = ["#ffffff", "#bdbdbd", "#4daf4a", "#ffbf00", "#d73027"]


def generate_health_heatmaps(combined_csv: Path, output_dir: Path) -> dict[str, Path]:
    """Generate one run-by-channel heatmap for every result column."""
    frame = pd.read_csv(combined_csv, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"No channel-health rows found in {combined_csv}")
    frame["Run"] = frame["Run"].astype(int)
    frame["Channel"] = frame["Channel"].astype(int)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_columns = [column for column in frame.columns if column not in ("Station", "Run", "Channel")]
    paths = {}
    for column in result_columns:
        filename = _filename_for(column) + "_by_run.png"
        path = output_dir / filename
        _plot_column(frame, column, path)
        paths[column] = path
    return paths


def _plot_column(frame: pd.DataFrame, column: str, output_path: Path) -> None:
    runs = sorted(frame["Run"].unique())
    channels = sorted(frame["Channel"].unique())
    run_index = {run: index for index, run in enumerate(runs)}
    channel_index = {channel: index for index, channel in enumerate(channels)}
    matrix = np.full((len(channels), len(runs)), -1, dtype=int)

    for row in frame[["Run", "Channel", column]].itertuples(index=False, name=None):
        run, channel, status = row
        matrix[channel_index[channel], run_index[run]] = STATUS_VALUE.get(str(status).strip(), -1)

    width = max(12, min(42, len(runs) * 0.09))
    figure, axis = plt.subplots(figsize=(width, 8))
    cmap = ListedColormap(STATUS_COLORS)
    norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    axis.set_yticks(np.arange(len(channels)), labels=[str(channel) for channel in channels])
    tick_step = max(1, len(runs) // 20)
    ticks = np.arange(0, len(runs), tick_step)
    axis.set_xticks(ticks, labels=[str(runs[index]) for index in ticks], rotation=40, ha="right")
    axis.set_xlabel("Run number")
    axis.set_ylabel("Channel")
    axis.set_title(f"{column} by run")
    colorbar = figure.colorbar(image, ax=axis, ticks=[-2, -1, 0, 1, 2], pad=0.01)
    colorbar.ax.set_yticklabels(["N/A", "Missing", "OK", "Warning", "Failure"])
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _filename_for(label: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return value or "health"
