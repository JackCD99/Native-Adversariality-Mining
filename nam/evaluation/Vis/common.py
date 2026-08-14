"""Shared, data-preserving helpers for evaluation figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def read_metric_csv(path: str | Path, metric: str) -> dict[str, dict[str, Any]]:
    """Index one evaluation CSV by sample ID and validate the requested metric."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "id" not in reader.fieldnames or metric not in reader.fieldnames:
            raise KeyError(f"{source} must contain 'id' and '{metric}' columns.")
        rows: dict[str, dict[str, Any]] = {}
        for row in reader:
            sample_id = str(row["id"])
            if sample_id in rows:
                raise ValueError(f"Duplicate sample ID '{sample_id}' in {source}.")
            row[metric] = float(row[metric])
            rows[sample_id] = row
    return rows


def import_plotting():
    """Import optional plotting packages with an actionable install message."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install visualization dependencies with: pip install -e '.[visualization]'") from error
    return plt


def save_figure(figure: Any, output: str | Path, dpi: int = 300) -> list[str]:
    """Save matching vector PDF and review PNG files."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stem = destination.with_suffix("")
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=dpi, bbox_inches="tight")
    return [str(path) for path in paths]


def finite_values(rows: dict[str, dict[str, Any]], metric: str) -> np.ndarray:
    """Return finite metric values without clipping, perturbing, or resampling."""
    values = np.asarray([row[metric] for row in rows.values()], dtype=np.float64)
    return values[np.isfinite(values)]
