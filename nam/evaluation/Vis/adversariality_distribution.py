"""Plot unchanged sample-level adversariality distributions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.stats import gaussian_kde

from nam.evaluation.Vis.common import finite_values, import_plotting, read_metric_csv, save_figure


def plot_distributions(
    csv_paths: list[str], labels: list[str], metric: str, output: str
) -> dict[str, float]:
    """Draw comparable KDE curves and return the exact arithmetic means."""
    if labels and len(labels) != len(csv_paths):
        raise ValueError("--labels must have the same length as --csv.")
    labels = labels or [f"Set {index + 1}" for index in range(len(csv_paths))]
    series = [finite_values(read_metric_csv(path, metric), metric) for path in csv_paths]
    if any(values.size == 0 for values in series):
        raise ValueError("Every input CSV must contain at least one finite metric value.")

    plt = import_plotting()
    figure, axis = plt.subplots(figsize=(6.4, 4.5))
    combined = np.concatenate(series)
    lower, upper = float(combined.min()), float(combined.max())
    padding = max((upper - lower) * 0.05, 1e-3)
    grid = np.linspace(lower - padding, upper + padding, 512)
    means: dict[str, float] = {}
    for label, values in zip(labels, series):
        means[label] = float(values.mean())
        if values.size > 1 and not np.allclose(values, values[0]):
            density = gaussian_kde(values, bw_method="scott")(grid)
            axis.plot(grid, density, linewidth=2.0, label=f"{label} (mean={means[label]:.3f})")
            axis.fill_between(grid, density, alpha=0.12)
        else:
            axis.axvline(values[0], linewidth=2.0, label=f"{label} (mean={means[label]:.3f})")
    axis.set_xlabel(metric.upper())
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    save_figure(figure, output)
    plt.close(figure)
    return means


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot adversariality distributions from evaluation CSVs.")
    parser.add_argument("--csv", nargs="+", default=["outputs/evaluation/adversariality.csv"])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--metric", default="adv")
    parser.add_argument("--output", default="outputs/evaluation/Vis/adversariality_distribution.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(plot_distributions(arguments.csv, arguments.labels, arguments.metric, arguments.output))
