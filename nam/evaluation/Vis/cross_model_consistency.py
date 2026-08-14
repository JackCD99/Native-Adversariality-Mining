"""Measure and visualize adversariality consistency across downstream models."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from nam.evaluation.Vis.common import import_plotting, read_metric_csv, save_figure


def plot_consistency(
    csv_paths: list[str], model_names: list[str], metric: str, output: str
) -> list[dict[str, float | int | str]]:
    """Align exact sample IDs, draw pairwise plots, and save correlation statistics."""
    if len(csv_paths) < 2:
        raise ValueError("Cross-model consistency requires at least two CSV files.")
    if model_names and len(model_names) != len(csv_paths):
        raise ValueError("--model-names must have the same length as --csv.")
    model_names = model_names or [f"Model {index + 1}" for index in range(len(csv_paths))]
    tables = [read_metric_csv(path, metric) for path in csv_paths]
    shared_ids = set(tables[0])
    for table in tables[1:]:
        shared_ids &= set(table)
    identifiers = sorted(shared_ids)
    if len(identifiers) < 2:
        raise ValueError("Fewer than two sample IDs are shared by all model CSVs.")
    for identifier in identifiers:
        paths = {str(table[identifier].get("path", "")) for table in tables}
        paths.discard("")
        if len(paths) > 1:
            raise ValueError(
                f"Sample ID '{identifier}' refers to different paths across model CSVs."
            )

    plt = import_plotting()
    comparison_count = len(tables) - 1
    figure, axes = plt.subplots(1, comparison_count, figsize=(5.0 * comparison_count, 4.5), squeeze=False)
    reference = np.asarray([tables[0][key][metric] for key in identifiers], dtype=float)
    results: list[dict[str, float | int | str]] = []
    for index, (table, name) in enumerate(zip(tables[1:], model_names[1:])):
        compared = np.asarray([table[key][metric] for key in identifiers], dtype=float)
        valid = np.isfinite(reference) & np.isfinite(compared)
        x, y = reference[valid], compared[valid]
        if x.size < 2:
            raise ValueError(f"Insufficient finite paired values for {model_names[0]} and {name}.")
        pearson = float(pearsonr(x, y).statistic) if not (np.allclose(x, x[0]) or np.allclose(y, y[0])) else float("nan")
        spearman = float(spearmanr(x, y).statistic) if not (np.allclose(x, x[0]) or np.allclose(y, y[0])) else float("nan")
        result = {
            "reference_model": model_names[0],
            "compared_model": name,
            "metric": metric,
            "samples": int(x.size),
            "pearson": pearson,
            "spearman": spearman,
            "mean_absolute_difference": float(np.mean(np.abs(x - y))),
        }
        results.append(result)
        axis = axes[0, index]
        axis.scatter(x, y, s=18, alpha=0.55, edgecolors="none")
        low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        axis.set_xlabel(f"{model_names[0]} {metric.upper()}")
        axis.set_ylabel(f"{name} {metric.upper()}")
        axis.set_title(f"Pearson={pearson:.3f} | Spearman={spearman:.3f}")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)

    summary_path = Path(output).with_suffix("").with_name(Path(output).stem + "_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cross-model adversariality consistency.")
    parser.add_argument("--csv", nargs="+", default=["outputs/evaluation/nnunet_adv.csv", "outputs/evaluation/swinunet_adv.csv"])
    parser.add_argument("--model-names", nargs="*", default=["nnU-Net", "Swin-Unet"])
    parser.add_argument("--metric", default="adv")
    parser.add_argument("--output", default="outputs/evaluation/Vis/cross_model_consistency.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(plot_consistency(arguments.csv, arguments.model_names, arguments.metric, arguments.output))
