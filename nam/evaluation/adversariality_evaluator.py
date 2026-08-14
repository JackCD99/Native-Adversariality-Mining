"""Fixed-budget adversariality evaluation for generated medical datasets.

LCBCE is the default distribution-analysis proxy because it prevents large
structures from dominating per-sample rankings. NAM optimization itself uses
ordinary cross-entropy by default, following the main experimental protocol.
The module also reports conventional cross-entropy and soft Dice losses. All
proxy columns are oriented so that a larger value means a harder sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from nam.config import apply_overrides, load_config
from nam.data import NAMBatch, build_dataset, collate_medical_batch
from nam.downstream import build_downstream


LossFunction = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


def _prepare_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Convert canonical masks to integer labels at the prediction resolution."""
    if logits.ndim == 2:
        return target.reshape(-1).long()
    if target.ndim == logits.ndim and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape[1:] != logits.shape[2:]:
        target = F.interpolate(target.unsqueeze(1).float(), size=logits.shape[2:], mode="nearest")[:, 0]
    return target.long()


def _class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return two-class probabilities for binary logits and softmax otherwise."""
    if logits.shape[1] == 1:
        foreground = torch.sigmoid(logits[:, 0])
        return torch.stack((1.0 - foreground, foreground), dim=1)
    return torch.softmax(logits, dim=1)


def cross_entropy_loss(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Return ordinary per-sample cross entropy before score normalization."""
    target = _prepare_target(logits, target)
    probabilities = _class_probabilities(logits).clamp_min(1e-7)
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0).clamp(0, probabilities.shape[1] - 1)
    nll = -probabilities.log().gather(1, safe_target.unsqueeze(1)).squeeze(1)
    if target.ndim == 1:
        return nll * valid
    spatial = tuple(range(1, target.ndim))
    return (nll * valid).sum(dim=spatial) / valid.sum(dim=spatial).clamp_min(1)


def class_balanced_ce(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Return raw per-sample class-balanced cross entropy."""
    target = _prepare_target(logits, target)
    probabilities = _class_probabilities(logits).clamp_min(1e-7)
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0).clamp(0, probabilities.shape[1] - 1)
    nll = -probabilities.log().gather(1, safe_target.unsqueeze(1)).squeeze(1)
    if target.ndim == 1:
        return nll * valid
    spatial = tuple(range(1, target.ndim))
    losses, present = [], []
    for class_index in range(probabilities.shape[1]):
        class_mask = (safe_target == class_index) & valid
        count = class_mask.sum(dim=spatial)
        losses.append((nll * class_mask).sum(dim=spatial) / count.clamp_min(1))
        present.append(count > 0)
    stacked_losses = torch.stack(losses, dim=1)
    stacked_present = torch.stack(present, dim=1)
    return (stacked_losses * stacked_present).sum(1) / stacked_present.sum(1).clamp_min(1)


def soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Return a per-sample foreground soft Dice loss.

    Classes present in each target receive equal weight.  For an all-background
    target, the background class is used, making the result defined for every
    sample without filtering or altering the dataset.
    """
    target = _prepare_target(logits, target)
    probabilities = _class_probabilities(logits)
    if target.ndim == 1:
        safe_target = target.clamp(0, probabilities.shape[1] - 1)
        return 1.0 - probabilities.gather(1, safe_target[:, None]).squeeze(1)
    class_count = probabilities.shape[1]
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0).clamp(0, class_count - 1)
    one_hot = F.one_hot(safe_target, class_count).movedim(-1, 1).to(probabilities.dtype)
    valid_channels = valid.unsqueeze(1)
    probabilities = probabilities * valid_channels
    one_hot = one_hot * valid_channels
    spatial = tuple(range(2, probabilities.ndim))
    intersection = (probabilities * one_hot).sum(dim=spatial)
    denominator = probabilities.sum(dim=spatial) + one_hot.sum(dim=spatial)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    present = one_hot.sum(dim=spatial) > 0
    foreground = present.clone()
    if class_count > 1:
        foreground[:, 0] = False
    use_background = ~foreground.any(dim=1)
    foreground[use_background, 0] = True
    return 1.0 - (dice * foreground).sum(dim=1) / foreground.sum(dim=1).clamp_min(1)


PROXY_LOSSES: dict[str, LossFunction] = {
    "lce": cross_entropy_loss,
    "lcbce": class_balanced_ce,
    "ldice": soft_dice_loss,
}


def proxy_to_adversariality(name: str, loss: torch.Tensor) -> torch.Tensor:
    """Map heterogeneous losses to comparable higher-is-harder ``[0, 1]`` scores."""
    return loss.clamp(0.0, 1.0) if name == "ldice" else 1.0 - torch.exp(-loss.clamp_min(0.0))


def _hard_dice(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Compute one audit Dice value per sample without changing proxy losses."""
    target = _prepare_target(logits, target)
    if logits.ndim == 2:
        return (logits.argmax(1) == target).float()
    prediction = (torch.sigmoid(logits[:, 0]) >= 0.5).long() if logits.shape[1] == 1 else logits.argmax(1)
    valid = target != ignore_index
    foreground_prediction = (prediction > 0) & valid
    foreground_target = (target > 0) & valid
    spatial = tuple(range(1, target.ndim))
    intersection = (foreground_prediction & foreground_target).sum(dim=spatial).float()
    denominator = foreground_prediction.sum(dim=spatial) + foreground_target.sum(dim=spatial)
    return (2.0 * intersection + 1e-6) / (denominator.float() + 1e-6)


def _metadata_path(metadata: dict[str, Any]) -> str:
    """Read a source path from common dataset metadata keys without assuming layout."""
    for key in ("path", "image_path", "image", "source_path", "source"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _settings(config: Any) -> Any:
    return getattr(config, "adversariality", {})


@torch.inference_mode()
def evaluate_adversariality(config: Any, spatial_dims: int) -> dict[str, Any]:
    """Evaluate one frozen downstream model on an unchanged synthetic subset."""
    settings = _settings(config)
    proxy = str(settings.get("proxy", "lcbce")).lower()
    if proxy not in PROXY_LOSSES:
        raise KeyError(f"Unknown proxy '{proxy}'. Available: {sorted(PROXY_LOSSES)}")

    dataset_key = str(settings.get("dataset", "synthetic_dataset"))
    if not hasattr(config, dataset_key):
        raise KeyError(f"Configuration has no dataset section named '{dataset_key}'.")
    split = str(settings.get("split", "train"))
    dataset = build_dataset(getattr(config, dataset_key), split, spatial_dims)
    requested_budget = int(settings.get("budget", 0))
    budget = len(dataset) if requested_budget <= 0 else requested_budget
    if budget > len(dataset):
        raise ValueError(f"Fixed budget {budget} exceeds dataset size {len(dataset)}.")
    # Subset preserves dataset order and performs no resampling or augmentation.
    dataset = Subset(dataset, range(budget))

    device_name = str(getattr(config.runtime, "device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    adapter = build_downstream(config.downstream)
    adapter.model.to(device)
    adapter.freeze()
    loader = DataLoader(
        dataset,
        batch_size=int(settings.get("batch_size", getattr(config.evaluation, "batch_size", 1))),
        shuffle=False,
        num_workers=int(getattr(config.runtime, "num_workers", 0)),
        collate_fn=collate_medical_batch,
        pin_memory=device.type == "cuda",
    )
    ignore_index = int(getattr(getattr(config, dataset_key), "ignore_index", -100))
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for batch in tqdm(loader, desc=f"Adversariality ({proxy.upper()})"):
        if batch.image is None:
            raise ValueError("Adversariality evaluation requires each dataset item to provide 'image'.")
        device_batch: NAMBatch = batch.to(device)
        logits = adapter.logits(device_batch.image)
        losses = {
            name: function(logits, device_batch.target, ignore_index).detach().cpu()
            for name, function in PROXY_LOSSES.items()
        }
        scores = {name: proxy_to_adversariality(name, value) for name, value in losses.items()}
        dice = _hard_dice(logits, device_batch.target, ignore_index).detach().cpu()
        metadata_items = batch.metadata.get("items", [{}] * len(batch.sample_id))
        for index, sample_id in enumerate(batch.sample_id):
            if str(sample_id) in seen_ids:
                raise ValueError(f"Duplicate sample ID '{sample_id}' in the fixed-budget set.")
            seen_ids.add(str(sample_id))
            item = metadata_items[index] if index < len(metadata_items) else {}
            row = {
                "id": str(sample_id),
                "path": _metadata_path(item),
                "model": adapter.metadata.name,
                "proxy": proxy,
                "adv": float(scores[proxy][index]),
                "lce": float(losses["lce"][index]),
                "lcbce": float(losses["lcbce"][index]),
                "ldice": float(losses["ldice"][index]),
                "adv_lce": float(scores["lce"][index]),
                "adv_lcbce": float(scores["lcbce"][index]),
                "adv_ldice": float(scores["ldice"][index]),
                "prediction_score": float(dice[index]),
            }
            rows.append(row)

    output = Path(str(settings.get("output", "outputs/evaluation/adversariality.csv")))
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "path", "lce", "lcbce", "ldice", "model", "proxy", "adv",
        "adv_lce", "adv_lcbce", "adv_ldice", "prediction_score",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    means = {
        name: sum(float(row[name]) for row in rows) / len(rows) if rows else float("nan")
        for name in PROXY_LOSSES
    }
    mean_scores = {
        name: sum(float(row[f"adv_{name}"]) for row in rows) / len(rows) if rows else float("nan")
        for name in PROXY_LOSSES
    }
    summary: dict[str, Any] = {
        "mean_adv": mean_scores[proxy],
        "proxy": proxy,
        "mean_losses": means,
        "mean_adversariality": mean_scores,
        "samples": len(rows),
        "budget": budget,
        "dataset": dataset_key,
        "split": split,
        "model": adapter.metadata.name,
        "csv": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    """Define publication defaults so the module runs without CLI overrides."""
    parser = argparse.ArgumentParser(description="Evaluate fixed-budget synthetic adversariality.")
    parser.add_argument("--config", default="configs/table1_2d.yaml")
    parser.add_argument("--spatial-dims", type=int, choices=(2, 3), default=2)
    parser.add_argument("--proxy", choices=tuple(PROXY_LOSSES), default="lcbce")
    parser.add_argument("--budget", type=int, default=None, help="Override the configured fixed budget.")
    parser.add_argument("--output", default=None, help="Override the configured CSV path.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    experiment = apply_overrides(load_config(arguments.config), arguments.set)
    if "adversariality" not in experiment:
        experiment.adversariality = {}
    experiment.adversariality["proxy"] = arguments.proxy
    if arguments.budget is not None:
        experiment.adversariality["budget"] = arguments.budget
    if arguments.output is not None:
        experiment.adversariality["output"] = arguments.output
    print(json.dumps(evaluate_adversariality(experiment, arguments.spatial_dims), indent=2))
