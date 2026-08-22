"""Dataset-level DSC and ASD evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nam.data import build_dataset, collate_medical_batch
from nam.downstream import build_downstream
from nam.evaluation.metrics import average_surface_distance, dice_per_class


@torch.no_grad()
def evaluate_segmentation(config: Any, spatial_dims: int) -> dict[str, Any]:
    """Evaluate a configured downstream checkpoint on the held-out test set."""
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    adapter = build_downstream(config.downstream)
    adapter.model.to(device)
    adapter.freeze()
    dataset = build_dataset(config.dataset, "test", spatial_dims)
    loader = DataLoader(
        dataset,
        batch_size=int(config.evaluation.batch_size),
        shuffle=False,
        num_workers=int(config.runtime.num_workers),
        collate_fn=collate_medical_batch,
    )
    dice_values: list[float] = []
    asd_values: list[float] = []
    asd_absent_objects = 0
    asd_empty_failures = 0
    confusion = None
    num_classes = int(config.downstream.num_classes)
    ignore_index = int(getattr(config.dataset, "ignore_index", -100))
    for batch in tqdm(loader, desc="Evaluating segmentation"):
        batch = batch.to(device)
        logits = adapter.logits(batch.image)
        prediction = (torch.sigmoid(logits[:, 0]) > 0.5).long() if logits.shape[1] == 1 else logits.argmax(1)
        target = batch.target[:, 0] if batch.target.ndim == logits.ndim else batch.target
        if str(getattr(config.evaluation, "metric", "dsc")).lower() == "miou":
            if confusion is None:
                confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)
            valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
            indices = target[valid].cpu() * num_classes + prediction[valid].cpu()
            confusion += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
            continue
        dice_values.extend(dice_per_class(prediction, target, num_classes).cpu().tolist())
        pred_numpy = prediction.cpu().numpy()
        target_numpy = target.cpu().numpy()
        for sample_index in range(pred_numpy.shape[0]):
            spacing = tuple(batch.metadata["items"][sample_index].get("spacing", [1.0] * spatial_dims))
            valid_region = target_numpy[sample_index] != ignore_index
            for class_index in range(1, num_classes):
                # ignore 区域不应因模型任意输出而形成额外表面或空掩膜失败。
                prediction_mask = (pred_numpy[sample_index] == class_index) & valid_region
                target_mask = (target_numpy[sample_index] == class_index) & valid_region
                if not prediction_mask.any() and not target_mask.any():
                    asd_absent_objects += 1
                elif not prediction_mask.any() or not target_mask.any():
                    asd_empty_failures += 1
                value = average_surface_distance(
                    prediction_mask,
                    target_mask,
                    spacing,
                )
                if np.isfinite(value):
                    asd_values.append(value)
    if confusion is not None:
        intersection = confusion.diag()
        union = confusion.sum(0) + confusion.sum(1) - intersection
        valid = union > 0
        result = {
            "miou": float((intersection[valid] / union[valid]).mean() * 100.0) if valid.any() else float("nan"),
            "samples": len(dataset),
        }
    else:
        result = {
            "dsc": float(np.mean(dice_values) * 100.0) if dice_values else float("nan"),
            "asd": float(np.mean(asd_values)) if asd_values else float("nan"),
            "samples": len(dataset),
            "asd_valid_objects": len(asd_values),
            "asd_empty_failures": asd_empty_failures,
            "asd_absent_objects": asd_absent_objects,
            "asd_empty_penalty": "physical_fov_diagonal",
            "asd_aggregation": "sample_class_macro_equal_directed_means",
        }
    output = Path(config.evaluation.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result
