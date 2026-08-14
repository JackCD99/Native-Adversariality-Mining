"""Dataset conversion, boundary extraction, and prompt routing for DiffBoost."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nam.data import NAMBatch, build_dataset, collate_medical_batch


OFFICIAL_AUGMENTATION_PROMPTS = (
    "Enhance contrast", "High resolution", "Low resolution", "Clear edges",
    "Blurred edges", "Noise reduction", "Sharpened details", "Smoothed texture",
    "Brightened image", "Darkened image", "Balanced lighting", "Grayscale colormap",
    "Gamma correction", "Image normalization", "Edge detection", "Image enhancement",
)


def build_diffboost_loader(
    dataset_config: Any, split: str, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Build a loader from any dataset implementing the NAM contract."""
    return DataLoader(
        build_dataset(dataset_config, split, spatial_dims=2),
        batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), pin_memory=True, drop_last=bool(shuffle),
        persistent_workers=int(num_workers) > 0, collate_fn=collate_medical_batch,
    )


def labels_to_edges(labels: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Create a deterministic structure boundary when no precomputed edge is supplied."""
    labels = labels.unsqueeze(1) if labels.ndim == 3 else labels
    if labels.shape[1] > 1:
        labels = labels.argmax(1, keepdim=True)
    labels = labels.float()
    kernel = max(int(width), 1)
    if kernel % 2 == 0:
        kernel += 1
    maximum = F.max_pool2d(labels, kernel, stride=1, padding=kernel // 2)
    minimum = -F.max_pool2d(-labels, kernel, stride=1, padding=kernel // 2)
    return (maximum != minimum).float()


def _condition_parts(
    batch: NAMBatch,
    default_prompt: str,
    default_augmentation_prompt: str,
    condition_mode: str,
    num_classes: int,
) -> tuple[torch.Tensor, list[str], list[str]]:
    condition = batch.condition
    edge = None
    prompts: Any = default_prompt
    augmentation: Any = default_augmentation_prompt
    if isinstance(condition, dict):
        edge = condition.get("edge", condition.get("hint"))
        prompts = condition.get("prompt", condition.get("txt", default_prompt))
        augmentation = condition.get(
            "augmentation_prompt", condition.get("appearance_prompt", default_augmentation_prompt)
        )
    condition_mode = str(condition_mode).lower()
    if condition_mode == "mask":
        labels = batch.target.unsqueeze(1) if batch.target.ndim == 3 else batch.target
        if labels.shape[1] > 1:
            labels = labels.argmax(1, keepdim=True)
        edge = labels.float() / max(int(num_classes) - 1, 1)
    elif condition_mode == "edge":
        if edge is None or not torch.is_tensor(edge):
            edge = labels_to_edges(batch.target)
    else:
        raise ValueError("DiffBoost condition_mode must be 'mask' or 'edge'.")
    edge = edge.unsqueeze(1) if edge.ndim == 3 else edge
    if edge.shape[1] == 1:
        edge = edge.repeat(1, 3, 1, 1)
    if edge.shape[1] != 3:
        raise ValueError("DiffBoost edge controls must contain one or three channels.")

    def expand(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] * edge.shape[0]
        return [str(item) for item in value]

    return edge.float(), expand(prompts), expand(augmentation)


def prepare_diffboost_condition(
    batch: NAMBatch,
    resolution: int,
    default_prompt: str,
    default_augmentation_prompt: str,
    condition_mode: str = "mask",
    num_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Return resized edge control, aligned labels, and per-sample prompts."""
    edge, prompts, augmentation = _condition_parts(
        batch, default_prompt, default_augmentation_prompt, condition_mode, num_classes
    )
    edge = F.interpolate(edge, (resolution, resolution), mode="bilinear", align_corners=False)
    if edge.max() > 1.5:
        edge = edge / 255.0
    target = batch.target
    target = target.unsqueeze(1) if target.ndim == 3 else target
    if target.shape[1] > 1:
        target = target.argmax(1, keepdim=True)
    target = F.interpolate(target.float(), (resolution, resolution), mode="nearest")[:, 0].long()
    return edge.clamp(0.0, 1.0), target, prompts, augmentation


def to_official_batch(
    batch: NAMBatch,
    augmentation: Any,
    default_prompt: str,
    training: bool,
    condition_mode: str = "mask",
    num_classes: int = 2,
) -> dict[str, Any]:
    """Convert a canonical batch to official ControlLDM BHWC fields."""
    if training and batch.image is None:
        raise ValueError("DiffBoost fine-tuning requires paired real images and labels.")
    edge, prompts, _ = _condition_parts(
        batch, default_prompt, "Image enhancement", condition_mode, num_classes
    )
    target = batch.target.unsqueeze(1) if batch.target.ndim == 3 else batch.target
    image = (
        torch.zeros((edge.shape[0], 3, *edge.shape[-2:]), device=edge.device)
        if batch.image is None else batch.image.float()
    )
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    image, edge, _ = augmentation(image, edge, target)
    return {
        "jpg": image.permute(0, 2, 3, 1).contiguous(),
        "hint": edge.permute(0, 2, 3, 1).contiguous(),
        "txt": prompts,
    }
