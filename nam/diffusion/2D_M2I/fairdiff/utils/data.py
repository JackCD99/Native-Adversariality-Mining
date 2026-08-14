"""Dataset conversion and RGB mask encoding for official FairDiff."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nam.data import NAMBatch, build_dataset, collate_medical_batch


def build_fairdiff_loader(
    dataset_config: Any, split: str, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Build a loader from any dataset implementing the canonical NAM contract."""
    return DataLoader(
        build_dataset(dataset_config, split, spatial_dims=2),
        batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), pin_memory=True, drop_last=bool(shuffle),
        persistent_workers=int(num_workers) > 0, collate_fn=collate_medical_batch,
    )


def _labels(target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(1) if target.ndim == 3 else target
    if target.shape[1] > 1:
        target = target.argmax(1, keepdim=True)
    return target.long()


def _pascal_palette(num_classes: int, device: torch.device) -> torch.Tensor:
    """Return a deterministic RGB palette compatible with label-map PNGs."""
    colors = torch.zeros((max(int(num_classes), 1), 3), dtype=torch.float32, device=device)
    for class_index in range(colors.shape[0]):
        value = class_index
        red = green = blue = 0
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        colors[class_index] = torch.tensor((red, green, blue), device=device) / 255.0
    return colors


def labels_to_control(
    target: torch.Tensor, num_classes: int, encoding: str = "palette"
) -> torch.Tensor:
    """Convert categorical labels to the three-channel hint read by ControlNet."""
    labels = _labels(target)[:, 0]
    if labels.min() < 0 or labels.max() >= int(num_classes):
        raise ValueError("FairDiff labels must be in [0, num_classes - 1].")
    mode = str(encoding).lower()
    if mode == "palette":
        palette = _pascal_palette(num_classes, labels.device)
        return palette[labels].permute(0, 3, 1, 2).contiguous()
    if mode == "scalar":
        scaled = labels.float().unsqueeze(1) / max(int(num_classes) - 1, 1)
        return scaled.repeat(1, 3, 1, 1)
    raise ValueError("FairDiff mask_encoding must be 'palette' or 'scalar'.")


def _condition_parts(
    batch: NAMBatch, default_prompt: str, num_classes: int, mask_encoding: str
) -> tuple[torch.Tensor, list[str]]:
    condition = batch.condition
    control = None
    prompts: Any = default_prompt
    if isinstance(condition, dict):
        control = condition.get("mask", condition.get("hint", condition.get("control")))
        prompts = condition.get("prompt", condition.get("txt", default_prompt))
    if control is None or not torch.is_tensor(control):
        control = labels_to_control(batch.target, num_classes, mask_encoding)
    else:
        control = control.unsqueeze(1) if control.ndim == 3 else control.float()
        if control.shape[1] == 1:
            # A categorical one-channel user hint is encoded consistently with
            # canonical labels; an RGB hint is preserved exactly as upstream.
            control = labels_to_control(control.long(), num_classes, mask_encoding)
        if control.shape[1] != 3:
            raise ValueError("FairDiff mask controls must contain one or three channels.")
    if isinstance(prompts, str):
        prompt_list = [prompts] * control.shape[0]
    else:
        prompt_list = [str(item) for item in prompts]
    if len(prompt_list) != control.shape[0]:
        raise ValueError("The number of FairDiff prompts must match the batch size.")
    return control.float(), prompt_list


def prepare_fairdiff_condition(
    batch: NAMBatch,
    resolution: int,
    default_prompt: str,
    num_classes: int,
    mask_encoding: str = "palette",
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return the resized RGB control, aligned categorical target, and prompts."""
    control, prompts = _condition_parts(batch, default_prompt, num_classes, mask_encoding)
    control = F.interpolate(control, (resolution, resolution), mode="nearest")
    if control.max() > 1.5:
        control = control / 255.0
    target = F.interpolate(
        _labels(batch.target).float(), (resolution, resolution), mode="nearest"
    )[:, 0].long()
    return control.clamp(0.0, 1.0), target, prompts


def to_official_batch(
    batch: NAMBatch,
    augmentation: Any,
    default_prompt: str,
    num_classes: int,
    mask_encoding: str,
    training: bool,
) -> dict[str, Any]:
    """Convert a canonical batch to official ControlLDM BHWC fields."""
    if training and batch.image is None:
        raise ValueError("FairDiff fine-tuning requires paired real images and labels.")
    control, prompts = _condition_parts(batch, default_prompt, num_classes, mask_encoding)
    target = _labels(batch.target)
    image = (
        torch.zeros((control.shape[0], 3, *control.shape[-2:]), device=control.device)
        if batch.image is None else batch.image.float()
    )
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    if image.shape[1] != 3:
        raise ValueError("FairDiff images must contain one or three channels.")
    image, control, _ = augmentation(image, control, target)
    return {
        "jpg": image.permute(0, 2, 3, 1).contiguous(),
        "hint": control.permute(0, 2, 3, 1).contiguous(),
        "txt": prompts,
    }
