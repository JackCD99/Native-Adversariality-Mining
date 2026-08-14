"""Canonical 3D dataset bridge for the official VolDiT implementation."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nam.data import NAMBatch, build_dataset, collate_medical_batch


def build_voldit_loader(
    dataset_config: Any, split: str, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Build a loader from any dataset satisfying the canonical NAM contract."""
    return DataLoader(
        build_dataset(dataset_config, split, spatial_dims=3),
        batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=int(num_workers),
        pin_memory=True, drop_last=bool(shuffle), persistent_workers=int(num_workers) > 0,
        collate_fn=collate_medical_batch,
    )


def _labels(target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(1) if target.ndim == 4 else target
    if target.shape[1] > 1:
        target = target.argmax(1, keepdim=True)
    return target.long()


def prepare_voldit_condition(
    batch: NAMBatch, volume_size: tuple[int, int, int], default_prompt: str
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return aligned one-channel mask controls, labels, and provenance prompts."""
    condition, prompts = batch.condition, default_prompt
    control = None
    if isinstance(condition, dict):
        control = condition.get("mask", condition.get("control"))
        prompts = condition.get("prompt", default_prompt)
    elif torch.is_tensor(condition):
        control = condition
    if control is None:
        control = _labels(batch.target).float()
    control = control.unsqueeze(1) if control.ndim == 4 else control.float()
    if control.shape[1] > 1:
        control = control.argmax(1, keepdim=True).float()
    control = F.interpolate(control, size=volume_size, mode="nearest")
    target = F.interpolate(_labels(batch.target).float(), size=volume_size, mode="nearest")[:, 0].long()
    prompt_list = [prompts] * control.shape[0] if isinstance(prompts, str) else [str(value) for value in prompts]
    if len(prompt_list) != control.shape[0]:
        raise ValueError("The number of VolDiT prompts must match the batch size.")
    return control, target, prompt_list


def prepare_training_tensors(
    batch: NAMBatch, volume_size: tuple[int, int, int], default_prompt: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Resize a paired volume and its control without dataset-specific assumptions."""
    if batch.image is None:
        raise ValueError("VolDiT TGCA fine-tuning requires paired real images and masks.")
    control, target, prompts = prepare_voldit_condition(batch, volume_size, default_prompt)
    image = batch.image.float()
    image = image.unsqueeze(1) if image.ndim == 4 else image
    if image.shape[1] != 1:
        raise ValueError("VolDiT expects single-channel medical volumes.")
    image = F.interpolate(image, size=volume_size, mode="trilinear", align_corners=False)
    if image.min() >= 0.0 and image.max() <= 1.0:
        image = image.mul(2.0).sub(1.0)
    return image.clamp(-1.0, 1.0), control, target, prompts
