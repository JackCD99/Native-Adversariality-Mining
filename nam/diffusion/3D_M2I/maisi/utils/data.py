"""Canonical volumetric data conversion for official MAISI ControlNet."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nam.data import NAMBatch, build_dataset, collate_medical_batch


def build_maisi_loader(dataset_config: Any, split: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        build_dataset(dataset_config, split, spatial_dims=3), batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), pin_memory=True, drop_last=bool(shuffle), persistent_workers=int(num_workers) > 0,
        collate_fn=collate_medical_batch,
    )


def _labels(target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(1) if target.ndim == 4 else target
    if target.shape[1] > 1:
        target = target.argmax(1, keepdim=True)
    return target.long()


def binary_encode_labels(labels: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Match official MAISI's little-endian eight-channel mask encoding."""
    mask = 2 ** torch.arange(bits, device=labels.device, dtype=labels.dtype)
    return labels.unsqueeze(-1).bitwise_and(mask).ne(0).byte().squeeze(1).permute(0, 4, 1, 2, 3).float()


def prepare_maisi_condition(
    batch: NAMBatch, volume_size: tuple[int, int, int], default_prompt: str,
    default_spacing: tuple[float, float, float], default_top: tuple[float, ...],
    default_bottom: tuple[float, ...], default_modality: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    condition, prompts = batch.condition, default_prompt
    raw_mask = None
    if isinstance(condition, dict):
        raw_mask = condition.get("mask", condition.get("control"))
        prompts = condition.get("prompt", default_prompt)
    elif torch.is_tensor(condition):
        raw_mask = condition
    labels = _labels(batch.target if raw_mask is None else raw_mask)
    labels = F.interpolate(labels.float(), size=volume_size, mode="nearest").long()
    target = F.interpolate(_labels(batch.target).float(), size=volume_size, mode="nearest")[:, 0].long()
    batch_size = labels.shape[0]
    def field(name: str, default: tuple[float, ...]) -> torch.Tensor:
        value = condition.get(name) if isinstance(condition, dict) else None
        if value is None:
            value = torch.tensor(default, dtype=torch.float32).repeat(batch_size, 1)
        elif not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.float() * 100.0
    modality = condition.get("modality") if isinstance(condition, dict) else None
    if modality is None:
        modality = torch.full((batch_size,), int(default_modality), dtype=torch.long)
    elif not torch.is_tensor(modality):
        modality = torch.as_tensor(modality)
    prompt_list = [prompts] * batch_size if isinstance(prompts, str) else [str(value) for value in prompts]
    extras = {"spacing": field("spacing", default_spacing), "top_region": field("top_region", default_top), "bottom_region": field("bottom_region", default_bottom), "modality": modality.long(), "prompts": prompt_list}
    return binary_encode_labels(labels), target, extras


def prepare_training_tensors(batch: NAMBatch, volume_size: tuple[int, int, int]) -> torch.Tensor:
    """Convert canonical normalized images to the official MAISI VAE range [0,1]."""
    if batch.image is None:
        raise ValueError("MAISI ControlNet fine-tuning requires paired real images and masks.")
    image = batch.image.float()
    image = image.unsqueeze(1) if image.ndim == 4 else image
    image = F.interpolate(image, size=volume_size, mode="trilinear", align_corners=False)
    if image.min() < 0.0:
        image = image.add(1.0).mul(0.5)
    return image.clamp(0.0, 1.0)
