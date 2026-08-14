"""Canonical paired-data bridge for MedSegFactory."""
from __future__ import annotations
from typing import Any
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from nam.data import NAMBatch, build_dataset, collate_medical_batch


def build_loader(config: Any, split: str, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(build_dataset(config, split, 2), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, drop_last=shuffle, persistent_workers=workers > 0,
        collate_fn=collate_medical_batch)


def prompts_from_batch(batch: NAMBatch, image_default: str, mask_default: str) -> tuple[list[str], list[str]]:
    condition = batch.condition if isinstance(batch.condition, dict) else {}
    def collect(key: str, default: str) -> list[str]:
        value = condition.get(key, default)
        if isinstance(value, str): return [value] * batch.target.shape[0]
        return [str(item) for item in value]
    return collect("image_prompt", image_default), collect("mask_prompt", mask_default)


def prepare_pair(batch: NAMBatch, resolution: int, num_classes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch.image is None: raise ValueError("MedSegFactory training requires paired images and masks.")
    image = F.interpolate(batch.image.float(), (resolution, resolution), mode="bilinear", align_corners=False)
    if image.shape[1] == 1: image = image.repeat(1, 3, 1, 1)
    if image.max() > 1.5: image = image / 127.5 - 1
    elif image.min() >= 0: image = image * 2 - 1
    target = batch.target.argmax(1) if batch.target.ndim == 4 and batch.target.shape[1] > 1 else batch.target.squeeze(1) if batch.target.ndim == 4 else batch.target
    target = F.interpolate(target[:, None].float(), (resolution, resolution), mode="nearest")[:, 0].long()
    mask = target.float().div(max(num_classes - 1, 1)).mul(2).sub(1).unsqueeze(1).repeat(1, 3, 1, 1)
    return image.clamp(-1, 1), target, mask
