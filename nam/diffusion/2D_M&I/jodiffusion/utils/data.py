"""Canonical paired-data bridge for JoDiffusion."""

from __future__ import annotations

from typing import Any
import math
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from nam.data import NAMBatch, build_dataset, collate_medical_batch


def build_loader(config: Any, split: str, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    """Build a loader for every dataset implementing the public NAM contract."""
    return DataLoader(build_dataset(config, split, 2), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, drop_last=shuffle,
                      persistent_workers=workers > 0, collate_fn=collate_medical_batch)


def prompts_from_batch(batch: NAMBatch, default: str) -> list[str]:
    """Read optional text prompts without imposing a dataset-specific schema."""
    value = batch.condition.get("prompt") if isinstance(batch.condition, dict) else None
    if value is None:
        return [default] * batch.target.shape[0]
    if isinstance(value, str):
        return [value] * batch.target.shape[0]
    return [str(item) for item in value]


def prepare_pair(batch: NAMBatch, resolution: int, num_classes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return image, categorical mask, and the official color-map label tensor."""
    if batch.image is None:
        raise ValueError("JoDiffusion pre-training requires paired real images and masks.")
    image = F.interpolate(batch.image.float(), (resolution, resolution), mode="bilinear", align_corners=False)
    if image.max() > 1.5:
        image = image / 127.5 - 1.0
    elif image.min() >= 0:
        image = image * 2.0 - 1.0
    target = batch.target.argmax(1) if batch.target.ndim == 4 and batch.target.shape[1] > 1 else batch.target.squeeze(1) if batch.target.ndim == 4 else batch.target
    target = F.interpolate(target[:, None].float(), (resolution, resolution), mode="nearest")[:, 0].long()
    if target.max() >= num_classes:
        raise ValueError("A target class index exceeds diffusion.num_classes.")
    # Official EncodeBitMap uses ceil(log2(C)) binary channels in [-1, 1].
    bits = max(1, math.ceil(math.log2(max(num_classes, 2))))
    bitmap = torch.stack([((target >> bit) & 1) for bit in range(bits)], 1).float()
    return image.clamp(-1, 1), target, bitmap * 2.0 - 1.0
