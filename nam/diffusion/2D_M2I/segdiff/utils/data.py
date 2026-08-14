"""Canonical dataset conversion for all 2D SegDiff datasets."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nam.data import NAMBatch, build_dataset, collate_medical_batch


def build_segdiff_loader(
    dataset_config: Any, split: str, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Build a loader from any dataset implementing the public NAM contract."""
    return DataLoader(
        build_dataset(dataset_config, split, spatial_dims=2),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=bool(shuffle),
        persistent_workers=int(num_workers) > 0,
        collate_fn=collate_medical_batch,
    )


def _mask_from_batch(batch: NAMBatch) -> torch.Tensor:
    condition = batch.condition
    if isinstance(condition, dict):
        condition = condition.get("mask", condition.get("segmentation", batch.target))
    if not torch.is_tensor(condition):
        condition = batch.target
    return condition


def prepare_condition_tensors(
    batch: NAMBatch,
    resolution: int,
    encoding: str,
    num_classes: int,
    condition_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize a canonical mask and encode the official concatenated condition."""
    mask = _mask_from_batch(batch)
    mask = mask.unsqueeze(1) if mask.ndim == 3 else mask
    labels = mask.argmax(1, keepdim=True) if mask.shape[1] > 1 else mask.long()
    labels = F.interpolate(labels.float(), (resolution, resolution), mode="nearest").long()
    encoding = encoding.lower()
    if encoding == "binary":
        condition = (labels > 0).float()
    elif encoding == "class_index":
        condition = labels.float()
    elif encoding == "official_uint8":
        condition = labels.float() / 255.0
    elif encoding == "normalized_index":
        condition = labels.float() / max(int(num_classes) - 1, 1)
    elif encoding == "one_hot_foreground":
        condition = F.one_hot(labels[:, 0].clamp(0, num_classes - 1), num_classes)
        condition = condition.permute(0, 3, 1, 2).float()[:, 1:]
    else:
        raise ValueError(f"Unknown SegDiff condition encoding '{encoding}'.")
    if condition.shape[1] != int(condition_channels):
        raise ValueError(
            f"Encoded condition has {condition.shape[1]} channels, expected {condition_channels}."
        )
    return condition, labels[:, 0]


def prepare_training_pair(
    batch: NAMBatch, augmentation: Any, encoding: str, num_classes: int, condition_channels: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized image, encoded condition, and categorical target."""
    if batch.image is None:
        raise ValueError("SegDiff pre-training requires paired real images and masks.")
    mask = _mask_from_batch(batch)
    mask = mask.unsqueeze(1) if mask.ndim == 3 else mask
    image, augmented_mask = augmentation(batch.image, mask)
    augmented_batch = NAMBatch(
        image=image,
        target=augmented_mask[:, 0].long(),
        condition=augmented_mask,
        sample_id=batch.sample_id,
        metadata=batch.metadata,
    )
    condition, target = prepare_condition_tensors(
        augmented_batch, augmentation.resolution, encoding, num_classes, condition_channels
    )
    return image, condition.to(image.device), target.to(image.device)
