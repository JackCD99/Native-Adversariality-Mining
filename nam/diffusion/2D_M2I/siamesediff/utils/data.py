"""Canonical-dataset conversion for official SiameseDiff training and sampling."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import DataLoader

from nam.data import NAMBatch, build_dataset, collate_medical_batch
from .augmentations import PairedAugmentation


def build_siamesediff_loader(
    dataset_config: Any,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Build one 2D loader from any dataset implementing the NAM contract."""
    return DataLoader(
        build_dataset(dataset_config, split, spatial_dims=2),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=int(num_workers) > 0,
        collate_fn=collate_medical_batch,
    )


def _condition_parts(batch: NAMBatch, default_prompt: str) -> tuple[torch.Tensor, list[str]]:
    condition = batch.condition
    if isinstance(condition, dict):
        mask = condition.get("hint", condition.get("mask", batch.target))
        prompt_value = condition.get("txt", condition.get("prompt", default_prompt))
    else:
        mask, prompt_value = condition, default_prompt
    if mask is None or not torch.is_tensor(mask):
        mask = batch.target
    if isinstance(prompt_value, str):
        prompts = [prompt_value] * mask.shape[0]
    else:
        prompts = [str(value) for value in prompt_value]
    return mask, prompts


def _label_map_to_rgb(labels: torch.Tensor) -> torch.Tensor:
    """Encode categorical masks with the deterministic PASCAL color palette."""
    labels = labels[:, 0].long().clamp_min(0)
    palette = torch.zeros((256, 3), dtype=torch.float32, device=labels.device)
    for class_index in range(256):
        value, bit = class_index, 0
        while value:
            for channel in range(3):
                palette[class_index, channel] += ((value >> channel) & 1) << (7 - bit)
            value >>= 3
            bit += 1
    rgb = palette[labels.clamp_max(255)] / 255.0
    return rgb.permute(0, 3, 1, 2).contiguous()


def to_official_batch(
    batch: NAMBatch,
    augmentation: PairedAugmentation,
    default_prompt: str,
    prompt_dropout: float,
    training: bool,
) -> dict[str, Any]:
    """Convert BCHW tensors to the BHWC dictionary expected by ``ControlLDM``."""
    if batch.image is None and training:
        raise ValueError("SiameseDiff pre-training requires paired real images and masks.")
    masks, prompts = _condition_parts(batch, default_prompt)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.shape[1] != 1:
        masks = masks.argmax(1, keepdim=True) if masks.shape[1] > 1 else masks[:, :1]
    categorical = masks.max() > 1
    masks = masks.float()
    images = (
        torch.zeros((masks.shape[0], 3, *masks.shape[2:]), device=masks.device)
        if batch.image is None
        else batch.image.float()
    )
    if batch.image is not None:
        if images.max() > 1.5:
            images = images / 127.5 - 1.0
        elif images.min() >= 0.0:
            images = images * 2.0 - 1.0
    images, masks = augmentation(images, masks)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if training and prompt_dropout > 0:
        prompts = ["" if random.random() < prompt_dropout else prompt for prompt in prompts]
    hints = _label_map_to_rgb(masks) if categorical else masks.clamp(0, 1).repeat(1, 3, 1, 1)
    return {
        "jpg": images.permute(0, 2, 3, 1).contiguous(),
        "hint": hints.permute(0, 2, 3, 1).contiguous(),
        "txt": prompts,
    }
