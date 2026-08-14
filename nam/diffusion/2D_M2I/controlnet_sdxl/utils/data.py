"""Canonical PASCAL VOC/SBD loader used by ControlNet-SDXL."""

from typing import Any

from nam.data import build_dataset, collate_medical_batch
from torch.utils.data import DataLoader


def build_loader(
    config: Any, split: str, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        build_dataset(config, split, spatial_dims=2),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        collate_fn=collate_medical_batch,
    )
