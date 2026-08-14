"""PASCAL VOC 2012 and SBD semantic segmentation with class-aware prompts."""

from __future__ import annotations

from typing import Any

import torch

from nam.data.common import (
    ManifestItem,
    ManifestSegmentationDataset,
    build_nam_dataloader,
    config_value,
    package_root,
)


CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train",
    "television monitor",
)


def _prompt(target: torch.Tensor, item: ManifestItem) -> str:
    """Build the paper's `a photo of [VOC object]` prompt from visible labels."""
    present = torch.unique(target).tolist()
    names = [CLASS_NAMES[index] for index in present if 0 < index < len(CLASS_NAMES)]
    if not names:
        names = ["objects"]
    return f"a photo of [{', '.join(names)}]"


class PascalVOCSBDDataset(ManifestSegmentationDataset):
    """Unified 21-class VOC/SBD split after duplicate-image removal."""

    def __init__(self, config: Any, split: str = "train", spatial_dims: int = 2) -> None:
        if spatial_dims != 2:
            raise ValueError("PASCAL VOC + SBD is a 2D semantic-segmentation benchmark.")
        super().__init__(
            package_root(__file__, config),
            split,
            spatial_dims=2,
            modality="RGB",
            prompt=_prompt,
            image_size=config_value(config, "image_size", (512, 512)),
            train_augmentation=config_value(config, "augment", True),
        )


def build_dataset(
    config: Any, split: str = "train", spatial_dims: int = 2
) -> PascalVOCSBDDataset:
    return PascalVOCSBDDataset(config, split, spatial_dims)


def build_dataloader(config: Any, split: str = "train", spatial_dims: int = 2):
    dataset = build_dataset(config, split, spatial_dims)
    return build_nam_dataloader(
        dataset,
        batch_size=int(config_value(config, "batch_size", 4)),
        num_workers=int(config_value(config, "num_workers", 4)),
    )
