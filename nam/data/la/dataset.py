"""Cropped 3D left-atrium MRI volumes for NAM."""

from __future__ import annotations

from typing import Any

from nam.data.common import ManifestSegmentationDataset, build_nam_dataloader, config_value, package_root


class LADataset(ManifestSegmentationDataset):
    def __init__(self, config: Any, split: str = "train", spatial_dims: int = 3) -> None:
        if spatial_dims != 3:
            raise ValueError("LA is a volumetric 3D benchmark in the paper.")
        super().__init__(package_root(__file__, config), split, 3, "MRI", "a MRI imaging of [left atrium]",
                         config_value(config, "image_size", (192, 192, 96)), config_value(config, "augment", True),
                         label_map={0: 0, 1: 1, 255: 1})


def build_dataset(config: Any, split: str = "train", spatial_dims: int = 3) -> LADataset:
    return LADataset(config, split, spatial_dims)


def build_dataloader(config: Any, split: str = "train", spatial_dims: int = 3):
    return build_nam_dataloader(build_dataset(config, split, spatial_dims), int(config_value(config, "batch_size", 1)), num_workers=int(config_value(config, "num_workers", 4)))
