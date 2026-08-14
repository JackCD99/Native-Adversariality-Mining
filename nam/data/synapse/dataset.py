"""Synapse multi-organ axial CT dataset for NAM."""

from __future__ import annotations

from typing import Any

from nam.data.common import ManifestSegmentationDataset, build_nam_dataloader, config_value, package_root


CLASS_NAMES = (
    "background",
    "spleen",
    "right kidney",
    "left kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "pancreas",
)


def _prompt(target, item) -> str:
    names = [CLASS_NAMES[index] for index in range(1, len(CLASS_NAMES)) if (target == index).any()]
    return f"a CT imaging of [{', '.join(names or CLASS_NAMES[1:])}]"


class SynapseDataset(ManifestSegmentationDataset):
    def __init__(self, config: Any, split: str = "train", spatial_dims: int = 2) -> None:
        if spatial_dims != 2:
            raise ValueError("The paper uses Synapse as foreground axial 2D slices.")
        super().__init__(package_root(__file__, config), split, 2, "CT", _prompt,
                         config_value(config, "image_size", (256, 256)), config_value(config, "augment", True))


def build_dataset(config: Any, split: str = "train", spatial_dims: int = 2) -> SynapseDataset:
    return SynapseDataset(config, split, spatial_dims)


def build_dataloader(config: Any, split: str = "train", spatial_dims: int = 2):
    return build_nam_dataloader(build_dataset(config, split, spatial_dims), int(config_value(config, "batch_size", 8)), num_workers=int(config_value(config, "num_workers", 4)))
