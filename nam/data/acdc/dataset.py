"""ACDC 2D cardiac-structure segmentation for NAM experiments."""

from __future__ import annotations

from typing import Any

from nam.data.common import ManifestSegmentationDataset, build_nam_dataloader, config_value, package_root


CLASS_NAMES = ("background", "right ventricle", "myocardium", "left ventricle")


def _prompt(target, item) -> str:
    names = [CLASS_NAMES[index] for index in range(1, len(CLASS_NAMES)) if (target == index).any()]
    return f"a MRI imaging of [{', '.join(names or CLASS_NAMES[1:])}]"


class ACDCDataset(ManifestSegmentationDataset):
    def __init__(self, config: Any, split: str = "train", spatial_dims: int = 2) -> None:
        if spatial_dims != 2:
            raise ValueError("The paper uses ACDC as axial 2D slices.")
        super().__init__(package_root(__file__, config), split, 2, "MRI", _prompt,
                         config_value(config, "image_size", (256, 256)),
                         config_value(config, "augment", True))


def build_dataset(config: Any, split: str = "train", spatial_dims: int = 2) -> ACDCDataset:
    return ACDCDataset(config, split, spatial_dims)


def build_dataloader(config: Any, split: str = "train", spatial_dims: int = 2):
    dataset = build_dataset(config, split, spatial_dims)
    return build_nam_dataloader(dataset, int(config_value(config, "batch_size", 8)), num_workers=int(config_value(config, "num_workers", 4)))
