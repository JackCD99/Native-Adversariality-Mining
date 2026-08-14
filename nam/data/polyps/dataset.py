"""Mixed-source polyp segmentation dataset used by SiameseDiff experiments."""

from __future__ import annotations

from typing import Any

from nam.data.common import ManifestSegmentationDataset, build_nam_dataloader, config_value, package_root


class PolypsDataset(ManifestSegmentationDataset):
    def __init__(self, config: Any, split: str = "train", spatial_dims: int = 2) -> None:
        if spatial_dims != 2:
            raise ValueError("Polyps is a 2D RGB segmentation benchmark.")
        super().__init__(package_root(__file__, config), split, 2, "RGB", "a colonoscopy image of a polyp",
                         config_value(config, "image_size", (256, 256)), config_value(config, "augment", True),
                         label_map={0: 0, 1: 1, 255: 1})


def build_dataset(config: Any, split: str = "train", spatial_dims: int = 2) -> PolypsDataset:
    return PolypsDataset(config, split, spatial_dims)


def build_dataloader(config: Any, split: str = "train", spatial_dims: int = 2):
    return build_nam_dataloader(build_dataset(config, split, spatial_dims), int(config_value(config, "batch_size", 8)), num_workers=int(config_value(config, "num_workers", 4)))
