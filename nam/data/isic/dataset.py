"""ISIC benign/malignant classification and class-aware prompts."""

from __future__ import annotations

from typing import Any

from nam.data.common import ManifestClassificationDataset, build_nam_dataloader, config_value, package_root


PROMPTS = {
    0: ("a dermoscopic image of a benign skin lesion", "a dermoscopy photograph of a benign skin lesion", "a detailed dermoscopic image of a benign melanocytic skin lesion"),
    1: ("a dermoscopic image of a malignant skin lesion", "a dermoscopy photograph of a malignant skin lesion", "a detailed dermoscopic image of a malignant melanocytic skin lesion"),
}


class ISICDataset(ManifestClassificationDataset):
    def __init__(self, config: Any, split: str = "train", spatial_dims: int = 2) -> None:
        if spatial_dims != 2:
            raise ValueError("ISIC is a 2D classification benchmark.")
        super().__init__(package_root(__file__, config), split, PROMPTS, "Dermoscopy",
                         config_value(config, "image_size", (256, 256)), config_value(config, "augment", True),
                         config_value(config, "caption_dropout_probability", 0.0))


def build_dataset(config: Any, split: str = "train", spatial_dims: int = 2) -> ISICDataset:
    return ISICDataset(config, split, spatial_dims)


def build_dataloader(config: Any, split: str = "train", spatial_dims: int = 2):
    return build_nam_dataloader(build_dataset(config, split, spatial_dims), int(config_value(config, "batch_size", 32)), num_workers=int(config_value(config, "num_workers", 4)))
