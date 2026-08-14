"""PneumoniaMNIST-224 binary classification and T2I prompts."""

from __future__ import annotations

from typing import Any

from nam.data.common import ManifestClassificationDataset, build_nam_dataloader, config_value, package_root


PROMPTS = {
    0: ("a grayscale chest X-ray image of normal", "a frontal chest X-ray image of normal lungs", "a pediatric chest X-ray image without pneumonia"),
    1: ("a grayscale chest X-ray image of pneumonia", "a frontal chest X-ray image showing pneumonia", "a pediatric chest X-ray image with pneumonia"),
}


class PneumoniaMNISTDataset(ManifestClassificationDataset):
    def __init__(self, config: Any, split: str = "train", spatial_dims: int = 2) -> None:
        if spatial_dims != 2:
            raise ValueError("PneumoniaMNIST-224 is a 2D classification benchmark.")
        super().__init__(package_root(__file__, config), split, PROMPTS, "X-ray",
                         config_value(config, "image_size", (224, 224)), config_value(config, "augment", True),
                         config_value(config, "caption_dropout_probability", 0.0))


def build_dataset(config: Any, split: str = "train", spatial_dims: int = 2) -> PneumoniaMNISTDataset:
    return PneumoniaMNISTDataset(config, split, spatial_dims)


def build_dataloader(config: Any, split: str = "train", spatial_dims: int = 2):
    return build_nam_dataloader(build_dataset(config, split, spatial_dims), int(config_value(config, "batch_size", 64)), num_workers=int(config_value(config, "num_workers", 4)))
