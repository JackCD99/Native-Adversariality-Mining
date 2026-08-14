"""Aligned image-mask augmentation for MedSegFactory fine-tuning."""
from __future__ import annotations
import random
import torch


class MedSegFactoryAugmentation:
    """Optional paired flip; the official default probability is zero."""
    def __init__(self, horizontal_flip_probability: float = 0.0) -> None:
        self.horizontal_flip_probability = float(horizontal_flip_probability)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.horizontal_flip_probability:
            return image.flip(-1), mask.flip(-1)
        return image, mask
