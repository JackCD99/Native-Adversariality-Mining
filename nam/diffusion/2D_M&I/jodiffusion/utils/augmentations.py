"""Official paired spatial augmentation used by JoDiffusion fine-tuning."""
from __future__ import annotations
import random
import torch


class JoDiffusionAugmentation:
    """Apply the official aligned horizontal flip to an image-label pair."""
    def __init__(self, horizontal_flip_probability: float = 0.5) -> None:
        self.horizontal_flip_probability = float(horizontal_flip_probability)

    def __call__(self, image: torch.Tensor, bitmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.horizontal_flip_probability:
            return image.flip(-1), bitmap.flip(-1)
        return image, bitmap
