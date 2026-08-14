"""Paired image-mask preprocessing for SegDiff.

Official SegDiff resizes images bilinearly, masks with nearest-neighbor
interpolation, maps images to [-1, 1], and disables random flipping.
"""

from __future__ import annotations

import random

import torch
from torch.nn import functional as F


class SegDiffAugmentation:
    """Apply identical spatial transforms to a BCHW image-mask pair."""

    def __init__(self, resolution: int, horizontal_flip_probability: float = 0.0) -> None:
        self.resolution = int(resolution)
        self.horizontal_flip_probability = float(horizontal_flip_probability)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = F.interpolate(
            image.float(), (self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        mask = F.interpolate(mask.float(), (self.resolution, self.resolution), mode="nearest")
        if random.random() < self.horizontal_flip_probability:
            image, mask = image.flip(-1), mask.flip(-1)
        if image.max() > 1.5:
            image = image / 127.5 - 1.0
        elif image.min() >= 0.0:
            image = image * 2.0 - 1.0
        return image.clamp(-1.0, 1.0), mask
