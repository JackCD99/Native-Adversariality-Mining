"""Paired image-mask augmentation policies for SiameseDiff.

The official training recipe uses resizing and 5% empty-prompt dropout. The
optional horizontal flip and isotropic scaling hooks are provided for unified
dataset studies but remain disabled in the published SiameseDiff configuration.
"""

from __future__ import annotations

import random

import torch
from torch.nn import functional as F


class PairedAugmentation:
    """Apply geometry-identical transformations to images and masks."""

    def __init__(
        self,
        resolution: int,
        horizontal_flip_probability: float = 0.0,
        scale_range: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.resolution = int(resolution)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.scale_range = tuple(float(value) for value in scale_range)

    def __call__(
        self, images: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize, optionally scale/crop, and flip a BCHW image-mask batch."""
        images = F.interpolate(
            images, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        masks = F.interpolate(masks.float(), size=(self.resolution, self.resolution), mode="nearest")
        scale = random.uniform(*self.scale_range)
        if abs(scale - 1.0) > 1e-6:
            scaled = max(1, round(self.resolution * scale))
            images = F.interpolate(images, size=(scaled, scaled), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks, size=(scaled, scaled), mode="nearest")
            images, masks = self._center_fit(images, masks)
        if random.random() < self.horizontal_flip_probability:
            images, masks = images.flip(-1), masks.flip(-1)
        return images, masks

    def _center_fit(
        self, images: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = images.shape[-1]
        if size >= self.resolution:
            start = (size - self.resolution) // 2
            index = (..., slice(start, start + self.resolution), slice(start, start + self.resolution))
            return images[index], masks[index]
        total = self.resolution - size
        before, after = total // 2, total - total // 2
        padding = (before, after, before, after)
        return F.pad(images, padding), F.pad(masks, padding)
