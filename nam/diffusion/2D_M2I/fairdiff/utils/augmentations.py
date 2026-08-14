"""Geometry-aligned image, mask-control, and label transforms for FairDiff."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class FairDiffAugmentation:
    """Apply only FairDiff's aligned resize and value normalization.

    The public ``medical_dataset.py`` performs no random geometric or intensity
    augmentation. The configured medical pipeline uses a 256-pixel resize.
    """

    def __init__(self, resolution: int) -> None:
        self.resolution = int(resolution)

    def __call__(
        self, image: torch.Tensor, control: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = F.interpolate(
            image.float(), (self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        control = F.interpolate(
            control.float(), (self.resolution, self.resolution), mode="nearest"
        )
        target = F.interpolate(
            target.float(), (self.resolution, self.resolution), mode="nearest"
        )
        if image.max() > 1.5:
            image = image / 127.5 - 1.0
        elif image.min() >= 0.0:
            image = image * 2.0 - 1.0
        if control.max() > 1.5:
            control = control / 255.0
        return image.clamp(-1.0, 1.0), control.clamp(0.0, 1.0), target.long()
