"""Geometry-aligned image, edge, and label transforms for DiffBoost."""

from __future__ import annotations

import random

import torch
from torch.nn import functional as F


class DiffBoostAugmentation:
    """Apply the official centered scale and rotation with tensor operations."""

    def __init__(
        self,
        resolution: int,
        rotation_degrees: float = 20.0,
        scale_range: tuple[float, float] = (0.75, 1.05),
        enabled: bool = True,
    ) -> None:
        self.resolution = int(resolution)
        self.rotation_degrees = float(rotation_degrees)
        self.scale_range = tuple(float(value) for value in scale_range)
        self.enabled = bool(enabled)

    def __call__(
        self, image: torch.Tensor, edge: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = F.interpolate(
            image.float(), (self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        edge = F.interpolate(
            edge.float(), (self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        target = F.interpolate(
            target.float(), (self.resolution, self.resolution), mode="nearest"
        )
        if self.enabled:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            scale = random.uniform(*self.scale_range)
            radians = torch.deg2rad(torch.tensor(angle, device=image.device, dtype=image.dtype))
            cosine, sine = torch.cos(radians) / scale, torch.sin(radians) / scale
            zero = torch.zeros((), device=image.device, dtype=image.dtype)
            transform = torch.stack(
                (torch.stack((cosine, -sine, zero)), torch.stack((sine, cosine, zero)))
            ).unsqueeze(0).repeat(image.shape[0], 1, 1)
            grid = F.affine_grid(transform, image.shape, align_corners=False)
            image = F.grid_sample(image, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
            edge = F.grid_sample(edge, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            target_grid = F.affine_grid(transform, target.shape, align_corners=False)
            target = F.grid_sample(target, target_grid, mode="nearest", padding_mode="zeros", align_corners=False)
        if image.max() > 1.5:
            image = image / 127.5 - 1.0
        elif image.min() >= 0.0:
            image = image * 2.0 - 1.0
        if edge.max() > 1.5:
            edge = edge / 255.0
        return image.clamp(-1.0, 1.0), edge.clamp(0.0, 1.0), target.long()
