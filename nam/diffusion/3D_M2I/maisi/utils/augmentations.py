"""Condition-only 3D augmentation used for fixed-budget MAISI synthesis."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from nam.data import NAMBatch


def _fit(tensor: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    output = tensor
    slices = [slice(None), slice(None)]
    for current, wanted in zip(output.shape[2:], size):
        start = max((current - wanted) // 2, 0)
        slices.append(slice(start, start + min(current, wanted)))
    output = output[tuple(slices)]
    padding = []
    for current, wanted in reversed(list(zip(output.shape[2:], size))):
        total = max(wanted - current, 0)
        padding.extend((total // 2, total - total // 2))
    return F.pad(output, padding)


class MAISIConditionAugmentation:
    """Apply random horizontal flipping and isotropic scaling to MAISI masks."""

    def __init__(self, flip_probability: float = 0.5, scale_range: tuple[float, float] = (0.9, 1.1)) -> None:
        self.flip_probability, self.scale_range = float(flip_probability), tuple(scale_range)

    def __call__(self, batch: NAMBatch) -> NAMBatch:
        target = batch.target.unsqueeze(1) if batch.target.ndim == 4 else batch.target
        source_size = tuple(target.shape[2:])
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        scaled_size = tuple(max(1, round(value * scale)) for value in source_size)
        target = _fit(F.interpolate(target.float(), scaled_size, mode="nearest"), source_size)
        if torch.rand(()) < self.flip_probability:
            target = target.flip(-2)
        condition = dict(batch.condition) if isinstance(batch.condition, dict) else {}
        condition["mask"] = target
        return NAMBatch(batch.image, target[:, 0].long(), condition, batch.sample_id, batch.metadata)
