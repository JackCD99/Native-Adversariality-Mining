"""Paper-aligned 2.5D FID utilities for volumetric validation."""

from __future__ import annotations

from typing import Any

import torch

from nam.evaluation.metrics import frechet_distance


def _matrix(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("features", output.get("pooler_output", output.get("logits")))
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.flatten(2).mean(-1) if output.ndim > 2 else output


@torch.no_grad()
def volume_features(extractor: torch.nn.Module, volumes: torch.Tensor, slice_batch: int = 16) -> dict[str, torch.Tensor]:
    """Extract RadImageNet-style features from all three anatomical planes."""
    if volumes.ndim != 5:
        raise ValueError("2.5D FID expects BxCxDxHxW volumes.")
    planes = {
        "axial": volumes.permute(0, 4, 1, 2, 3).reshape(-1, volumes.shape[1], volumes.shape[2], volumes.shape[3]),
        "coronal": volumes.permute(0, 3, 1, 2, 4).reshape(-1, volumes.shape[1], volumes.shape[2], volumes.shape[4]),
        "sagittal": volumes.permute(0, 2, 1, 3, 4).reshape(-1, volumes.shape[1], volumes.shape[3], volumes.shape[4]),
    }
    output = {}
    for name, slices in planes.items():
        chunks = [_matrix(extractor(chunk)).float().cpu() for chunk in slices.split(int(slice_batch))]
        output[name] = torch.cat(chunks)
    return output


def fid_2p5d(real: dict[str, list[torch.Tensor]], synthetic: dict[str, list[torch.Tensor]]) -> float:
    """Average independent axial, coronal, and sagittal Frechet distances."""
    values = []
    for plane in ("axial", "coronal", "sagittal"):
        values.append(frechet_distance(torch.cat(real[plane]).numpy(), torch.cat(synthetic[plane]).numpy()))
    return float(sum(values) / len(values))
