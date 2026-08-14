"""Paper-aligned axial/coronal/sagittal 2.5D FID for MAISI."""

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
    planes = {
        "axial": volumes.permute(0, 4, 1, 2, 3).reshape(-1, volumes.shape[1], volumes.shape[2], volumes.shape[3]),
        "coronal": volumes.permute(0, 3, 1, 2, 4).reshape(-1, volumes.shape[1], volumes.shape[2], volumes.shape[4]),
        "sagittal": volumes.permute(0, 2, 1, 3, 4).reshape(-1, volumes.shape[1], volumes.shape[3], volumes.shape[4]),
    }
    return {name: torch.cat([_matrix(extractor(chunk)).float().cpu() for chunk in slices.split(int(slice_batch))]) for name, slices in planes.items()}


def fid_2p5d(real: dict[str, list[torch.Tensor]], synthetic: dict[str, list[torch.Tensor]]) -> float:
    values = [frechet_distance(torch.cat(real[key]).numpy(), torch.cat(synthetic[key]).numpy()) for key in ("axial", "coronal", "sagittal")]
    return float(sum(values) / 3.0)
