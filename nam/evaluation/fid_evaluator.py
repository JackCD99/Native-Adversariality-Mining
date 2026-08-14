"""Modality-aware 2D and 2.5D FID evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nam.data import build_dataset, collate_medical_batch
from nam.evaluation.metrics import frechet_distance
from nam.utils.imports import import_factory


def _feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("features", output.get("pooler_output", output.get("logits")))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim > 2:
        output = output.flatten(2).mean(-1)
    return output


@torch.no_grad()
def _extract_2d(loader: DataLoader, extractor: torch.nn.Module, device: torch.device) -> np.ndarray:
    features = []
    for batch in tqdm(loader, desc="Extracting FID features", leave=False):
        batch = batch.to(device)
        if batch.image is None:
            raise ValueError("FID datasets must return an image tensor.")
        features.append(_feature_tensor(extractor(batch.image)).float().cpu())
    return torch.cat(features).numpy()


@torch.no_grad()
def _extract_3d_views(
    loader: DataLoader, extractor: torch.nn.Module, device: torch.device
) -> dict[str, np.ndarray]:
    collected: dict[str, list[torch.Tensor]] = {"axial": [], "coronal": [], "sagittal": []}
    for batch in tqdm(loader, desc="Extracting 2.5D FID features", leave=False):
        batch = batch.to(device)
        images = batch.image
        if images is None or images.ndim != 5:
            raise ValueError("2.5D FID expects BCDHW image volumes.")
        batch_size, channels, depth, height, width = images.shape
        planes = {
            "axial": images.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width),
            "coronal": images.permute(0, 3, 1, 2, 4).reshape(batch_size * height, channels, depth, width),
            "sagittal": images.permute(0, 4, 1, 2, 3).reshape(batch_size * width, channels, depth, height),
        }
        for name, slices in planes.items():
            collected[name].append(_feature_tensor(extractor(slices)).float().cpu())
    return {name: torch.cat(values).numpy() for name, values in collected.items()}


def evaluate_fid(config: Any, spatial_dims: int) -> dict[str, float]:
    """Evaluate modality-specific FID using a configured feature extractor.

    The extractor factory receives ``config.fid`` and returns an evaluation-mode
    PyTorch module. MRI/CT experiments should use RadImageNet ResNet-50; RGB uses
    Inception-v3. Three-dimensional experiments report the mean of axial,
    coronal, and sagittal FID, matching the paper's 2.5D protocol.
    """
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    factory = import_factory(config.fid.feature_factory, getattr(config.fid, "project_dir", None))
    extractor = factory(config=config.fid).to(device).eval()

    def make_loader(dataset_config: Any, split: str) -> DataLoader:
        return DataLoader(
            build_dataset(dataset_config, split, spatial_dims),
            batch_size=int(config.fid.batch_size),
            shuffle=False,
            num_workers=int(config.runtime.num_workers),
            collate_fn=collate_medical_batch,
        )

    real_loader = make_loader(config.dataset, str(config.fid.real_split))
    synthetic_loader = make_loader(config.synthetic_dataset, str(config.fid.synthetic_split))
    if spatial_dims == 2:
        real = _extract_2d(real_loader, extractor, device)
        synthetic = _extract_2d(synthetic_loader, extractor, device)
        result = {"fid": frechet_distance(real, synthetic)}
    else:
        real_views = _extract_3d_views(real_loader, extractor, device)
        synthetic_views = _extract_3d_views(synthetic_loader, extractor, device)
        result = {
            f"fid_{view}": frechet_distance(real_views[view], synthetic_views[view])
            for view in real_views
        }
        result["fid"] = float(np.mean(list(result.values())))

    output = Path(config.fid.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result

