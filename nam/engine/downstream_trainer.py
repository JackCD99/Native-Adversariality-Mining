"""Route Table I runs to model-native real or synthetic training pipelines."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable


PIPELINES = {
    "nnunet": "nam.downstream.nnunet",
    "swinunet": "nam.downstream.swinunet",
    "swinunetr": "nam.downstream.swinunet",
    "samed": "nam.downstream.samed",
    "deeplabv3": "nam.downstream.deeplabv3",
    "mask2former": "nam.downstream.mask2former",
    "resnet50": "nam.downstream.resnet50",
    "vits16": "nam.downstream.vit_s16",
}


def _resolve(config: Any, phase: str) -> Callable[[Any, int], Path]:
    name = str(config.downstream.name).lower().replace("-", "").replace("_", "")
    if name not in PIPELINES:
        raise KeyError(f"Unknown downstream model '{name}'. Available: {sorted(PIPELINES)}")
    if phase == "real":
        return import_module(f"{PIPELINES[name]}.train_real").train_real
    if phase in {"synthetic", "syn"}:
        return import_module(f"{PIPELINES[name]}.train_syn").train_synthetic
    raise ValueError("phase must be 'real' or 'synthetic'.")


def run_downstream_training(config: Any, spatial_dims: int, phase: str = "synthetic") -> Path:
    """Run one architecture's real or real-plus-synthetic training pipeline."""
    configured_dims = int(config.downstream.spatial_dims)
    if configured_dims != spatial_dims:
        raise ValueError(
            f"Script requests {spatial_dims}D but downstream.spatial_dims={configured_dims}."
        )
    config.downstream.dataset_name = Path(str(config.dataset.root)).name
    config.downstream.generator_name = str(getattr(config.diffusion, "name", "generator"))
    return _resolve(config, phase)(config, spatial_dims)
