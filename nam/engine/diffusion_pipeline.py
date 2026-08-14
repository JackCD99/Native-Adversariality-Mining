"""Dispatch generator-specific training, NAM optimization, and sampling pipelines."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable


PIPELINES = {
    "segdiff": ("nam.diffusion.2D_M2I.segdiff", 2),
    "diffboost": ("nam.diffusion.2D_M2I.diffboost", 2),
    "fairdiff": ("nam.diffusion.2D_M2I.fairdiff", 2),
    "siamesediff": ("nam.diffusion.2D_M2I.siamesediff", 2),
    "jodiffusion": ("nam.diffusion.2D_M&I.jodiffusion", 2),
    "medsegfactory": ("nam.diffusion.2D_M&I.medsegfactory", 2),
    "voldit": ("nam.diffusion.3D_M2I.voldit", 3),
    "maisi": ("nam.diffusion.3D_M2I.maisi", 3),
    "controlnet_sdxl": ("nam.diffusion.2D_M2I.controlnet_sdxl", 2),
    "sd15_lora": ("nam.diffusion.2D_T2I.sd15_lora", 2),
}


def _resolve(config: Any, spatial_dims: int, module: str, function: str) -> Callable[..., Path | None]:
    name = str(config.diffusion.name).lower()
    if name not in PIPELINES:
        raise KeyError(f"Unknown diffusion pipeline '{name}'. Available: {sorted(PIPELINES)}")
    package, expected_dims = PIPELINES[name]
    if spatial_dims != expected_dims:
        raise ValueError(f"{name} is a {expected_dims}D pipeline, not a {spatial_dims}D pipeline.")
    entry = getattr(import_module(f"{package}.{module}"), function, None)
    if not callable(entry):
        raise AttributeError(f"{package}.{module} does not define callable '{function}'.")
    return entry


def run_diffusion_pretraining(config: Any, spatial_dims: int) -> Path | None:
    """Run the selected generator's model-native pre-training pipeline."""
    return _resolve(config, spatial_dims, "pre_training", "train_pretrained_model")(config)


def run_nam_training(config: Any, spatial_dims: int) -> Path | None:
    """Run the selected generator's compatible NAM miner pipeline."""
    return _resolve(config, spatial_dims, "NAM_training", "train_nam")(config)


def run_sampling(config: Any, spatial_dims: int, use_nam: bool = True) -> Path:
    """Run fixed-budget Base or NAM sampling for the selected generator."""
    result = _resolve(config, spatial_dims, "sampling", "sample_dataset")(config, use_nam)
    if result is None:
        raise RuntimeError("The sampling pipeline completed without returning an output directory.")
    return result
