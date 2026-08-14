"""Runtime builders for official Diffusers and the authors' legacy UNet.

The official repository is CC BY-NC 4.0, so its source and weights are not
redistributed here. The public backend reconstructs its documented Diffusers
UNet exactly. The legacy bridge loads the minimal ``semi_diffseg`` source used
by the authors' existing server checkpoints without copying unrelated code.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class SegDiffRuntime:
    model: torch.nn.Module
    scheduler: Any
    backend: str


def _diffusers() -> tuple[Any, Any]:
    try:
        from diffusers import DDIMScheduler, UNet2DModel
    except ImportError as error:
        raise ImportError(
            "SegDiff requires Diffusers. Install the diffusion extra with "
            "`pip install -e .[diffusion]`."
        ) from error
    return UNet2DModel, DDIMScheduler


def _official_model(config: Any) -> torch.nn.Module:
    UNet2DModel, _ = _diffusers()
    return UNet2DModel(
        sample_size=int(getattr(config, "resolution", 256)),
        in_channels=int(getattr(config, "image_channels", 3))
        + int(getattr(config, "condition_channels", 1)),
        out_channels=int(getattr(config, "image_channels", 3)),
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D",
        ),
    )


def _legacy_model(config: Any) -> torch.nn.Module:
    root = Path(config.project_dir).expanduser().resolve()
    if not (root / "semi_diffseg").is_dir():
        raise FileNotFoundError(
            f"Legacy SegDiff source was not found at '{root}'. See the SegDiff guide."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    factory_path = str(
        getattr(config, "legacy_factory", "semi_diffseg.denoise_model.denoise_unet:UNetModel")
    )
    module_name, attribute = factory_path.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(**dict(getattr(config, "legacy_model_kwargs", {})))


def _checkpoint_path(config: Any) -> Path:
    path = Path(str(config.checkpoint)).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _unwrap_state(payload: Any) -> dict[str, torch.Tensor]:
    state = payload
    for key in ("state_dict", "model", "module", "denoise_model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError("SegDiff checkpoint does not contain a PyTorch state dictionary.")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def build_runtime(config: Any, load_checkpoint: bool) -> SegDiffRuntime:
    """Build one supported architecture and optionally restore its checkpoint."""
    UNet2DModel, DDIMScheduler = _diffusers()
    backend = str(getattr(config, "backend", "official_diffusers")).lower()
    checkpoint = _checkpoint_path(config) if load_checkpoint else None
    if backend == "official_diffusers" and checkpoint is not None and checkpoint.is_dir():
        unet_dir = checkpoint / "unet" if (checkpoint / "unet").is_dir() else checkpoint
        model = UNet2DModel.from_pretrained(str(unet_dir))
    elif backend == "official_diffusers":
        model = _official_model(config)
    elif backend == "legacy_unet":
        model = _legacy_model(config)
    else:
        raise ValueError("SegDiff backend must be 'official_diffusers' or 'legacy_unet'.")

    scheduler_dir = checkpoint / "scheduler" if checkpoint is not None else None
    if scheduler_dir is not None and scheduler_dir.is_dir():
        scheduler = DDIMScheduler.from_pretrained(str(scheduler_dir))
    else:
        scheduler = DDIMScheduler(
            num_train_timesteps=int(getattr(config, "num_train_timesteps", 1000)),
            beta_schedule=str(getattr(config, "beta_schedule", "linear")),
            prediction_type="epsilon",
            clip_sample=bool(getattr(config, "clip_sample", True)),
        )
    if load_checkpoint:
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"SegDiff checkpoint was not found: {checkpoint}")
        if checkpoint.is_file():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(
                _unwrap_state(payload), strict=bool(getattr(config, "checkpoint_strict", True))
            )
    return SegDiffRuntime(model=model, scheduler=scheduler, backend=backend)


def model_output_tensor(output: Any) -> torch.Tensor:
    """Normalize Diffusers, tuple, and legacy UNet outputs to a tensor."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unsupported SegDiff model output: {type(output)!r}")
