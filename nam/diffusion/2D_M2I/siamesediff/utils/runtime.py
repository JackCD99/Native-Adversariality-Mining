"""Runtime integration with the official Siamese-Diffusion repository.

Official source: https://github.com/Qiukunpeng/Siamese-Diffusion
The upstream project is MIT licensed. This module imports it at runtime and
does not redistribute Stable Diffusion, CLIP, or pretrained weights.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def resolve_project_path(project_dir: str | Path, path: str | Path) -> Path:
    """Resolve an official configuration or checkpoint relative to its repository."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    repository_local = candidate.resolve()
    if repository_local.exists() or candidate.parts[0] in {
        "nam",
        "checkpoints",
        "outputs",
        "pretrained_weights",
    }:
        return repository_local
    return (Path(project_dir).expanduser() / candidate).resolve()


def import_official_siamesediff(project_dir: str | Path) -> SimpleNamespace:
    """Expose the official repository and return only APIs used by this package."""
    root = Path(project_dir).expanduser().resolve()
    if not (root / "cldm").is_dir():
        raise FileNotFoundError(
            f"Official Siamese-Diffusion source was not found at '{root}'. Clone "
            "https://github.com/Qiukunpeng/Siamese-Diffusion into that directory."
        )
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    model_api = importlib.import_module("cldm.model")
    try:
        sampler_class = importlib.import_module("cldm.ddim_hacked").DDIMSampler
    except (ImportError, AttributeError):
        sampler_class = importlib.import_module("ldm.models.diffusion.ddim").DDIMSampler
    return SimpleNamespace(
        create_model=model_api.create_model,
        load_state_dict=model_api.load_state_dict,
        sampler_class=sampler_class,
    )


def load_official_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    loader: Any,
    strict: bool,
) -> tuple[list[str], list[str]]:
    """Load standard, Lightning, DeepSpeed-merged, or raw upstream weights."""
    state = loader(str(checkpoint), location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(cleaned, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


class TrainerState:
    """Minimal Lightning trainer state required by upstream Siamese losses."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.global_step = 0


def attach_trainer_state(model: torch.nn.Module, max_steps: int) -> TrainerState:
    """Attach the two trainer fields read by the official ``p_losses`` method."""
    state = TrainerState(max_steps)
    model._trainer = state
    return state
