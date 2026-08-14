"""Runtime bridge to the official DiffBoost ControlNet repository.

Official source: https://github.com/NUBagciLab/DiffBoost
Pinned reference commit: 32da5619c9ff03b9f33d521b83254f7a60236e15.
The upstream source and Stable-Diffusion/RadImageNet weights are not copied.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def resolve_path(project_dir: str | Path, path: str | Path) -> Path:
    """Resolve repository files while keeping experiment checkpoints local."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    local = candidate.resolve()
    if local.exists() or (candidate.parts and candidate.parts[0] in {"nam", "outputs", "checkpoints"}):
        return local
    return (Path(project_dir).expanduser() / candidate).resolve()


def import_official_diffboost(project_dir: str | Path) -> SimpleNamespace:
    """Import only the model and DDIM APIs used by the structured package."""
    repository = Path(project_dir).expanduser().resolve()
    root = repository / "diffusion" / "ControlNet"
    if not root.is_dir() and (repository / "cldm").is_dir():
        root = repository
    if not (root / "cldm").is_dir():
        raise FileNotFoundError(
            f"Official DiffBoost ControlNet source was not found at '{root}'. Clone "
            "https://github.com/NUBagciLab/DiffBoost into the configured project_dir."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    loaded_cldm = sys.modules.get("cldm")
    loaded_file = Path(getattr(loaded_cldm, "__file__", "")).resolve() if loaded_cldm else None
    if loaded_file is not None and root not in loaded_file.parents:
        # DiffBoost and several ControlNet projects expose top-level ``cldm``
        # and ``ldm`` packages. Remove a previously imported foreign bridge so
        # this process cannot silently construct the wrong implementation.
        for module_name in list(sys.modules):
            if module_name == "cldm" or module_name.startswith("cldm."):
                sys.modules.pop(module_name, None)
            if module_name == "ldm" or module_name.startswith("ldm."):
                sys.modules.pop(module_name, None)
    model_api = importlib.import_module("cldm.model")
    sampler_class = importlib.import_module("cldm.ddim_hacked").DDIMSampler
    return SimpleNamespace(
        root=root,
        create_model=model_api.create_model,
        load_state_dict=model_api.load_state_dict,
        sampler_class=sampler_class,
    )


def load_checkpoint(
    model: torch.nn.Module, checkpoint: str | Path, loader: Any, strict: bool
) -> tuple[list[str], list[str]]:
    """Load raw, Lightning, or safetensors weights through the official loader."""
    state = loader(str(checkpoint), location="cpu")
    for key in ("state_dict", "model", "module"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(cleaned, strict=bool(strict))
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def build_official_model(config: Any, load_checkpoint: bool) -> torch.nn.Module:
    """Construct the official ControlLDM and optionally restore fine-tuned weights."""
    api = import_official_diffboost(config.project_dir)
    model_config = resolve_path(config.project_dir, config.config)
    if not model_config.is_file():
        raise FileNotFoundError(f"DiffBoost model configuration was not found: {model_config}")
    model = api.create_model(str(model_config)).cpu()
    if load_checkpoint:
        checkpoint = resolve_path(config.project_dir, config.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"DiffBoost checkpoint was not found: {checkpoint}. See the DiffBoost guide."
            )
        load_checkpoint_fn = globals()["load_checkpoint"]
        load_checkpoint_fn(
            model, checkpoint, api.load_state_dict,
            strict=bool(getattr(config, "checkpoint_strict", False)),
        )
    return model
