"""Runtime bridge to the official FairDiff repository.

Official source: https://github.com/wenyi-li/FairDiff
Pinned reference commit: 3a0a67ad1f1a3be719b6d529178eeb217a2868a0.
The upstream source and licensed Stable-Diffusion weights are not redistributed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def resolve_path(project_dir: str | Path, path: str | Path) -> Path:
    """Resolve official-repository files and local experiment checkpoints."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    local = candidate.resolve()
    if local.exists() or (candidate.parts and candidate.parts[0] in {"nam", "outputs", "checkpoints"}):
        return local
    return (Path(project_dir).expanduser() / candidate).resolve()


def _remove_foreign_controlnet_modules(root: Path) -> None:
    """Prevent top-level ``cldm``/``ldm`` collisions between public baselines."""
    for package in ("cldm", "ldm"):
        loaded = sys.modules.get(package)
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve() if loaded else None
        if loaded_file is not None and root not in loaded_file.parents:
            for module_name in list(sys.modules):
                if module_name == package or module_name.startswith(f"{package}."):
                    sys.modules.pop(module_name, None)


def import_official_fairdiff(project_dir: str | Path) -> SimpleNamespace:
    """Import the minimal ControlLDM and DDIM APIs used by FairDiff."""
    repository = Path(project_dir).expanduser().resolve()
    root = repository / "MaskImageGen"
    if not (root / "cldm").is_dir() and (repository / "cldm").is_dir():
        root = repository
    if not (root / "cldm").is_dir():
        raise FileNotFoundError(
            f"Official FairDiff MaskImageGen source was not found at '{root}'. Clone "
            "https://github.com/wenyi-li/FairDiff into the configured project_dir."
        )
    _remove_foreign_controlnet_modules(root)
    # Reinsert at position zero even if a previous call left this path later
    # in sys.path; another baseline may have since prepended its own ``cldm``.
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != root]
    sys.path.insert(0, str(root))
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
    """Load official raw, Lightning, or safetensors checkpoint layouts."""
    state = loader(str(checkpoint), location="cpu")
    for key in ("state_dict", "model", "module"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(cleaned, strict=bool(strict))
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def build_official_model(config: Any, load_checkpoint: bool) -> torch.nn.Module:
    """Construct official FairDiff ControlLDM and optionally restore weights."""
    api = import_official_fairdiff(config.project_dir)
    model_config = resolve_path(config.project_dir, config.config)
    if not model_config.is_file():
        raise FileNotFoundError(f"FairDiff model configuration was not found: {model_config}")
    model = api.create_model(str(model_config)).cpu()
    if load_checkpoint:
        checkpoint = resolve_path(config.project_dir, config.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"FairDiff checkpoint was not found: {checkpoint}. See fairdiff/README.md."
            )
        globals()["load_checkpoint"](
            model, checkpoint, api.load_state_dict,
            strict=bool(getattr(config, "checkpoint_strict", False)),
        )
    return model
