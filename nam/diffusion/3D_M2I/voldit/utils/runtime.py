"""Pinned official VolDiT import and checkpoint construction helpers."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

OFFICIAL_COMMIT = "76c7063a1d51884dfeb7cd51c63d4191b5358839"


def _require(path: str | Path, purpose: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{purpose} was not found: {resolved}")
    return resolved


def import_official_voldit(project_dir: str | Path) -> dict[str, Any]:
    """Import VolDiT from a user-cloned official repository without vendoring it."""
    root = Path(project_dir).expanduser().resolve()
    if not (root / "src" / "models" / "tgca.py").is_file():
        raise FileNotFoundError(
            f"Official VolDiT source was not found under {root}. See voldit/README.md."
        )
    loaded_src = sys.modules.get("src")
    loaded_file = Path(getattr(loaded_src, "__file__", "") or ".").resolve()
    if loaded_src is not None and root not in loaded_file.parents:
        # Several research repositories use the generic top-level name
        # ``src``. Remove a previously imported foreign namespace before
        # resolving the pinned VolDiT package.
        for name in [key for key in sys.modules if key == "src" or key.startswith("src.")]:
            del sys.modules[name]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return {
            "OmegaConf": importlib.import_module("omegaconf").OmegaConf,
            "VQVAE": importlib.import_module("src.models.vqvae").VQVAE,
            "DiT3D": importlib.import_module("src.models.dit").DiT3D,
            "TGCA3D": importlib.import_module("src.models.tgca").TGCA3D,
            "DDIMScheduler": importlib.import_module("src.models.ddimscheduler").DDIMScheduler,
            "DDPMScheduler": importlib.import_module("src.models.ddpmscheduler").DDPMScheduler,
            "config_utils": importlib.import_module("src.config_utils"),
        }
    except ImportError as error:
        raise ImportError(
            "VolDiT dependencies are unavailable. Install the pinned official requirements."
        ) from error


def _checkpoint_state(path: str | Path, preferred_key: str = "model") -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(_require(path, "VolDiT checkpoint"), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return payload, {}
    state = dict(payload.get(preferred_key, payload.get("state_dict", payload)))
    ema = payload.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
        state.update(ema["shadow"])
    return state, payload


def _load_tgca_checkpoint(model: nn.Module, path: str | Path, strict: bool) -> dict[str, Any]:
    """Load EMA weights exactly as the official VolDiT sampler does."""
    payload = torch.load(_require(path, "VolDiT checkpoint"), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        model.load_state_dict(payload, strict=strict)
        return {}
    ema = payload.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
        shadow = ema["shadow"]
        unknown = set(shadow) - dict(model.named_parameters()).keys()
        if strict and unknown:
            raise RuntimeError(f"Unexpected VolDiT TGCA EMA keys: {sorted(unknown)}")
        for name, parameter in model.named_parameters():
            if name in shadow:
                parameter.data.copy_(shadow[name])
    else:
        model.load_state_dict(payload.get("model", payload.get("state_dict", payload)), strict=strict)
    return payload


class VolDiTComponents(nn.Module):
    """Register all official modules while keeping their native APIs visible."""

    def __init__(self, stage1: nn.Module, tgca: nn.Module, scheduler: Any, scale_factor: float) -> None:
        super().__init__()
        self.stage1, self.tgca = stage1, tgca
        self.scheduler, self.scale_factor = scheduler, float(scale_factor)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.tgca(*args, **kwargs)


def build_official_components(config: Any, load_tgca: bool) -> VolDiTComponents:
    """Instantiate the official VQ-GAN, DiT3D, TGCA, and schedulers."""
    api = import_official_voldit(config.project_dir)
    OmegaConf, utilities = api["OmegaConf"], api["config_utils"]
    root = Path(config.project_dir).expanduser().resolve()
    stage1_config = OmegaConf.load(root / str(getattr(config, "stage1_config", "configs/stage1/vqgan_ds8.yaml")))
    diffusion_config = OmegaConf.load(root / str(getattr(config, "dit_config", "configs/transformer/dit_ds8_l4.yaml")))
    tgca_config = OmegaConf.load(root / str(getattr(config, "tgca_config", "configs/tgca/tgca_ds8.yaml")))
    # Paper data are 192x192x96 and VQ-GAN downsamples by eight. The official
    # public config targets 512x512x256; only positional-grid size is changed.
    diffusion_config.dit.params.input_size = list(getattr(config, "noise_size", (24, 24, 12)))
    stage1 = api["VQVAE"](**utilities.get_stage1_params(stage1_config))
    stage1_state, _ = _checkpoint_state(config.stage1_checkpoint)
    stage1.load_state_dict({k: v for k, v in stage1_state.items() if k in stage1.state_dict()}, strict=False)
    base_dit = api["DiT3D"](**utilities.get_dit_params(diffusion_config))
    dit_state, dit_payload = _checkpoint_state(config.dit_checkpoint)
    # The public checkpoint contains a fixed 512x512x256 sinusoidal grid.
    # The paper crop has a smaller token grid, which DiT3D regenerates during
    # construction; every learned tensor must still load strictly.
    if "pos_embed" in dit_state and dit_state["pos_embed"].shape != base_dit.pos_embed.shape:
        dit_state.pop("pos_embed")
    incompatible = base_dit.load_state_dict(dit_state, strict=False)
    allowed_missing = {"pos_embed"}
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(
            "VolDiT base checkpoint is incompatible: "
            f"missing={sorted(unexpected_missing)}, unexpected={incompatible.unexpected_keys}"
        )
    tgca = api["TGCA3D"](base_model=base_dit, **tgca_config.tgca.params)
    if load_tgca:
        tgca_path = str(getattr(config, "checkpoint", "")).strip()
        if not tgca_path:
            raise ValueError("diffusion.checkpoint must point to a trained TGCA checkpoint.")
        payload = _load_tgca_checkpoint(
            tgca, tgca_path, strict=bool(getattr(config, "strict_checkpoint", True))
        )
    else:
        payload = {}
    scheduler_config = dict(utilities.get_dit_scheduler(diffusion_config))
    scheduler = api["DDIMScheduler"](**scheduler_config)
    scale_factor = float(payload.get("scale_factor", dit_payload.get("scale_factor", getattr(config, "scale_factor", 1.0))))
    return VolDiTComponents(stage1, tgca, scheduler, scale_factor)


def build_training_components(config: Any) -> tuple[VolDiTComponents, Any]:
    """Build a trainable TGCA initialized from official VQ-GAN and DiT weights."""
    components = build_official_components(config, load_tgca=False)
    api = import_official_voldit(config.project_dir)
    root = Path(config.project_dir).expanduser().resolve()
    OmegaConf, utilities = api["OmegaConf"], api["config_utils"]
    diffusion_config = OmegaConf.load(root / str(getattr(config, "dit_config", "configs/transformer/dit_ds8_l4.yaml")))
    scheduler = api["DDPMScheduler"](**dict(utilities.get_dit_scheduler(diffusion_config)))
    return components, scheduler
