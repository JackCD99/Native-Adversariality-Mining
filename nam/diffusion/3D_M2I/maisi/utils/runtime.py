"""Official MAISI model construction and paper DDIM schedule utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

OFFICIAL_COMMIT = "81dcf0f63e2a3a064e882ef0f26d5889b7bedf53"


class DeterministicDDIMScheduler:
    """DDIM eta=0 over MAISI's official scaled-linear DDPM beta schedule."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0015, beta_end: float = 0.0195) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps).square()
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        if not 1 <= int(steps) <= self.num_train_timesteps:
            raise ValueError("DDIM steps must be between one and the training timestep count.")
        stride = self.num_train_timesteps // int(steps)
        return (torch.arange(int(steps), device=device) * stride).round().long().flip(0)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha = self.alphas_cumprod.to(clean.device, clean.dtype)[timesteps]
        while alpha.ndim < clean.ndim:
            alpha = alpha.unsqueeze(-1)
        return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise


class MAISIComponents(nn.Module):
    def __init__(self, autoencoder: nn.Module, diffusion_unet: nn.Module, controlnet: nn.Module, scheduler: Any, scale_factor: torch.Tensor) -> None:
        super().__init__()
        self.autoencoder, self.diffusion_unet, self.controlnet = autoencoder, diffusion_unet, controlnet
        self.scheduler = scheduler
        self.register_buffer("scale_factor", scale_factor.float())

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.diffusion_unet(*args, **kwargs)


def _require(path: str | Path, purpose: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{purpose} was not found: {result}")
    return result


def _instantiate(config: dict[str, Any], key: str) -> Any:
    try:
        from monai.bundle import ConfigParser
    except ImportError as error:
        raise ImportError("MAISI requires MONAI with monai.apps.generation.maisi support.") from error
    parser = ConfigParser(config)
    parser.parse(True)
    return parser.get_parsed_content(key, instantiate=True)


def build_official_components(config: Any, load_controlnet: bool) -> MAISIComponents:
    """Instantiate models from the official DDPM JSON and load its checkpoint schema."""
    root = Path(config.project_dir).expanduser().resolve()
    configuration_path = root / str(getattr(config, "official_config", "configs/config_maisi3d-ddpm.json"))
    with _require(configuration_path, "MAISI DDPM configuration").open("r", encoding="utf-8") as stream:
        definition = json.load(stream)
    autoencoder = _instantiate(definition, "autoencoder_def")
    diffusion_unet = _instantiate(definition, "diffusion_unet_def")
    controlnet = _instantiate(definition, "controlnet_def")
    autoencoder_state = torch.load(_require(config.autoencoder_checkpoint, "MAISI autoencoder checkpoint"), map_location="cpu", weights_only=False)
    autoencoder.load_state_dict(autoencoder_state.get("state_dict", autoencoder_state) if isinstance(autoencoder_state, dict) else autoencoder_state)
    diffusion_payload = torch.load(_require(config.diffusion_checkpoint, "MAISI diffusion checkpoint"), map_location="cpu", weights_only=False)
    diffusion_unet.load_state_dict(diffusion_payload["unet_state_dict"], strict=True)
    from monai.networks.utils import copy_model_state
    copy_model_state(controlnet, diffusion_unet.state_dict())
    if load_controlnet:
        payload = torch.load(_require(config.checkpoint, "MAISI ControlNet checkpoint"), map_location="cpu", weights_only=False)
        controlnet.load_state_dict(payload["controlnet_state_dict"], strict=True)
    scale = diffusion_payload.get("scale_factor", torch.tensor(float(getattr(config, "scale_factor", 1.0))))
    scale = scale.detach().clone() if torch.is_tensor(scale) else torch.tensor(float(scale))
    schedule = definition["noise_scheduler"]
    scheduler = DeterministicDDIMScheduler(int(schedule["num_train_timesteps"]), float(schedule["beta_start"]), float(schedule["beta_end"]))
    return MAISIComponents(autoencoder, diffusion_unet, controlnet, scheduler, scale)
