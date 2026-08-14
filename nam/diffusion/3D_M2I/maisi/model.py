"""Official MAISI DDPM/ControlNet adapter with the paper's NAM interface.

This implementation intentionally uses the DDPM-era MAISI configuration cited
by the paper, not the later rectified-flow replacement.

Official source: https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi
Pinned reference commit: 81dcf0f63e2a3a064e882ef0f26d5889b7bedf53
"""

from __future__ import annotations

from typing import Any

import torch

from nam.diffusion.base import (
    DiffusionCondition, DiffusionMetadata, MedicalDiffusionAdapter, ScoreOutput, epsilon_to_score,
)
from .utils.data import prepare_maisi_condition
from .utils.runtime import build_official_components


class MAISIAdapter(MedicalDiffusionAdapter):
    """Frozen official MAISI VAE + diffusion U-Net + ControlNet."""

    def __init__(self, config: Any) -> None:
        components = build_official_components(config, load_controlnet=True)
        self.autoencoder = components.autoencoder
        self.diffusion_unet = components.diffusion_unet
        self.controlnet = components.controlnet
        self.scheduler = components.scheduler
        self.scale_factor = components.scale_factor
        self.volume_size = tuple(int(value) for value in getattr(config, "volume_size", (192, 192, 96)))
        self.full_schedule_steps = int(getattr(config, "full_schedule_steps", 50))
        self.metadata = DiffusionMetadata(
            name="MAISI", official_repository="https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi",
            synthesis_paradigm="M2I-SD", spatial_dims=3, prediction_type="epsilon",
            noise_channels=int(getattr(config, "noise_channels", 4)),
            noise_size=tuple(int(value) for value in getattr(config, "noise_size", (48, 48, 24))),
        )
        super().__init__(components, config)

    def prepare_condition(self, batch: Any) -> DiffusionCondition:
        values = prepare_maisi_condition(
            batch, self.volume_size, str(getattr(self.config, "default_prompt", "a medical volume")),
            tuple(getattr(self.config, "spacing", (1.0, 1.0, 1.0))),
            tuple(getattr(self.config, "top_region", (0.0, 1.0, 0.0, 0.0))),
            tuple(getattr(self.config, "bottom_region", (0.0, 1.0, 0.0, 0.0))),
            int(getattr(self.config, "modality", 1)),
        )
        control, target, extras = values
        control = control.to(self.device, dtype=next(self.model.parameters()).dtype)
        extras = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in extras.items()}
        return DiffusionCondition(control, target=target.to(self.device), extras=extras)

    def _predict(self, sample: torch.Tensor, timestep: torch.Tensor, condition: DiffusionCondition) -> torch.Tensor:
        inputs = {"x": sample, "timesteps": timestep, "controlnet_cond": condition.conditional}
        if self.diffusion_unet.num_class_embeds is not None:
            inputs["class_labels"] = condition.extras["modality"]
        down, middle = self.controlnet(**inputs)
        inputs = {
            "x": sample, "timesteps": timestep, "spacing_tensor": condition.extras["spacing"],
            "down_block_additional_residuals": down, "mid_block_additional_residual": middle,
        }
        if self.diffusion_unet.include_top_region_index_input:
            inputs.update({"top_region_index_tensor": condition.extras["top_region"], "bottom_region_index_tensor": condition.extras["bottom_region"]})
        if self.diffusion_unet.num_class_embeds is not None:
            inputs["class_labels"] = condition.extras["modality"]
        return self.diffusion_unet(**inputs)

    @torch.no_grad()
    def initial_score(self, probe_noise: torch.Tensor, condition: DiffusionCondition, cfg_scale: float) -> ScoreOutput:
        del cfg_scale
        timestep_value = self.scheduler.timesteps(self.full_schedule_steps, self.device)[0]
        timestep = torch.full((probe_noise.shape[0],), int(timestep_value), device=self.device, dtype=torch.long)
        epsilon = self._predict(probe_noise, timestep, condition)
        alpha = self.scheduler.alphas_cumprod[int(timestep_value)].to(probe_noise.device, probe_noise.dtype).expand(probe_noise.shape[0])
        return ScoreOutput(epsilon_to_score(epsilon, alpha), timestep, epsilon)

    def _rollout(self, noise: torch.Tensor, condition: DiffusionCondition, schedule_steps: int, executed_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = self.scheduler.timesteps(schedule_steps, noise.device)
        if not 1 <= executed_steps <= len(timesteps):
            raise ValueError(f"executed_steps must be in [1, {len(timesteps)}].")
        sample, clean = noise, noise
        for index, raw_timestep in enumerate(timesteps[:executed_steps]):
            timestep = torch.full((sample.shape[0],), int(raw_timestep), device=sample.device, dtype=torch.long)
            with torch.no_grad():
                epsilon = self._predict(sample.detach(), timestep, condition)
            alpha = self.scheduler.alphas_cumprod[int(raw_timestep)].to(sample.device, sample.dtype)
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            alpha_previous = self.scheduler.alphas_cumprod[int(previous)].to(sample.device, sample.dtype) if int(previous) >= 0 else sample.new_tensor(1.0)
            clean = (sample - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt()
            sample = alpha_previous.sqrt() * clean + (1.0 - alpha_previous).sqrt() * epsilon
        return sample, clean

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = self.autoencoder.decode_stage_2_outputs(latent / self.model.scale_factor)
        return image.clamp(0.0, 1.0).mul(2.0).sub(1.0)

    def truncated_rollout(self, initial_noise: torch.Tensor, condition: DiffusionCondition, steps: int, cfg_scale: float) -> torch.Tensor:
        del cfg_scale
        _, clean = self._rollout(initial_noise, condition, self.full_schedule_steps, int(steps))
        return self._decode(clean)

    @torch.no_grad()
    def sample(self, initial_noise: torch.Tensor, condition: DiffusionCondition, steps: int, cfg_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        del cfg_scale
        latent, _ = self._rollout(initial_noise, condition, int(steps), int(steps))
        return self._decode(latent), condition.target


def build_adapter(config: Any) -> MAISIAdapter:
    """Build the official MAISI adapter registered by NAM."""
    return MAISIAdapter(config)
