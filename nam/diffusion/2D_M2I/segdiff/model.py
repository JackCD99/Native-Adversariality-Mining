"""SegDiff adapter for official training, deterministic sampling, and NAM.

The upstream method concatenates a segmentation map with the noisy image and
predicts epsilon with a conditional UNet. This module keeps that formulation
and adds only the NAM score query and differentiable early-DDIM seed path.

Official source: https://github.com/mazurowski-lab/segmentation-guided-diffusion
Reference: Konz et al., MICCAI 2024.
"""

from __future__ import annotations

from typing import Any

import torch

from nam.diffusion.base import (
    DiffusionCondition,
    DiffusionMetadata,
    MedicalDiffusionAdapter,
    ScoreOutput,
    epsilon_to_score,
)
from .utils.data import prepare_condition_tensors
from .utils.runtime import build_runtime, model_output_tensor


class SegDiffAdapter(MedicalDiffusionAdapter):
    """Frozen SegDiff epsilon predictor with the common 2D M2I interface."""

    def __init__(self, config: Any) -> None:
        runtime = build_runtime(config, load_checkpoint=True)
        self.scheduler = runtime.scheduler
        self.backend = runtime.backend
        self.full_schedule_steps = int(getattr(config, "full_schedule_steps", 50))
        self.image_channels = int(getattr(config, "image_channels", 3))
        self.condition_channels = int(getattr(config, "condition_channels", 1))
        self.resolution = int(getattr(config, "resolution", 256))
        self.condition_encoding = str(getattr(config, "condition_encoding", "binary"))
        self.num_classes = int(getattr(config, "num_classes", 2))
        self.unconditional_value = float(getattr(config, "unconditional_value", 0.0))
        self.clip_sample = bool(getattr(config, "clip_sample", True))
        self.metadata = DiffusionMetadata(
            name="SegDiff",
            official_repository="https://github.com/mazurowski-lab/segmentation-guided-diffusion",
            synthesis_paradigm="M2I-DPM",
            spatial_dims=2,
            prediction_type="epsilon",
            noise_channels=self.image_channels,
            noise_size=(self.resolution, self.resolution),
        )
        super().__init__(runtime.model, config)

    def prepare_condition(self, batch: Any) -> DiffusionCondition:
        condition, target = prepare_condition_tensors(
            batch,
            resolution=self.resolution,
            encoding=self.condition_encoding,
            num_classes=self.num_classes,
            condition_channels=self.condition_channels,
        )
        condition = condition.to(device=self.device, dtype=next(self.model.parameters()).dtype)
        unconditional = torch.full_like(condition, self.unconditional_value)
        return DiffusionCondition(
            conditional=condition,
            unconditional=unconditional,
            target=target.to(self.device),
            extras={"backend": self.backend},
        )

    def _epsilon(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        condition: DiffusionCondition,
        cfg_scale: float,
    ) -> torch.Tensor:
        conditional = model_output_tensor(
            self.model(torch.cat((sample, condition.conditional), dim=1), timestep)
        )
        if cfg_scale == 1.0 or condition.unconditional is None:
            return conditional
        unconditional = model_output_tensor(
            self.model(torch.cat((sample, condition.unconditional), dim=1), timestep)
        )
        return unconditional + cfg_scale * (conditional - unconditional)

    def _timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        self.scheduler.set_timesteps(int(steps), device=device)
        return torch.as_tensor(self.scheduler.timesteps, device=device, dtype=torch.long)

    @torch.no_grad()
    def initial_score(
        self, probe_noise: torch.Tensor, condition: DiffusionCondition, cfg_scale: float
    ) -> ScoreOutput:
        timesteps = self._timesteps(self.full_schedule_steps, probe_noise.device)
        timestep_index = int(timesteps[0].item())
        timestep = timesteps[0].expand(probe_noise.shape[0])
        epsilon = self._epsilon(probe_noise, timestep, condition, cfg_scale)
        alpha_bar = self.scheduler.alphas_cumprod[timestep_index].to(
            device=probe_noise.device, dtype=probe_noise.dtype
        ).expand(probe_noise.shape[0])
        return ScoreOutput(epsilon_to_score(epsilon, alpha_bar), timestep, epsilon)

    def _rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        schedule_steps: int,
        executed_steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run eta-zero DDIM while stopping gradients through UNet queries."""
        timesteps = self._timesteps(schedule_steps, initial_noise.device)
        if executed_steps < 1 or executed_steps > len(timesteps):
            raise ValueError(f"executed_steps must be in [1, {len(timesteps)}].")
        sample = initial_noise
        clean = initial_noise
        for index, scalar_timestep in enumerate(timesteps[:executed_steps]):
            timestep_index = int(scalar_timestep.item())
            timestep = scalar_timestep.expand(sample.shape[0])
            with torch.no_grad():
                epsilon = self._epsilon(sample.detach(), timestep, condition, cfg_scale)
            alpha = self.scheduler.alphas_cumprod[timestep_index].to(sample).clamp_min(1e-12)
            clean = (sample - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt()
            if self.clip_sample:
                clean = clean.clamp(-1.0, 1.0)
            if index + 1 < len(timesteps):
                previous_timestep = timesteps[index + 1]
                previous_alpha = self.scheduler.alphas_cumprod[
                    int(previous_timestep.item())
                ].to(sample)
            else:
                previous_alpha = torch.as_tensor(
                    getattr(self.scheduler, "final_alpha_cumprod", 1.0),
                    device=sample.device,
                    dtype=sample.dtype,
                )
            sample = previous_alpha.sqrt() * clean + (1.0 - previous_alpha).sqrt() * epsilon
        return sample, clean

    def truncated_rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Return the early clean estimate used by the frozen downstream anchor."""
        _, clean = self._rollout(
            initial_noise, condition, self.full_schedule_steps, int(steps), cfg_scale
        )
        return clean

    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = self._timesteps(int(steps), initial_noise.device)
        images = initial_noise
        for scalar_timestep in timesteps:
            timestep = scalar_timestep.expand(images.shape[0])
            epsilon = self._epsilon(images, timestep, condition, cfg_scale)
            images = self.scheduler.step(
                epsilon, int(scalar_timestep.item()), images, eta=0.0
            ).prev_sample
        return images.clamp(-1.0, 1.0), condition.target


def build_adapter(config: Any) -> SegDiffAdapter:
    """Build the specialized SegDiff adapter registered by NAM."""
    return SegDiffAdapter(config)
