"""VolDiT TGCA adapter for paper-aligned 3D mask-to-image synthesis.

The architecture is instantiated directly from the official VolDiT source.
NAM only adds score extraction and a differentiable deterministic-DDIM path.

Official source: https://github.com/Cardio-AI/voldit
Pinned reference commit: 76c7063a1d51884dfeb7cd51c63d4191b5358839
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
    v_prediction_to_epsilon,
)
from .utils.data import prepare_voldit_condition
from .utils.runtime import build_official_components


class VolDiTAdapter(MedicalDiffusionAdapter):
    """Frozen VQ-GAN + DiT3D + TGCA implementation used by NAM."""

    def __init__(self, config: Any) -> None:
        components = build_official_components(config, load_tgca=True)
        self.scheduler = components.scheduler
        self.stage1 = components.stage1
        self.tgca = components.tgca
        self.scale_factor = float(components.scale_factor)
        self.volume_size = tuple(int(value) for value in getattr(config, "volume_size", (192, 192, 96)))
        self.full_schedule_steps = int(getattr(config, "full_schedule_steps", 50))
        self.metadata = DiffusionMetadata(
            name="VolDiT",
            official_repository="https://github.com/Cardio-AI/voldit",
            synthesis_paradigm="M2I-DiT",
            spatial_dims=3,
            prediction_type="v_prediction",
            noise_channels=int(getattr(config, "noise_channels", 8)),
            noise_size=tuple(int(value) for value in getattr(config, "noise_size", (24, 24, 12))),
        )
        super().__init__(components, config)

    def prepare_condition(self, batch: Any) -> DiffusionCondition:
        control, target, prompts = prepare_voldit_condition(
            batch, self.volume_size, str(getattr(self.config, "default_prompt", "a medical volume"))
        )
        dtype = next(self.model.parameters()).dtype
        return DiffusionCondition(
            conditional=control.to(self.device, dtype=dtype),
            target=target.to(self.device),
            extras={"prompts": prompts},
        )

    def _timesteps(self, steps: int) -> torch.Tensor:
        self.scheduler.set_timesteps(int(steps), self.device)
        return self.scheduler.timesteps

    def _velocity(
        self, sample: torch.Tensor, timestep: torch.Tensor, condition: DiffusionCondition
    ) -> torch.Tensor:
        return self.tgca(sample, timestep, y=None, condition_input=condition.conditional)

    def _alpha(self, timestep: int | torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        return self.scheduler.alphas_cumprod[int(timestep)].to(sample.device, sample.dtype)

    @torch.no_grad()
    def initial_score(
        self, probe_noise: torch.Tensor, condition: DiffusionCondition, cfg_scale: float
    ) -> ScoreOutput:
        del cfg_scale  # TGCA is mask-conditioned and has no classifier-free text branch.
        timestep_value = self._timesteps(self.full_schedule_steps)[0]
        timestep = torch.full(
            (probe_noise.shape[0],), int(timestep_value), device=probe_noise.device, dtype=torch.long
        )
        velocity = self._velocity(probe_noise, timestep, condition)
        alpha_bar = self._alpha(timestep_value, probe_noise).expand(probe_noise.shape[0])
        epsilon = v_prediction_to_epsilon(velocity, probe_noise, alpha_bar)
        return ScoreOutput(epsilon_to_score(epsilon, alpha_bar), timestep, velocity)

    def _rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        schedule_steps: int,
        executed_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = self._timesteps(schedule_steps)
        if not 1 <= executed_steps <= len(timesteps):
            raise ValueError(f"executed_steps must be in [1, {len(timesteps)}].")
        sample, clean = initial_noise, initial_noise
        stride = self.scheduler.num_train_timesteps // int(schedule_steps)
        for raw_timestep in timesteps[:executed_steps]:
            timestep = torch.full(
                (sample.shape[0],), int(raw_timestep), device=sample.device, dtype=torch.long
            )
            # The frozen denoiser is a stop-gradient vector field; only the
            # analytic DDIM transition remains connected to the selected seed.
            with torch.no_grad():
                velocity = self._velocity(sample.detach(), timestep, condition)
            alpha = self._alpha(raw_timestep, sample)
            previous_timestep = int(raw_timestep) - stride
            alpha_previous = (
                self.scheduler.alphas_cumprod[previous_timestep].to(sample.device, sample.dtype)
                if previous_timestep >= 0 else self.scheduler.final_alpha_cumprod.to(sample.device, sample.dtype)
            )
            beta = 1.0 - alpha
            clean = alpha.sqrt() * sample - beta.sqrt() * velocity
            epsilon = alpha.sqrt() * velocity + beta.sqrt() * sample
            sample = alpha_previous.sqrt() * clean + (1.0 - alpha_previous).sqrt() * epsilon
        return sample, clean

    def _decode_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        # Direct decoder access avoids the non-differentiable nearest-codebook
        # quantization used by the official convenience method.
        return self.stage1.decode(latent / self.scale_factor).clamp(-1.0, 1.0)

    def _decode_official(self, latent: torch.Tensor) -> torch.Tensor:
        """Use the official VQ codebook path for final dataset generation."""
        return self.stage1.decode_stage_2_outputs(latent / self.scale_factor).clamp(-1.0, 1.0)

    def truncated_rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        del cfg_scale
        _, clean = self._rollout(initial_noise, condition, self.full_schedule_steps, int(steps))
        return self._decode_differentiable(clean)

    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cfg_scale
        latent, _ = self._rollout(initial_noise, condition, int(steps), int(steps))
        return self._decode_official(latent), condition.target


def build_adapter(config: Any) -> VolDiTAdapter:
    """Build the official VolDiT adapter registered by NAM."""
    return VolDiTAdapter(config)
