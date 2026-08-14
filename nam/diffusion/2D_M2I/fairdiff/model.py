"""Official FairDiff ControlNet adapter with the NAM interface.

FairDiff trains a Stable-Diffusion v1.5 ControlNet to synthesize a medical
image from an RGB segmentation mask and a text prompt. This module preserves
that public implementation and adds the paper's deterministic DDIM and NAM
score-query interfaces.

Official source: https://github.com/wenyi-li/FairDiff
Reference: Li et al., MICCAI 2024, "FairDiff: Fair Segmentation with
Point-Image Diffusion."
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from nam.diffusion.base import (
    DiffusionCondition,
    DiffusionMetadata,
    MedicalDiffusionAdapter,
    ScoreOutput,
    epsilon_to_score,
)
from .utils.data import prepare_fairdiff_condition
from .utils.runtime import build_official_model, import_official_fairdiff


class FairDiffAdapter(MedicalDiffusionAdapter):
    """Frozen official FairDiff model specialized for 2D M2I synthesis."""

    def __init__(self, config: Any) -> None:
        api = import_official_fairdiff(config.project_dir)
        model = build_official_model(config, load_checkpoint=True).cpu()
        model.sd_locked = bool(getattr(config, "sd_locked", False))
        model.only_mid_control = bool(getattr(config, "only_mid_control", False))
        model.control_scales = [float(getattr(config, "control_strength", 1.0))] * 13
        self._sampler_class = api.sampler_class
        self.full_schedule_steps = int(getattr(config, "full_schedule_steps", 50))
        self.resolution = int(getattr(config, "resolution", 256))
        self.default_prompt = str(getattr(config, "default_prompt", "a medical image"))
        self.positive_prompt = str(getattr(config, "positive_prompt", "low quality, blurry"))
        self.negative_prompt = str(
            getattr(config, "negative_prompt", "lowres,extra digit, fewer digits, cropped, worst quality")
        )
        self.mask_encoding = str(getattr(config, "mask_encoding", "palette"))
        self.num_classes = int(getattr(config, "num_classes", 2))
        self.metadata = DiffusionMetadata(
            name="FairDiff",
            official_repository="https://github.com/wenyi-li/FairDiff",
            synthesis_paradigm="M2I-LDM",
            spatial_dims=2,
            prediction_type="epsilon",
            noise_channels=int(getattr(config, "noise_channels", 4)),
            noise_size=tuple(getattr(config, "noise_size", (32, 32))),
        )
        super().__init__(model, config)

    def prepare_condition(self, batch: Any) -> DiffusionCondition:
        """Build FairDiff's RGB mask control and class-aware text condition."""
        control, target, prompts = prepare_fairdiff_condition(
            batch,
            resolution=self.resolution,
            default_prompt=self.default_prompt,
            num_classes=self.num_classes,
            mask_encoding=self.mask_encoding,
        )
        control = control.to(device=self.device, dtype=next(self.model.parameters()).dtype)
        positive = [f"{prompt}, {self.positive_prompt}" for prompt in prompts]
        conditional_text = self.model.get_learned_conditioning(positive)
        negative_text = self.model.get_learned_conditioning([self.negative_prompt] * len(prompts))
        conditional = {"c_concat": [control], "c_crossattn": [conditional_text]}
        # FairDiff uses non-guess-mode CFG, so both branches retain the mask.
        unconditional = {"c_concat": [control], "c_crossattn": [negative_text]}
        return DiffusionCondition(
            conditional=conditional,
            unconditional=unconditional,
            target=target.to(self.device),
            extras={"control": control, "prompts": prompts},
        )

    def _schedule(self, steps: int) -> tuple[Any, np.ndarray]:
        sampler = self._sampler_class(self.model)
        sampler.make_schedule(ddim_num_steps=int(steps), ddim_eta=0.0, verbose=False)
        return sampler, np.flip(sampler.ddim_timesteps)

    def _cfg_epsilon(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        condition: DiffusionCondition,
        scale: float,
    ) -> torch.Tensor:
        conditional = self.model.apply_model(sample, timestep, condition.conditional)
        if scale == 1.0 or condition.unconditional is None:
            return conditional
        unconditional = self.model.apply_model(sample, timestep, condition.unconditional)
        return unconditional + scale * (conditional - unconditional)

    @torch.no_grad()
    def initial_score(
        self, probe_noise: torch.Tensor, condition: DiffusionCondition, cfg_scale: float
    ) -> ScoreOutput:
        sampler, timesteps = self._schedule(self.full_schedule_steps)
        timestep = torch.full(
            (probe_noise.shape[0],), int(timesteps[0]),
            device=probe_noise.device, dtype=torch.long,
        )
        epsilon = self._cfg_epsilon(probe_noise, timestep, condition, cfg_scale)
        alpha_bar = torch.as_tensor(
            sampler.ddim_alphas[len(timesteps) - 1],
            device=probe_noise.device, dtype=probe_noise.dtype,
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
        """Run deterministic DDIM and keep the analytic seed path differentiable."""
        sampler, timesteps = self._schedule(schedule_steps)
        if executed_steps < 1 or executed_steps > len(timesteps):
            raise ValueError(f"executed_steps must be in [1, {len(timesteps)}].")
        sample = initial_noise
        clean = initial_noise
        for reverse_index, raw_timestep in enumerate(timesteps[:executed_steps]):
            schedule_index = len(timesteps) - reverse_index - 1
            timestep = torch.full(
                (sample.shape[0],), int(raw_timestep),
                device=sample.device, dtype=torch.long,
            )
            # NAM does not optimize the 860M-parameter diffusion network; its
            # score is treated as a fixed vector field while gradients follow
            # the deterministic DDIM state transition back to the seed.
            with torch.no_grad():
                epsilon = self._cfg_epsilon(sample.detach(), timestep, condition, cfg_scale)
            alpha = torch.as_tensor(
                sampler.ddim_alphas[schedule_index], device=sample.device, dtype=sample.dtype
            )
            alpha_previous = torch.as_tensor(
                sampler.ddim_alphas_prev[schedule_index], device=sample.device, dtype=sample.dtype
            )
            sigma = torch.as_tensor(
                sampler.ddim_sigmas[schedule_index], device=sample.device, dtype=sample.dtype
            )
            sqrt_one_minus = torch.as_tensor(
                sampler.ddim_sqrt_one_minus_alphas[schedule_index],
                device=sample.device, dtype=sample.dtype,
            )
            clean = (sample - sqrt_one_minus * epsilon) / alpha.sqrt()
            direction = (1.0 - alpha_previous - sigma.square()).clamp_min(0).sqrt() * epsilon
            sample = alpha_previous.sqrt() * clean + direction
        return sample, clean

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        differentiable = getattr(self.model, "differentiable_decode_first_stage", None)
        if differentiable is not None:
            return differentiable(latent)
        scale = float(getattr(self.model, "scale_factor", 1.0))
        return self.model.first_stage_model.decode(latent / scale)

    def truncated_rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        _, clean = self._rollout(
            initial_noise, condition, self.full_schedule_steps, int(steps), cfg_scale
        )
        return self._decode(clean).clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, _ = self._rollout(initial_noise, condition, int(steps), int(steps), cfg_scale)
        return self.model.decode_first_stage(latent).clamp(-1.0, 1.0), condition.target


def build_adapter(config: Any) -> FairDiffAdapter:
    """Build the official FairDiff adapter registered by NAM."""
    return FairDiffAdapter(config)
