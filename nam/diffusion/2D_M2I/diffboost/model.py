"""Official DiffBoost ControlNet adapter with the NAM interface.

DiffBoost fine-tunes a Stable-Diffusion v1.5 ControlNet on paired medical
images, structure edges, and text prompts. This adapter preserves the official
three-way prompt mixture and adds only NAM's score query and differentiable
early-DDIM seed path.

Official source: https://github.com/NUBagciLab/DiffBoost
Reference: Zhang et al., IEEE Transactions on Medical Imaging, 2024.
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
from .utils.data import prepare_diffboost_condition
from .utils.runtime import build_official_model, import_official_diffboost


class DiffBoostAdapter(MedicalDiffusionAdapter):
    """Frozen official DiffBoost model specialized for 2D M2I synthesis."""

    def __init__(self, config: Any) -> None:
        api = import_official_diffboost(config.project_dir)
        model = build_official_model(config, load_checkpoint=True).cpu()
        model.sd_locked = bool(getattr(config, "sd_locked", True))
        model.only_mid_control = bool(getattr(config, "only_mid_control", False))
        model.control_scales = [float(getattr(config, "control_strength", 1.0))] * 13
        self._sampler_class = api.sampler_class
        self.full_schedule_steps = int(getattr(config, "full_schedule_steps", 20))
        self.resolution = int(getattr(config, "resolution", 256))
        self.default_prompt = str(getattr(config, "default_prompt", "a medical image"))
        self.augmentation_prompt = str(
            getattr(config, "augmentation_prompt", "Image enhancement")
        )
        self.positive_prompt = str(
            getattr(config, "positive_prompt", "best quality, extremely detailed")
        )
        self.negative_prompt = str(
            getattr(
                config,
                "negative_prompt",
                "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
            )
        )
        self.prompt_weights = tuple(
            float(value) for value in getattr(config, "prompt_weights", (0.045, 0.005, 0.95))
        )
        if len(self.prompt_weights) != 3:
            raise ValueError("DiffBoost prompt_weights must contain exactly three values.")
        self.condition_mode = str(getattr(config, "condition_mode", "mask"))
        self.num_classes = int(getattr(config, "num_classes", 2))
        self.metadata = DiffusionMetadata(
            name="DiffBoost",
            official_repository="https://github.com/NUBagciLab/DiffBoost",
            synthesis_paradigm="M2I-SD",
            spatial_dims=2,
            prediction_type="epsilon",
            noise_channels=int(getattr(config, "noise_channels", 4)),
            noise_size=tuple(getattr(config, "noise_size", (32, 32))),
        )
        super().__init__(model, config)

    def prepare_condition(self, batch: Any) -> DiffusionCondition:
        """Build the edge ControlNet hint and official three-way text mixture."""
        control, target, prompts, augmentation_prompts = prepare_diffboost_condition(
            batch,
            resolution=self.resolution,
            default_prompt=self.default_prompt,
            default_augmentation_prompt=self.augmentation_prompt,
            condition_mode=self.condition_mode,
            num_classes=self.num_classes,
        )
        control = control.to(device=self.device, dtype=next(self.model.parameters()).dtype)
        base = self.model.get_learned_conditioning(
            [f"{prompt},{self.positive_prompt}" for prompt in prompts]
        )
        augmentation = self.model.get_learned_conditioning(augmentation_prompts)
        sketch = self.model.get_learned_conditioning(["gray, sketch"] * len(prompts))
        w_base, w_augmentation, w_sketch = self.prompt_weights
        mixed_text = w_base * base + w_augmentation * augmentation + w_sketch * sketch
        negative = self.model.get_learned_conditioning([self.negative_prompt] * len(prompts))
        conditional = {"c_concat": [control], "c_crossattn": [mixed_text]}
        # Official non-guess-mode CFG retains the same structure control.
        unconditional = {"c_concat": [control], "c_crossattn": [negative]}
        return DiffusionCondition(
            conditional=conditional,
            unconditional=unconditional,
            target=target.to(self.device),
            extras={
                "control": control,
                "prompts": prompts,
                "augmentation_prompts": augmentation_prompts,
            },
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
        """Run deterministic DDIM with stop-gradient through score evaluations."""
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


def build_adapter(config: Any) -> DiffBoostAdapter:
    """Build the official DiffBoost adapter registered by NAM."""
    return DiffBoostAdapter(config)
