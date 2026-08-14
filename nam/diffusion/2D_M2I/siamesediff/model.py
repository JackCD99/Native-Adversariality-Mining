"""Official SiameseDiff model adapter with the NAM score/rollout interface.

SiameseDiff is loaded from its official repository instead of being copied.
This module preserves its SD-v1.5 VAE, CLIP text encoder, DHI-conditioned
ControlNet, and mask-conditioned epsilon predictor. NAM only adds initial-score
queries and a differentiable DDIM seed path.

Official implementation: https://github.com/Qiukunpeng/Siamese-Diffusion
Reference: Qiu et al., Noise-Consistent Siamese-Diffusion, CVPR 2025.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from nam.diffusion.base import (
    DiffusionCondition,
    DiffusionMetadata,
    MedicalDiffusionAdapter,
    ScoreOutput,
    epsilon_to_score,
)
from .utils.augmentations import PairedAugmentation
from .utils.data import to_official_batch
from .utils.runtime import (
    import_official_siamesediff,
    load_official_checkpoint,
    resolve_project_path,
)


class SiameseDiffAdapter(MedicalDiffusionAdapter):
    """Frozen official SiameseDiff model specialized for 2D M2I synthesis."""

    def __init__(self, config: Any) -> None:
        api = import_official_siamesediff(config.project_dir)
        model_config = resolve_project_path(config.project_dir, config.config)
        checkpoint = resolve_project_path(config.project_dir, config.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SiameseDiff checkpoint was not found at '{checkpoint}'. See "
                "nam/diffusion/2D_M2I/siamesediff/README.md."
            )
        model = api.create_model(str(model_config)).cpu()
        load_official_checkpoint(model, checkpoint, api.load_state_dict, strict=False)
        model.sd_locked = bool(getattr(config, "sd_locked", False))
        model.only_mid_control = bool(getattr(config, "only_mid_control", False))
        model.learning_rate = float(getattr(config, "learning_rate", 1e-5))
        metadata = DiffusionMetadata(
            name="SiameseDiff",
            official_repository="https://github.com/Qiukunpeng/Siamese-Diffusion",
            synthesis_paradigm="M2I-SD",
            spatial_dims=2,
            prediction_type="epsilon",
            noise_channels=int(getattr(config, "noise_channels", 4)),
            noise_size=tuple(getattr(config, "noise_size", (32, 32))),
        )
        self._sampler_class = api.sampler_class
        self.full_schedule_steps = int(getattr(config, "full_schedule_steps", 50))
        self.default_prompt = str(
            getattr(config, "default_prompt", "a colonoscopy image of a polyp")
        )
        self.condition_augmentation = PairedAugmentation(
            int(getattr(config, "resolution", 256))
        )
        super().__init__(model, config)

    def prepare_condition(self, batch: Any) -> DiffusionCondition:
        """Build official mask/text conditional and unconditional dictionaries."""
        official = to_official_batch(
            batch,
            self.condition_augmentation,
            self.default_prompt,
            prompt_dropout=0.0,
            training=False,
        )
        hint = official["hint"].permute(0, 3, 1, 2).contiguous()
        prompts = official["txt"]
        text = self.model.get_learned_conditioning(prompts)
        empty_text = self.model.get_unconditional_conditioning(len(prompts))
        conditional = {"c_concat": [hint], "c_crossattn": [text]}
        # Official inference retains the mask for unconditional CFG.
        unconditional = {"c_concat": [hint], "c_crossattn": [empty_text]}
        target = batch.target
        if tuple(target.shape[-2:]) != tuple(hint.shape[-2:]):
            channel_target = target.unsqueeze(1) if target.ndim == 3 else target
            target = F.interpolate(channel_target.float(), size=hint.shape[-2:], mode="nearest")
            if batch.target.ndim == 3:
                target = target[:, 0]
            target = target.long()
        return DiffusionCondition(
            conditional=conditional,
            unconditional=unconditional,
            target=target,
            extras={"hint": hint, "prompts": prompts},
        )

    def _schedule(self, steps: int) -> tuple[Any, np.ndarray]:
        sampler = self._sampler_class(self.model)
        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)
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
        """Query the first reverse-time epsilon and convert it to a VP score."""
        sampler, timesteps = self._schedule(self.full_schedule_steps)
        timestep = torch.full(
            (probe_noise.shape[0],),
            int(timesteps[0]),
            device=probe_noise.device,
            dtype=torch.long,
        )
        epsilon = self._cfg_epsilon(probe_noise, timestep, condition, cfg_scale)
        alpha_bar = torch.as_tensor(
            sampler.ddim_alphas[len(timesteps) - 1],
            device=probe_noise.device,
            dtype=probe_noise.dtype,
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
        """Run deterministic DDIM with temporal stop-gradient on score queries."""
        sampler, timesteps = self._schedule(schedule_steps)
        sample = initial_noise
        clean_estimate = initial_noise
        for reverse_index, raw_timestep in enumerate(timesteps[:executed_steps]):
            schedule_index = len(timesteps) - reverse_index - 1
            timestep = torch.full(
                (sample.shape[0],), int(raw_timestep), device=sample.device, dtype=torch.long
            )
            # The state is detached only for the frozen denoiser query. The
            # explicit DDIM path from selected seed to x0 stays differentiable.
            with torch.no_grad():
                epsilon = self._cfg_epsilon(sample.detach(), timestep, condition, cfg_scale)
            shape = (sample.shape[0],) + (1,) * (sample.ndim - 1)
            alpha = torch.as_tensor(
                sampler.ddim_alphas[schedule_index], device=sample.device, dtype=sample.dtype
            ).view((1,) * sample.ndim).expand(shape)
            alpha_previous = torch.as_tensor(
                sampler.ddim_alphas_prev[schedule_index], device=sample.device, dtype=sample.dtype
            ).view((1,) * sample.ndim).expand(shape)
            sigma = torch.as_tensor(
                sampler.ddim_sigmas[schedule_index], device=sample.device, dtype=sample.dtype
            ).view((1,) * sample.ndim).expand(shape)
            sqrt_one_minus = torch.as_tensor(
                sampler.ddim_sqrt_one_minus_alphas[schedule_index],
                device=sample.device,
                dtype=sample.dtype,
            ).view((1,) * sample.ndim).expand(shape)
            clean_estimate = (sample - sqrt_one_minus * epsilon) / alpha.sqrt()
            direction = (1.0 - alpha_previous - sigma.square()).clamp_min(0).sqrt() * epsilon
            sample = alpha_previous.sqrt() * clean_estimate + direction
        return sample, clean_estimate

    def _ema_context(self) -> Any:
        return self.model.ema_scope() if hasattr(self.model, "ema_scope") else nullcontext()

    def truncated_rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Decode the early-DDIM clean estimate while retaining seed gradients."""
        _, clean_estimate = self._rollout(
            initial_noise, condition, self.full_schedule_steps, steps, cfg_scale
        )
        decoder = getattr(self.model, "differentiable_decode_first_stage", None)
        if decoder is None:
            # Stable-Diffusion's VAE decoder is differentiable; the official
            # method's no-grad wrapper is bypassed when available.
            decoder = self.model.first_stage_model.decode
            scale = float(getattr(self.model, "scale_factor", 1.0))
            return decoder(clean_estimate / scale).clamp(-1.0, 1.0)
        return decoder(clean_estimate).clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the final M2I pair with 50-step eta-zero DDIM."""
        with self._ema_context():
            latent, _ = self._rollout(initial_noise, condition, steps, steps, cfg_scale)
            images = self.model.decode_first_stage(latent).clamp(-1.0, 1.0)
        return images, condition.target


def build_adapter(config: Any) -> SiameseDiffAdapter:
    """Registry factory for the specialized SiameseDiff implementation."""
    return SiameseDiffAdapter(config)
