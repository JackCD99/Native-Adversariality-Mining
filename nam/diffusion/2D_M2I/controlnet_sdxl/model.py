"""Diffusers-based ControlNet-SDXL adapter for semantic-mask conditioning.

Sources:
https://github.com/lllyasviel/ControlNet
https://github.com/Stability-AI/generative-models
https://github.com/huggingface/diffusers
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from nam.data import NAMBatch
from nam.diffusion.base import (
    DiffusionCondition,
    DiffusionMetadata,
    MedicalDiffusionAdapter,
    ScoreOutput,
    epsilon_to_score,
)
from .utils.conditioning import (
    colorize_masks,
    encode_sdxl_prompts,
    prompts_from_batch,
)


class ControlNetSDXLComponents(nn.Module):
    """Register the frozen SDXL modules and the task-tuned ControlNet."""

    def __init__(self, pipe: Any, resolution: int) -> None:
        super().__init__()
        self.unet = pipe.unet
        self.controlnet = pipe.controlnet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.pipe = pipe
        self.scheduler = pipe.scheduler
        self.resolution = resolution


def load_components(config: Any) -> ControlNetSDXLComponents:
    """Load SDXL and initialize or restore its semantic ControlNet."""
    try:
        from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionXLControlNetPipeline
    except ImportError as error:
        raise ImportError("ControlNet-SDXL requires `pip install diffusers transformers accelerate`.") from error

    dtype_name = str(getattr(config, "dtype", "float16"))
    dtype = getattr(torch, dtype_name)
    controlnet_path = str(getattr(config, "checkpoint", "")).strip()
    if controlnet_path and Path(controlnet_path).is_dir():
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
    else:
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(
            str(config.base_model), subfolder="unet", torch_dtype=dtype
        )
        controlnet = ControlNetModel.from_unet(unet)
        del unet
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        str(config.base_model), controlnet=controlnet, torch_dtype=dtype, use_safetensors=True
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return ControlNetSDXLComponents(pipe, int(config.resolution))


class ControlNetSDXLAdapter(MedicalDiffusionAdapter):
    """Expose SDXL score prediction and deterministic DDIM to NAM."""

    metadata = DiffusionMetadata(
        name="controlnet_sdxl",
        official_repository="https://github.com/lllyasviel/ControlNet",
        synthesis_paradigm="M2I-SDXL",
        spatial_dims=2,
        prediction_type="epsilon",
        noise_channels=4,
        noise_size=(64, 64),
    )

    @property
    def components(self) -> ControlNetSDXLComponents:
        return self.model

    def sample_probe_noise(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Draw latent seeds in the precision required by the SDXL U-Net."""
        shape = (batch_size, self.metadata.noise_channels, *self.metadata.noise_size)
        dtype = next(self.components.unet.parameters()).dtype
        return torch.randn(shape, device=self.device, dtype=dtype, generator=generator)

    def prepare_condition(self, batch: NAMBatch) -> DiffusionCondition:
        prompts = prompts_from_batch(batch, str(self.config.default_prompt))
        prompt_embeds, negative_embeds, pooled, negative_pooled, time_ids = encode_sdxl_prompts(
            self.components, prompts, self.device
        )
        control = colorize_masks(batch.target).to(self.device, dtype=prompt_embeds.dtype)
        return DiffusionCondition(
            conditional=prompt_embeds,
            unconditional=negative_embeds,
            target=batch.target,
            extras={
                "pooled": pooled,
                "negative_pooled": negative_pooled,
                "time_ids": time_ids,
                "control": control,
                "prompts": prompts,
            },
        )

    def _predict_epsilon(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        pooled: torch.Tensor,
        condition: DiffusionCondition,
    ) -> torch.Tensor:
        extras = condition.extras
        added = {"text_embeds": pooled, "time_ids": extras["time_ids"]}
        down, middle = self.components.controlnet(
            latent,
            timestep,
            encoder_hidden_states=text_embeds,
            controlnet_cond=extras["control"],
            conditioning_scale=float(getattr(self.config, "controlnet_scale", 1.0)),
            added_cond_kwargs=added,
            return_dict=False,
        )
        return self.components.unet(
            latent,
            timestep,
            encoder_hidden_states=text_embeds,
            added_cond_kwargs=added,
            down_block_additional_residuals=down,
            mid_block_additional_residual=middle,
        ).sample

    def _epsilon(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: DiffusionCondition,
        cfg_scale: float,
    ) -> torch.Tensor:
        conditional = self._predict_epsilon(
            latent, timestep, condition.conditional, condition.extras["pooled"], condition
        )
        if cfg_scale == 1.0 or condition.unconditional is None:
            return conditional
        unconditional = self._predict_epsilon(
            latent,
            timestep,
            condition.unconditional,
            condition.extras["negative_pooled"],
            condition,
        )
        return unconditional + cfg_scale * (conditional - unconditional)

    def initial_score(
        self, probe_noise: torch.Tensor, condition: DiffusionCondition, cfg_scale: float
    ) -> ScoreOutput:
        self.components.scheduler.set_timesteps(
            int(self.config.full_schedule_steps), device=probe_noise.device
        )
        timestep = self.components.scheduler.timesteps[0].expand(probe_noise.shape[0])
        epsilon = self._epsilon(probe_noise, timestep, condition, cfg_scale)
        alpha = self.components.scheduler.alphas_cumprod.to(probe_noise.device)[timestep]
        # The miner is optimized in FP32 even when SDXL inference uses FP16.
        return ScoreOutput(epsilon_to_score(epsilon, alpha).float(), timestep, epsilon)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        scaling = float(self.components.vae.config.scaling_factor)
        image = self.components.vae.decode(latent / scaling).sample
        return image.add(1).div(2).clamp(0, 1)

    def truncated_rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        scheduler = self.components.scheduler
        scheduler.set_timesteps(int(self.config.full_schedule_steps), device=initial_noise.device)
        latent = initial_noise.to(dtype=next(self.components.unet.parameters()).dtype)
        for timestep in scheduler.timesteps[:steps]:
            with torch.no_grad():
                epsilon = self._epsilon(latent.detach(), timestep, condition, cfg_scale)
            latent = scheduler.step(epsilon.detach(), timestep, latent, eta=0.0).prev_sample
        return self._decode(latent)

    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scheduler = self.components.scheduler
        scheduler.set_timesteps(steps, device=initial_noise.device)
        latent = initial_noise.to(dtype=next(self.components.unet.parameters()).dtype)
        for timestep in scheduler.timesteps:
            epsilon = self._epsilon(latent, timestep, condition, cfg_scale)
            latent = scheduler.step(epsilon, timestep, latent, eta=0.0).prev_sample
        return self._decode(latent), condition.target


def build_adapter(config: Any) -> ControlNetSDXLAdapter:
    return ControlNetSDXLAdapter(load_components(config), config)
