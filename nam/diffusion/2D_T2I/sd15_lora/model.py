"""Stable Diffusion v1.5 LoRA adapter for class-conditional synthesis.

Sources:
https://github.com/CompVis/stable-diffusion
https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
https://github.com/microsoft/LoRA
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
    v_prediction_to_epsilon,
)


class SD15LoRAComponents(nn.Module):
    """Register the frozen modules required by the SD-v1.5 latent sampler."""

    def __init__(self, pipeline: Any, resolution: int) -> None:
        super().__init__()
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.pipeline = pipeline
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler
        self.resolution = int(resolution)


def _load_lora(pipeline: Any, checkpoint: str) -> None:
    path = Path(checkpoint)
    if not checkpoint or not path.exists():
        return
    try:
        pipeline.load_lora_weights(str(path))
    except (AttributeError, OSError, ValueError) as error:
        raise RuntimeError(f"Unable to load the SD-v1.5 LoRA checkpoint at {path}.") from error


def load_components(config: Any, load_lora: bool = True) -> SD15LoRAComponents:
    """Load SD-v1.5 and optionally attach a trained Diffusers LoRA adapter."""
    try:
        from diffusers import DDIMScheduler, StableDiffusionPipeline
    except ImportError as error:
        raise ImportError(
            "SD-v1.5 LoRA requires `pip install diffusers transformers accelerate peft`."
        ) from error
    dtype = getattr(torch, str(getattr(config, "dtype", "float16")))
    pipeline = StableDiffusionPipeline.from_pretrained(
        str(config.base_model), torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False, use_safetensors=True,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    if load_lora:
        checkpoint = str(getattr(config, "checkpoint", ""))
        if bool(getattr(config, "require_checkpoint", True)) and not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"SD-v1.5 LoRA checkpoint was not found: {checkpoint}. "
                "Run the method pre-training stage first."
            )
        _load_lora(pipeline, checkpoint)
    return SD15LoRAComponents(pipeline, int(getattr(config, "resolution", 256)))


def prompts_from_batch(batch: NAMBatch, default_prompt: str) -> list[str]:
    values = batch.condition.get("prompt", batch.condition.get("txt", []))
    return [str(value) if value else default_prompt for value in values]


class SD15LoRAAdapter(MedicalDiffusionAdapter):
    """Expose LoRA-tuned SD-v1.5 score queries and deterministic DDIM to NAM."""

    metadata = DiffusionMetadata(
        name="sd15_lora",
        official_repository="https://github.com/CompVis/stable-diffusion",
        synthesis_paradigm="T2I-SD-LoRA",
        spatial_dims=2,
        prediction_type="epsilon",
        noise_channels=4,
        noise_size=(32, 32),
    )

    def __init__(self, config: Any, load_lora: bool = True) -> None:
        resolution = int(getattr(config, "resolution", 256))
        self.metadata = DiffusionMetadata(
            **{**type(self).metadata.__dict__, "prediction_type": str(getattr(config, "prediction_type", "epsilon")),
               "noise_size": (resolution // 8, resolution // 8)}
        )
        self.full_schedule_steps = int(getattr(config, "full_schedule_steps", 50))
        self.default_prompt = str(getattr(config, "default_prompt", "a medical image"))
        super().__init__(load_components(config, load_lora=load_lora), config)

    @property
    def components(self) -> SD15LoRAComponents:
        return self.model

    def sample_probe_noise(self, batch_size: int, generator=None) -> torch.Tensor:
        shape = (batch_size, self.metadata.noise_channels, *self.metadata.noise_size)
        dtype = next(self.components.unet.parameters()).dtype
        return torch.randn(shape, device=self.device, dtype=dtype, generator=generator)

    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.components.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=self.components.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids.to(self.device)
        return self.components.text_encoder(tokens)[0]

    def prepare_condition(self, batch: NAMBatch) -> DiffusionCondition:
        prompts = prompts_from_batch(batch, self.default_prompt)
        negative = [str(getattr(self.config, "negative_prompt", ""))] * len(prompts)
        with torch.no_grad():
            conditional = self._encode_text(prompts)
            unconditional = self._encode_text(negative)
        return DiffusionCondition(
            conditional=conditional,
            unconditional=unconditional,
            target=batch.target,
            extras={"prompts": prompts},
        )

    def _epsilon(self, latent: torch.Tensor, timestep: torch.Tensor, condition: DiffusionCondition,
                 cfg_scale: float) -> torch.Tensor:
        conditional = self.components.unet(
            latent, timestep, encoder_hidden_states=condition.conditional
        ).sample
        if cfg_scale == 1.0:
            prediction = conditional
        else:
            unconditional = self.components.unet(
                latent, timestep, encoder_hidden_states=condition.unconditional
            ).sample
            prediction = unconditional + cfg_scale * (conditional - unconditional)
        if self.metadata.prediction_type == "v_prediction":
            index = timestep.long()
            alpha = self.components.scheduler.alphas_cumprod.to(latent.device)[index]
            return v_prediction_to_epsilon(prediction, latent, alpha)
        return prediction

    def initial_score(self, probe_noise: torch.Tensor, condition: DiffusionCondition,
                      cfg_scale: float) -> ScoreOutput:
        scheduler = self.components.scheduler
        scheduler.set_timesteps(self.full_schedule_steps, device=probe_noise.device)
        timestep = scheduler.timesteps[0].expand(probe_noise.shape[0])
        epsilon = self._epsilon(probe_noise, timestep, condition, cfg_scale)
        alpha = scheduler.alphas_cumprod.to(probe_noise.device)[timestep.long()]
        return ScoreOutput(epsilon_to_score(epsilon, alpha).float(), timestep, epsilon)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        scale = float(self.components.vae.config.scaling_factor)
        return self.components.vae.decode(latent / scale).sample.add(1).div(2).clamp(0, 1)

    def truncated_rollout(self, initial_noise: torch.Tensor, condition: DiffusionCondition,
                          steps: int, cfg_scale: float) -> torch.Tensor:
        scheduler = self.components.scheduler
        scheduler.set_timesteps(self.full_schedule_steps, device=initial_noise.device)
        latent = initial_noise.to(next(self.components.unet.parameters()).dtype)
        for timestep in scheduler.timesteps[: int(steps)]:
            with torch.no_grad():
                epsilon = self._epsilon(latent.detach(), timestep, condition, cfg_scale)
            latent = scheduler.step(epsilon.detach(), timestep, latent, eta=0.0).prev_sample
        return self._decode(latent)

    @torch.no_grad()
    def sample(self, initial_noise: torch.Tensor, condition: DiffusionCondition,
               steps: int, cfg_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        scheduler = self.components.scheduler
        scheduler.set_timesteps(int(steps), device=initial_noise.device)
        latent = initial_noise.to(next(self.components.unet.parameters()).dtype)
        for timestep in scheduler.timesteps:
            epsilon = self._epsilon(latent, timestep, condition, cfg_scale)
            latent = scheduler.step(epsilon, timestep, latent, eta=0.0).prev_sample
        return self._decode(latent), condition.target


def build_adapter(config: Any) -> SD15LoRAAdapter:
    return SD15LoRAAdapter(config)
