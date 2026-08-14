"""MedSegFactory dual-stream architecture and NAM adapter.

Official code: https://github.com/jwmao1/MedSegFactory
"""

from __future__ import annotations
import importlib
from typing import Any
import torch
from nam.data import NAMBatch
from nam.diffusion.base import (
    DiffusionCondition,
    DiffusionMetadata,
    MedicalDiffusionAdapter,
    ScoreOutput,
    epsilon_to_score,
)

_package = __package__ or "nam.diffusion.2D_M&I.medsegfactory"
_runtime = importlib.import_module(f"{_package}.utils.runtime")
_data = importlib.import_module(f"{_package}.utils.data")


class MedSegFactoryAdapter(MedicalDiffusionAdapter):
    """Expose two coupled 4-channel diffusion streams to two NAM miners."""

    def __init__(self, components: Any, config: Any) -> None:
        self.components, self.scheduler = components, components.scheduler
        self.metadata = DiffusionMetadata(
            "MedSegFactory",
            _runtime.OFFICIAL_REPOSITORY,
            "M&I-LDM-dual-noise",
            2,
            "epsilon",
            4,
            tuple(getattr(config, "noise_size", (32, 32))),
            True,
        )
        super().__init__(components, config)

    def sample_probe_noise(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        # Concatenation is an adapter boundary only; each branch remains an independent N(0,I).
        return torch.randn(
            (batch_size, 8, *self.metadata.noise_size), device=self.device, generator=generator
        )

    def _encode(self, prompts: list[str]) -> torch.Tensor:
        ids = self.components.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.components.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.components.text_encoder(ids)[0]

    def prepare_condition(self, batch: NAMBatch) -> DiffusionCondition:
        images, masks = _data.prompts_from_batch(
            batch, str(self.config.image_prompt), str(self.config.mask_prompt)
        )
        empty = [str(getattr(self.config, "negative_prompt", ""))] * len(images)
        return DiffusionCondition(
            (self._encode(images), self._encode(masks)),
            (self._encode(empty), self._encode(empty)),
            extras={"image_prompts": images, "mask_prompts": masks},
        )

    def _predict(
        self,
        joint: torch.Tensor,
        timestep: torch.Tensor,
        condition: DiffusionCondition,
        cfg_scale: float,
    ) -> torch.Tensor:
        image, mask = joint.chunk(2, 1)
        image_cond, mask_cond = condition.conditional
        image_null, mask_null = condition.unconditional
        latent = torch.cat((image, image, mask, mask), 0)
        text = torch.cat((image_cond, image_null, mask_cond, mask_null), 0)
        repeated_t = timestep.repeat(4) if timestep.numel() == image.shape[0] else timestep
        prediction = self.components.unet(latent, repeated_t, encoder_hidden_states=text).sample
        image_cond_pred, image_null_pred, mask_cond_pred, mask_null_pred = prediction.chunk(4, 0)
        image_pred = image_null_pred + cfg_scale * (image_cond_pred - image_null_pred)
        mask_pred = mask_null_pred + cfg_scale * (mask_cond_pred - mask_null_pred)
        return torch.cat((image_pred, mask_pred), 1)

    @torch.no_grad()
    def initial_score(
        self, probe_noise: torch.Tensor, condition: DiffusionCondition, cfg_scale: float
    ) -> ScoreOutput:
        self.scheduler.set_timesteps(
            int(getattr(self.config, "full_schedule_steps", 50)), device=self.device
        )
        timestep = self.scheduler.timesteps[:1].expand(probe_noise.shape[0])
        epsilon = self._predict(probe_noise, timestep, condition, cfg_scale)
        alpha = self.scheduler.alphas_cumprod.to(self.device)[timestep]
        return ScoreOutput(epsilon_to_score(epsilon, alpha), timestep, epsilon)

    def _rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full = int(getattr(self.config, "full_schedule_steps", 50))
        self.scheduler.set_timesteps(full, device=self.device)
        schedule, joint = self.scheduler.timesteps, initial_noise
        for index, timestep in enumerate(schedule[: min(int(steps), len(schedule))]):
            with torch.no_grad():
                epsilon = self._predict(
                    joint.detach(), timestep.expand(joint.shape[0]), condition, cfg_scale
                )
            previous = schedule[index + 1] if index + 1 < len(schedule) else -1
            image, mask = joint.chunk(2, 1)
            image_eps, mask_eps = epsilon.chunk(2, 1)
            image = _runtime.ddim_step(
                image, image_eps, timestep, previous, self.scheduler.alphas_cumprod
            )
            mask = _runtime.ddim_step(
                mask, mask_eps, timestep, previous, self.scheduler.alphas_cumprod
            )
            joint = torch.cat((image, mask), 1)
        return self._decode(joint)

    def _decode(self, joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_latent, mask_latent = joint.chunk(2, 1)
        scale = float(self.components.vae.config.scaling_factor)
        image = self.components.vae.decode(image_latent / scale).sample.clamp(-1, 1)
        mask_rgb = self.components.vae.decode(mask_latent / scale).sample
        normalized = (mask_rgb + 1).div(2)
        # Match skimage.color.rgb2gray used by the official medical pipeline.
        weights = normalized.new_tensor((0.2125, 0.7154, 0.0721)).view(1, 3, 1, 1)
        gray = (normalized * weights).sum(1)
        target = (
            (gray * max(int(self.config.num_classes) - 1, 1))
            .round()
            .clamp(0, int(self.config.num_classes) - 1)
            .long()
        )
        return image, target

    def truncated_rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rollout(initial_noise, condition, steps, cfg_scale)

    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(steps) != int(getattr(self.config, "full_schedule_steps", 50)):
            raise ValueError("sampling.ddim_steps must equal diffusion.full_schedule_steps.")
        return self._rollout(initial_noise, condition, steps, cfg_scale)


def build_adapter(config: Any) -> MedSegFactoryAdapter:
    return MedSegFactoryAdapter(_runtime.build_official_components(config, True), config)
