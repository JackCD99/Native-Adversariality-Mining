"""JoDiffusion adapter with the paper's single-stream NAM modification.

Official code: https://github.com/00why00/JoDiffusion
"""

from __future__ import annotations
import importlib
from typing import Any
import torch
from torch import nn
from nam.data import NAMBatch
from nam.diffusion.base import DiffusionCondition, DiffusionMetadata, MedicalDiffusionAdapter, ScoreOutput, epsilon_to_score

_package = __package__ or "nam.diffusion.2D_M&I.jodiffusion"
_runtime = importlib.import_module(f"{_package}.utils.runtime")
_data = importlib.import_module(f"{_package}.utils.data")


class JoDiffusionAdapter(MedicalDiffusionAdapter):
    """Expose JoDiffusion's joint 8-channel spatial seed to NAM."""
    def __init__(self, components: nn.Module, config: Any) -> None:
        self.components, self.scheduler = components, components.scheduler
        self.metadata = DiffusionMetadata("JoDiffusion", _runtime.OFFICIAL_REPOSITORY,
            "M&I-LDM-shared-noise", 2, "epsilon", 8,
            tuple(getattr(config, "noise_size", (32, 32))))
        super().__init__(components, config)

    def _prompt_embeddings(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.components.tokenizer(prompts, padding="max_length",
            max_length=self.components.tokenizer.model_max_length, truncation=True,
            return_tensors="pt").input_ids.to(self.device)
        return self.components.text_encoder(tokens)[0]

    def prepare_condition(self, batch: NAMBatch) -> DiffusionCondition:
        prompts = _data.prompts_from_batch(batch, str(getattr(self.config, "default_prompt", "a medical image")))
        conditional = self._prompt_embeddings(prompts)
        # Official JoDiffusion uses Gaussian text for the CFG null branch.
        unconditional = torch.randn_like(conditional)
        clip_dim = int(getattr(self.components.unet.config, "clip_img_dim", 512))
        clip_noise = torch.randn((len(prompts), 1, clip_dim), device=self.device, dtype=conditional.dtype)
        return DiffusionCondition(conditional, unconditional, extras={"prompts": prompts, "clip_noise": clip_noise})

    def _predict_parts(self, joint: torch.Tensor, clip: torch.Tensor, timestep: torch.Tensor,
                       condition: DiffusionCondition, cfg_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = joint.chunk(2, 1)
        def run(text: torch.Tensor, text_timestep: int) -> tuple[torch.Tensor, torch.Tensor]:
            _, image_out, clip_out, label_out = self.components.unet(
                text, image, clip, label, timestep_img=timestep,
                timestep_text=text_timestep, data_type=1)
            return torch.cat((image_out, label_out), 1), clip_out
        conditional, clip_conditional = run(condition.conditional, 0)
        if cfg_scale <= 1.0:
            return conditional, clip_conditional
        maximum = int(self.scheduler.config.num_train_timesteps)
        unconditional, clip_unconditional = run(condition.unconditional, maximum)
        return (unconditional + cfg_scale * (conditional - unconditional),
                clip_unconditional + cfg_scale * (clip_conditional - clip_unconditional))

    @torch.no_grad()
    def initial_score(self, probe_noise: torch.Tensor, condition: DiffusionCondition, cfg_scale: float) -> ScoreOutput:
        schedule = _runtime.ddim_timesteps(self.scheduler, int(getattr(self.config, "full_schedule_steps", 50)), self.device)
        timestep = schedule[:1].expand(probe_noise.shape[0])
        epsilon, _ = self._predict_parts(probe_noise, condition.extras["clip_noise"], timestep, condition, cfg_scale)
        alpha = self.scheduler.alphas_cumprod.to(self.device)[timestep]
        return ScoreOutput(epsilon_to_score(epsilon, alpha), timestep, epsilon)

    def _rollout(self, initial_noise: torch.Tensor, condition: DiffusionCondition, steps: int,
                 cfg_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        full_steps = int(getattr(self.config, "full_schedule_steps", 50))
        schedule = _runtime.ddim_timesteps(self.scheduler, full_steps, self.device)
        count, joint, clip = min(int(steps), len(schedule)), initial_noise, condition.extras["clip_noise"]
        alphas = self.scheduler.alphas_cumprod
        for index, timestep in enumerate(schedule[:count]):
            # Stop score-network Jacobians while preserving the analytic DDIM
            # path from the selected seed, as required by truncated NAM.
            with torch.no_grad():
                epsilon, clip_epsilon = self._predict_parts(
                    joint.detach(), clip.detach(), timestep.expand(joint.shape[0]), condition, cfg_scale)
            previous = schedule[index + 1] if index + 1 < len(schedule) else -1
            joint = _runtime.ddim_step(joint, epsilon, timestep, previous, alphas)
            # NAM reselects the paper-reported spatial seed; the official CLIP
            # image token remains an auxiliary diffusion state.
            clip = _runtime.ddim_step(clip, clip_epsilon, timestep, previous, alphas)
        return self._decode(joint)

    def _decode(self, joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_latent, label_latent = joint.chunk(2, 1)
        image = self.components.vae.decode(image_latent / float(self.components.vae.config.scaling_factor)).sample.clamp(-1, 1)
        label_logits = self.components.label_vae.decode(label_latent / float(self.components.label_vae.config.scaling_factor)).sample
        probabilities, target = label_logits.softmax(1).max(1)
        target = target.masked_fill(
            probabilities < float(getattr(self.config, "label_confidence_threshold", 0.5)),
            int(getattr(self.config, "ignore_label", 255)),
        )
        return image, target

    def truncated_rollout(self, initial_noise: torch.Tensor, condition: DiffusionCondition,
                          steps: int, cfg_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rollout(initial_noise, condition, steps, cfg_scale)

    @torch.no_grad()
    def sample(self, initial_noise: torch.Tensor, condition: DiffusionCondition,
               steps: int, cfg_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        if int(steps) != int(getattr(self.config, "full_schedule_steps", 50)):
            raise ValueError("JoDiffusion sampling.ddim_steps must equal diffusion.full_schedule_steps.")
        return self._rollout(initial_noise, condition, steps, cfg_scale)


def build_adapter(config: Any) -> JoDiffusionAdapter:
    return JoDiffusionAdapter(_runtime.build_official_components(config), config)
