"""Official JoDiffusion model construction and deterministic DDIM utilities."""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from typing import Any
import torch
from torch import nn

OFFICIAL_REPOSITORY = "https://github.com/00why00/JoDiffusion"
OFFICIAL_COMMIT = "9fef37099c982e0fa512e84456e4d717d797b593"


class JoComponents(nn.Module):
    """Register all official modules so device and freeze operations are atomic."""
    def __init__(self, pipeline: nn.Module) -> None:
        super().__init__()
        self.unet = pipeline.unet
        self.vae = pipeline.image_vae
        self.label_vae = pipeline.label_vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.clip_tokenizer
        self.image_encoder = pipeline.image_encoder
        self.image_processor = pipeline.clip_image_processor
        self.scheduler = pipeline.scheduler


def _load_official_module(root: Path, name: str, relative: str):
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"JoDiffusion source file was not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_official_components(config: Any) -> JoComponents:
    """Load the pinned official pipeline while keeping its source outside this repository."""
    root = Path(config.project_dir).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from pipelines.pipeline_jodiffusion import JoDiffusionPipeline
        from pipelines.modeling_uvit import JoDiffusionModel
        from pipelines.modeling_lightweight_vae import LightweightLabelVAE
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(f"Cannot import the official JoDiffusion code from {root}. See README.md.") from error
    unet_path = str(getattr(config, "checkpoint", ""))
    label_path = str(config.label_vae_checkpoint)
    if unet_path and not Path(unet_path).exists():
        raise FileNotFoundError(f"JoDiffusion U-ViT checkpoint directory was not found: {unet_path}")
    unet = (JoDiffusionModel.from_pretrained(unet_path) if unet_path
            else JoDiffusionModel.from_pretrained(config.base_pipeline, subfolder="unet"))
    label_vae = LightweightLabelVAE.from_pretrained(label_path)
    pipeline = JoDiffusionPipeline.from_pretrained(config.base_pipeline, unet=unet, label_vae=label_vae)
    from diffusers import DDIMScheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    return JoComponents(pipeline)


def ddim_timesteps(scheduler: Any, steps: int, device: torch.device) -> torch.Tensor:
    scheduler.set_timesteps(int(steps), device=device)
    return scheduler.timesteps


def ddim_step(sample: torch.Tensor, epsilon: torch.Tensor, timestep: torch.Tensor,
              previous: torch.Tensor | int, alphas: torch.Tensor) -> torch.Tensor:
    """Differentiable DDIM eta=0 update used by full and truncated rollouts."""
    t = int(timestep.item())
    p = int(previous.item()) if torch.is_tensor(previous) else int(previous)
    alpha_t = alphas[t].to(sample.device, sample.dtype)
    alpha_p = alphas[p].to(sample.device, sample.dtype) if p >= 0 else sample.new_tensor(1.0)
    clean = (sample - (1 - alpha_t).sqrt() * epsilon) / alpha_t.sqrt()
    return alpha_p.sqrt() * clean + (1 - alpha_p).sqrt() * epsilon
