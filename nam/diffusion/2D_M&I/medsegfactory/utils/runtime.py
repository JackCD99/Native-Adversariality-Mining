"""Official MedSegFactory JCA U-Net construction and DDIM utilities."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import torch
from torch import nn

OFFICIAL_REPOSITORY = "https://github.com/jwmao1/MedSegFactory"
OFFICIAL_COMMIT = "b227c6b5f0ff6b02d6046a1cdf57fc47cb74ae96"


class Components(nn.Module):
    def __init__(self, unet: nn.Module, vae: nn.Module, text_encoder: nn.Module, tokenizer: Any, scheduler: Any) -> None:
        super().__init__(); self.unet, self.vae, self.text_encoder = unet, vae, text_encoder
        self.tokenizer, self.scheduler = tokenizer, scheduler


def build_official_components(config: Any, load_checkpoint: bool = True) -> Components:
    root = Path(config.project_dir).expanduser().resolve()
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    try:
        from StableDiffusion.Our_UNet import UNet2DConditionModel
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as error:
        raise ImportError(f"Cannot import MedSegFactory from {root}. See README.md.") from error
    base = str(config.base_model)
    unet = (UNet2DConditionModel.from_config(base, subfolder="unet") if load_checkpoint
            else UNet2DConditionModel.from_pretrained(base, subfolder="unet", low_cpu_mem_usage=False, device_map=None))
    if load_checkpoint:
        path = Path(config.checkpoint).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(f"MedSegFactory U-Net checkpoint was not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        unet.load_state_dict(payload.get("unet", payload) if isinstance(payload, dict) else payload, strict=True)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
    text = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    return Components(unet, vae, text, tokenizer, scheduler)


def ddim_step(sample: torch.Tensor, epsilon: torch.Tensor, timestep: torch.Tensor,
              previous: torch.Tensor | int, alphas: torch.Tensor) -> torch.Tensor:
    t, p = int(timestep.item()), int(previous.item()) if torch.is_tensor(previous) else int(previous)
    alpha_t = alphas[t].to(sample.device, sample.dtype)
    # Official scheduler uses alpha_cumprod[0] at the final step because
    # MedSegFactory configures set_alpha_to_one=False.
    alpha_p = alphas[p].to(sample.device, sample.dtype) if p >= 0 else alphas[0].to(sample.device, sample.dtype)
    clean = (sample - (1 - alpha_t).sqrt() * epsilon) / alpha_t.sqrt()
    return alpha_p.sqrt() * clean + (1 - alpha_p).sqrt() * epsilon
