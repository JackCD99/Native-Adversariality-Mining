"""Train a semantic ControlNet on SDXL with PASCAL VOC/SBD pairs."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nam.config import apply_overrides, load_config
from nam.utils.seed import seed_everything
from nam.utils.monitoring import log_diffusion_diagnostics

_package = __package__ or "nam.diffusion.2D_M2I.controlnet_sdxl"
load_components = importlib.import_module(f"{_package}.model").load_components
_conditioning = importlib.import_module(f"{_package}.utils.conditioning")
colorize_masks = _conditioning.colorize_masks
encode_sdxl_prompts = _conditioning.encode_sdxl_prompts
prompts_from_batch = _conditioning.prompts_from_batch
build_loader = importlib.import_module(f"{_package}.utils.data").build_loader


def train_pretrained_model(config: Any) -> Path:
    """Optimize ControlNet while the SDXL backbone and text encoders stay frozen."""
    settings = config.controlnet_sdxl.pre_training
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    components = load_components(config.diffusion).to(device)
    components.requires_grad_(False)
    components.controlnet.requires_grad_(True)
    components.controlnet.train()
    optimizer = torch.optim.AdamW(
        components.controlnet.parameters(),
        lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )
    loader = build_loader(
        config.dataset, "train", int(settings.batch_size_per_gpu),
        int(config.runtime.num_workers), True,
    )
    run = Path(settings.output_dir) / (
        f"{config.experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    writer = SummaryWriter(run / "tensorboard")
    iterator = iter(loader)
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))
    dtype = next(components.vae.parameters()).dtype
    for step in tqdm(range(1, int(settings.max_iterations) + 1), desc="ControlNet-SDXL"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        prompts = prompts_from_batch(batch, str(config.diffusion.default_prompt))
        with torch.no_grad():
            images = batch.image.to(dtype=dtype).mul(2).sub(1)
            latents = components.vae.encode(images).latent_dist.sample(generator=generator)
            latents = latents * float(components.vae.config.scaling_factor)
            prompt_embeds, _, pooled, _, time_ids = encode_sdxl_prompts(
                components, prompts, device
            )
        noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=generator)
        timesteps = torch.randint(
            0, components.scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device, generator=generator,
        ).long()
        noisy = components.scheduler.add_noise(latents, noise, timesteps)
        added = {"text_embeds": pooled, "time_ids": time_ids}
        down, middle = components.controlnet(
            noisy,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=colorize_masks(batch.target).to(device, dtype=dtype),
            conditioning_scale=1.0,
            added_cond_kwargs=added,
            return_dict=False,
        )
        prediction = components.unet(
            noisy,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added,
            down_block_additional_residuals=down,
            mid_block_additional_residual=middle,
        ).sample
        loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            components.controlnet.parameters(), float(settings.gradient_clip)
        )
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/gradient_norm", float(gradient), step)
        log_diffusion_diagnostics(
            writer,
            components.controlnet,
            optimizer,
            step,
            {
                "image": images,
                "condition": colorize_masks(batch.target).to(device, dtype=dtype),
                "prediction": prediction,
                "target_noise": noise,
            },
        )
        if step % int(settings.save_every) == 0 or step == int(settings.max_iterations):
            destination = Path(config.controlnet_sdxl.checkpoints.diffusion_root)
            components.controlnet.save_pretrained(destination)
            components.controlnet.save_pretrained(run / "checkpoints" / f"step_{step}")
            torch.save(
                {"step": step, "optimizer": optimizer.state_dict()},
                destination / "training_state.pt",
            )
    writer.close()
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ControlNet-SDXL on VOC/SBD masks.")
    parser.add_argument("--config", default="configs/controlnet_sdxl_voc.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
