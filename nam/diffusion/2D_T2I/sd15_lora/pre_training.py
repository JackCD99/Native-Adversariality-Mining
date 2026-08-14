"""LoRA fine-tuning for class-aware Stable Diffusion v1.5 synthesis."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nam.config import apply_overrides, load_config
from nam.utils.monitoring import log_diffusion_diagnostics
from nam.utils.seed import seed_everything

_package = __package__ or "nam.diffusion.2D_T2I.sd15_lora"
_model = importlib.import_module(f"{_package}.model")
load_components, prompts_from_batch = _model.load_components, _model.prompts_from_batch
build_loader = importlib.import_module(f"{_package}.utils.data").build_loader


def _attach_lora(unet: torch.nn.Module, rank: int) -> list[torch.nn.Parameter]:
    try:
        from peft import LoraConfig
    except ImportError as error:
        raise ImportError("LoRA training requires a compatible Diffusers installation.") from error
    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=("to_k", "to_q", "to_v", "to_out.0"),
        )
    )
    parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable LoRA attention parameters were created.")
    return parameters


def _encode_prompts(components: Any, prompts: list[str], device: torch.device) -> torch.Tensor:
    tokens = components.tokenizer(
        prompts, padding="max_length", truncation=True,
        max_length=components.tokenizer.model_max_length, return_tensors="pt",
    ).input_ids.to(device)
    return components.text_encoder(tokens)[0]


def train_pretrained_model(config: Any) -> Path:
    """Train only LoRA attention matrices while keeping SD-v1.5 frozen."""
    settings = config.sd15_lora.pre_training
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    components = load_components(config.diffusion, load_lora=False).to(device)
    components.requires_grad_(False)
    parameters = _attach_lora(components.unet, int(settings.rank))
    components.unet.train()
    optimizer = torch.optim.AdamW(
        parameters, lr=float(settings.learning_rate), weight_decay=float(settings.weight_decay)
    )
    loader = build_loader(config.dataset, "train", int(settings.batch_size_per_gpu),
                          int(config.runtime.num_workers), True)
    run = Path(settings.output_dir) / f"{config.experiment_name}-{datetime.now():%Y%m%d-%H%M%S}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    writer = SummaryWriter(run / "tensorboard")
    iterator = iter(loader)
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))
    dtype = next(components.vae.parameters()).dtype
    for step in tqdm(range(1, int(settings.max_iterations) + 1), desc="SD-v1.5 LoRA"):
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
            embeddings = _encode_prompts(components, prompts, device)
        noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=generator)
        timesteps = torch.randint(
            0, components.scheduler.config.num_train_timesteps, (latents.shape[0],),
            device=device, generator=generator,
        ).long()
        noisy = components.scheduler.add_noise(latents, noise, timesteps)
        prediction = components.unet(noisy, timesteps, encoder_hidden_states=embeddings).sample
        loss = F.mse_loss(prediction.float(), noise.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, float(settings.gradient_clip))
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/gradient_norm", float(gradient), step)
        log_diffusion_diagnostics(
            writer, components.unet, optimizer, step,
            {"image": images, "prediction": prediction, "target_noise": noise},
        )
        if step % int(settings.save_every) == 0 or step == int(settings.max_iterations):
            from diffusers import StableDiffusionPipeline
            from diffusers.utils import convert_state_dict_to_diffusers
            from peft.utils import get_peft_model_state_dict

            state = convert_state_dict_to_diffusers(get_peft_model_state_dict(components.unet))
            destination = Path(config.sd15_lora.checkpoints.diffusion_root)
            destination.mkdir(parents=True, exist_ok=True)
            StableDiffusionPipeline.save_lora_weights(destination, unet_lora_layers=state)
            snapshot = run / "checkpoints" / f"step_{step:07d}"
            snapshot.mkdir(parents=True, exist_ok=True)
            StableDiffusionPipeline.save_lora_weights(snapshot, unet_lora_layers=state)
            torch.save({"step": step, "optimizer": optimizer.state_dict()}, destination / "training_state.pt")
    writer.close()
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sd15_lora_pneumoniamnist.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
