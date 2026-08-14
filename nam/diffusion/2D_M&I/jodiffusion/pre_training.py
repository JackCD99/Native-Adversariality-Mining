"""Official JoDiffusion joint-noise fine-tuning on canonical medical datasets."""

from __future__ import annotations
import argparse, importlib, json, sys
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
from nam.evaluation.metrics import frechet_distance
from nam.utils.imports import import_factory
from nam.utils.seed import seed_everything
from nam.utils.monitoring import log_diffusion_diagnostics

_package = __package__ or "nam.diffusion.2D_M&I.jodiffusion"
_runtime = importlib.import_module(f"{_package}.utils.runtime")
_data = importlib.import_module(f"{_package}.utils.data")
_io = importlib.import_module(f"{_package}.utils.io")
JoDiffusionAugmentation = importlib.import_module(
    f"{_package}.utils.augmentations"
).JoDiffusionAugmentation
JoDiffusionAdapter = importlib.import_module(f"{_package}.model").JoDiffusionAdapter


def _encode_real(
    components: Any, image: torch.Tensor, label: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        image_latent = (
            components.vae.encode(image).latent_dist.sample() * components.vae.config.scaling_factor
        )
        label_latent = (
            components.label_vae.encode(label).latent_dist.sample()
            * components.label_vae.config.scaling_factor
        )
        pixels = components.image_processor.preprocess(
            (image + 1) / 2, return_tensors="pt"
        ).pixel_values.to(image.device, components.image_encoder.dtype)
        image_embed = components.image_encoder(pixels).image_embeds.unsqueeze(1)
    return image_latent, image_embed, label_latent


def _adapter_view(components: Any, config: Any) -> Any:
    adapter = object.__new__(JoDiffusionAdapter)
    adapter.components = adapter.model = components
    adapter.scheduler = components.scheduler
    adapter.config = config
    adapter.metadata = type(
        "Metadata", (), {"noise_channels": 8, "noise_size": tuple(config.noise_size)}
    )()
    return adapter


@torch.no_grad()
def _validate_fid(config: Any, components: Any, device: torch.device) -> float:
    settings = config.jodiffusion.pre_training
    extractor = (
        import_factory(config.fid.feature_factory, getattr(config.fid, "project_dir", None))(
            config=config.fid
        )
        .to(device)
        .eval()
    )
    loader = _data.build_loader(
        config.dataset,
        "val",
        int(settings.validation_batch_size),
        int(config.runtime.num_workers),
        False,
    )
    adapter, real, fake = _adapter_view(components, config.diffusion), [], []
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))
    for index, batch in enumerate(loader):
        if index >= int(settings.validation_batches):
            break
        batch = batch.to(device)
        image, _, _ = _data.prepare_pair(
            batch, int(config.diffusion.resolution), int(config.diffusion.num_classes)
        )
        condition = adapter.prepare_condition(batch)
        noise = adapter.sample_probe_noise(image.shape[0], generator)
        generated, _ = adapter.sample(
            noise,
            condition,
            int(settings.validation_ddim_steps),
            float(settings.validation_cfg_scale),
        )

        def features(value: torch.Tensor) -> torch.Tensor:
            output = extractor(value)
            if isinstance(output, dict):
                output = output.get("features", output.get("logits"))
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output.flatten(2).mean(-1) if output.ndim > 2 else output

        real.append(features(image).float().cpu())
        fake.append(features(generated).float().cpu())
    if not real:
        raise RuntimeError("JoDiffusion validation loader produced no samples.")
    return frechet_distance(torch.cat(real).numpy(), torch.cat(fake).numpy())


def train_pretrained_model(config: Any) -> Path:
    """Train only the official U-ViT with its joint text/image/label objective."""
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    settings = config.jodiffusion.pre_training
    # Pre-training starts from the official pipeline unless a resumable
    # initialization directory is explicitly configured.
    diffusion_settings = type("Config", (), dict(config.diffusion))()
    diffusion_settings.checkpoint = str(getattr(settings, "initialization_checkpoint", ""))
    components = _runtime.build_official_components(diffusion_settings).to(device)
    for module in (
        components.vae,
        components.label_vae,
        components.text_encoder,
        components.image_encoder,
    ):
        module.eval().requires_grad_(False)
    components.unet.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        components.unet.parameters(),
        lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )
    loader = _data.build_loader(
        config.dataset,
        "train",
        int(settings.batch_size_per_gpu),
        int(config.runtime.num_workers),
        True,
    )
    maximum = int(settings.max_iterations) or int(settings.max_epochs) * len(loader)
    run = (
        Path(config.jodiffusion.checkpoints.diffusion_root)
        / f"{config.experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    writer, iterator, best = SummaryWriter(run / "tensorboard"), iter(loader), float("inf")
    augmentation = JoDiffusionAugmentation(
        float(getattr(settings, "horizontal_flip_probability", 0.5))
    )
    for step in tqdm(range(1, maximum + 1), desc="JoDiffusion fine-tuning"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        image, _, bitmap = _data.prepare_pair(
            batch, int(config.diffusion.resolution), int(config.diffusion.num_classes)
        )
        image, bitmap = augmentation(image, bitmap)
        image_latent, image_embed, label_latent = _encode_real(components, image, bitmap)
        prompts = _data.prompts_from_batch(batch, str(config.diffusion.default_prompt))
        adapter = _adapter_view(components, config.diffusion)
        text = adapter._prompt_embeddings(prompts)
        text_noise, image_noise, clip_noise, label_noise = (
            torch.randn_like(text),
            torch.randn_like(image_latent),
            torch.randn_like(image_embed),
            torch.randn_like(label_latent),
        )
        timesteps = torch.randint(
            0, components.scheduler.config.num_train_timesteps, (image.shape[0],), device=device
        )
        text_steps = torch.randint_like(
            timesteps, 0, components.scheduler.config.num_train_timesteps
        )
        noisy_text = components.scheduler.add_noise(text, text_noise, text_steps)
        noisy_image = components.scheduler.add_noise(image_latent, image_noise, timesteps)
        noisy_clip = components.scheduler.add_noise(image_embed, clip_noise, timesteps)
        noisy_label = components.scheduler.add_noise(label_latent, label_noise, timesteps)
        outputs = components.unet(
            noisy_text,
            noisy_image,
            noisy_clip,
            noisy_label,
            timestep_img=timesteps,
            timestep_text=text_steps,
            data_type=1,
        )
        loss = F.mse_loss(
            torch.cat([item.flatten(1) for item in outputs], 1).float(),
            torch.cat(
                [
                    text_noise.flatten(1),
                    image_noise.flatten(1),
                    clip_noise.flatten(1),
                    label_noise.flatten(1),
                ],
                1,
            ).float(),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(components.unet.parameters(), float(settings.gradient_clip))
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), step)
        log_diffusion_diagnostics(
            writer,
            components.unet,
            optimizer,
            step,
            {
                "image_latent": image_latent,
                "label_latent": label_latent,
                "image_noise": image_noise,
                "label_noise": label_noise,
            },
        )
        if step % int(settings.validation_every) == 0 or step == maximum:
            components.unet.eval()
            fid = _validate_fid(config, components, device)
            components.unet.train()
            writer.add_scalar("validation/fid", fid, step)
            state = {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "best_validation_fid": min(best, fid),
                "official_commit": _runtime.OFFICIAL_COMMIT,
            }
            stable = Path(config.jodiffusion.checkpoints.diffusion_root)
            _io.save_unet(components.unet, stable / "last", state)
            _io.save_unet(components.unet, run / "checkpoints" / "last", state)
            if fid < best:
                best = fid
                _io.save_unet(components.unet, stable / "best", state)
    writer.close()
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune official JoDiffusion on canonical paired data."
    )
    parser.add_argument("--config", default="configs/jodiffusion_2d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
