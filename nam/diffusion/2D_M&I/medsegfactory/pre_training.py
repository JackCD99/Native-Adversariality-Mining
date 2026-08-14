"""Official MedSegFactory paired latent-diffusion training pipeline."""

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

_package = __package__ or "nam.diffusion.2D_M&I.medsegfactory"
_runtime = importlib.import_module(f"{_package}.utils.runtime")
_data = importlib.import_module(f"{_package}.utils.data")
_io = importlib.import_module(f"{_package}.utils.io")
Adapter = importlib.import_module(f"{_package}.model").MedSegFactoryAdapter
MedSegFactoryAugmentation = importlib.import_module(
    f"{_package}.utils.augmentations"
).MedSegFactoryAugmentation


def _view(components: Any, config: Any) -> Any:
    adapter = object.__new__(Adapter)
    adapter.components = adapter.model = components
    adapter.scheduler = components.scheduler
    adapter.config = config
    adapter.metadata = type(
        "Metadata", (), {"noise_channels": 4, "noise_size": tuple(config.noise_size)}
    )()
    return adapter


@torch.no_grad()
def _validation_fid(config: Any, components: Any, device: torch.device) -> float:
    settings = config.medsegfactory.pre_training
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
    adapter, real, fake = _view(components, config.diffusion), [], []
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))

    def feature(value: torch.Tensor) -> torch.Tensor:
        output = extractor(value)
        if isinstance(output, dict):
            output = output.get("features", output.get("logits"))
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.flatten(2).mean(-1) if output.ndim > 2 else output

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
        real.append(feature(image).float().cpu())
        fake.append(feature(generated).float().cpu())
    if not real:
        raise RuntimeError("MedSegFactory validation loader produced no samples.")
    return frechet_distance(torch.cat(real).numpy(), torch.cat(fake).numpy())


def train_pretrained_model(config: Any) -> Path:
    """Train the official JCA U-Net with independent image and mask noises."""
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    settings = config.medsegfactory.pre_training
    components = _runtime.build_official_components(config.diffusion, False).to(device)
    components.vae.eval().requires_grad_(False)
    components.text_encoder.eval().requires_grad_(False)
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
    iterator = iter(loader)
    run = (
        Path(config.medsegfactory.checkpoints.diffusion_root)
        / f"{config.experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    writer, best = SummaryWriter(run / "tensorboard"), float("inf")
    adapter = _view(components, config.diffusion)
    augmentation = MedSegFactoryAugmentation(
        float(getattr(settings, "horizontal_flip_probability", 0.0))
    )
    for step in tqdm(range(1, maximum + 1), desc="MedSegFactory fine-tuning"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        image, _, mask = _data.prepare_pair(
            batch, int(config.diffusion.resolution), int(config.diffusion.num_classes)
        )
        image, mask = augmentation(image, mask)
        with torch.no_grad():
            latent = (
                components.vae.encode(torch.cat((image, mask), 0)).latent_dist.sample()
                * components.vae.config.scaling_factor
            )
            image_prompts, mask_prompts = _data.prompts_from_batch(
                batch, str(config.diffusion.image_prompt), str(config.diffusion.mask_prompt)
            )
            text = torch.cat((adapter._encode(image_prompts), adapter._encode(mask_prompts)), 0)
        noise = torch.randn_like(latent)
        timesteps = torch.randint(
            0, components.scheduler.config.num_train_timesteps, (latent.shape[0],), device=device
        )
        noisy = components.scheduler.add_noise(latent, noise, timesteps)
        prediction = components.unet(noisy, timesteps, encoder_hidden_states=text).sample
        loss = F.mse_loss(prediction.float(), noise.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            components.unet.parameters(), float(settings.gradient_clip)
        )
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/gradient_norm", float(gradient), step)
        log_diffusion_diagnostics(
            writer,
            components.unet,
            optimizer,
            step,
            {"noisy_latent": noisy, "prediction": prediction, "target_noise": noise},
        )
        if step % int(settings.validation_every) == 0 or step == maximum:
            components.unet.eval()
            fid = _validation_fid(config, components, device)
            components.unet.train()
            writer.add_scalar("validation/fid", fid, step)
            state = {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "best_validation_fid": min(best, fid),
                "official_commit": _runtime.OFFICIAL_COMMIT,
            }
            stable = Path(config.medsegfactory.checkpoints.diffusion_root)
            _io.save_checkpoint(components.unet, stable / "last.pt", state)
            _io.save_checkpoint(components.unet, run / "checkpoints" / "last.pt", state)
            if fid < best:
                best = fid
                _io.save_checkpoint(components.unet, stable / "best.pt", state)
    writer.close()
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune official MedSegFactory on canonical paired data."
    )
    parser.add_argument("--config", default="configs/medsegfactory_2d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
