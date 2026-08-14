"""Official MAISI DDPM ControlNet fine-tuning for canonical 3D datasets.

The VAE and diffusion U-Net remain frozen. ControlNet is initialized from the
official U-Net and trained using the official L1 epsilon objective, AdamW at
1e-5, and polynomial decay. Paper checkpoints are ranked by 2.5D FID every
500 iterations rather than by training loss.
"""

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
from nam.utils.imports import import_factory
from nam.utils.seed import seed_everything
from nam.utils.monitoring import log_diffusion_diagnostics

_package = __package__ or "nam.diffusion.3D_M2I.maisi"
_data = importlib.import_module(f"{_package}.utils.data")
_runtime = importlib.import_module(f"{_package}.utils.runtime")
_metrics = importlib.import_module(f"{_package}.utils.metrics")
_io = importlib.import_module(f"{_package}.utils.io")


def _adapter_view(components: Any, diffusion: Any) -> Any:
    adapter_class = importlib.import_module(f"{_package}.model").MAISIAdapter
    adapter = object.__new__(adapter_class)
    adapter.autoencoder, adapter.diffusion_unet, adapter.controlnet = components.autoencoder, components.diffusion_unet, components.controlnet
    adapter.scheduler, adapter.scale_factor = components.scheduler, components.scale_factor
    adapter.volume_size, adapter.full_schedule_steps = tuple(diffusion.volume_size), int(diffusion.full_schedule_steps)
    adapter.metadata = type("Metadata", (), {"noise_channels": int(diffusion.noise_channels), "noise_size": tuple(diffusion.noise_size)})()
    adapter.model, adapter.config = components, diffusion
    return adapter


@torch.no_grad()
def _validate_fid(config: Any, components: Any, device: torch.device) -> float:
    extractor = import_factory(config.fid.feature_factory, getattr(config.fid, "project_dir", None))(config=config.fid).to(device).eval()
    settings, diffusion = config.maisi.pre_training, config.diffusion
    loader = _data.build_maisi_loader(config.dataset, "val", int(settings.validation_batch_size), int(config.runtime.num_workers), False)
    adapter = _adapter_view(components, diffusion)
    components.autoencoder.eval()
    real = {key: [] for key in ("axial", "coronal", "sagittal")}
    fake = {key: [] for key in real}
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))
    for index, batch in enumerate(loader):
        if index >= int(settings.validation_batches):
            break
        batch = batch.to(device)
        image = _data.prepare_training_tensors(batch, tuple(diffusion.volume_size)).mul(2.0).sub(1.0)
        condition = adapter.prepare_condition(batch)
        noise = torch.randn((image.shape[0], int(diffusion.noise_channels), *tuple(diffusion.noise_size)), device=device, generator=generator)
        synthetic, _ = adapter.sample(noise, condition, int(settings.validation_ddim_steps), 1.0)
        for key, value in _metrics.volume_features(extractor, image, int(config.fid.batch_size)).items():
            real[key].append(value)
        for key, value in _metrics.volume_features(extractor, synthetic, int(config.fid.batch_size)).items():
            fake[key].append(value)
    if not real["axial"]:
        raise RuntimeError("MAISI validation loader produced no samples for FID.")
    return _metrics.fid_2p5d(real, fake)


def train_pretrained_model(config: Any) -> Path:
    """Fine-tune official MAISI ControlNet and publish last/best checkpoints."""
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    settings, diffusion = config.maisi.pre_training, config.diffusion
    components = _runtime.build_official_components(diffusion, load_controlnet=False).to(device)
    components.autoencoder.eval().requires_grad_(False)
    components.diffusion_unet.eval().requires_grad_(False)
    components.controlnet.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(components.controlnet.parameters(), lr=float(settings.learning_rate))
    loader = _data.build_maisi_loader(config.dataset, "train", int(settings.batch_size_per_gpu), int(config.runtime.num_workers), True)
    maximum = int(settings.max_iterations) or int(settings.max_epochs) * len(loader)
    learning_rate = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=maximum, power=2.0)
    adapter = _adapter_view(components, diffusion)
    run_dir = Path(config.maisi.checkpoints.diffusion_root) / f"{config.experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
    writer, best_fid, iterator = SummaryWriter(run_dir / "tensorboard"), float("inf"), iter(loader)
    for step in tqdm(range(1, maximum + 1), desc="MAISI ControlNet fine-tuning"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        image = _data.prepare_training_tensors(batch, tuple(diffusion.volume_size))
        condition = adapter.prepare_condition(batch)
        with torch.no_grad():
            latent = components.autoencoder.encode_stage_2_inputs(image) * components.scale_factor
        noise = torch.randn_like(latent)
        timesteps = torch.randint(0, components.scheduler.num_train_timesteps, (latent.shape[0],), device=device)
        noisy = components.scheduler.add_noise(latent, noise, timesteps)
        prediction = adapter._predict(noisy, timesteps, condition)
        loss = F.l1_loss(prediction.float(), noise.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(components.controlnet.parameters(), float(settings.gradient_clip))
        optimizer.step()
        learning_rate.step()
        # Avoid retaining the frozen U-Net graph after gradients have reached
        # ControlNet through its additional residual tensors.
        for parameter in components.diffusion_unet.parameters():
            parameter.grad = None
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
        log_diffusion_diagnostics(
            writer,
            components.controlnet,
            optimizer,
            step,
            {"image": image, "noisy_latent": noisy, "prediction": prediction, "target_noise": noise},
        )
        evaluate = step % int(settings.validation_every) == 0 or step == maximum
        if evaluate:
            components.controlnet.eval()
            fid = _validate_fid(config, components, device)
            components.controlnet.train()
            writer.add_scalar("validation/fid_2p5d", fid, step)
            state = {"step": step, "optimizer": optimizer.state_dict(), "lr_scheduler": learning_rate.state_dict(), "best_validation_fid": min(best_fid, fid), "official_commit": _runtime.OFFICIAL_COMMIT}
            stable = Path(config.maisi.checkpoints.diffusion_root)
            _io.save_checkpoint(components.controlnet, stable / "last.pt", state)
            _io.save_checkpoint(components.controlnet, run_dir / "checkpoints" / "last.pt", state)
            if fid < best_fid:
                best_fid = fid
                state["best_validation_fid"] = fid
                _io.save_checkpoint(components.controlnet, stable / "best.pt", state)
    writer.close()
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune official MAISI ControlNet on canonical 3D data.")
    parser.add_argument("--config", default="configs/maisi_3d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
