"""Official VolDiT TGCA fine-tuning with paper-aligned checkpoint selection.

The frozen VQ-GAN and unconditional DiT are initialized from official weights.
Only TGCA parameters are optimized with the official Smooth-L1 velocity loss,
AdamW/ExponentialLR recipe, and EMA. Validation 2.5D FID is measured every
500 iterations and the lowest-FID checkpoint is published as ``best.pth``.
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

_package = __package__ or "nam.diffusion.3D_M2I.voldit"
_utils = importlib.import_module(f"{_package}.utils")
_runtime = importlib.import_module(f"{_package}.utils.runtime")
_metrics = importlib.import_module(f"{_package}.utils.metrics")
build_voldit_loader = _utils.build_voldit_loader
prepare_training_tensors = _utils.prepare_training_tensors
save_checkpoint = _utils.save_checkpoint
build_training_components = _runtime.build_training_components
volume_features, fid_2p5d = _metrics.volume_features, _metrics.fid_2p5d


class EMA:
    """Minimal equivalent of the official VolDiT parameter EMA."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def apply(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {}
        for name, value in model.named_parameters():
            if name in self.shadow:
                backup[name] = value.detach().clone()
                value.data.copy_(self.shadow[name].to(value.device, value.dtype))
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, value in model.named_parameters():
            if name in backup:
                value.data.copy_(backup[name])


@torch.no_grad()
def _validate_fid(config: Any, components: Any, device: torch.device) -> float:
    factory = import_factory(config.fid.feature_factory, getattr(config.fid, "project_dir", None))
    extractor = factory(config=config.fid).to(device).eval()
    settings, diffusion = config.voldit.pre_training, config.diffusion
    loader = build_voldit_loader(config.dataset, "val", int(settings.validation_batch_size), int(config.runtime.num_workers), False)
    real = {key: [] for key in ("axial", "coronal", "sagittal")}
    fake = {key: [] for key in real}
    from_model = importlib.import_module(f"{_package}.model").VolDiTAdapter
    # Avoid rebuilding gigabyte-scale models: temporarily expose the trained components.
    adapter = object.__new__(VolDiTAdapter)
    adapter.scheduler, adapter.stage1, adapter.tgca = components.scheduler, components.stage1, components.tgca
    adapter.scale_factor = float(components.scale_factor)
    adapter.volume_size = tuple(diffusion.volume_size)
    adapter.full_schedule_steps = int(diffusion.full_schedule_steps)
    adapter.metadata = type("Metadata", (), {"noise_channels": int(diffusion.noise_channels), "noise_size": tuple(diffusion.noise_size)})()
    adapter.model, adapter.config = components, diffusion
    components.stage1.eval()
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))
    for index, batch in enumerate(loader):
        if index >= int(settings.validation_batches):
            break
        batch = batch.to(device)
        image, _, _, _ = prepare_training_tensors(batch, tuple(diffusion.volume_size), str(diffusion.default_prompt))
        condition = adapter.prepare_condition(batch)
        noise = torch.randn((image.shape[0], int(diffusion.noise_channels), *tuple(diffusion.noise_size)), device=device, generator=generator)
        synthetic, _ = adapter.sample(noise, condition, int(settings.validation_ddim_steps), 1.0)
        for key, value in volume_features(extractor, image, int(config.fid.batch_size)).items():
            real[key].append(value)
        for key, value in volume_features(extractor, synthetic, int(config.fid.batch_size)).items():
            fake[key].append(value)
    if not real["axial"]:
        raise RuntimeError("VolDiT validation loader produced no samples for FID.")
    return fid_2p5d(real, fake)


def train_pretrained_model(config: Any) -> Path:
    """Fine-tune official TGCA on any canonical paired 3D dataset."""
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    settings, diffusion = config.voldit.pre_training, config.diffusion
    components, scheduler = build_training_components(diffusion)
    components.to(device)
    components.stage1.eval().requires_grad_(False)
    parameters = [value for value in components.tgca.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(settings.learning_rate))
    learning_rate = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(settings.lr_gamma))
    ema = EMA(components.tgca, float(settings.ema_decay))
    loader = build_voldit_loader(config.dataset, "train", int(settings.batch_size_per_gpu), int(config.runtime.num_workers), True)
    maximum = int(settings.max_iterations) or int(settings.max_epochs) * len(loader)
    run_dir = Path(config.voldit.checkpoints.diffusion_root) / f"{config.experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
    writer, best_fid = SummaryWriter(run_dir / "tensorboard"), float("inf")
    iterator = iter(loader)
    components.tgca.train()
    for step in tqdm(range(1, maximum + 1), desc="VolDiT TGCA fine-tuning"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        image, control, _, _ = prepare_training_tensors(batch, tuple(diffusion.volume_size), str(diffusion.default_prompt))
        with torch.no_grad():
            latent = components.stage1.encode_stage_2_inputs(image) * components.scale_factor
        noise = torch.randn_like(latent)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (latent.shape[0],), device=device)
        noisy = scheduler.add_noise(latent, noise, timesteps)
        velocity = components.tgca(noisy, timesteps, y=None, condition_input=control)
        target = scheduler.get_velocity(latent, noise, timesteps)
        loss = F.smooth_l1_loss(velocity.float(), target.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(settings.gradient_clip))
        optimizer.step()
        learning_rate.step()
        ema.update(components.tgca)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
        log_diffusion_diagnostics(
            writer,
            components.tgca,
            optimizer,
            step,
            {"noisy_latent": noisy, "prediction": velocity, "target_velocity": target, "condition": control},
        )
        evaluate = step % int(settings.validation_every) == 0 or step == maximum
        if evaluate:
            components.tgca.eval()
            backup = ema.apply(components.tgca)
            fid = _validate_fid(config, components, device)
            ema.restore(components.tgca, backup)
            components.tgca.train()
            writer.add_scalar("validation/fid_2p5d", fid, step)
            state = {"step": step, "optimizer": optimizer.state_dict(), "lr_scheduler": learning_rate.state_dict(), "ema": ema.state_dict(), "scale_factor": components.scale_factor, "best_validation_fid": min(best_fid, fid), "official_commit": _runtime.OFFICIAL_COMMIT}
            stable = Path(config.voldit.checkpoints.diffusion_root)
            save_checkpoint(components.tgca, stable / "last.pth", state)
            save_checkpoint(components.tgca, run_dir / "checkpoints" / "last.pth", state)
            if fid < best_fid:
                best_fid = fid
                state["best_validation_fid"] = fid
                save_checkpoint(components.tgca, stable / "best.pth", state)
    writer.close()
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune official VolDiT TGCA on canonical 3D data.")
    parser.add_argument("--config", default="configs/voldit_3d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
