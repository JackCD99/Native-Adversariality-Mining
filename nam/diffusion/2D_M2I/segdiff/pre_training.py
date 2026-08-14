"""Official SegDiff epsilon training on any canonical 2D medical dataset.

This dependency-minimal implementation follows the upstream objective: add
scheduled Gaussian noise, concatenate the segmentation condition, and regress
epsilon with MSE. The optional condition dropout is the paper-code adaptation
needed for mask classifier-free guidance; set it to zero for upstream behavior.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nam.config import apply_overrides, load_config
from nam.utils.distributed import finalize, initialize, reduce_mean
from nam.utils.seed import seed_everything
from nam.utils.monitoring import log_diffusion_diagnostics

_package = __package__ or "nam.diffusion.2D_M2I.segdiff"
_runtime = importlib.import_module(f"{_package}.utils.runtime")
_utils = importlib.import_module(f"{_package}.utils")
build_runtime = _runtime.build_runtime
model_output_tensor = _runtime.model_output_tensor
SegDiffAugmentation = _utils.SegDiffAugmentation
build_segdiff_loader = _utils.build_segdiff_loader
prepare_training_pair = _utils.prepare_training_pair
save_diffusers_checkpoint = _utils.save_diffusers_checkpoint


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Match the upstream cosine schedule without importing Transformers."""
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(float(warmup_steps), 1.0)
        progress = float(step - warmup_steps) / max(float(total_steps - warmup_steps), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _forward_loss(
    model: torch.nn.Module,
    scheduler: Any,
    images: torch.Tensor,
    condition: torch.Tensor,
    dropout_probability: float,
    unconditional_value: float,
) -> torch.Tensor:
    noise = torch.randn_like(images)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
    ).long()
    noisy = scheduler.add_noise(images, noise, timesteps)
    if dropout_probability > 0:
        dropped = torch.rand(images.shape[0], device=images.device) < dropout_probability
        condition = condition.clone()
        condition[dropped] = float(unconditional_value)
    prediction = model_output_tensor(model(torch.cat((noisy, condition), dim=1), timesteps))
    return torch.nn.functional.mse_loss(prediction.float(), noise.float())


@torch.no_grad()
def _validation_loss(
    model: torch.nn.Module,
    scheduler: Any,
    loader: Any,
    augmentation: Any,
    config: Any,
    device: torch.device,
    max_batches: int,
) -> torch.Tensor:
    model.eval()
    losses = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = batch.to(device)
        images, condition, _ = prepare_training_pair(
            batch,
            augmentation,
            str(config.diffusion.condition_encoding),
            int(config.diffusion.num_classes),
            int(config.diffusion.condition_channels),
        )
        losses.append(_forward_loss(model, scheduler, images, condition, 0.0, 0.0))
    model.train()
    return torch.stack(losses).mean() if losses else torch.tensor(float("inf"), device=device)


def _save_checkpoint(
    model: torch.nn.Module,
    scheduler: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    destination: Path,
    step: int,
    best_loss: float,
    config: Any,
) -> Path:
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "best_validation_noise_mse": best_loss,
        "config": dict(config),
    }
    if str(config.diffusion.backend).lower() == "official_diffusers":
        return save_diffusers_checkpoint(raw_model, scheduler, destination, state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {**state, "state_dict": raw_model.state_dict()}
    target = destination.with_suffix(".pth")
    torch.save(payload, target)
    return target


def train_pretrained_model(config: Any) -> Path | None:
    """Train SegDiff and publish stable ``last`` and ``best`` checkpoints."""
    settings = config.segdiff.pre_training
    context = initialize(config.runtime.device)
    seed_everything(int(config.runtime.seed) + context.rank, bool(config.runtime.deterministic))
    resume_value = str(getattr(settings, "resume_checkpoint", "")).strip()
    if resume_value:
        resume_config = SimpleNamespace(**dict(config.diffusion))
        resume_config.checkpoint = resume_value
        runtime = build_runtime(resume_config, load_checkpoint=True)
    else:
        runtime = build_runtime(config.diffusion, load_checkpoint=False)
    model = runtime.model.to(context.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.learning_rate))
    lr_scheduler = _cosine_scheduler(
        optimizer, int(settings.lr_warmup_steps), int(settings.max_iterations)
    )
    start_step = 0
    best_loss = float("inf")
    if resume_value:
        resume_path = Path(resume_value)
        state_path = resume_path / "training_state.pt" if resume_path.is_dir() else resume_path
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            lr_scheduler.load_state_dict(state["lr_scheduler"])
        start_step = int(state.get("step", 0))
        best_loss = float(state.get("best_validation_noise_mse", float("inf")))
    train_loader = build_segdiff_loader(
        config.dataset, "train", int(settings.batch_size_per_gpu),
        int(config.runtime.num_workers), True,
    )
    validation_loader = build_segdiff_loader(
        config.dataset, "val", int(settings.validation_batch_size),
        int(config.runtime.num_workers), False,
    )
    augmentation = SegDiffAugmentation(
        int(config.diffusion.resolution),
        float(getattr(settings, "horizontal_flip_probability", 0.0)),
    )
    if context.world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[context.local_rank], output_device=context.local_rank
        )
    run_dir = Path(settings.output_dir) / (
        f"{config.experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    writer = None
    if context.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, ensure_ascii=False)
        writer = SummaryWriter(run_dir / "tensorboard")
    iterator = iter(train_loader)
    progress = (
        tqdm(range(start_step + 1, int(settings.max_iterations) + 1), desc="SegDiff pre-training")
        if context.is_main else range(start_step + 1, int(settings.max_iterations) + 1)
    )
    model.train()
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(context.device)
        images, condition, _ = prepare_training_pair(
            batch,
            augmentation,
            str(config.diffusion.condition_encoding),
            int(config.diffusion.num_classes),
            int(config.diffusion.condition_channels),
        )
        loss = _forward_loss(
            model, runtime.scheduler, images, condition,
            float(settings.condition_dropout_probability),
            float(config.diffusion.unconditional_value),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings.gradient_clip)
        )
        optimizer.step()
        lr_scheduler.step()
        reduced_loss = reduce_mean(loss, context)
        if context.is_main:
            writer.add_scalar("train/noise_mse", reduced_loss.item(), step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
            writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
            progress.set_postfix(mse=f"{reduced_loss.item():.4f}")
            log_diffusion_diagnostics(
                writer,
                model.module if hasattr(model, "module") else model,
                optimizer,
                step,
                {"image": images, "condition": condition},
            )
        evaluate = step % int(settings.validation_every) == 0 or step == int(settings.max_iterations)
        if evaluate and context.is_main:
            raw_model = model.module if hasattr(model, "module") else model
            validation = _validation_loss(
                raw_model, runtime.scheduler, validation_loader, augmentation,
                config, context.device, int(settings.validation_batches),
            )
            value = float(validation.item())
            writer.add_scalar("validation/noise_mse", value, step)
            stable_root = Path(config.segdiff.checkpoints.diffusion_root)
            _save_checkpoint(
                model, runtime.scheduler, optimizer, lr_scheduler,
                stable_root / "last", step, min(best_loss, value), config,
            )
            _save_checkpoint(
                model, runtime.scheduler, optimizer, lr_scheduler,
                run_dir / "checkpoints" / "last", step, min(best_loss, value), config,
            )
            if value < best_loss:
                best_loss = value
                _save_checkpoint(
                    model, runtime.scheduler, optimizer, lr_scheduler,
                    stable_root / "best", step, best_loss, config,
                )
        if evaluate and context.world_size > 1:
            torch.distributed.barrier()
    if writer is not None:
        writer.close()
    finalize(context)
    return run_dir if context.is_main else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train SegDiff on a canonical 2D dataset.")
    parser.add_argument("--config", default="configs/segdiff_2d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
