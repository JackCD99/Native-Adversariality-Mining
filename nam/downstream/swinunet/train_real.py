"""Training pipeline for Swin-Unet (2D) and SwinUNETR (3D).

Sources:
    https://github.com/HuCaoFighting/Swin-Unet
    https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from tqdm import tqdm

from nam.downstream.swinunet.model import build_model
from nam.downstream.training import (
    DiceCrossEntropyLoss,
    DownstreamTrainingMonitor,
    PolynomialLRScheduler,
    WarmupCosineScheduler,
    build_loader,
    create_run_directory,
    mean_foreground_dice,
    save_checkpoint,
    should_validate,
)
from nam.downstream.training import target_without_channel
from nam.utils.seed import seed_everything


def _settings(config: Any, spatial_dims: int) -> Any:
    return getattr(config.swinunet, f"real_training_{spatial_dims}d")


def _optimizer(model: torch.nn.Module, settings: Any, spatial_dims: int) -> torch.optim.Optimizer:
    if spatial_dims == 2:
        return torch.optim.SGD(
            model.parameters(),
            lr=float(settings.learning_rate),
            momentum=0.9,
            weight_decay=float(settings.weight_decay),
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, settings: Any, spatial_dims: int, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    if spatial_dims == 2:
        return PolynomialLRScheduler(
            optimizer, int(settings.epochs) * steps_per_epoch, power=0.9
        )
    return WarmupCosineScheduler(
        optimizer,
        total_steps=int(settings.epochs),
        warmup_steps=int(settings.warmup_epochs),
    )


def _scan_starts(size: int, window: int, overlap: float) -> list[int]:
    """Return full-coverage sliding-window starts with a final boundary patch."""
    window = min(window, size)
    stride = max(int(window * (1.0 - overlap)), 1)
    starts = list(range(0, max(size - window, 0) + 1, stride))
    boundary = size - window
    if not starts or starts[-1] != boundary:
        starts.append(boundary)
    return starts


@torch.no_grad()
def _sliding_window(model: torch.nn.Module, image: torch.Tensor, roi_size: tuple[int, ...], overlap: float) -> torch.Tensor:
    """Constant-weight sliding-window inference matching MONAI's default mode."""
    spatial = tuple(image.shape[2:])
    windows = tuple(min(roi, size) for roi, size in zip(roi_size, spatial))
    starts = [_scan_starts(size, window, overlap) for size, window in zip(spatial, windows)]
    logits_sum = None
    count = torch.zeros((image.shape[0], 1, *spatial), device=image.device)
    for location in itertools.product(*starts):
        index = (slice(None), slice(None), *(slice(start, start + window) for start, window in zip(location, windows)))
        patch_logits = model(image[index])
        if logits_sum is None:
            logits_sum = torch.zeros(
                (image.shape[0], patch_logits.shape[1], *spatial),
                device=image.device,
                dtype=patch_logits.dtype,
            )
        logits_sum[(slice(None), slice(None), *index[2:])] += patch_logits
        count[(slice(None), slice(None), *index[2:])] += 1
    return logits_sum / count.clamp_min(1)


@torch.no_grad()
def _validation_dice(
    model: torch.nn.Module, loader: Any, device: torch.device, settings: Any, spatial_dims: int
) -> float:
    """Use full images in 2D and official sliding-window validation in 3D."""
    if spatial_dims == 2:
        return mean_foreground_dice(model, loader, device)
    model.eval()
    values = []
    roi_size = tuple(getattr(settings, "roi_size", (96, 96, 96)))
    overlap = float(getattr(settings, "sliding_window_overlap", 0.5))
    for batch in loader:
        batch = batch.to(device)
        logits = _sliding_window(model, batch.image, roi_size, overlap)
        target = target_without_channel(batch.target, logits)
        prediction = logits.argmax(1)
        for class_index in range(1, logits.shape[1]):
            predicted, truth = prediction == class_index, target == class_index
            axes = tuple(range(1, predicted.ndim))
            numerator = 2.0 * (predicted & truth).sum(dim=axes).float()
            denominator = predicted.sum(dim=axes) + truth.sum(dim=axes)
            valid = denominator > 0
            if valid.any():
                values.append(numerator[valid] / denominator[valid])
    model.train()
    return torch.cat(values).mean().item() if values else 0.0


def train_real(config: Any, spatial_dims: int) -> Path:
    """Train the appropriate official Swin segmentation baseline."""
    settings = _settings(config, spatial_dims)
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(getattr(config.runtime, "deterministic", False)))
    model = build_model(config.downstream).to(device)
    train_loader = build_loader(
        config.dataset,
        "train",
        spatial_dims,
        int(settings.batch_size),
        int(config.runtime.num_workers),
        True,
    )
    validation_loader = build_loader(
        config.dataset, "val", spatial_dims, 1, int(config.runtime.num_workers), False
    )
    optimizer = _optimizer(model, settings, spatial_dims)
    scheduler = _scheduler(optimizer, settings, spatial_dims, len(train_loader))
    loss_function = DiceCrossEntropyLoss(
        int(config.downstream.num_classes),
        dice_weight=0.6 if spatial_dims == 2 else 1.0,
        ce_weight=0.4 if spatial_dims == 2 else 1.0,
        include_background=spatial_dims == 3,
        ignore_index=int(getattr(config.dataset, "ignore_index", -100)),
    )
    run_dir = create_run_directory(config, "swinunet" if spatial_dims == 2 else "swinunetr", "real")
    model_name = "swinunet" if spatial_dims == 2 else "swinunetr"
    monitor = DownstreamTrainingMonitor(run_dir, config, model, model_name, "real")
    global_step, best_dice = 0, -1.0

    for epoch in range(int(settings.epochs)):
        model.train()
        progress = tqdm(train_loader, desc=f"Swin real {epoch + 1}/{int(settings.epochs)}")
        for batch in progress:
            batch = batch.to(device)
            outputs = model(batch.image)
            loss = loss_function(outputs, batch.target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings.gradient_clip)
            )
            optimizer.step()
            global_step += 1
            monitor.training_step(
                global_step,
                model,
                optimizer,
                loss,
                batch.image,
                batch.target,
                outputs,
                gradient_norm,
            )
            if spatial_dims == 2:
                scheduler.step()
            progress.set_postfix(loss=f"{loss.item():.3f}")
        if spatial_dims == 3:
            scheduler.step()
        if should_validate(epoch + 1, int(settings.epochs), int(settings.validation_every)):
            dice = _validation_dice(model, validation_loader, device, settings, spatial_dims)
            monitor.validation(epoch + 1, global_step, dice, "foreground_dice")
            save_checkpoint(run_dir, "latest.pt", model, optimizer, scheduler, epoch + 1, global_step, dice)
            if dice > best_dice:
                best_dice = dice
                save_checkpoint(run_dir, "best.pt", model, optimizer, scheduler, epoch + 1, global_step, dice)
    monitor.close()
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/table1_2d.yaml")
    parser.add_argument("--spatial-dims", type=int, choices=(2, 3), default=2)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    from nam.config import apply_overrides, load_config
    arguments = parse_args()
    train_real(apply_overrides(load_config(arguments.config), arguments.set), arguments.spatial_dims)
