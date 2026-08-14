"""Shared optimization and mIoU validation for natural-image segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn import functional as F
from tqdm import tqdm

from nam.downstream.training import (
    DownstreamTrainingMonitor,
    build_loader,
    create_run_directory,
    cycle,
    load_model_checkpoint,
    paired_cutmix,
    save_checkpoint,
    should_validate,
    target_without_channel,
)
from nam.utils.seed import seed_everything


def _logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, dict):
        output = output.get("out", output.get("logits"))
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def _loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Route query-based models through their native matching objective."""
    targets = target_without_channel(targets)
    if hasattr(model, "compute_loss"):
        return model.compute_loss(images, targets, ignore_index)
    return F.cross_entropy(_logits(model, images), targets, ignore_index=ignore_index)


@torch.no_grad()
def mean_iou(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> float:
    """Compute dataset-level mean IoU over classes present in the union."""
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)
    for batch in loader:
        batch = batch.to(device)
        prediction = _logits(model, batch.image).argmax(1)
        target = target_without_channel(batch.target)
        valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
        indices = target[valid] * num_classes + prediction[valid]
        confusion += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    valid_classes = union > 0
    model.train()
    return float((intersection[valid_classes] / union[valid_classes]).mean()) if valid_classes.any() else 0.0


def _settings(config: Any, model_name: str, phase: str) -> Any:
    return getattr(getattr(config, model_name), f"{phase}_training")


def train_natural_segmentation(
    config: Any,
    spatial_dims: int,
    model_name: str,
    build_model: Callable[[Any], torch.nn.Module],
    phase: str,
) -> Path:
    """Train a real baseline or continue it with 1:1 real/synthetic streams."""
    if spatial_dims != 2:
        raise ValueError(f"{model_name} is configured for 2D natural images.")
    settings = _settings(config, model_name, phase)
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    model = build_model(config.downstream).to(device)
    if phase == "synthetic":
        checkpoint = str(settings.real_checkpoint)
        if not checkpoint:
            raise ValueError(f"{model_name} synthetic training requires real_checkpoint.")
        load_model_checkpoint(model, checkpoint)
    batch_size = int(
        settings.batch_size if phase == "real" else settings.batch_size_per_stream
    )
    real_loader = build_loader(
        config.dataset, "train", 2, batch_size, int(config.runtime.num_workers), True
    )
    synthetic_loader = None
    if phase == "synthetic":
        synthetic_loader = build_loader(
            config.synthetic_dataset, "train", 2, batch_size,
            int(config.runtime.num_workers), True,
        )
    validation_loader = build_loader(
        config.dataset, "val", 2, 1, int(config.runtime.num_workers), False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )
    configured_steps = int(getattr(settings, "iterations_per_epoch", 0))
    steps_per_epoch = configured_steps if configured_steps > 0 else len(real_loader)
    total_steps = max(int(settings.epochs) * steps_per_epoch, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / total_steps) ** 0.9
    )
    ignore_index = int(config.dataset.ignore_index)
    run = create_run_directory(config, model_name, phase)
    monitor = DownstreamTrainingMonitor(run, config, model, model_name, phase)
    real_iterator = cycle(real_loader)
    synthetic_iterator = cycle(synthetic_loader) if synthetic_loader is not None else None
    global_step, best_iou = 0, -1.0
    for epoch in range(int(settings.epochs)):
        model.train()
        progress = tqdm(range(steps_per_epoch), desc=f"{model_name} {phase} {epoch + 1}")
        for _ in progress:
            real = next(real_iterator).to(device)
            if synthetic_iterator is None:
                real_logits = _logits(model, real.image)
                losses = [
                    model.compute_loss(real.image, target_without_channel(real.target), ignore_index)
                    if hasattr(model, "compute_loss")
                    else F.cross_entropy(
                        real_logits,
                        target_without_channel(real.target),
                        ignore_index=ignore_index,
                    )
                ]
                monitor_images, monitor_targets, monitor_logits = (
                    real.image,
                    real.target,
                    real_logits,
                )
            else:
                synthetic = next(synthetic_iterator).to(device)
                real_x, real_y, syn_x, syn_y = paired_cutmix(
                    real, synthetic, float(settings.cutmix_probability)
                )
                losses = [
                    _loss(model, real_x, real_y, ignore_index),
                    _loss(model, syn_x, syn_y, ignore_index),
                ]
                with torch.no_grad():
                    monitor_logits = torch.cat((_logits(model, real_x), _logits(model, syn_x)), 0)
                monitor_images = torch.cat((real_x, syn_x), 0)
                monitor_targets = torch.cat((real_y, syn_y), 0)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings.gradient_clip)
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            monitor.training_step(
                global_step,
                model,
                optimizer,
                loss,
                monitor_images,
                monitor_targets,
                monitor_logits,
                gradient_norm,
                {f"stream_{index}_loss": value for index, value in enumerate(losses)},
            )
            progress.set_postfix(loss=f"{loss.item():.3f}")
        if should_validate(epoch + 1, int(settings.epochs), int(settings.validation_every)):
            value = mean_iou(
                model, validation_loader, device, int(config.downstream.num_classes), ignore_index
            )
            monitor.validation(epoch + 1, global_step, value, "mean_iou")
            save_checkpoint(run, "latest.pt", model, optimizer, scheduler, epoch + 1, global_step, value)
            if value > best_iou:
                best_iou = value
                save_checkpoint(run, "best.pt", model, optimizer, scheduler, epoch + 1, global_step, value)
    monitor.close()
    return run
