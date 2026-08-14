"""Shared real and real-plus-synthetic training for image classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import random

import torch
from torch.nn import functional as F
from tqdm import tqdm

from nam.downstream.training import (
    build_loader,
    create_run_directory,
    cycle,
    load_model_checkpoint,
    save_checkpoint,
    should_validate,
)
from nam.utils.monitoring import ExperimentMonitor, logging_interval
from nam.utils.seed import seed_everything


@torch.no_grad()
def classification_metrics(model: torch.nn.Module, loader: Any,
                           device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)
    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch.image).argmax(1)
        index = batch.target.long() * num_classes + prediction
        confusion += torch.bincount(index, minlength=num_classes**2).reshape(num_classes, num_classes)
    accuracy = confusion.diag().sum() / confusion.sum().clamp_min(1)
    recall = confusion.diag() / confusion.sum(1).clamp_min(1)
    specificity = []
    total = confusion.sum()
    for class_index in range(num_classes):
        tp = confusion[class_index, class_index]
        fn = confusion[class_index].sum() - tp
        fp = confusion[:, class_index].sum() - tp
        tn = total - tp - fn - fp
        specificity.append(tn / (tn + fp).clamp_min(1))
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(recall.mean()),
        "specificity": float(torch.stack(specificity).mean()),
    }


def _paired_cutmix(real: Any, synthetic: Any, probability: float):
    """Swap one rectangle and retain the exact area-weighted class targets."""
    count = min(real.image.shape[0], synthetic.image.shape[0])
    real_images = real.image[:count].clone()
    synthetic_images = synthetic.image[:count].clone()
    real_targets = real.target[:count].long()
    synthetic_targets = synthetic.target[:count].long()
    if random.random() >= probability:
        return real_images, synthetic_images, real_targets, synthetic_targets, 1.0
    height, width = real_images.shape[-2:]
    cut_ratio = random.uniform(0.25, 0.75)
    cut_height, cut_width = max(1, int(height * cut_ratio)), max(1, int(width * cut_ratio))
    top = random.randint(0, height - cut_height)
    left = random.randint(0, width - cut_width)
    region = (..., slice(top, top + cut_height), slice(left, left + cut_width))
    real_source = real_images[region].clone()
    real_images[region] = synthetic_images[region]
    synthetic_images[region] = real_source
    retained = 1.0 - (cut_height * cut_width) / float(height * width)
    return real_images, synthetic_images, real_targets, synthetic_targets, retained


def train_classification(config: Any, model_name: str, build_model: Callable[[Any], torch.nn.Module],
                         phase: str) -> Path:
    settings = getattr(
        getattr(config, model_name),
        "real_training" if phase == "real" else "synthetic_training",
    )
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    model = build_model(config.downstream).to(device)
    if phase == "synthetic":
        load_model_checkpoint(model, str(settings.real_checkpoint))
    batch_size = int(settings.batch_size if phase == "real" else settings.batch_size_per_stream)
    real_loader = build_loader(config.dataset, "train", 2, batch_size,
                               int(config.runtime.num_workers), True)
    synthetic_loader = None
    if phase == "synthetic":
        synthetic_loader = build_loader(config.synthetic_dataset, "train", 2, batch_size,
                                        int(config.runtime.num_workers), True)
    validation_loader = build_loader(config.dataset, "val", 2, batch_size,
                                     int(config.runtime.num_workers), False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings.learning_rate), weight_decay=float(settings.weight_decay)
    )
    epochs = int(settings.epochs)
    steps = int(getattr(settings, "iterations_per_epoch", 0)) or len(real_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs * steps, 1))
    run = create_run_directory(config, model_name, phase)
    monitor = ExperimentMonitor(run, config)
    monitor.describe_model(model_name, model)
    real_iterator = cycle(real_loader)
    synthetic_iterator = cycle(synthetic_loader) if synthetic_loader is not None else None
    scalar_every = logging_interval(config, "scalar_every", 10)
    image_every = logging_interval(config, "downstream_image_every", 200)
    global_step, best_accuracy = 0, -1.0
    for epoch in range(epochs):
        model.train()
        progress = tqdm(range(steps), desc=f"{model_name} {phase} {epoch + 1}/{epochs}")
        for _ in progress:
            real = next(real_iterator).to(device)
            if synthetic_iterator is None:
                real_logits = model(real.image)
                losses = [F.cross_entropy(real_logits, real.target.long())]
                images, targets, logits = real.image, real.target, real_logits
            else:
                synthetic = next(synthetic_iterator).to(device)
                real_images, synthetic_images, real_targets, synthetic_targets, retained = _paired_cutmix(
                    real, synthetic, float(getattr(settings, "cutmix_probability", 0.5))
                )
                real_logits = model(real_images)
                synthetic_logits = model(synthetic_images)
                losses = [
                    retained * F.cross_entropy(real_logits, real_targets)
                    + (1.0 - retained) * F.cross_entropy(real_logits, synthetic_targets),
                    retained * F.cross_entropy(synthetic_logits, synthetic_targets)
                    + (1.0 - retained) * F.cross_entropy(synthetic_logits, real_targets),
                ]
                images = torch.cat((real_images, synthetic_images), 0)
                targets = torch.cat((real_targets, synthetic_targets), 0)
                logits = torch.cat((real_logits, synthetic_logits), 0)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.gradient_clip))
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step == 1 or global_step % scalar_every == 0:
                monitor.log_metrics(
                    f"{model_name}/{phase}/train",
                    {"loss": loss, "accuracy": (logits.argmax(1) == targets).float().mean(),
                     "gradient_norm": gradient}, global_step,
                )
                monitor.log_optimizer(optimizer, global_step)
            if global_step == 1 or global_step % image_every == 0:
                monitor.writer.add_images(
                    f"{model_name}/{phase}/examples", images[:8].detach().float().clamp(0, 1), global_step
                )
            progress.set_postfix(loss=f"{loss.item():.3f}")
        if should_validate(epoch + 1, epochs, int(settings.validation_every)):
            metrics = classification_metrics(
                model, validation_loader, device, int(config.downstream.num_classes)
            )
            monitor.log_metrics(f"{model_name}/{phase}/validation", metrics, global_step)
            save_checkpoint(
                run, "latest.pt", model, optimizer, scheduler, epoch + 1, global_step,
                metrics["accuracy"], {"metrics": metrics},
            )
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                save_checkpoint(
                    run, "best.pt", model, optimizer, scheduler, epoch + 1, global_step,
                    metrics["accuracy"], {"metrics": metrics},
                )
    monitor.close()
    return run
