"""Real-plus-synthetic continuation pipeline for nnU-Net.

The optimizer, loss, deep supervision, and PolyLR remain nnU-Net-native. The
only NAM protocol additions are baseline initialization, matched real/synthetic
streams, and paired CutMix with probability 0.5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from tqdm import tqdm

from nam.downstream.nnunet.model import build_model
from nam.downstream.training import (
    DeepSupervisionLoss,
    DownstreamTrainingMonitor,
    DiceCrossEntropyLoss,
    PolynomialLRScheduler,
    build_loader,
    create_run_directory,
    cycle,
    load_model_checkpoint,
    mean_foreground_dice,
    paired_cutmix,
    primary_logits,
    save_checkpoint,
    should_validate,
)
from nam.utils.seed import seed_everything


def train_synthetic(config: Any, spatial_dims: int) -> Path:
    """Continue nnU-Net from a converged real-data checkpoint."""
    settings = config.nnunet.synthetic_training
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(getattr(config.runtime, "deterministic", False)))
    model = build_model(config.downstream).to(device)
    checkpoint = getattr(settings, "real_checkpoint", getattr(config.downstream, "checkpoint", None))
    if not checkpoint:
        raise ValueError("nnU-Net synthetic training requires a converged real_checkpoint.")
    load_model_checkpoint(model, checkpoint)

    batch_size = int(settings.batch_size_per_stream)
    real_loader = build_loader(config.dataset, "train", spatial_dims, batch_size, int(config.runtime.num_workers), True)
    synthetic_loader = build_loader(
        config.synthetic_dataset, "train", spatial_dims, batch_size, int(config.runtime.num_workers), True
    )
    validation_loader = build_loader(config.dataset, "val", spatial_dims, 1, int(config.runtime.num_workers), False)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(settings.learning_rate),
        momentum=0.99,
        nesterov=True,
        weight_decay=float(settings.weight_decay),
    )
    epochs = int(settings.epochs)
    scheduler = PolynomialLRScheduler(optimizer, epochs, power=0.9)
    base_loss = DiceCrossEntropyLoss(
        int(config.downstream.num_classes),
        include_background=False,
        ignore_index=int(getattr(config.dataset, "ignore_index", -100)),
        batch_dice=bool(getattr(settings, "batch_dice", False)),
    )
    loss_function = DeepSupervisionLoss(base_loss, len(model.segmentation_heads))
    run_dir = create_run_directory(config, "nnunet", "synthetic")
    monitor = DownstreamTrainingMonitor(run_dir, config, model, "nnunet", "synthetic")
    real_iterator, synthetic_iterator = cycle(real_loader), cycle(synthetic_loader)
    steps_per_epoch = int(getattr(settings, "iterations_per_epoch", len(real_loader)))
    global_step, best_dice = 0, -1.0

    for epoch in range(epochs):
        model.train()
        progress = tqdm(range(steps_per_epoch), desc=f"nnU-Net synthetic {epoch + 1}/{epochs}")
        for _ in progress:
            real = next(real_iterator).to(device)
            synthetic = next(synthetic_iterator).to(device)
            real_x, real_y, syn_x, syn_y = paired_cutmix(
                real, synthetic, float(settings.cutmix_probability)
            )
            real_outputs = model(real_x)
            synthetic_outputs = model(syn_x)
            real_loss = loss_function(real_outputs, real_y)
            synthetic_loss = loss_function(synthetic_outputs, syn_y)
            loss = 0.5 * (real_loss + synthetic_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            optimizer.step()
            global_step += 1
            monitor.training_step(
                global_step,
                model,
                optimizer,
                loss,
                torch.cat((real_x, syn_x), 0),
                torch.cat((real_y, syn_y), 0),
                torch.cat((primary_logits(real_outputs), primary_logits(synthetic_outputs)), 0),
                gradient_norm,
                {"real_loss": real_loss, "synthetic_loss": synthetic_loss},
            )
            progress.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()
        if should_validate(epoch + 1, epochs, int(settings.validation_every)):
            dice = mean_foreground_dice(model, validation_loader, device)
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
    train_synthetic(apply_overrides(load_config(arguments.config), arguments.set), arguments.spatial_dims)
