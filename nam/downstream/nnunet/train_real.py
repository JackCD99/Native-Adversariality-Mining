"""Real-data training pipeline for nnU-Net v2.

Defaults follow nnUNetTrainer: SGD with Nesterov momentum 0.99, initial
learning rate 1e-2, weight decay 3e-5, PolyLR, 1000 epochs, 250 iterations per
epoch, Dice+CE, and exponentially weighted deep supervision.
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
    mean_foreground_dice,
    save_checkpoint,
    should_validate,
)
from nam.utils.seed import resolve_stage_seed, seed_everything


def train_real(config: Any, spatial_dims: int) -> Path:
    """Train the leakage-free real-data baseline used by NAM."""
    settings = config.nnunet.real_training
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(resolve_stage_seed(config, "downstream"), bool(getattr(config.runtime, "deterministic", False)))
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
        config.dataset,
        "val",
        spatial_dims,
        1,
        int(config.runtime.num_workers),
        False,
    )
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
    output_count = len(model.segmentation_heads)
    loss_function = DeepSupervisionLoss(base_loss, output_count)
    run_dir = create_run_directory(config, "nnunet", "real")
    monitor = DownstreamTrainingMonitor(run_dir, config, model, "nnunet", "real")
    iterator = cycle(train_loader)
    global_step, best_dice = 0, -1.0
    iterations_per_epoch = int(settings.iterations_per_epoch)

    for epoch in range(epochs):
        model.train()
        progress = tqdm(range(iterations_per_epoch), desc=f"nnU-Net real {epoch + 1}/{epochs}")
        for _ in progress:
            batch = next(iterator).to(device)
            outputs = model(batch.image)
            loss = loss_function(outputs, batch.target)
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
                batch.image,
                batch.target,
                outputs,
                gradient_norm,
            )
            progress.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()
        if should_validate(epoch + 1, epochs, int(settings.validation_every)):
            dice = mean_foreground_dice(model, validation_loader, device)
            monitor.validation(epoch + 1, global_step, dice, "foreground_dice")
            save_checkpoint(
                run_dir, "latest.pt", model, optimizer, scheduler, epoch + 1, global_step, dice
            )
            if dice > best_dice:
                best_dice = dice
                save_checkpoint(
                    run_dir, "best.pt", model, optimizer, scheduler, epoch + 1, global_step, dice
                )
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
