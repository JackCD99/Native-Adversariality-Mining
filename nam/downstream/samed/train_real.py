"""Official SAMed real-data training pipeline.

The pipeline follows the public SAMed defaults: AdamW, 5e-3 peak learning
rate, 0.1 weight decay, 250-step warmup, power-0.9 polynomial decay, and
0.2 cross-entropy plus 0.8 Dice loss on native low-resolution logits.

Source: https://github.com/hitachinsk/SAMed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.nn import functional as F
from tqdm import tqdm

from nam.downstream.samed.model import SAMed, build_model
from nam.downstream.training import (
    DiceCrossEntropyLoss,
    DownstreamTrainingMonitor,
    WarmupPolynomialScheduler,
    build_loader,
    create_run_directory,
    mean_foreground_dice,
    save_checkpoint,
    should_validate,
    target_without_channel,
)
from nam.utils.seed import resolve_stage_seed, seed_everything


def _native_loss(
    model: SAMed, images: torch.Tensor, targets: torch.Tensor, loss_function: DiceCrossEntropyLoss
) -> torch.Tensor:
    logits = model.forward_low_resolution(images)
    targets = target_without_channel(targets)
    resized = F.interpolate(targets.unsqueeze(1).float(), size=logits.shape[2:], mode="nearest")[:, 0].long()
    return loss_function(logits, resized)


def _optimizer(model: SAMed, settings: Any) -> torch.optim.AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=float(settings.learning_rate),
        betas=(0.9, 0.999),
        weight_decay=float(settings.weight_decay),
    )


def train_real(config: Any, spatial_dims: int) -> Path:
    """Train LoRA and mask-decoder parameters on real images only."""
    settings = config.samed.real_training
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(resolve_stage_seed(config, "downstream"), bool(getattr(config.runtime, "deterministic", False)))
    model = build_model(config.downstream).to(device)
    train_loader = build_loader(
        config.dataset, "train", spatial_dims, int(settings.batch_size), int(config.runtime.num_workers), True
    )
    validation_loader = build_loader(
        config.dataset, "val", spatial_dims, 1, int(config.runtime.num_workers), False
    )
    optimizer = _optimizer(model, settings)
    total_steps = int(settings.epochs) * len(train_loader)
    scheduler = WarmupPolynomialScheduler(
        optimizer, total_steps, int(settings.warmup_iterations), power=0.9
    )
    loss_function = DiceCrossEntropyLoss(
        int(config.downstream.num_classes),
        dice_weight=0.8,
        ce_weight=0.2,
        include_background=True,
        ignore_index=int(getattr(config.dataset, "ignore_index", -100)),
    )
    run_dir = create_run_directory(config, "samed", "real")
    monitor = DownstreamTrainingMonitor(run_dir, config, model, "samed", "real")
    global_step, best_dice = 0, -1.0

    for epoch in range(int(settings.epochs)):
        model.train()
        progress = tqdm(train_loader, desc=f"SAMed real {epoch + 1}/{int(settings.epochs)}")
        for batch in progress:
            batch = batch.to(device)
            low_resolution_logits = model.forward_low_resolution(batch.image)
            resized_target = F.interpolate(
                target_without_channel(batch.target).unsqueeze(1).float(),
                size=low_resolution_logits.shape[2:],
                mode="nearest",
            )[:, 0].long()
            loss = loss_function(low_resolution_logits, resized_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            monitor.training_step(
                global_step,
                model,
                optimizer,
                loss,
                batch.image,
                batch.target,
                F.interpolate(
                    low_resolution_logits,
                    size=batch.target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                gradient_norm,
            )
            progress.set_postfix(loss=f"{loss.item():.3f}")
        if should_validate(epoch + 1, int(settings.epochs), int(settings.validation_every)):
            dice = mean_foreground_dice(model, validation_loader, device)
            monitor.validation(epoch + 1, global_step, dice, "foreground_dice")
            save_checkpoint(run_dir, "latest.pt", model, optimizer, scheduler, epoch + 1, global_step, dice)
            if dice > best_dice:
                best_dice = dice
                save_checkpoint(run_dir, "best.pt", model, optimizer, scheduler, epoch + 1, global_step, dice)
        if epoch + 1 >= int(getattr(settings, "stop_epoch", settings.epochs)):
            break
    monitor.close()
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/table1_2d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    from nam.config import apply_overrides, load_config
    arguments = parse_args()
    train_real(apply_overrides(load_config(arguments.config), arguments.set), 2)
