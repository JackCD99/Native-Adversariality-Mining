"""NAM real-plus-synthetic continuation pipeline for SAMed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from tqdm import tqdm

from nam.downstream.samed.model import build_model
from nam.downstream.samed.train_real import _native_loss, _optimizer
from nam.downstream.training import (
    DiceCrossEntropyLoss,
    DownstreamTrainingMonitor,
    WarmupPolynomialScheduler,
    build_loader,
    create_run_directory,
    cycle,
    load_model_checkpoint,
    mean_foreground_dice,
    paired_cutmix,
    save_checkpoint,
    should_validate,
)
from nam.utils.seed import resolve_stage_seed, seed_everything


def train_synthetic(config: Any, spatial_dims: int) -> Path:
    """Continue a converged SAMed checkpoint on 1:1 real/synthetic batches."""
    settings = config.samed.synthetic_training
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(resolve_stage_seed(config, "downstream"), bool(getattr(config.runtime, "deterministic", False)))
    model = build_model(config.downstream).to(device)
    checkpoint = getattr(settings, "real_checkpoint", getattr(config.downstream, "checkpoint", None))
    if not checkpoint:
        raise ValueError("SAMed synthetic training requires a converged real_checkpoint.")
    load_model_checkpoint(model, checkpoint)
    batch_size = int(settings.batch_size_per_stream)
    real_loader = build_loader(config.dataset, "train", spatial_dims, batch_size, int(config.runtime.num_workers), True)
    synthetic_loader = build_loader(
        config.synthetic_dataset, "train", spatial_dims, batch_size, int(config.runtime.num_workers), True
    )
    validation_loader = build_loader(config.dataset, "val", spatial_dims, 1, int(config.runtime.num_workers), False)
    optimizer = _optimizer(model, settings)
    steps_per_epoch = int(getattr(settings, "iterations_per_epoch", len(real_loader)))
    scheduler = WarmupPolynomialScheduler(
        optimizer, int(settings.epochs) * steps_per_epoch, int(settings.warmup_iterations), power=0.9
    )
    loss_function = DiceCrossEntropyLoss(
        int(config.downstream.num_classes),
        dice_weight=0.8,
        ce_weight=0.2,
        include_background=True,
        ignore_index=int(getattr(config.dataset, "ignore_index", -100)),
    )
    run_dir = create_run_directory(config, "samed", "synthetic")
    monitor = DownstreamTrainingMonitor(run_dir, config, model, "samed", "synthetic")
    real_iterator, synthetic_iterator = cycle(real_loader), cycle(synthetic_loader)
    global_step, best_dice = 0, -1.0

    for epoch in range(int(settings.epochs)):
        model.train()
        progress = tqdm(range(steps_per_epoch), desc=f"SAMed synthetic {epoch + 1}/{int(settings.epochs)}")
        for _ in progress:
            real, synthetic = next(real_iterator).to(device), next(synthetic_iterator).to(device)
            real_x, real_y, syn_x, syn_y = paired_cutmix(real, synthetic, float(settings.cutmix_probability))
            real_logits = model.forward_low_resolution(real_x)
            synthetic_logits = model.forward_low_resolution(syn_x)
            real_target = torch.nn.functional.interpolate(
                real_y.unsqueeze(1).float() if real_y.ndim == 3 else real_y.float(),
                size=real_logits.shape[2:], mode="nearest",
            )[:, 0].long()
            synthetic_target = torch.nn.functional.interpolate(
                syn_y.unsqueeze(1).float() if syn_y.ndim == 3 else syn_y.float(),
                size=synthetic_logits.shape[2:], mode="nearest",
            )[:, 0].long()
            real_loss = loss_function(real_logits, real_target)
            synthetic_loss = loss_function(synthetic_logits, synthetic_target)
            loss = 0.5 * (real_loss + synthetic_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            full_size = real_x.shape[-2:]
            monitor.training_step(
                global_step,
                model,
                optimizer,
                loss,
                torch.cat((real_x, syn_x), 0),
                torch.cat((real_y, syn_y), 0),
                torch.cat(
                    (
                        torch.nn.functional.interpolate(real_logits, full_size, mode="bilinear", align_corners=False),
                        torch.nn.functional.interpolate(synthetic_logits, full_size, mode="bilinear", align_corners=False),
                    ),
                    0,
                ),
                gradient_norm,
                {"real_loss": real_loss, "synthetic_loss": synthetic_loss},
            )
            progress.set_postfix(loss=f"{loss.item():.3f}")
        if should_validate(epoch + 1, int(settings.epochs), int(settings.validation_every)):
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
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    from nam.config import apply_overrides, load_config
    arguments = parse_args()
    train_synthetic(apply_overrides(load_config(arguments.config), arguments.set), 2)
