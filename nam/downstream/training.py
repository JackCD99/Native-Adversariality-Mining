"""Shared data, loss, checkpoint, and validation primitives.

Model-specific optimization policies live in each model package. This module
centralizes invariant data, metric, and checkpoint mechanics to avoid divergent
implementations of data loading, Dice measurement, and checkpoint I/O.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nam.data import NAMBatch, build_dataset, collate_medical_batch
from nam.utils.monitoring import ExperimentMonitor, logging_interval


def target_without_channel(target: torch.Tensor, logits: torch.Tensor | None = None) -> torch.Tensor:
    """Convert B1HW/B1DHW labels to BHW/BDHW integer maps."""
    if target.ndim >= 4 and target.shape[1] == 1:
        if logits is None or target.ndim == logits.ndim:
            target = target[:, 0]
    return target.long()


def primary_logits(output: Any) -> torch.Tensor:
    """Extract the full-resolution logits from common segmentation outputs."""
    if isinstance(output, dict):
        output = output.get("out", output.get("logits", output.get("pred")))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError("The segmentation model did not return a logits tensor.")
    return output


@torch.no_grad()
def batch_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """Compute inexpensive batch diagnostics without affecting optimization."""
    target = target_without_channel(target, logits)
    prediction = primary_logits(logits).argmax(1)
    valid = target != ignore_index
    pixel_accuracy = ((prediction == target) & valid).sum() / valid.sum().clamp_min(1)
    dice_values = []
    for class_index in range(1, logits.shape[1]):
        predicted = (prediction == class_index) & valid
        truth = (target == class_index) & valid
        axes = tuple(range(1, predicted.ndim))
        numerator = 2.0 * (predicted & truth).sum(dim=axes).float()
        denominator = predicted.sum(dim=axes) + truth.sum(dim=axes)
        present = denominator > 0
        if present.any():
            dice_values.append(numerator[present] / denominator[present])
    foreground_dice = (
        torch.cat(dice_values).mean()
        if dice_values
        else torch.zeros((), device=logits.device)
    )
    return {"pixel_accuracy": pixel_accuracy, "foreground_dice": foreground_dice}


class DownstreamTrainingMonitor:
    """Record standardized diagnostics for real and synthetic segmentation runs."""

    def __init__(
        self,
        run_dir: Path,
        config: Any,
        model: nn.Module,
        model_name: str,
        phase: str,
    ) -> None:
        self.scope = f"{model_name}/{phase}"
        self.scalar_every = logging_interval(config, "scalar_every", 10)
        self.image_every = logging_interval(config, "downstream_image_every", 200)
        self.histogram_every = logging_interval(config, "histogram_every", 500)
        self.monitor = ExperimentMonitor(run_dir, config)
        self.monitor.describe_model(model_name, model)

    def training_step(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        images: torch.Tensor,
        targets: torch.Tensor,
        outputs: Any,
        gradient_norm: torch.Tensor | float,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        logits = primary_logits(outputs)
        if step == 1 or step % self.scalar_every == 0:
            metrics = {
                "loss": loss,
                "gradient_norm": gradient_norm,
                **batch_segmentation_metrics(logits, targets),
                **(dict(extra) if extra else {}),
            }
            self.monitor.log_metrics(f"{self.scope}/train", metrics, step)
            self.monitor.log_optimizer(optimizer, step)
        if step == 1 or step % self.image_every == 0:
            self.monitor.log_segmentation(
                f"{self.scope}/train_examples", images, targets, logits, step
            )
        if step == 1 or step % self.histogram_every == 0:
            parameters = {
                name.replace(".", "/"): value
                for name, value in model.named_parameters()
                if value.requires_grad and value.numel() > 0
            }
            self.monitor.log_histograms(f"{self.scope}/parameters", parameters, step)

    def validation(self, epoch: int, step: int, value: float, metric: str) -> None:
        self.monitor.log_metrics(
            f"{self.scope}/validation", {metric: value, "epoch": epoch}, step
        )
        self.monitor.flush()

    def close(self) -> None:
        self.monitor.close()


class DiceCrossEntropyLoss(nn.Module):
    """Memory-efficient foreground Dice plus cross entropy.

    This follows nnU-Net v2's DC_and_CE_loss convention: equal Dice/CE weights,
    softmax Dice, foreground classes only, and a smoothing constant of 1e-5.
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = False,
        ignore_index: int = -100,
        smooth: float = 1e-5,
        batch_dice: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.batch_dice = batch_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target_without_channel(target, logits)
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0).clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(safe_target, self.num_classes)
        order = (0, target.ndim, *range(1, target.ndim))
        one_hot = one_hot.permute(order).to(logits.dtype)
        one_hot = one_hot * valid.unsqueeze(1)
        probability = logits.softmax(1) * valid.unsqueeze(1)
        axes = (0, *range(2, logits.ndim)) if self.batch_dice else tuple(range(2, logits.ndim))
        intersection = (probability * one_hot).sum(dim=axes)
        denominator = probability.sum(dim=axes) + one_hot.sum(dim=axes)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background and dice.shape[-1] > 1:
            dice = dice[..., 1:]
        dice_loss = 1.0 - dice.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


class DeepSupervisionLoss(nn.Module):
    """Apply exponentially decaying nnU-Net deep-supervision weights."""

    def __init__(self, base_loss: nn.Module, output_count: int) -> None:
        super().__init__()
        weights = torch.tensor([1.0 / (2**index) for index in range(output_count)])
        if output_count > 1:
            weights[-1] = 0.0
        self.register_buffer("weights", weights / weights.sum())
        self.base_loss = base_loss

    def forward(self, outputs: torch.Tensor | Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return self.base_loss(outputs, target)
        loss = target.new_zeros((), dtype=torch.float32)
        for weight, output in zip(self.weights, outputs):
            resized_target = target
            if tuple(output.shape[2:]) != tuple(target.shape[-len(output.shape[2:]) :]):
                source = target.unsqueeze(1).float() if target.ndim == output.ndim - 1 else target.float()
                resized_target = F.interpolate(source, size=output.shape[2:], mode="nearest")
                if target.ndim == output.ndim - 1:
                    resized_target = resized_target[:, 0]
            loss = loss + weight * self.base_loss(output, resized_target)
        return loss


class PolynomialLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Iteration-level polynomial decay used by nnU-Net and Swin-Unet."""

    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, power: float = 0.9):
        self.total_steps = max(int(total_steps), 1)
        self.power = power
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        factor = max(0.0, 1.0 - self.last_epoch / self.total_steps) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine decay used by SwinUNETR and SAMed."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        minimum_factor: float = 0.0,
    ) -> None:
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.minimum_factor = minimum_factor
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        step = max(self.last_epoch, 0)
        if self.warmup_steps and step < self.warmup_steps:
            factor = float(step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            factor = self.minimum_factor + (1.0 - self.minimum_factor) * cosine
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupPolynomialScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by iteration-level polynomial decay."""

    def __init__(
        self, optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, power: float = 0.9
    ) -> None:
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.power = power
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        step = max(self.last_epoch, 0)
        if self.warmup_steps and step < self.warmup_steps:
            factor = float(step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            factor = max(0.0, 1.0 - progress) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


def build_loader(
    dataset_config: Any,
    split: str,
    spatial_dims: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Build one canonical medical DataLoader."""
    return DataLoader(
        build_dataset(dataset_config, split, spatial_dims),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        collate_fn=collate_medical_batch,
    )


def paired_cutmix(
    real: NAMBatch, synthetic: NAMBatch, probability: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one symmetric rectangular CutMix operation to a paired mini-batch."""
    if random.random() >= probability:
        return real.image, real.target, synthetic.image, synthetic.target
    real_image, real_target = real.image.clone(), real.target.clone()
    synthetic_image, synthetic_target = synthetic.image.clone(), synthetic.target.clone()
    starts, ends = [], []
    for size in real_image.shape[2:]:
        length = random.randint(max(1, size // 4), max(1, 3 * size // 4))
        start = random.randint(0, max(size - length, 0))
        starts.append(start)
        ends.append(start + length)
    image_slice = (slice(None), slice(None), *(slice(a, b) for a, b in zip(starts, ends)))
    target_prefix = (slice(None), slice(None)) if real_target.ndim == real_image.ndim else (slice(None),)
    target_slice = (*target_prefix, *(slice(a, b) for a, b in zip(starts, ends)))
    real_image[image_slice], synthetic_image[image_slice] = (
        synthetic.image[image_slice],
        real.image[image_slice],
    )
    real_target[target_slice], synthetic_target[target_slice] = (
        synthetic.target[target_slice],
        real.target[target_slice],
    )
    return real_image, real_target, synthetic_image, synthetic_target


@torch.no_grad()
def mean_foreground_dice(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute mean sample/class foreground Dice for checkpoint selection."""
    model.eval()
    values: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.image)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        target = target_without_channel(batch.target, logits)
        prediction = logits.argmax(1)
        for class_index in range(1, logits.shape[1]):
            pred = prediction == class_index
            truth = target == class_index
            axes = tuple(range(1, pred.ndim))
            numerator = 2.0 * (pred & truth).sum(dim=axes).float()
            denominator = pred.sum(dim=axes) + truth.sum(dim=axes)
            valid = denominator > 0
            if valid.any():
                values.append(numerator[valid] / denominator[valid])
    model.train()
    return torch.cat(values).mean().item() if values else 0.0


def create_run_directory(config: Any, model_name: str, phase: str) -> Path:
    """Create a run below the phase-specific, release-visible checkpoint root."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    normalized_phase = "real" if phase == "real" else "syn"
    root_key = "real_checkpoint_root" if normalized_phase == "real" else "syn_checkpoint_root"
    configured_root = getattr(config.downstream, root_key, None)
    dataset_name = Path(str(config.dataset.root)).name
    generator_name = str(getattr(getattr(config, "diffusion", {}), "name", "generator"))
    if configured_root:
        configured_root = str(configured_root).format(
            dataset=dataset_name,
            generator=generator_name,
            model=model_name,
            phase=normalized_phase,
        )
    root = (
        Path(configured_root)
        if configured_root
        else Path(__file__).resolve().parent
        / f"{normalized_phase}_checkpoint"
        / dataset_name
        / (generator_name if normalized_phase == "syn" else "")
        / model_name
    )
    directory = root / f"{config.experiment_name}-{phase}-{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
    return directory


def save_checkpoint(
    directory: Path,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    validation_dice: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a stable checkpoint and a 20-epoch archival snapshot."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "validation_dice": validation_dice,
        "extra": extra or {},
    }
    torch.save(state, directory / name)
    torch.save(state, directory.parent / name)
    if name == "latest.pt" and epoch % 20 == 0:
        stage_name = f"epoch_{epoch:04d}.pt"
        torch.save(state, directory / stage_name)
        torch.save(state, directory.parent / stage_name)


def should_validate(epoch: int, total_epochs: int, validation_every: int) -> bool:
    """Validate at the configured cadence, final epoch, and every 20 epochs."""
    return epoch % validation_every == 0 or epoch % 20 == 0 or epoch == total_epochs


def load_model_checkpoint(model: nn.Module, path: str | Path, strict: bool = True) -> dict[str, Any]:
    """Load a real-data baseline or model-only state dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "network_weights", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def cycle(loader: DataLoader) -> Iterable[NAMBatch]:
    """Yield batches indefinitely without caching dataset contents."""
    while True:
        yield from loader
