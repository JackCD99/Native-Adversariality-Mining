"""TensorBoard, JSONL, and image diagnostics shared by training pipelines."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        value = value.detach().float().mean().cpu().item()
    return float(value)


def _image_plane(tensor: torch.Tensor) -> torch.Tensor:
    """Convert BCHW/BCHWD tensors into a representative BCHW plane."""
    tensor = tensor.detach().float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim == 5:
        tensor = tensor[:, :, tensor.shape[2] // 2]
    if tensor.ndim != 4:
        raise ValueError(f"Expected a batched 2D or 3D tensor, received shape {tuple(tensor.shape)}.")
    return tensor


def _normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _image_plane(tensor)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    elif tensor.shape[1] >= 3:
        tensor = tensor[:, :3]
    else:
        tensor = torch.cat((tensor, tensor[:, :1]), dim=1)
    minimum = tensor.amin(dim=(1, 2, 3), keepdim=True)
    maximum = tensor.amax(dim=(1, 2, 3), keepdim=True)
    return ((tensor - minimum) / (maximum - minimum).clamp_min(1e-6)).clamp(0, 1)


def _label_plane(target: torch.Tensor) -> torch.Tensor:
    target = target.detach()
    if target.ndim == 5:
        target = target[:, :, target.shape[2] // 2]
    elif target.ndim == 4 and target.shape[1] != 1:
        # A 4D target can be BxDxHxW or BxCxHxW. Integer tensors are label
        # volumes; floating tensors with a small channel axis are one-hot maps.
        if target.is_floating_point() and target.shape[1] <= 32:
            target = target.argmax(1, keepdim=True)
        else:
            target = target[:, target.shape[1] // 2].unsqueeze(1)
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4:
        raise ValueError(f"Expected a batched target map, received shape {tuple(target.shape)}.")
    if target.shape[1] > 1:
        target = target.argmax(1, keepdim=True)
    return target[:, :1].long()


def _palette(target: torch.Tensor) -> torch.Tensor:
    labels = _label_plane(target).clamp_min(0)
    red = ((labels * 37) % 255).float() / 255.0
    green = ((labels * 17 + 83) % 255).float() / 255.0
    blue = ((labels * 29 + 151) % 255).float() / 255.0
    colors = torch.cat((red, green, blue), dim=1)
    return torch.where(labels.eq(0).expand_as(colors), torch.zeros_like(colors), colors)


def _prediction(logits: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if isinstance(logits, dict):
        logits = logits.get("out", logits.get("logits"))
    if logits.ndim == 5:
        logits = logits[:, :, logits.shape[2] // 2]
    if logits.shape[1] == 1:
        return (logits.sigmoid() >= 0.5).long()
    return logits.argmax(1, keepdim=True)


def _heatmap(values: torch.Tensor) -> torch.Tensor:
    values = _image_plane(values)
    values = values.abs().mean(1, keepdim=True)
    maximum = values.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    values = (values / maximum).clamp(0, 1)
    return torch.cat((values, 0.25 * (1.0 - (2.0 * values - 1.0).abs()), 1.0 - values), 1)


def _grid(columns: list[torch.Tensor], max_items: int) -> torch.Tensor:
    count = min(max_items, *(column.shape[0] for column in columns))
    target_size = columns[0].shape[-2:]
    resized = [
        F.interpolate(column[:count].float(), size=target_size, mode="bilinear", align_corners=False)
        if column.shape[-2:] != target_size
        else column[:count].float()
        for column in columns
    ]
    rows = [torch.cat([column[index] for column in resized], dim=2) for index in range(count)]
    return torch.cat(rows, dim=1).clamp(0, 1).cpu()


def _save_png(tensor: torch.Tensor, destination: Path) -> None:
    array = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).round().astype("uint8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(destination)


def logging_interval(config: Any, name: str, default: int) -> int:
    """Read an optional interval from ``logging`` without requiring the section."""
    section = getattr(config, "logging", {})
    return max(int(getattr(section, name, default)), 1)


class ExperimentMonitor:
    """Persist scalar, distribution, and visual diagnostics for one run."""

    def __init__(
        self,
        run_dir: str | Path,
        config: Any,
        enabled: bool = True,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.enabled = enabled
        self.writer = writer
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.preview_dir = self.run_dir / "visualizations"
        if not enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        if self.writer is None:
            self.writer = SummaryWriter(self.run_dir / "tensorboard")
        config_text = json.dumps(config, indent=2, default=_json_default)
        (self.run_dir / "config.json").write_text(config_text, encoding="utf-8")
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
        (self.run_dir / "environment.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        self.writer.add_text("run/config", f"```json\n{config_text}\n```", 0)
        self.writer.add_text("run/environment", json.dumps(metadata, indent=2), 0)

    def describe_model(self, name: str, model: nn.Module, step: int = 0) -> None:
        if not self.enabled:
            return
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        self.writer.add_text(
            f"models/{name}",
            f"total_parameters: {total:,}  \ntrainable_parameters: {trainable:,}",
            step,
        )

    def log_metrics(self, scope: str, metrics: Mapping[str, Any], step: int) -> None:
        if not self.enabled:
            return
        values = {name: _scalar(value) for name, value in metrics.items()}
        for name, value in values.items():
            self.writer.add_scalar(f"{scope}/{name}", value, step)
        record = {"step": int(step), "scope": scope, **values}
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def log_histograms(self, scope: str, tensors: Mapping[str, torch.Tensor], step: int) -> None:
        if not self.enabled:
            return
        for name, value in tensors.items():
            if value is None or not torch.is_tensor(value) or value.numel() == 0:
                continue
            flattened = value.detach().float().flatten()
            if flattened.numel() > 262_144:
                stride = max(flattened.numel() // 262_144, 1)
                flattened = flattened[::stride]
            if torch.isfinite(flattened).any():
                self.writer.add_histogram(f"{scope}/{name}", flattened.cpu(), step)

    def log_optimizer(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        if not self.enabled:
            return
        for index, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"optimizer/group_{index}_learning_rate", group["lr"], step)
            if "weight_decay" in group:
                self.writer.add_scalar(
                    f"optimizer/group_{index}_weight_decay", group["weight_decay"], step
                )

    def log_segmentation(
        self,
        tag: str,
        images: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
        step: int,
        max_items: int = 4,
    ) -> None:
        if not self.enabled:
            return
        prediction = _prediction(logits)
        truth = _label_plane(targets)
        error = prediction.ne(truth).float()
        grid = _grid(
            [_normalize_image(images), _palette(truth), _palette(prediction), _heatmap(error)],
            max_items,
        )
        self.writer.add_image(tag, grid, step)
        _save_png(grid, self.preview_dir / f"{tag.replace('/', '_')}_{step:07d}.png")

    def log_nam_comparison(
        self,
        step: int,
        base_images: torch.Tensor,
        nam_images: torch.Tensor,
        targets: torch.Tensor,
        base_logits: torch.Tensor,
        nam_logits: torch.Tensor,
        tensors: Mapping[str, torch.Tensor],
        max_items: int = 4,
    ) -> None:
        if not self.enabled:
            return
        difference = _heatmap(_image_plane(nam_images) - _image_plane(base_images))
        if targets.ndim == 1:
            grid = _grid(
                [_normalize_image(base_images), _normalize_image(nam_images), difference],
                max_items,
            )
            self.writer.add_text(
                "nam/classification_predictions",
                "  \n".join(
                    f"target={int(targets[index])}, base={int(base_logits[index].argmax())}, "
                    f"nam={int(nam_logits[index].argmax())}"
                    for index in range(min(max_items, targets.shape[0]))
                ),
                step,
            )
            self.writer.add_image("nam/base_vs_selected", grid, step)
            _save_png(grid, self.preview_dir / f"nam_comparison_{step:07d}.png")
            self.log_histograms("nam/distributions", tensors, step)
            return
        base_prediction = _prediction(base_logits)
        nam_prediction = _prediction(nam_logits)
        truth = _label_plane(targets)
        grid = _grid(
            [
                _palette(truth),
                _normalize_image(base_images),
                _palette(base_prediction),
                _normalize_image(nam_images),
                _palette(nam_prediction),
                difference,
            ],
            max_items,
        )
        self.writer.add_image("nam/base_vs_selected", grid, step)
        _save_png(grid, self.preview_dir / f"nam_comparison_{step:07d}.png")
        self.log_histograms("nam/distributions", tensors, step)

    def flush(self) -> None:
        if self.enabled:
            self.writer.flush()

    def close(self) -> None:
        if self.enabled and self.writer is not None:
            self.writer.flush()
            self.writer.close()


class SamplingMonitor:
    """Track generation progress, condition alignment, and noise reselection."""

    def __init__(self, output_dir: str | Path, config: Any, method: str, variant: str) -> None:
        self.method = method
        self.variant = variant
        self.interval = logging_interval(config, "sampling_image_every", 100)
        self.monitor = ExperimentMonitor(Path(output_dir) / "logs", config)
        self.count = 0

    def log_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        probe_noise: torch.Tensor,
        selected_noise: torch.Tensor,
        sample_ids: list[str],
    ) -> None:
        batch_size = int(images.shape[0])
        self.count += batch_size
        self.monitor.log_metrics(
            f"sampling/{self.method}/{self.variant}",
            {
                "samples_written": self.count,
                "image_mean": images.mean(),
                "image_standard_deviation": images.std(),
                "noise_l1_shift": (selected_noise - probe_noise).abs().mean(),
                "noise_l2_shift": (selected_noise - probe_noise).square().mean().sqrt(),
            },
            self.count,
        )
        if self.count == batch_size or self.count % self.interval < batch_size:
            if targets.ndim == 1:
                grid = _grid([_normalize_image(images)], max_items=4)
            else:
                target_palette = _palette(targets)
                grid = _grid([target_palette, _normalize_image(images)], max_items=4)
            tag = f"sampling/{self.method}/{self.variant}_examples"
            self.monitor.writer.add_image(tag, grid, self.count)
            _save_png(
                grid,
                self.monitor.preview_dir / f"sampling_{self.variant}_{self.count:07d}.png",
            )
            self.monitor.log_histograms(
                f"sampling/{self.method}/{self.variant}_noise",
                {
                    "probe": probe_noise,
                    "selected": selected_noise,
                    "shift": selected_noise - probe_noise,
                },
                self.count,
            )
            self.monitor.writer.add_text(
                f"sampling/{self.method}/sample_ids",
                "  \n".join(sample_ids[:8]),
                self.count,
            )

    def close(self) -> None:
        self.monitor.log_metrics(
            f"sampling/{self.method}/{self.variant}",
            {"final_sample_count": self.count},
            self.count,
        )
        self.monitor.close()


def log_training_state(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    interval: int = 100,
) -> None:
    """Log optimizer state and representative parameter/gradient distributions."""
    for index, group in enumerate(optimizer.param_groups):
        writer.add_scalar(f"optimizer/group_{index}_learning_rate", group["lr"], step)
    if step != 1 and step % max(interval, 1) != 0:
        return
    for name, parameter in model.named_parameters():
        if parameter.numel() == 0:
            continue
        safe_name = name.replace(".", "/")
        values = parameter.detach().float().flatten()
        if values.numel() > 65_536:
            values = values[:: max(values.numel() // 65_536, 1)]
        writer.add_histogram(f"parameters/{safe_name}", values.cpu(), step)
        if parameter.grad is not None:
            gradients = parameter.grad.detach().float().flatten()
            if gradients.numel() > 65_536:
                gradients = gradients[:: max(gradients.numel() // 65_536, 1)]
            writer.add_histogram(f"gradients/{safe_name}", gradients.cpu(), step)


def log_diffusion_diagnostics(
    writer: SummaryWriter,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tensors: Mapping[str, torch.Tensor],
    interval: int = 100,
) -> None:
    """Log model state and representative denoising tensors at a fixed cadence."""
    log_training_state(writer, model, optimizer, step, interval)
    if step != 1 and step % max(interval, 1) != 0:
        return
    for name, tensor in tensors.items():
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            continue
        values = tensor.detach().float().flatten()
        if values.numel() > 131_072:
            values = values[:: max(values.numel() // 131_072, 1)]
        if torch.isfinite(values).any():
            writer.add_histogram(f"diffusion/distributions/{name}", values.cpu(), step)
        if tensor.ndim not in (4, 5):
            continue
        try:
            if any(token in name.lower() for token in ("target", "mask", "label", "hint")):
                preview = _grid([_palette(tensor)], max_items=4)
            else:
                preview = _grid([_normalize_image(tensor)], max_items=4)
            writer.add_image(f"diffusion/previews/{name}", preview, step)
        except (RuntimeError, ValueError):
            continue
