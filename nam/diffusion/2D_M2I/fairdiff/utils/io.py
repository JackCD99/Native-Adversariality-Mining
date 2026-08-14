"""Checkpoint and synthetic-pair writers specialized for FairDiff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def save_checkpoint(model: torch.nn.Module, destination: str | Path, state: dict[str, Any]) -> Path:
    """Save official model weights together with optimizer and provenance state."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({**state, "state_dict": raw_model.state_dict()}, path)
    return path


def prepare_output_directory(root: str | Path, experiment: str, method: str) -> Path:
    directory = Path(root) / f"{experiment}-{method}"
    for name in ("images", "masks", "tensors", "metadata"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    return directory


def save_pair(
    directory: Path,
    sample_id: str,
    image: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    """Save exact image-label tensors and portable PNG previews."""
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    torch.save({"image": image.cpu(), "target": target.cpu()}, directory / "tensors" / f"{safe_id}.pt")
    preview = ((image.detach().float().clamp(-1, 1) + 1.0) * 127.5).byte()
    if preview.shape[0] == 1:
        preview = preview.repeat(3, 1, 1)
    Image.fromarray(preview.permute(1, 2, 0).cpu().numpy()).save(directory / "images" / f"{safe_id}.png")
    mask = target.detach().cpu().squeeze().long().numpy()
    maximum = max(int(mask.max()), 1)
    Image.fromarray((mask.astype(np.float32) / maximum * 255).astype(np.uint8)).save(
        directory / "masks" / f"{safe_id}.png"
    )
    with (directory / "metadata" / f"{safe_id}.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
