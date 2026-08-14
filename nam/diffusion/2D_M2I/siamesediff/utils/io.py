"""Lossless and viewable output writers specialized for 2D M2I synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def prepare_output_directory(root: str | Path, experiment: str, method: str) -> Path:
    """Create tensor, PNG, mask, and metadata subdirectories."""
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
    """Save exact tensors together with publication-friendly PNG previews."""
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    torch.save({"image": image.cpu(), "target": target.cpu()}, directory / "tensors" / f"{safe_id}.pt")
    image_uint8 = ((image.detach().float().clamp(-1, 1) + 1.0) * 127.5).byte()
    if image_uint8.shape[0] == 1:
        image_uint8 = image_uint8.repeat(3, 1, 1)
    Image.fromarray(image_uint8.permute(1, 2, 0).cpu().numpy()).save(directory / "images" / f"{safe_id}.png")
    mask = target.detach().cpu()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask = mask.long().numpy()
    maximum = max(int(mask.max()), 1)
    Image.fromarray((mask.astype(np.float32) / maximum * 255).astype(np.uint8)).save(
        directory / "masks" / f"{safe_id}.png"
    )
    with (directory / "metadata" / f"{safe_id}.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
