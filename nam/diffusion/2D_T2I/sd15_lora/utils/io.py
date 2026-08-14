"""Portable storage for class-conditional synthetic images."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def prepare_output_directory(root: str | Path, experiment: str, method: str) -> Path:
    directory = Path(root) / f"{experiment}-{method}"
    for name in ("images", "tensors", "metadata"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    return directory


def save_sample(root: Path, sample_id: str, image: torch.Tensor, target: torch.Tensor,
                metadata: dict[str, Any]) -> None:
    safe = re.sub(r"[^\w.-]+", "_", sample_id).strip("._") or "sample"
    image = image.detach().cpu().float().clamp(0, 1)
    target = target.detach().cpu().long()
    payload = {"image": image, "target": target, "metadata": metadata}
    torch.save(payload, root / "tensors" / f"{safe}.pt")
    Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(
        root / "images" / f"{safe}.png"
    )
    (root / "metadata" / f"{safe}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
