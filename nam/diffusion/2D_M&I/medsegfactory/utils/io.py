"""MedSegFactory checkpoint and synthetic-pair writers."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json, numpy as np, torch
from PIL import Image


def save_checkpoint(unet: torch.nn.Module, path: str | Path, state: dict[str, Any]) -> Path:
    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"unet": unet.state_dict(), **state}, result)
    return result


def output_directory(root: str | Path, experiment: str, method: str) -> Path:
    result = Path(root) / f"{experiment}-{method}"
    for name in ("images", "masks", "tensors", "metadata"):
        (result / name).mkdir(parents=True, exist_ok=True)
    return result


def save_pair(
    root: Path, sample_id: str, image: torch.Tensor, target: torch.Tensor, metadata: dict[str, Any]
) -> None:
    safe = sample_id.replace("/", "_").replace("\\", "_")
    torch.save({"image": image.cpu(), "target": target.cpu()}, root / "tensors" / f"{safe}.pt")
    preview = ((image.clamp(-1, 1) + 1) * 127.5).byte()
    Image.fromarray(preview.permute(1, 2, 0).cpu().numpy()).save(root / "images" / f"{safe}.png")
    mask = target.squeeze().cpu().numpy()
    Image.fromarray((mask / max(int(mask.max()), 1) * 255).astype(np.uint8)).save(
        root / "masks" / f"{safe}.png"
    )
    (root / "metadata" / f"{safe}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
