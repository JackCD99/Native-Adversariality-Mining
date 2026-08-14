"""Checkpoint and volumetric output writers specialized for MAISI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def save_checkpoint(model: torch.nn.Module, destination: str | Path, state: dict[str, Any]) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    torch.save({**state, "controlnet_state_dict": raw.state_dict()}, path)
    return path


def prepare_output_directory(root: str | Path, experiment: str, method: str) -> Path:
    directory = Path(root) / f"{experiment}-{method}"
    for name in ("volumes", "masks", "tensors", "previews", "metadata"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    return directory


def save_volume_pair(directory: Path, sample_id: str, image: torch.Tensor, target: torch.Tensor, metadata: dict[str, Any]) -> None:
    """Save exact tensors, center-slice PNGs, NIfTI files, and provenance."""
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    torch.save({"image": image.cpu(), "target": target.cpu()}, directory / "tensors" / f"{safe_id}.pt")
    preview = image.detach().float().squeeze().cpu()
    preview = preview[..., preview.shape[-1] // 2]
    Image.fromarray(((preview.clamp(-1, 1) + 1.0) * 127.5).byte().numpy()).save(directory / "previews" / f"{safe_id}.png")
    mask = target.detach().long().squeeze().cpu()
    Image.fromarray(((mask[..., mask.shape[-1] // 2] > 0).byte() * 255).numpy()).save(directory / "masks" / f"{safe_id}.png")
    try:
        import nibabel as nib
        nib.save(nib.Nifti1Image(image.squeeze().float().cpu().numpy(), np.eye(4)), directory / "volumes" / f"{safe_id}.nii.gz")
        nib.save(nib.Nifti1Image(mask.numpy().astype(np.int16), np.eye(4)), directory / "masks" / f"{safe_id}.nii.gz")
    except ImportError:
        metadata["nifti_warning"] = "Install nibabel to export NIfTI files; exact tensors were saved."
    with (directory / "metadata" / f"{safe_id}.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
