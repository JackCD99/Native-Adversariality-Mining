"""Checkpoint and NIfTI/portable-tensor writers for VolDiT volumes."""

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
    torch.save({**state, "model": raw.state_dict()}, path)
    return path


def prepare_output_directory(root: str | Path, experiment: str, method: str) -> Path:
    directory = Path(root) / f"{experiment}-{method}"
    for name in ("volumes", "masks", "tensors", "previews", "metadata"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    return directory


def _preview(volume: torch.Tensor) -> np.ndarray:
    data = volume.detach().float().squeeze().cpu()
    image = data[..., data.shape[-1] // 2] if data.ndim == 3 else data
    image = ((image.clamp(-1, 1) + 1.0) * 127.5).byte().numpy()
    return image


def save_volume_pair(
    directory: Path, sample_id: str, image: torch.Tensor, target: torch.Tensor,
    metadata: dict[str, Any], affine: np.ndarray | None = None,
) -> None:
    """Save exact tensors, center-slice previews, and optional NIfTI files."""
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    torch.save({"image": image.cpu(), "target": target.cpu()}, directory / "tensors" / f"{safe_id}.pt")
    Image.fromarray(_preview(image)).save(directory / "previews" / f"{safe_id}.png")
    mask = target.detach().long().squeeze().cpu()
    mask_slice = mask[..., mask.shape[-1] // 2] if mask.ndim == 3 else mask
    Image.fromarray((mask_slice.numpy() > 0).astype(np.uint8) * 255).save(directory / "masks" / f"{safe_id}.png")
    try:
        import nibabel as nib
        matrix = np.eye(4) if affine is None else affine
        nib.save(nib.Nifti1Image(image.squeeze().float().cpu().numpy(), matrix), directory / "volumes" / f"{safe_id}.nii.gz")
        nib.save(nib.Nifti1Image(mask.numpy().astype(np.int16), matrix), directory / "masks" / f"{safe_id}.nii.gz")
    except ImportError:
        metadata["nifti_warning"] = "Install nibabel to export NIfTI files; exact tensors were saved."
    with (directory / "metadata" / f"{safe_id}.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
