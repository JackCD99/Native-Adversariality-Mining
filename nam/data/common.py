"""Shared manifest, medical I/O, augmentation, and prompt utilities.

Dataset-specific label semantics stay in each dataset package. This module only
implements storage formats and transformations shared across all benchmarks.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from nam.data.base import collate_medical_batch


@dataclass(frozen=True)
class ManifestItem:
    """One portable list entry resolved relative to a dataset directory."""

    sample_id: str
    image: str
    target: str | None = None
    prompt: str | None = None
    class_id: int | None = None
    metadata: Mapping[str, Any] | None = None


def config_value(config: Any, name: str, default: Any) -> Any:
    """Read an attribute or mapping field from publication YAML configuration."""
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def package_root(module_file: str, config: Any) -> Path:
    """Use ``config.root`` when provided, otherwise the dataset package itself."""
    value = config_value(config, "root", "")
    return Path(value).expanduser() if value else Path(module_file).resolve().parent


def read_manifest(path: Path) -> list[ManifestItem]:
    """Read whitespace lists or JSONL manifests without machine-specific paths.

    Plain rows support ``id image target [prompt...]``. JSON rows support the
    explicit fields in :class:`ManifestItem`. Empty files are intentional in a
    source release and produce an actionable error at dataset construction.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split manifest was not found: {path}")
    items: list[ManifestItem] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            row = json.loads(line)
            image = row.get("image", row.get("file_name"))
            if not image:
                raise ValueError(f"Manifest row {number} has no image/file_name: {path}")
            class_value = row.get("class_id", row.get("label", row.get("binary_class_value")))
            if isinstance(class_value, str) and not class_value.lstrip("+-").isdigit():
                names = {"normal": 0, "benign": 0, "pneumonia": 1, "malignant": 1}
                class_value = names.get(class_value.lower())
            items.append(ManifestItem(
                str(row.get("sample_id", row.get("id", Path(str(image)).stem))),
                str(image), None if row.get("target") is None else str(row["target"]),
                row.get("prompt", row.get("text")),
                None if class_value is None else int(class_value), row,
            ))
            continue
        parts = line.split(maxsplit=3)
        if len(parts) == 1:
            items.append(ManifestItem(parts[0], parts[0]))
        elif len(parts) >= 2:
            target = parts[2] if len(parts) >= 3 and parts[2] != "-" else None
            class_id = int(target) if target is not None and target.lstrip("+-").isdigit() else None
            items.append(ManifestItem(parts[0], parts[1], None if class_id is not None else target,
                                      parts[3] if len(parts) == 4 else None, class_id))
    return items


def _load_h5(path: Path, key: str) -> np.ndarray:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("HDF5 datasets require `pip install h5py`.") from error
    with h5py.File(path, "r") as stream:
        candidates = (key, "label" if key == "target" else key, "seg" if key == "target" else key)
        for candidate in candidates:
            if candidate in stream:
                return np.asarray(stream[candidate])
    raise KeyError(f"None of {candidates} exists in {path}")


def load_array(path: Path, key: str = "image") -> np.ndarray:
    """Load PNG/JPEG, NPY/NPZ, PT, HDF5, or NIfTI arrays."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {path}")
    lower = path.name.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        with Image.open(path) as value:
            return np.asarray(value).copy()
    if lower.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    if lower.endswith(".npz"):
        with np.load(path, allow_pickle=False) as value:
            for candidate in (key, "label" if key == "target" else key, "arr_0"):
                if candidate in value:
                    return np.asarray(value[candidate])
        raise KeyError(f"No '{key}' array exists in {path}")
    if lower.endswith((".h5", ".hdf5")):
        return _load_h5(path, key)
    if lower.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as error:
            raise ImportError("NIfTI datasets require `pip install nibabel`.") from error
        return np.asarray(nib.load(str(path)).get_fdata())
    if lower.endswith((".pt", ".pth")):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, Mapping):
            value = value.get(key, value.get("label" if key == "target" else key))
        if value is None:
            raise KeyError(f"No '{key}' tensor exists in {path}")
        return torch.as_tensor(value).cpu().numpy()
    raise ValueError(f"Unsupported medical data format: {path}")


def _channels_first_image(array: np.ndarray, spatial_dims: int) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(array)).float()
    if value.ndim == spatial_dims:
        value = value.unsqueeze(0)
    elif value.ndim == spatial_dims + 1 and value.shape[-1] in (1, 3, 4):
        value = value.movedim(-1, 0)[:3]
    if value.ndim != spatial_dims + 1:
        raise ValueError(f"Expected a {spatial_dims}D image, got shape {tuple(value.shape)}")
    if value.max() > 1.0:
        value = value / 255.0 if value.max() <= 255 else value
    return value


def _label_tensor(array: np.ndarray, spatial_dims: int) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(array))
    while value.ndim > spatial_dims and value.shape[0] == 1:
        value = value[0]
    if value.ndim == spatial_dims + 1 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim == spatial_dims + 1 and value.shape[-1] in (3, 4):
        if not torch.equal(value[..., 0], value[..., 1]):
            raise ValueError("RGB label masks require an explicit dataset palette conversion.")
        value = value[..., 0]
    if value.ndim != spatial_dims:
        raise ValueError(f"Expected a {spatial_dims}D label, got shape {tuple(value.shape)}")
    return value.long()


def robust_normalize(image: torch.Tensor, modality: str) -> torch.Tensor:
    """Apply robust normalization using only statistics from the current sample."""
    image = image.float()
    if modality.upper() in {"RGB", "DERMOSCOPY"}:
        return image.clamp(0.0, 1.0)
    finite = image[torch.isfinite(image)]
    if finite.numel() == 0:
        raise ValueError("Medical image contains no finite voxel.")
    low, high = torch.quantile(finite, torch.tensor([0.005, 0.995], device=finite.device, dtype=finite.dtype))
    image = image.clamp(low, high)
    mean, std = image.mean(), image.std().clamp_min(1e-6)
    return ((image - mean) / std).clamp(-5.0, 5.0).add(5.0).div(10.0)


def resize_pair(image: torch.Tensor, target: torch.Tensor, size: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize intensity and discrete-label tensors with appropriate interpolation."""
    dimensionality = len(size)
    mode = "bilinear" if dimensionality == 2 else "trilinear"
    image = F.interpolate(image[None], tuple(size), mode=mode, align_corners=False)[0]
    target = F.interpolate(target[None, None].float(), tuple(size), mode="nearest")[0, 0].long()
    return image, target


class JointAugmentation:
    """Synchronized geometric/intensity augmentation for 2D or 3D pairs."""

    def __init__(self, spatial_dims: int, flip_probability: float = 0.5, scale_range: Sequence[float] = (0.9, 1.1), intensity_probability: float = 0.2) -> None:
        self.spatial_dims = spatial_dims
        self.flip_probability = float(flip_probability)
        self.scale_range = tuple(float(value) for value in scale_range)
        self.intensity_probability = float(intensity_probability)

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for axis in range(-self.spatial_dims, 0):
            if random.random() < self.flip_probability:
                image, target = image.flip(axis), target.flip(axis)
        if self.spatial_dims == 2 and random.random() < 0.5:
            turns = random.randrange(4)
            image, target = torch.rot90(image, turns, (-2, -1)), torch.rot90(target, turns, (-2, -1))
        if random.random() < 0.5 and self.scale_range[0] != self.scale_range[1]:
            original = target.shape
            factor = random.uniform(*self.scale_range)
            scaled = tuple(max(2, round(value * factor)) for value in original)
            image, target = resize_pair(image, target, scaled)
            image, target = center_crop_or_pad(image, target, original)
        if random.random() < self.intensity_probability:
            image = image * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
        return image.clamp(0.0, 1.0), target


def center_crop_or_pad(image: torch.Tensor, target: torch.Tensor, size: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Center crop then symmetric-pad a pair to an exact 2D/3D spatial size."""
    slices: list[slice] = []
    for current, wanted in zip(target.shape, size):
        start = max((current - wanted) // 2, 0)
        slices.append(slice(start, start + min(current, wanted)))
    image = image[(slice(None), *slices)]
    target = target[tuple(slices)]
    padding: list[int] = []
    for current, wanted in reversed(list(zip(target.shape, size))):
        total = max(wanted - current, 0)
        padding.extend((total // 2, total - total // 2))
    return F.pad(image, padding), F.pad(target, padding)


class ManifestSegmentationDataset(Dataset):
    """Portable segmentation dataset returning the canonical NAM contract."""

    def __init__(
        self,
        root: Path,
        split: str,
        spatial_dims: int,
        modality: str,
        prompt: str | Callable[[torch.Tensor, ManifestItem], str],
        image_size: Sequence[int],
        train_augmentation: bool = True,
        label_map: Mapping[int, int] | None = None,
        manifest_name: str | None = None,
    ) -> None:
        self.root, self.split, self.spatial_dims = root, split, spatial_dims
        self.modality, self.prompt, self.image_size = modality, prompt, tuple(image_size)
        self.items = read_manifest(root / (manifest_name or f"{split}.list"))
        if not self.items:
            raise RuntimeError(f"{root / (manifest_name or f'{split}.list')} is empty. Follow README.md to prepare the dataset.")
        self.augmentation = JointAugmentation(spatial_dims) if split == "train" and train_augmentation else None
        self.label_map = dict(label_map or {})

    def __len__(self) -> int:
        return len(self.items)

    def _paths(self, item: ManifestItem) -> tuple[Path, Path]:
        if not item.target:
            raise ValueError(f"Segmentation sample '{item.sample_id}' has no target path.")
        return self.root / item.image, self.root / item.target

    def _read_pair(self, image_path: Path, target_path: Path) -> tuple[np.ndarray, np.ndarray]:
        return load_array(image_path, "image"), load_array(target_path, "target")

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image_path, target_path = self._paths(item)
        raw_image, raw_target = self._read_pair(image_path, target_path)
        image = _channels_first_image(raw_image, self.spatial_dims)
        target = _label_tensor(raw_target, self.spatial_dims)
        if self.label_map:
            mapped = torch.zeros_like(target)
            for source, destination in self.label_map.items():
                mapped[target == source] = destination
            target = mapped
        image = robust_normalize(image, self.modality)
        image, target = resize_pair(image, target, self.image_size)
        if self.augmentation is not None:
            image, target = self.augmentation(image, target)
        prompt = item.prompt or (self.prompt(target, item) if callable(self.prompt) else self.prompt)
        condition = {
            "mask": target.clone(), "segmentation": target.clone(), "prompt": prompt,
            "txt": prompt, "image_prompt": prompt, "mask_prompt": prompt.replace("image", "segmentation mask"),
            "modality": self.modality, "target_name": prompt,
        }
        return {
            "image": image.contiguous(), "target": target.contiguous(), "condition": condition,
            "sample_id": item.sample_id,
            "metadata": {"image_path": str(image_path), "target_path": str(target_path), "split": self.split, **dict(item.metadata or {})},
        }


class ManifestClassificationDataset(Dataset):
    """Class-aware image/prompt dataset adapted to the canonical NAM batch."""

    def __init__(self, root: Path, split: str, prompts: Mapping[int, Sequence[str]], modality: str, image_size: Sequence[int] = (224, 224), train_augmentation: bool = True, caption_dropout_probability: float = 0.0) -> None:
        self.root, self.split, self.prompts, self.modality = root, split, prompts, modality
        self.image_size = tuple(image_size)
        self.items = read_manifest(root / f"{split}.list")
        if not self.items:
            raise RuntimeError(f"{root / f'{split}.list'} is empty. Follow README.md to prepare the dataset.")
        self.train_augmentation = bool(train_augmentation and split == "train")
        self.caption_dropout_probability = float(caption_dropout_probability)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        if item.class_id is None:
            raise ValueError(f"Classification sample '{item.sample_id}' has no class_id/label.")
        image_path = self.root / item.image
        image = _channels_first_image(load_array(image_path), 2)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = F.interpolate(image[None], self.image_size, mode="bicubic", align_corners=False)[0]
        if self.train_augmentation and random.random() < 0.5:
            image = image.flip(-1)
        prompt = item.prompt or random.choice(tuple(self.prompts[item.class_id]))
        if self.train_augmentation and random.random() < self.caption_dropout_probability:
            prompt = ""
        target = torch.tensor(item.class_id, dtype=torch.long)
        return {
            "image": robust_normalize(image, self.modality), "target": target,
            "condition": {"prompt": prompt, "txt": prompt, "class_id": target.clone(), "modality": self.modality, "target_name": prompt},
            "sample_id": item.sample_id,
            "metadata": {"image_path": str(image_path), "split": self.split, **dict(item.metadata or {})},
        }


class GeneratedPairDataset(Dataset):
    """Read tensor pairs and manifests emitted by NAM sampling pipelines."""

    def __init__(self, root: Path, split: str = "train", spatial_dims: int = 2) -> None:
        self.root, self.split = root, split
        self.spatial_dims = int(spatial_dims)
        split_manifest = root / f"{split}.list"
        if split_manifest.is_file() and split_manifest.stat().st_size:
            self.items = read_manifest(split_manifest)
            self.tensor_files: list[Path] = []
        else:
            tensor_root = root / "tensors" if (root / "tensors").is_dir() else root
            self.tensor_files = sorted((*tensor_root.glob("*.pt"), *tensor_root.glob("*.pth")))
            self.items = []
        if not self.items and not self.tensor_files:
            raise RuntimeError(f"No generated NAM pairs were found under {root}.")

    def __len__(self) -> int:
        return len(self.items) if self.items else len(self.tensor_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.items:
            item = self.items[index]
            image = _channels_first_image(load_array(self.root / item.image, "image"), self.spatial_dims)
            if item.class_id is not None:
                target = torch.tensor(item.class_id, dtype=torch.long)
            elif item.target:
                target = _label_tensor(load_array(self.root / item.target, "target"), self.spatial_dims)
            else:
                raise ValueError(
                    f"Generated sample '{item.sample_id}' has neither a target path nor class_id."
                )
            sample_id, metadata = item.sample_id, dict(item.metadata or {})
        else:
            path = self.tensor_files[index]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            image, target = torch.as_tensor(payload["image"]), torch.as_tensor(payload["target"])
            if image.ndim >= 4 and image.shape[0] == 1:
                image = image[0]
            if target.ndim >= 3 and target.shape[0] == 1:
                target = target[0]
            image, target = image.float(), target.long()
            sample_id, metadata = path.stem, dict(payload.get("metadata", {}))
        prompt = str(metadata.get("prompt", "a synthetic medical image"))
        return {"image": image, "target": target, "condition": {"mask": target.clone(), "prompt": prompt, "txt": prompt}, "sample_id": sample_id, "metadata": metadata}


def build_generated_dataset(config: Any, split: str, spatial_dims: int) -> GeneratedPairDataset:
    """Factory for `synthetic_dataset` configuration sections."""
    return GeneratedPairDataset(Path(config_value(config, "root", "")), split, spatial_dims)


def build_nam_dataloader(dataset: Dataset, batch_size: int = 4, shuffle: bool | None = None, num_workers: int = 4, drop_last: bool = False, pin_memory: bool = True) -> DataLoader:
    """Build the canonical loader consumed by diffusion, miner, and downstream code."""
    if shuffle is None:
        shuffle = getattr(dataset, "split", "") == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory, persistent_workers=num_workers > 0, collate_fn=collate_medical_batch)
