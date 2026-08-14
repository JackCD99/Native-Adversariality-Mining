"""Canonical dataset contracts shared by all medical experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from nam.utils.imports import import_factory


@dataclass
class NAMBatch:
    """Canonical batch consumed by every NAM experiment."""

    image: torch.Tensor | None
    target: torch.Tensor
    condition: Any
    sample_id: list[str]
    metadata: Mapping[str, Any]

    def to(self, device: torch.device) -> "NAMBatch":
        image = None if self.image is None else self.image.to(device, non_blocking=True)
        target = self.target.to(device, non_blocking=True)

        def move(value: Any) -> Any:
            if torch.is_tensor(value):
                return value.to(device, non_blocking=True)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(move(item) for item in value)
            return value

        return NAMBatch(image, target, move(self.condition), self.sample_id, self.metadata)


class MedicalDataset(Dataset, ABC):
    """Base class for datasets that expose the NAM sample contract."""

    spatial_dims: int

    @abstractmethod
    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Return image, target, condition, sample_id, and optional metadata."""


def collate_medical_batch(samples: list[Mapping[str, Any]]) -> NAMBatch:
    """Collate the canonical dictionary returned by a medical dataset."""
    if not samples:
        raise ValueError("Cannot collate an empty medical batch.")
    required = {"target", "condition", "sample_id"}
    missing = required - set(samples[0])
    if missing:
        raise KeyError(f"Dataset sample is missing required fields: {sorted(missing)}")

    images = None
    if samples[0].get("image") is not None:
        images = torch.stack([sample["image"] for sample in samples])
    targets = torch.stack([sample["target"] for sample in samples])

    first_condition = samples[0]["condition"]
    if torch.is_tensor(first_condition):
        conditions: Any = torch.stack([sample["condition"] for sample in samples])
    elif isinstance(first_condition, dict):
        def collate_values(values: list[Any]) -> Any:
            first = values[0]
            if torch.is_tensor(first):
                return torch.stack(values)
            if isinstance(first, dict):
                return {key: collate_values([value[key] for value in values]) for key in first}
            return values
        conditions = {key: collate_values([sample["condition"][key] for sample in samples]) for key in first_condition}
    else:
        conditions = [sample["condition"] for sample in samples]

    return NAMBatch(
        image=images,
        target=targets,
        condition=conditions,
        sample_id=[str(sample["sample_id"]) for sample in samples],
        metadata={"items": [sample.get("metadata", {}) for sample in samples]},
    )


def build_dataset(config: Any, split: str, spatial_dims: int) -> Dataset:
    """Build a dataset through the configured factory function.

    The factory signature is ``build_dataset(config, split, spatial_dims)``. Keeping
    preprocessing outside NAM prevents dataset-specific assumptions and data leakage.
    """
    factory = import_factory(config.factory, getattr(config, "project_dir", None))
    dataset = factory(config=config, split=split, spatial_dims=spatial_dims)
    if not isinstance(dataset, Dataset):
        raise TypeError("The configured dataset factory must return torch.utils.data.Dataset.")
    return dataset
