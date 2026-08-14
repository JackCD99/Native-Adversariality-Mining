"""Canonical classification loader used by LoRA training and sampling."""

from typing import Any

from nam.downstream.training import build_loader as _build_loader


def build_loader(config: Any, split: str, batch_size: int, num_workers: int, shuffle: bool):
    return _build_loader(config, split, 2, batch_size, num_workers, shuffle)
