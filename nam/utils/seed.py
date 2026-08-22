"""Random-seed and deterministic-execution helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


_STAGE_KEYS = {
    "miner": "miner_seed",
    "downstream": "downstream_seed",
    "sampling": "sampling_seed",
}


def resolve_stage_seed(config: object, stage: str) -> int:
    normalized = stage.strip().lower()
    if normalized not in _STAGE_KEYS:
        raise ValueError(f"Unknown seed stage '{stage}'; expected {sorted(_STAGE_KEYS)}.")
    runtime = getattr(config, "runtime")
    value = getattr(runtime, _STAGE_KEYS[normalized], getattr(runtime, "seed"))
    seed = int(value)
    if seed < 0:
        raise ValueError(f"runtime.{_STAGE_KEYS[normalized]} must be non-negative.")
    return seed


def sampling_output_root(root: str | Path, seed: int) -> Path:
    return Path(root) / f"seed_{int(seed)}"


def build_sampling_generators(
    device: str | torch.device, seed: int
) -> tuple[torch.Generator, torch.Generator]:
    probe = torch.Generator(device=device).manual_seed(int(seed))
    reselection_seed = (int(seed) + 1_000_003) % (2**63 - 1)
    reselection = torch.Generator(device=device).manual_seed(reselection_seed)
    return probe, reselection


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without overriding user GPU visibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
