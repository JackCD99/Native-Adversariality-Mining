"""SiameseDiff-specific entry point for Native Adversariality Mining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nam.config import apply_overrides, load_config
from nam.engine.miner_trainer import run_miner_training


def train_nam(config: Any) -> Path | None:
    """Train the SiameseDiff adversariality miner with frozen generator and anchor."""
    return run_miner_training(config, spatial_dims=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NAM for SiameseDiff.")
    parser.add_argument("--config", default="configs/table1_2d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_nam(apply_overrides(load_config(arguments.config), arguments.set))
