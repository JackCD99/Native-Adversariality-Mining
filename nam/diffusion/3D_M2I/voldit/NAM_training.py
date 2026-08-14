"""Train the paper's explicit 3D ResUNet miner for frozen VolDiT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nam.config import apply_overrides, load_config
from nam.engine.miner_trainer import run_miner_training


def train_nam(config):
    """Train the volumetric ResUNet miner used by the VolDiT adapter."""
    return run_miner_training(config, spatial_dims=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the volumetric NAM ResUNet for VolDiT.")
    parser.add_argument("--config", default="configs/voldit_3d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_nam(apply_overrides(load_config(arguments.config), arguments.set))
