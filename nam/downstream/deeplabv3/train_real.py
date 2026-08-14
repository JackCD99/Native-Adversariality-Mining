"""Real-data training pipeline for DeepLabV3-ResNet50."""

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nam.config import apply_overrides, load_config
from nam.downstream.deeplabv3.model import build_model
from nam.downstream.natural_training import train_natural_segmentation


def train_real(config: Any, spatial_dims: int) -> Path:
    return train_natural_segmentation(config, spatial_dims, "deeplabv3", build_model, "real")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3 on real VOC/SBD data.")
    parser.add_argument("--config", default="configs/controlnet_sdxl_voc.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_real(apply_overrides(load_config(arguments.config), arguments.set), 2)
