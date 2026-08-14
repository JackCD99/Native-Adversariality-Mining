"""Real-plus-synthetic continuation for ResNet-50 classification."""

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nam.config import apply_overrides, load_config
from nam.downstream.classification_training import train_classification
from nam.downstream.resnet50.model import build_model


def train_synthetic(config: Any, spatial_dims: int) -> Path:
    if spatial_dims != 2:
        raise ValueError("ResNet-50 classification requires 2D images.")
    return train_classification(config, "resnet50", build_model, "synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sd15_lora_pneumoniamnist.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_synthetic(apply_overrides(load_config(arguments.config), arguments.set), 2)
