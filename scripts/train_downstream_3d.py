"""Train a volumetric real baseline or real-plus-synthetic continuation."""

import argparse

import _bootstrap  # noqa: F401

from nam.cli import add_common_arguments, run_configured
from nam.engine import run_downstream_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "configs/voldit_3d.yaml", ("dataset", "downstream"))
    parser.add_argument("--phase", choices=("real", "synthetic"), default="synthetic")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    required = ("dataset", "downstream")
    if arguments.phase == "synthetic":
        required += ("synthetic_dataset",)
    run_configured(
        arguments,
        lambda config: run_downstream_training(config, 3, arguments.phase),
        required_sections=required,
    )
