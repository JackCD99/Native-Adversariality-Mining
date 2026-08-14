"""Train NAM for a configured volumetric diffusion model."""

import argparse

import _bootstrap  # noqa: F401

from nam.cli import add_common_arguments, run_configured
from nam.engine import run_nam_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser, "configs/voldit_3d.yaml", ("dataset", "diffusion", "anchor", "miner")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_configured(arguments, lambda config: run_nam_training(config, spatial_dims=3))
