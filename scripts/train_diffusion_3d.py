"""Train a configured volumetric diffusion generator."""

import argparse

import _bootstrap  # noqa: F401

from nam.cli import add_common_arguments, run_configured
from nam.engine import run_diffusion_pretraining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "configs/voldit_3d.yaml", ("dataset", "diffusion"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_configured(arguments, lambda config: run_diffusion_pretraining(config, spatial_dims=3))
