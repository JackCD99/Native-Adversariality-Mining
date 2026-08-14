"""Generate a condition-aligned 2D Base or NAM dataset."""

import argparse

import _bootstrap  # noqa: F401

from nam.cli import add_common_arguments, run_configured
from nam.engine import run_sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "configs/table1_2d.yaml", ("dataset", "diffusion"))
    parser.add_argument("--method", choices=("base", "nam"), default="nam")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_configured(
        arguments,
        lambda config: run_sampling(
            config, spatial_dims=2, use_nam=arguments.method == "nam"
        ),
    )
