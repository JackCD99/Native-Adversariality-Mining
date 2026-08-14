"""Evaluate 2D or 2.5D FID with the configured modality-specific encoder."""

import argparse

import _bootstrap  # noqa: F401

from nam.cli import add_common_arguments, run_configured
from nam.evaluation import evaluate_fid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "configs/table1_2d.yaml", ("dataset", "synthetic_dataset", "fid"))
    parser.add_argument("--spatial-dims", type=int, choices=(2, 3), default=2)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_configured(arguments, lambda config: print(evaluate_fid(config, arguments.spatial_dims)))
