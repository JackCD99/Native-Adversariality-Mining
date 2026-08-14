"""Evaluate a fixed-budget synthetic set with configurable proxy losses."""

import argparse
import json

import _bootstrap  # noqa: F401

from nam.cli import add_common_arguments, run_configured
from nam.evaluation import evaluate_adversariality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        "configs/table1_2d.yaml",
        ("dataset", "synthetic_dataset", "downstream", "adversariality"),
    )
    parser.add_argument("--spatial-dims", type=int, choices=(2, 3), default=2)
    parser.add_argument("--proxy", choices=("lcbce", "lce", "ldice"), default="lcbce")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def evaluate(config, arguments):
    config.adversariality.proxy = arguments.proxy
    if arguments.budget is not None:
        config.adversariality.budget = arguments.budget
    if arguments.output is not None:
        config.adversariality.output = arguments.output
    result = evaluate_adversariality(config, arguments.spatial_dims)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    arguments = parse_args()
    run_configured(arguments, evaluate, arguments)
