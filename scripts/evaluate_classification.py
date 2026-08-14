"""Evaluate a held-out classification checkpoint."""

import argparse
import json

import _bootstrap  # noqa: F401

from nam.cli import add_common_arguments, run_configured
from nam.evaluation import evaluate_classification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser, "configs/sd15_lora_pneumoniamnist.yaml",
        ("dataset", "downstream", "evaluation"),
    )
    parser.add_argument("--checkpoint-phase", choices=("real", "syn"), default="syn")
    return parser.parse_args()


def evaluate(config, phase: str):
    key = "real_checkpoint" if phase == "real" else "syn_checkpoint"
    config.downstream.checkpoint = getattr(config.downstream, key)
    result = evaluate_classification(config)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    arguments = parse_args()
    run_configured(arguments, evaluate, arguments.checkpoint_phase)
