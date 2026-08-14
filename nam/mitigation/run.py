"""Generic fixed-budget runner for NAM exposure mitigation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nam.config import apply_overrides, load_config
from nam.cli import validate_config
from nam.mitigation.base import NAMMitigationBackend
from nam.mitigation.io import save_manifest, save_tensor_pairs
from nam.mitigation.registry import build_strategy
from nam.utils.imports import import_factory
from nam.utils.seed import seed_everything


def run_mitigation(config: Any) -> Path:
    """Build a method bridge, run one strategy, and save auditable outputs."""
    seed_everything(int(config.runtime.seed), bool(getattr(config.runtime, "deterministic", False)))
    factory = import_factory(config.backend.factory, getattr(config.backend, "project_dir", None))
    backend = factory(config=config)
    if not isinstance(backend, NAMMitigationBackend):
        raise TypeError("backend.factory must return a NAMMitigationBackend instance.")
    section = config[str(config.strategy.name).lower()]
    strategy = build_strategy(section)
    generator = torch.Generator(device=backend.device).manual_seed(int(config.runtime.seed))
    candidates = strategy.run(backend, generator=generator)
    output = Path(config.runtime.output_dir) / str(config.strategy.name).lower()
    save_tensor_pairs(candidates, output / "tensors")
    save_manifest(candidates, output / "manifest.csv")
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a NAM exposure-mitigation strategy.")
    parser.add_argument("--config", default="configs/mitigation.yaml")
    parser.add_argument("--strategy", choices=("hat", "qsf", "lsrs", "asg"), default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = apply_overrides(load_config(arguments.config), arguments.set)
    if arguments.strategy is not None:
        configuration.strategy.name = arguments.strategy
    validate_config(
        configuration,
        ("dataset", "diffusion", "anchor", "miner", "sampling", "backend", "strategy"),
    )
    if arguments.print_config:
        print(json.dumps(configuration, indent=2))
    if arguments.dry_run:
        print(f"Configuration valid: {configuration.config_path}")
        raise SystemExit(0)
    run_mitigation(configuration)
