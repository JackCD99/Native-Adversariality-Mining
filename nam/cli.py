"""Command-line validation and execution helpers for release entry points."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable, Iterable

from nam.config import ConfigNode, apply_overrides, load_config
from nam.utils.imports import import_factory


def add_common_arguments(
    parser: ArgumentParser,
    default_config: str,
    extra_sections: Iterable[str] = (),
) -> None:
    """Add portable configuration, validation, and logging flags."""
    parser.add_argument("--config", default=default_config, help="Experiment YAML or JSON file.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and import configured factories without loading data or weights.",
    )
    parser.add_argument("--print-config", action="store_true", help="Print the resolved configuration.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.set_defaults(required_sections=tuple(extra_sections))


def _require_sections(config: ConfigNode, sections: Iterable[str]) -> None:
    missing = [section for section in sections if section not in config]
    if missing:
        raise KeyError(f"Configuration is missing required sections: {', '.join(missing)}")


def validate_config(config: ConfigNode, sections: Iterable[str] = ()) -> None:
    """Validate stable cross-pipeline contracts without touching experiment data."""
    _require_sections(config, ("experiment_name", "runtime", *sections))
    runtime = config.runtime
    for key in ("device", "seed", "num_workers"):
        if key not in runtime:
            raise KeyError(f"runtime.{key} is required.")
    if int(runtime.num_workers) < 0:
        raise ValueError("runtime.num_workers must be non-negative.")
    # 三阶段种子彼此独立；缺省时仍允许旧配置回退到 runtime.seed。
    for key in ("seed", "miner_seed", "downstream_seed", "sampling_seed"):
        if key in runtime:
            value = runtime[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"runtime.{key} must be a non-negative integer.")
    for section_name in ("dataset", "synthetic_dataset"):
        section = getattr(config, section_name, None)
        if section is None:
            continue
        if not getattr(section, "factory", ""):
            raise KeyError(f"{section_name}.factory is required.")
        import_factory(section.factory, getattr(section, "project_dir", None))
    if hasattr(config, "diffusion"):
        name = str(getattr(config.diffusion, "name", "")).strip()
        if not name:
            raise KeyError("diffusion.name is required.")
    if hasattr(config, "downstream"):
        if int(getattr(config.downstream, "spatial_dims", 0)) not in (2, 3):
            raise ValueError("downstream.spatial_dims must be 2 or 3.")


def run_configured(
    arguments: Namespace,
    function: Callable[..., Any],
    *function_args: Any,
    required_sections: Iterable[str] | None = None,
    **function_kwargs: Any,
) -> Any:
    """Resolve, validate, report, and execute one configuration-driven command."""
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config_path = Path(arguments.config).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {config_path.resolve()}")
    config = apply_overrides(load_config(config_path), list(arguments.set))
    sections = tuple(required_sections or getattr(arguments, "required_sections", ()))
    validate_config(config, sections)
    if arguments.print_config:
        print(json.dumps(config, indent=2))
    if arguments.dry_run:
        print(f"Configuration valid: {config.config_path}")
        return config
    return function(config, *function_args, **function_kwargs)
