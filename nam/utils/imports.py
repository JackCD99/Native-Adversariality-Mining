"""Safe dynamic imports for optional official repositories."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable


def import_factory(specification: str, project_dir: str | None = None) -> Callable[..., Any]:
    """Import ``module:function`` after optionally exposing an official repository."""
    if not specification or ":" not in specification:
        raise ValueError(
            "An external factory must be configured as 'module:function'. "
            "See docs/ADDING_A_GENERATOR.md for the bridge contract."
        )
    if project_dir:
        resolved = str(Path(project_dir).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    module_name, function_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"Configured factory is not callable: {specification}")
    return factory

