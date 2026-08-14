"""Configuration loading and command-line override utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dictionary with attribute access used by the executable entry points."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({key: _to_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_node(item) for item in value]
    return value


def _read_mapping(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        if config_path.suffix.lower() == ".json":
            raw = json.load(stream)
        else:
            raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError(f"The configuration root must be a mapping: {config_path}")
    return raw


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_inherited(config_path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if config_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Circular configuration inheritance: {chain}")
    raw = _read_mapping(config_path)
    parents = raw.pop("extends", [])
    if isinstance(parents, (str, Path)):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (config_path.parent / str(parent)).resolve()
        merged = _merge(merged, _load_inherited(parent_path, (*stack, config_path)))
    return _merge(merged, raw)


def load_config(path: str | Path) -> ConfigNode:
    """Load a JSON/YAML experiment and recursively merge optional parent files."""
    config_path = Path(path).expanduser().resolve()
    raw = _load_inherited(config_path)
    config = _to_node(raw)
    config.config_path = str(config_path)
    return config


def apply_overrides(config: ConfigNode, overrides: list[str]) -> ConfigNode:
    """Apply dotted ``key=value`` command-line overrides."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'; expected key=value.")
        dotted_key, raw_value = item.split("=", 1)
        cursor: ConfigNode = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in cursor:
                cursor[part] = ConfigNode()
            cursor = cursor[part]
        cursor[parts[-1]] = _to_node(yaml.safe_load(raw_value))
    return config
