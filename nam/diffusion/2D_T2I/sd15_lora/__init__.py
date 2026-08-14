"""Stable Diffusion v1.5 with dataset-specific LoRA adaptation."""

from importlib import import_module


def build_adapter(config):
    return import_module(f"{__name__}.model").build_adapter(config)


__all__ = ["build_adapter"]
