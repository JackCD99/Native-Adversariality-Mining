"""Diffusion adapter registry for Table I."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from nam.diffusion.base import MedicalDiffusionAdapter


MODULES = {
    "segdiff": "nam.diffusion.2D_M2I.segdiff.model",
    "diffboost": "nam.diffusion.2D_M2I.diffboost.model",
    "fairdiff": "nam.diffusion.2D_M2I.fairdiff.model",
    "siamesediff": "nam.diffusion.2D_M2I.siamesediff.model",
    "jodiffusion": "nam.diffusion.2D_M&I.jodiffusion.model",
    "medsegfactory": "nam.diffusion.2D_M&I.medsegfactory.model",
    "voldit": "nam.diffusion.3D_M2I.voldit.model",
    "maisi": "nam.diffusion.3D_M2I.maisi.model",
    "controlnet_sdxl": "nam.diffusion.2D_M2I.controlnet_sdxl.model",
    "sd15_lora": "nam.diffusion.2D_T2I.sd15_lora.model",
}


def build_diffusion(config: Any) -> MedicalDiffusionAdapter:
    """Build one named Table I diffusion adapter."""
    name = str(config.name).lower()
    if name not in MODULES:
        raise KeyError(f"Unknown diffusion adapter '{name}'. Available: {sorted(MODULES)}")
    module = import_module(MODULES[name])
    return module.build_adapter(config)
