"""Validated wrapper for official-repository bridge objects."""

from __future__ import annotations

from typing import Any

import torch

from nam.data import NAMBatch
from nam.diffusion.base import (
    DiffusionCondition,
    DiffusionMetadata,
    MedicalDiffusionAdapter,
    ScoreOutput,
)
from nam.utils.imports import import_factory


class ExternalDiffusionAdapter(MedicalDiffusionAdapter):
    """Delegate numerical operations to a bridge living with the official code.

    The bridge object must expose ``model``, ``prepare_condition``,
    ``initial_score``, ``truncated_rollout``, and ``sample``. This avoids copying
    or silently modifying third-party implementations.
    """

    def __init__(self, config: Any, metadata: DiffusionMetadata) -> None:
        factory = import_factory(config.factory, getattr(config, "project_dir", None))
        self.bridge = factory(config=config, metadata=metadata)
        if not hasattr(self.bridge, "model"):
            raise TypeError("The diffusion bridge must expose its torch model as '.model'.")
        self.metadata = metadata
        super().__init__(self.bridge.model, config)

    def prepare_condition(self, batch: NAMBatch) -> DiffusionCondition:
        value = self.bridge.prepare_condition(batch)
        if isinstance(value, DiffusionCondition):
            return value
        return DiffusionCondition(**value)

    def initial_score(
        self, probe_noise: torch.Tensor, condition: DiffusionCondition, cfg_scale: float
    ) -> ScoreOutput:
        value = self.bridge.initial_score(probe_noise, condition, cfg_scale)
        if isinstance(value, ScoreOutput):
            return value
        return ScoreOutput(**value)

    def truncated_rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.bridge.truncated_rollout(initial_noise, condition, steps, cfg_scale)

    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bridge.sample(initial_noise, condition, steps, cfg_scale)
