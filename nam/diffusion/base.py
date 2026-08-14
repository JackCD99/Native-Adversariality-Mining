"""Unified diffusion interface required by NAM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from nam.data import NAMBatch


@dataclass(frozen=True)
class DiffusionMetadata:
    """Static properties of an adapted medical diffusion model."""

    name: str
    official_repository: str
    synthesis_paradigm: str
    spatial_dims: int
    prediction_type: str
    noise_channels: int
    noise_size: tuple[int, ...]
    dual_noise: bool = False


@dataclass
class DiffusionCondition:
    """Condition tensors and auxiliary model inputs."""

    conditional: Any
    unconditional: Any = None
    target: torch.Tensor | None = None
    extras: Mapping[str, Any] | None = None


@dataclass
class ScoreOutput:
    """Initial denoising response consumed by the miner."""

    score: torch.Tensor
    timestep: torch.Tensor
    raw_prediction: torch.Tensor | None = None


class MedicalDiffusionAdapter(ABC):
    """Abstract bridge from an official repository to NAM.

    Implementations must not update the diffusion model. The truncated rollout
    must preserve gradients from its output to ``initial_noise`` while stopping
    gradients through score evaluations with respect to the current state.
    """

    metadata: DiffusionMetadata

    def __init__(self, model: nn.Module, config: Any) -> None:
        self.model = model
        self.config = config
        self.freeze()

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def sample_probe_noise(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        shape = (batch_size, self.metadata.noise_channels, *self.metadata.noise_size)
        return torch.randn(shape, device=self.device, generator=generator)

    @abstractmethod
    def prepare_condition(self, batch: NAMBatch) -> DiffusionCondition:
        """Convert a canonical medical batch into official model conditioning."""

    @abstractmethod
    def initial_score(
        self, probe_noise: torch.Tensor, condition: DiffusionCondition, cfg_scale: float
    ) -> ScoreOutput:
        """Return the score at the first reverse sampling timestep."""

    @abstractmethod
    def truncated_rollout(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return a differentiable image or an M&I image-target pair estimate."""

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor,
        condition: DiffusionCondition,
        steps: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return complete synthetic images and targets using deterministic DDIM."""


def epsilon_to_score(epsilon: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """Convert epsilon prediction to the VP diffusion score."""
    sigma = (1.0 - alpha_bar).clamp_min(1e-12).sqrt()
    while sigma.ndim < epsilon.ndim:
        sigma = sigma.unsqueeze(-1)
    return -epsilon / sigma


def v_prediction_to_epsilon(
    velocity: torch.Tensor, noisy_sample: torch.Tensor, alpha_bar: torch.Tensor
) -> torch.Tensor:
    """Convert v-prediction to epsilon following the Diffusers convention."""
    alpha = alpha_bar.sqrt()
    sigma = (1.0 - alpha_bar).sqrt()
    while alpha.ndim < noisy_sample.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    return alpha * velocity + sigma * noisy_sample
