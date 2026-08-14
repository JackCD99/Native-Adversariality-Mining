"""Public contracts shared by all NAM exposure-mitigation strategies.

The four strategies never update the diffusion model, NAM miner, downstream
anchor, or VQA model. A method-specific bridge only needs to expose the small
sampling primitives declared by :class:`NAMMitigationBackend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch


@dataclass
class TrajectoryPoint:
    """One cached DDIM state used by lift-score rejection sampling."""

    latent: torch.Tensor
    timestep: torch.Tensor
    full_prediction: torch.Tensor | None = None


@dataclass
class AttentionBundle:
    """Standardized attention tensors returned by an attention-aware backend.

    ``cross_attention`` is ``B x N_tokens x H x W`` (or the volumetric
    counterpart), averaged over selected layers and heads. ``self_attention``
    is ``B x L x L``, where ``L`` is the flattened spatial length.
    """

    epsilon: torch.Tensor
    cross_attention: torch.Tensor
    self_attention: torch.Tensor | None = None


@dataclass
class MitigationCandidate:
    """A generated image-target pair and its sampling diagnostics."""

    image: torch.Tensor
    target: torch.Tensor | None
    condition: Any
    sample_id: str
    selected_noise: torch.Tensor | None = None
    trajectory: tuple[TrajectoryPoint, ...] = ()
    score: float | None = None
    accepted: bool = True
    trials: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class NAMMitigationBackend(ABC):
    """Method bridge used by HAT, QSF, LSRS, and ASG.

    Implementations should reuse the method's normal NAM seed reselection and
    deterministic DDIM sampler. Optional methods raise ``NotImplementedError``
    until the corresponding strategy is enabled for that diffusion backbone.
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device hosting the frozen generator and miner."""

    @abstractmethod
    def conditions(self) -> Iterable[Any]:
        """Yield conditions in the exact fixed-budget synthesis order."""

    @abstractmethod
    @torch.no_grad()
    def select_nam_noise(
        self, condition: Any, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Run the frozen NAM probe/miner and return one reselected seed."""

    @abstractmethod
    @torch.no_grad()
    def sample_from_noise(
        self,
        selected_noise: torch.Tensor,
        condition: Any,
        cache_timesteps: Sequence[int] = (),
    ) -> MitigationCandidate:
        """Run deterministic DDIM and optionally retain specified states."""

    @abstractmethod
    @torch.no_grad()
    def adversariality(self, candidate: MitigationCandidate) -> torch.Tensor:
        """Return the per-candidate frozen-anchor segmentation loss."""

    def unconditional_condition(self, condition: Any) -> Any:
        """Return the null condition used by classifier-free guidance."""
        raise NotImplementedError("This backend does not expose unconditional conditioning.")

    def component_conditions(self, condition: Any) -> Sequence[Any]:
        """Decompose a full condition into modality/target/mask components."""
        raise NotImplementedError("This backend does not expose component conditions.")

    @torch.no_grad()
    def predict_noise(
        self, latent: torch.Tensor, timestep: torch.Tensor, condition: Any
    ) -> torch.Tensor:
        """Predict epsilon for one cached state and one condition variant."""
        raise NotImplementedError("This backend does not expose epsilon prediction.")

    def sampling_timesteps(self) -> Sequence[torch.Tensor]:
        """Return the deterministic DDIM schedule used by ASG."""
        raise NotImplementedError("This backend does not expose its DDIM schedule.")

    def predict_with_attention(
        self, latent: torch.Tensor, timestep: torch.Tensor, condition: Any
    ) -> AttentionBundle:
        """Predict epsilon and differentiable attention maps for ASG."""
        raise NotImplementedError("This backend does not expose attention maps.")

    def ddim_step(
        self, latent: torch.Tensor, epsilon: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Apply the method's deterministic DDIM update."""
        raise NotImplementedError("This backend does not expose a DDIM step.")

    @torch.no_grad()
    def decode_candidate(
        self, latent: torch.Tensor, condition: Any, selected_noise: torch.Tensor
    ) -> MitigationCandidate:
        """Decode the terminal ASG latent into the public candidate format."""
        raise NotImplementedError("This backend does not expose latent decoding.")

    def target_token_indices(self, condition: Any) -> Sequence[int]:
        """Return token indices corresponding to target anatomy/object names."""
        raise NotImplementedError("This backend does not expose target token indices.")

    def conditioning_mask(self, condition: Any) -> torch.Tensor | None:
        """Return an optional Bx1xH(xW) conditioning mask for ASG."""
        return None


def generate_nam_candidate(
    backend: NAMMitigationBackend,
    condition: Any,
    generator: torch.Generator | None,
    cache_timesteps: Sequence[int] = (),
) -> MitigationCandidate:
    """Execute the shared frozen NAM seed-reselection path once."""
    with torch.no_grad():
        noise = backend.select_nam_noise(condition, generator)
        candidate = backend.sample_from_noise(noise, condition, cache_timesteps)
    candidate.selected_noise = noise.detach()
    return candidate


def scalar(value: torch.Tensor | float) -> float:
    """Validate and convert a one-candidate score to a Python float."""
    tensor = torch.as_tensor(value).detach().float().flatten()
    if tensor.numel() != 1:
        raise ValueError("Mitigation strategies require one scalar score per candidate.")
    result = float(tensor.item())
    if not torch.isfinite(tensor).all():
        raise ValueError("A mitigation score is non-finite.")
    return result
