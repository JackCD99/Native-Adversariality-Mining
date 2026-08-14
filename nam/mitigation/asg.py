"""Attention-score guidance (ASG), Appendix Algorithm 5.

ASG borrows the cross-attention response and self-attention conflict signals
from InitNO, but deliberately keeps the NAM-selected seed fixed and applies
normalized score guidance only at configured DDIM trajectory steps.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Sequence

import torch
from torch.nn import functional as F

from nam.mitigation.base import AttentionBundle, MitigationCandidate, NAMMitigationBackend


def soft_dice_loss(attention: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Dice mismatch between normalized attention and mask."""
    attention, mask = attention.float(), mask.float()
    reduce = tuple(range(1, attention.ndim))
    intersection = (attention * mask).sum(reduce)
    denominator = attention.sum(reduce) + mask.sum(reduce)
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def aggregate_cross_attention(
    cross_attention: torch.Tensor, token_indices: Sequence[int]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Average target-token maps while retaining each raw response map."""
    if cross_attention.ndim < 4:
        raise ValueError("Cross-attention must be BxTokensxSpatial...")
    if not token_indices:
        raise ValueError("ASG requires at least one target token.")
    components: list[torch.Tensor] = []
    for index in token_indices:
        if index < 0 or index >= cross_attention.shape[1]:
            raise IndexError(f"Target token {index} is outside the attention tensor.")
        # Keep the response magnitude for the class-conditioned max-response
        # term in Eq. B.5. Pairwise conflict normalizes maps separately below.
        components.append(cross_attention[:, index : index + 1])
    return torch.stack(components).mean(0), components


def self_attention_conflict(
    self_attention: torch.Tensor | None,
    component_maps: Sequence[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute normalized pair overlap from Eq. B.6.

    When a backend exposes spatial self-attention, each component's peak
    location indexes its corresponding self-attention row, following InitNO.
    Otherwise the normalized cross-attention components are used directly.
    """
    if len(component_maps) < 2:
        return component_maps[0].new_zeros(())
    spatial_maps: list[torch.Tensor] = []
    if self_attention is not None:
        if self_attention.ndim != 3 or self_attention.shape[-1] != self_attention.shape[-2]:
            raise ValueError("Self-attention must be BxLxL.")
        for component in component_maps:
            peak = component.flatten(2).mean(1).argmax(-1)
            row = self_attention[torch.arange(component.shape[0], device=component.device), peak]
            spatial_maps.append(row)
    else:
        spatial_maps = [item.flatten(1) for item in component_maps]
    normalized = [item / item.norm(dim=-1, keepdim=True).clamp_min(eps) for item in spatial_maps]
    overlaps = [(left * right).sum(-1).mean() for left, right in combinations(normalized, 2)]
    return torch.stack(overlaps).mean()


class AttentionScoreGuidance:
    """Guide NAM sampling with mask alignment and component-conflict losses."""

    name = "asg"

    def __init__(
        self,
        guidance_timesteps: Sequence[int],
        guidance_strength: float = 0.1,
        conflict_weight: float = 0.1,
        gradient_epsilon: float = 1e-8,
    ) -> None:
        if not guidance_timesteps:
            raise ValueError("ASG requires at least one guidance timestep.")
        if guidance_strength < 0 or conflict_weight < 0:
            raise ValueError("ASG guidance weights must be non-negative.")
        self.guidance_timesteps = frozenset(int(item) for item in guidance_timesteps)
        self.guidance_strength = float(guidance_strength)
        self.conflict_weight = float(conflict_weight)
        self.gradient_epsilon = float(gradient_epsilon)

    def run_one(
        self,
        backend: NAMMitigationBackend,
        condition: Any,
        generator: torch.Generator | None = None,
    ) -> MitigationCandidate:
        with torch.no_grad():
            selected_noise = backend.select_nam_noise(condition, generator)
        latent = selected_noise.detach()
        token_indices = tuple(backend.target_token_indices(condition))
        mask = backend.conditioning_mask(condition)
        diagnostics: list[dict[str, float]] = []
        for timestep in backend.sampling_timesteps():
            value = int(torch.as_tensor(timestep).flatten()[0].item())
            if value in self.guidance_timesteps:
                latent = latent.detach().requires_grad_(True)
                bundle = backend.predict_with_attention(latent, timestep, condition)
                aggregate, components = aggregate_cross_attention(
                    bundle.cross_attention, token_indices
                )
                if mask is not None:
                    resized = F.interpolate(mask.float(), size=aggregate.shape[2:], mode="nearest")
                    mask_loss = soft_dice_loss(aggregate, resized)
                else:
                    # Eq. B.5 class-conditioned form: every target token should
                    # activate at least one spatial location.
                    maxima = torch.stack([item.flatten(1).amax(1) for item in components])
                    mask_loss = (1.0 - maxima).mean()
                conflict = self_attention_conflict(bundle.self_attention, components)
                loss = mask_loss + self.conflict_weight * conflict
                gradient = torch.autograd.grad(loss, latent, retain_graph=False)[0]
                gradient = gradient / gradient.flatten(1).norm(dim=1).view(
                    gradient.shape[0], *([1] * (gradient.ndim - 1))
                ).clamp_min(self.gradient_epsilon)
                epsilon = bundle.epsilon.detach() + self.guidance_strength * gradient.detach()
                diagnostics.append({
                    "timestep": float(value),
                    "mask_loss": float(mask_loss.detach()),
                    "conflict_loss": float(conflict.detach()),
                })
            else:
                with torch.no_grad():
                    epsilon = backend.predict_noise(latent, timestep, condition)
            with torch.no_grad():
                latent = backend.ddim_step(latent.detach(), epsilon, timestep)
        with torch.no_grad():
            result = backend.decode_candidate(latent, condition, selected_noise)
        result.metadata.update({
            "strategy": self.name,
            "guidance_timesteps": sorted(self.guidance_timesteps, reverse=True),
            "guidance_strength": self.guidance_strength,
            "conflict_weight": self.conflict_weight,
            "diagnostics": diagnostics,
        })
        return result

    def run(
        self,
        backend: NAMMitigationBackend,
        conditions: Iterable[Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> list[MitigationCandidate]:
        return [self.run_one(backend, condition, generator) for condition in (
            backend.conditions() if conditions is None else conditions
        )]
