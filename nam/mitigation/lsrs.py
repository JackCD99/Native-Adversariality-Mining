"""Lift-score rejection sampling (LSRS), Appendix Algorithm 4.

This implementation follows CompLift's cached-prediction principle: all
condition variants reuse the exact same DDIM states, and the full-condition
prediction is cached once per selected timestep.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch

from nam.mitigation.base import (
    MitigationCandidate,
    NAMMitigationBackend,
    TrajectoryPoint,
    generate_nam_candidate,
)


def squared_l2_per_sample(value: torch.Tensor) -> torch.Tensor:
    """Return the feature-summed squared L2 norm for every batch item."""
    return value.float().flatten(1).square().sum(1)


def component_lift(
    full: torch.Tensor, unconditional: torch.Tensor, component: torch.Tensor
) -> torch.Tensor:
    """Compute Eq. B.3 using the shared full-condition prediction."""
    if full.shape != unconditional.shape or full.shape != component.shape:
        raise ValueError("Lift-score prediction tensors must have identical shapes.")
    return squared_l2_per_sample(full - unconditional) - squared_l2_per_sample(full - component)


class LiftScoreRejectionSampling:
    """Accept candidates supported by every component of the full condition."""

    name = "lsrs"

    def __init__(
        self,
        selected_timesteps: Sequence[int],
        threshold: float = 0.0,
        maximum_trials: int = 5,
    ) -> None:
        if not selected_timesteps:
            raise ValueError("LSRS requires at least one cached timestep.")
        if maximum_trials < 1:
            raise ValueError("LSRS maximum_trials must be positive.")
        self.selected_timesteps = tuple(int(item) for item in selected_timesteps)
        self.threshold, self.maximum_trials = float(threshold), int(maximum_trials)

    @torch.no_grad()
    def score(self, backend: NAMMitigationBackend, candidate: MitigationCandidate) -> float:
        components = tuple(backend.component_conditions(candidate.condition))
        if not components:
            raise ValueError("LSRS requires at least one component condition.")
        points = {int(point.timestep.flatten()[0].item()): point for point in candidate.trajectory}
        missing = set(self.selected_timesteps) - set(points)
        if missing:
            raise ValueError(f"The backend did not cache LSRS timesteps: {sorted(missing)}")
        unconditional = backend.unconditional_condition(candidate.condition)
        totals = torch.zeros(len(components), device=backend.device)
        for timestep_value in self.selected_timesteps:
            point = points[timestep_value]
            full = point.full_prediction
            if full is None:
                full = backend.predict_noise(point.latent, point.timestep, candidate.condition)
            null = backend.predict_noise(point.latent, point.timestep, unconditional)
            for index, component in enumerate(components):
                prediction = backend.predict_noise(point.latent, point.timestep, component)
                lift = component_lift(full, null, prediction)
                if lift.numel() != 1:
                    raise ValueError("LSRS currently evaluates one candidate per trial.")
                totals[index] += lift[0]
        totals /= len(self.selected_timesteps)
        return float(totals.min().item())

    @torch.no_grad()
    def run_one(
        self,
        backend: NAMMitigationBackend,
        condition: Any,
        generator: torch.Generator | None = None,
    ) -> MitigationCandidate:
        candidates: list[MitigationCandidate] = []
        for trial in range(1, self.maximum_trials + 1):
            candidate = generate_nam_candidate(
                backend, condition, generator, self.selected_timesteps
            )
            candidate.score, candidate.trials = self.score(backend, candidate), trial
            candidates.append(candidate)
            if candidate.score >= self.threshold:
                candidate.metadata.update({
                    "strategy": self.name,
                    "threshold": self.threshold,
                    "selected_timesteps": list(self.selected_timesteps),
                })
                return candidate
        fallback = max(candidates, key=lambda item: float(item.score))
        fallback.accepted = False
        fallback.metadata.update({
            "strategy": self.name,
            "threshold": self.threshold,
            "selected_timesteps": list(self.selected_timesteps),
            "fallback": True,
        })
        return fallback

    def run(
        self,
        backend: NAMMitigationBackend,
        conditions: Iterable[Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> list[MitigationCandidate]:
        return [self.run_one(backend, condition, generator) for condition in (
            backend.conditions() if conditions is None else conditions
        )]
