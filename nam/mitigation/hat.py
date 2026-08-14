"""High-adversariality truncation (HAT), Appendix Algorithm 2."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from nam.mitigation.base import (
    MitigationCandidate,
    NAMMitigationBackend,
    generate_nam_candidate,
    scalar,
)


class HighAdversarialityTruncation:
    """Trim only the empirical extreme tail of a complete NAM candidate set.

    The initial threshold is estimated globally before any replacement. A
    rejected slot keeps its original condition. If every retry fails, the
    lowest-adversariality retry is returned, exactly following Algorithm 2.
    """

    name = "hat"

    def __init__(self, quantile: float = 0.95, maximum_trials: int = 5) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("HAT quantile must lie in (0, 1).")
        if maximum_trials < 1:
            raise ValueError("HAT maximum_trials must be positive.")
        self.quantile = float(quantile)
        self.maximum_trials = int(maximum_trials)

    @torch.no_grad()
    def run(
        self,
        backend: NAMMitigationBackend,
        conditions: Iterable[Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> list[MitigationCandidate]:
        items = list(backend.conditions() if conditions is None else conditions)
        if not items:
            raise ValueError("HAT cannot estimate a percentile from an empty candidate set.")
        initial: list[MitigationCandidate] = []
        adversarialities: list[float] = []
        for condition in items:
            candidate = generate_nam_candidate(backend, condition, generator)
            value = scalar(backend.adversariality(candidate))
            candidate.score = value
            candidate.metadata["initial_adversariality"] = value
            initial.append(candidate)
            adversarialities.append(value)
        threshold = float(torch.quantile(torch.tensor(adversarialities), self.quantile).item())
        output: list[MitigationCandidate] = []
        for candidate in initial:
            candidate.metadata.update({"strategy": self.name, "threshold": threshold})
            if candidate.score <= threshold:
                output.append(candidate)
                continue
            retries: list[MitigationCandidate] = []
            accepted: MitigationCandidate | None = None
            for trial in range(1, self.maximum_trials + 1):
                replacement = generate_nam_candidate(backend, candidate.condition, generator)
                replacement.score = scalar(backend.adversariality(replacement))
                replacement.trials = trial
                retries.append(replacement)
                if replacement.score <= threshold:
                    accepted = replacement
                    break
            selected = accepted or min(retries, key=lambda item: float(item.score))
            selected.accepted = accepted is not None
            selected.metadata.update({
                "strategy": self.name,
                "threshold": threshold,
                "initial_adversariality": candidate.score,
                "fallback": accepted is None,
            })
            output.append(selected)
        return output
