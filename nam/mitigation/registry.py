"""Configuration registry for the four Appendix exposure mitigations."""

from __future__ import annotations

from typing import Any

from nam.mitigation.asg import AttentionScoreGuidance
from nam.mitigation.hat import HighAdversarialityTruncation
from nam.mitigation.lsrs import LiftScoreRejectionSampling
from nam.mitigation.qsf import MedGemmaYesScorer, VQAScoreFiltering


def build_strategy(config: Any, scorer: Any = None) -> Any:
    """Build HAT, QSF, LSRS, or ASG from one config section."""
    name = str(config.name).lower()
    if name == "hat":
        return HighAdversarialityTruncation(
            float(getattr(config, "quantile", 0.95)),
            int(getattr(config, "maximum_trials", 5)),
        )
    if name == "qsf":
        scorer = scorer or MedGemmaYesScorer(
            str(getattr(config, "model", "google/medgemma-4b-it")),
            str(getattr(config, "device", "cuda")),
            str(getattr(config, "dtype", "bfloat16")),
        )
        return VQAScoreFiltering(
            scorer,
            float(getattr(config, "threshold", 0.8)),
            int(getattr(config, "maximum_trials", 5)),
        )
    if name == "lsrs":
        return LiftScoreRejectionSampling(
            tuple(config.selected_timesteps),
            float(getattr(config, "threshold", 0.0)),
            int(getattr(config, "maximum_trials", 5)),
        )
    if name == "asg":
        return AttentionScoreGuidance(
            tuple(config.guidance_timesteps),
            float(config.guidance_strength),
            float(config.conflict_weight),
            float(getattr(config, "gradient_epsilon", 1e-8)),
        )
    raise KeyError("Unknown mitigation strategy. Available: hat, qsf, lsrs, asg.")
