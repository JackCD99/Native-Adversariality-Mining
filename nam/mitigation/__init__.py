"""Sampling-time exposure mitigation strategies for NAM."""

from nam.mitigation.asg import AttentionScoreGuidance
from nam.mitigation.base import (
    AttentionBundle,
    MitigationCandidate,
    NAMMitigationBackend,
    TrajectoryPoint,
)
from nam.mitigation.hat import HighAdversarialityTruncation
from nam.mitigation.lsrs import LiftScoreRejectionSampling
from nam.mitigation.qsf import MedGemmaYesScorer, VQAScoreFiltering
from nam.mitigation.registry import build_strategy

__all__ = [
    "AttentionBundle",
    "AttentionScoreGuidance",
    "HighAdversarialityTruncation",
    "LiftScoreRejectionSampling",
    "MedGemmaYesScorer",
    "MitigationCandidate",
    "NAMMitigationBackend",
    "TrajectoryPoint",
    "VQAScoreFiltering",
    "build_strategy",
]
