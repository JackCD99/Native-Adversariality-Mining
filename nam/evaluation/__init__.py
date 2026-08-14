"""Table I evaluation utilities."""

from nam.evaluation.adversariality_evaluator import evaluate_adversariality
from nam.evaluation.classification_evaluator import evaluate_classification
from nam.evaluation.evaluator import evaluate_segmentation
from nam.evaluation.fid_evaluator import evaluate_fid
from nam.evaluation.metrics import average_surface_distance, dice_per_class, frechet_distance

__all__ = [
    "average_surface_distance",
    "dice_per_class",
    "evaluate_adversariality",
    "evaluate_classification",
    "evaluate_fid",
    "evaluate_segmentation",
    "frechet_distance",
]
