"""Segmentation and distribution metrics used in Table I."""

from __future__ import annotations

import numpy as np
import torch
from scipy import linalg, ndimage


def dice_per_class(
    prediction: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Return foreground Dice values, excluding absent classes and background."""
    values = []
    spatial = tuple(range(1, prediction.ndim))
    for class_index in range(1, num_classes):
        pred = prediction == class_index
        truth = target == class_index
        denominator = pred.sum(dim=spatial) + truth.sum(dim=spatial)
        intersection = (pred & truth).sum(dim=spatial)
        valid = denominator > 0
        if valid.any():
            values.append((2.0 * intersection[valid].float()) / denominator[valid].float())
    return torch.cat(values) if values else torch.empty(0, device=prediction.device)


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    return mask ^ ndimage.binary_erosion(mask, structure=structure, border_value=0)


def average_surface_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """Compute a fair and general symmetric average surface distance (ASD/ASSD).

    The two directional distances are averaged separately and then equally
    weighted, preventing the side with more surface points from receiving
    greater influence. If exactly one side is empty, the physical
    field-of-view (FOV) diagonal is returned as a finite failure penalty.
    If both sides are empty, the object is absent from the sample and NaN
    is returned so that dataset-level evaluation can explicitly exclude it.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("ASD prediction and target must have identical shapes.")
    if prediction.ndim not in (2, 3):
        raise ValueError("ASD supports arbitrary 2D or 3D binary masks.")
    if spacing is None:
        spacing_array = np.ones(prediction.ndim, dtype=np.float64)
    else:
        spacing_array = np.asarray(spacing, dtype=np.float64)
        if spacing_array.shape != (prediction.ndim,):
            raise ValueError("ASD spacing must contain one value per spatial dimension.")
        if not np.isfinite(spacing_array).all() or np.any(spacing_array <= 0):
            raise ValueError("ASD spacing values must be positive and finite.")
    if not prediction.any() and not target.any():
        return float("nan")
    if not prediction.any() or not target.any():
        field_of_view = np.asarray(prediction.shape, dtype=np.float64) * spacing_array
        return float(np.linalg.norm(field_of_view))
    pred_surface = _surface(prediction)
    target_surface = _surface(target)
    distance_to_target = ndimage.distance_transform_edt(
        ~target_surface, sampling=spacing_array
    )
    distance_to_prediction = ndimage.distance_transform_edt(
        ~pred_surface, sampling=spacing_array
    )
    prediction_to_target = distance_to_target[pred_surface].mean()
    target_to_prediction = distance_to_prediction[target_surface].mean()
    return float(0.5 * (prediction_to_target + target_to_prediction))


def frechet_distance(
    real_features: np.ndarray, synthetic_features: np.ndarray, eps: float = 1e-6
) -> float:
    """Compute Fréchet distance from two feature matrices."""
    real = np.asarray(real_features, dtype=np.float64)
    synthetic = np.asarray(synthetic_features, dtype=np.float64)
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError("FID inputs must be [samples, features] matrices with equal width.")
    mean_real, mean_synthetic = real.mean(0), synthetic.mean(0)
    covariance_real = np.cov(real, rowvar=False)
    covariance_synthetic = np.cov(synthetic, rowvar=False)
    product_root, _ = linalg.sqrtm(covariance_real @ covariance_synthetic, disp=False)
    if not np.isfinite(product_root).all():
        offset = np.eye(covariance_real.shape[0]) * eps
        product_root = linalg.sqrtm((covariance_real + offset) @ (covariance_synthetic + offset))
    if np.iscomplexobj(product_root):
        product_root = product_root.real
    difference = mean_real - mean_synthetic
    return float(
        difference.dot(difference)
        + np.trace(covariance_real + covariance_synthetic - 2.0 * product_root)
    )
