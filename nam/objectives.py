"""NAM adversariality reward and noise-prior regularization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


def _target_map(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == logits.ndim and target.shape[1] == 1:
        target = target[:, 0]
    return target.long()


def cross_entropy_proxy(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Return bounded per-sample categorical cross entropy."""
    logits = logits[0] if isinstance(logits, (tuple, list)) else logits
    target = _target_map(logits, target)
    loss = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    valid = target != ignore_index
    if loss.ndim == 1:
        mean = loss * valid
    else:
        axes = tuple(range(1, loss.ndim))
        mean = (loss * valid).sum(dim=axes) / valid.sum(dim=axes).clamp_min(1)
    return 1.0 - torch.exp(-mean)


def soft_dice_proxy(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, eps: float = 1e-6
) -> torch.Tensor:
    """Return per-sample foreground soft-Dice loss in the unit interval."""
    logits = logits[0] if isinstance(logits, (tuple, list)) else logits
    target = _target_map(logits, target)
    if logits.ndim == 2:
        valid = target != ignore_index
        safe_target = target.masked_fill(~valid, 0).clamp(0, logits.shape[1] - 1)
        true_probability = logits.softmax(1).gather(1, safe_target[:, None]).squeeze(1)
        return (1.0 - true_probability) * valid
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0).clamp(0, logits.shape[1] - 1)
    probabilities = logits.softmax(1) * valid.unsqueeze(1)
    one_hot = F.one_hot(safe_target, logits.shape[1]).movedim(-1, 1).to(logits.dtype)
    one_hot = one_hot * valid.unsqueeze(1)
    axes = tuple(range(2, logits.ndim))
    numerator = 2.0 * (probabilities * one_hot).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + one_hot.sum(dim=axes)
    dice = (numerator + eps) / (denominator + eps)
    present = one_hot.sum(dim=axes) > 0
    if logits.shape[1] > 1:
        dice, present = dice[:, 1:], present[:, 1:]
    mean = (dice * present).sum(1) / present.sum(1).clamp_min(1)
    return 1.0 - mean


def focal_proxy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Return bounded per-sample focal loss for hard-region ablations."""
    logits = logits[0] if isinstance(logits, (tuple, list)) else logits
    target = _target_map(logits, target)
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0).clamp(0, logits.shape[1] - 1)
    probability = logits.softmax(1).gather(1, safe_target.unsqueeze(1)).squeeze(1)
    loss = -(1.0 - probability).pow(gamma) * probability.clamp_min(1e-6).log()
    if loss.ndim == 1:
        mean = loss * valid
    else:
        axes = tuple(range(1, loss.ndim))
        mean = (loss * valid).sum(dim=axes) / valid.sum(dim=axes).clamp_min(1)
    return 1.0 - torch.exp(-mean)


def class_balanced_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-sample class-balanced cross entropy.

    Every class present in a target receives equal weight regardless of its area,
    The result is monotonically mapped to [0, 1), so that ``kappa_up`` has a
    consistent interpretation across datasets. Classification logits fall back
    to the ordinary per-example cross entropy because each target contains one
    class rather than a spatial class distribution.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if target.ndim == logits.ndim and target.shape[1] == 1:
        target = target[:, 0]
    target = target.long()

    if logits.ndim == 2:
        loss = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
        return (1.0 - torch.exp(-loss)) * (target != ignore_index)

    if logits.shape[1] == 1:
        foreground = torch.sigmoid(logits[:, 0]).clamp(eps, 1.0 - eps)
        log_probabilities = torch.stack(
            [torch.log1p(-foreground), torch.log(foreground)], dim=1
        )
    else:
        log_probabilities = F.log_softmax(logits, dim=1)

    class_count = log_probabilities.shape[1]
    valid = target != ignore_index
    target_safe = target.masked_fill(~valid, 0).clamp(0, class_count - 1)
    negative_log_likelihood = -log_probabilities.gather(1, target_safe.unsqueeze(1)).squeeze(1)

    per_class = []
    spatial_dims = tuple(range(1, target.ndim))
    for class_index in range(class_count):
        class_mask = (target_safe == class_index) & valid
        denominator = class_mask.sum(dim=spatial_dims)
        numerator = (negative_log_likelihood * class_mask).sum(dim=spatial_dims)
        loss = numerator / denominator.clamp_min(1)
        per_class.append((loss, denominator > 0))

    losses = torch.stack([item[0] for item in per_class], dim=1)
    present = torch.stack([item[1] for item in per_class], dim=1)
    balanced = (losses * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)
    return 1.0 - torch.exp(-balanced)


ADVERSARIALITY_REWARDS = {
    "lcbce": class_balanced_cross_entropy,
    "lce": cross_entropy_proxy,
    "ldice": soft_dice_proxy,
    "lfocal": focal_proxy,
}


def adversariality_reward(
    name: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Evaluate one registered bounded adversariality reward."""
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized not in ADVERSARIALITY_REWARDS:
        raise KeyError(
            f"Unknown adversariality reward '{name}'. "
            f"Available: {sorted(ADVERSARIALITY_REWARDS)}"
        )
    return ADVERSARIALITY_REWARDS[normalized](logits, target, ignore_index)


@dataclass
class GaussianReselection:
    """Parameters and diagnostics of the NAM diagonal Gaussian."""

    sample: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    kl_per_sample: torch.Tensor


def reselect_noise(
    delta_mean: torch.Tensor,
    delta_variance_raw: torch.Tensor,
    variance_bound: float = 0.95,
    generator: torch.Generator | None = None,
) -> GaussianReselection:
    """Sample from N(delta_mean, I + delta_variance) by reparameterization.

    ``tanh`` constrains the diagonal covariance shift and guarantees strictly
    positive variance without changing the zero-initialization property.
    """
    if not 0.0 < variance_bound < 1.0:
        raise ValueError("variance_bound must be in the open interval (0, 1).")
    delta_variance = variance_bound * torch.tanh(delta_variance_raw)
    variance = 1.0 + delta_variance
    epsilon = torch.randn(
        delta_mean.shape,
        device=delta_mean.device,
        dtype=delta_mean.dtype,
        generator=generator,
    )
    sample = delta_mean + variance.sqrt() * epsilon
    kl_map = 0.5 * (delta_mean.square() + variance - variance.log() - 1.0)
    kl_per_sample = kl_map.flatten(1).sum(dim=1)
    return GaussianReselection(sample, delta_mean, variance, kl_per_sample)


def normalized_kl(kl_per_sample: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Normalize KL by latent dimensionality for resolution-stable optimization."""
    dimensions = math.prod(noise.shape[1:])
    return kl_per_sample / max(dimensions, 1)
