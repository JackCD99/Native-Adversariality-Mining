"""VOC palette conversion and SDXL text/time conditioning."""

from __future__ import annotations

from typing import Any, Sequence

import torch


# Standard PASCAL VOC color map for class IDs 0..20.
VOC_PALETTE = torch.tensor(
    [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
        [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
        [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
    ],
    dtype=torch.float32,
) / 255.0


def colorize_masks(masks: torch.Tensor) -> torch.Tensor:
    """Convert BxHxW class IDs into the RGB control images used for training."""
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    palette = VOC_PALETTE.to(device=masks.device, dtype=torch.float32)
    safe = masks.long()
    safe = torch.where(
        (safe >= 0) & (safe < len(VOC_PALETTE)), safe, torch.zeros_like(safe)
    )
    return palette[safe].permute(0, 3, 1, 2).contiguous()


def prompts_from_batch(batch: Any, default_prompt: str) -> list[str]:
    condition = batch.condition
    if isinstance(condition, dict):
        values = condition.get("prompt", condition.get("txt"))
        if isinstance(values, list):
            return [str(value) if value else default_prompt for value in values]
    return [default_prompt] * batch.target.shape[0]


def encode_sdxl_prompts(
    components: Any, prompts: Sequence[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode conditional/unconditional SDXL text streams and size embeddings."""
    prompt_embeds, negative_embeds, pooled, negative_pooled = components.pipe.encode_prompt(
        prompt=list(prompts),
        negative_prompt=[""] * len(prompts),
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    resolution = int(components.resolution)
    time_ids = components.pipe._get_add_time_ids(
        (resolution, resolution),
        (0, 0),
        (resolution, resolution),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pooled.shape[-1],
    ).to(device)
    return (
        prompt_embeds,
        negative_embeds,
        pooled,
        negative_pooled,
        time_ids.repeat(len(prompts), 1),
    )
