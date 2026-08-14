"""Mask2Former semantic segmentation with a ResNet-50 backbone.

Architecture source: https://github.com/facebookresearch/Mask2Former
Maintained implementation: https://github.com/huggingface/transformers/tree/main/src/transformers/models/mask2former
Reference: Cheng et al., Masked-attention Mask Transformer for Universal Image
Segmentation, CVPR 2022.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class Mask2Former(nn.Module):
    """Thin semantic-segmentation interface over the canonical architecture."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        try:
            from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, ResNetConfig
        except ImportError as error:
            raise ImportError("Mask2Former requires `pip install transformers`.") from error

        options = getattr(config, "model", config)
        backbone = ResNetConfig(
            num_channels=int(getattr(config, "in_channels", 3)),
            embedding_size=64,
            hidden_sizes=[256, 512, 1024, 2048],
            depths=[3, 4, 6, 3],
            layer_type="bottleneck",
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
        architecture = Mask2FormerConfig(
            backbone_config=backbone,
            num_labels=int(config.num_classes),
            ignore_value=int(getattr(config, "ignore_index", 255)),
            feature_size=int(getattr(options, "feature_size", 256)),
            mask_feature_size=int(getattr(options, "mask_feature_size", 256)),
            hidden_dim=int(getattr(options, "hidden_dim", 256)),
            encoder_layers=int(getattr(options, "encoder_layers", 6)),
            decoder_layers=int(getattr(options, "decoder_layers", 10)),
            num_attention_heads=int(getattr(options, "heads", 8)),
            num_queries=int(getattr(options, "queries", 100)),
            use_auxiliary_loss=True,
            output_auxiliary_logits=True,
        )
        self.model = Mask2FormerForUniversalSegmentation(architecture)
        self.num_classes = int(config.num_classes)

    @staticmethod
    def _set_targets(
        targets: torch.Tensor, ignore_index: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Convert dense semantic labels to set-prediction targets."""
        mask_labels: list[torch.Tensor] = []
        class_labels: list[torch.Tensor] = []
        for target in targets:
            classes = torch.unique(target)
            classes = classes[classes != ignore_index].long()
            class_labels.append(classes)
            if classes.numel():
                masks = torch.stack([(target == label) for label in classes]).float()
            else:
                masks = target.new_empty((0, *target.shape), dtype=torch.float32)
            mask_labels.append(masks)
        return mask_labels, class_labels

    def compute_loss(
        self, images: torch.Tensor, targets: torch.Tensor, ignore_index: int
    ) -> torch.Tensor:
        """Apply the official matching, mask, Dice, and class objectives."""
        masks, classes = self._set_targets(targets, ignore_index)
        return self.model(
            pixel_values=images, mask_labels=masks, class_labels=classes
        ).loss

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images)
        classes = output.class_queries_logits.softmax(-1)[..., : self.num_classes]
        masks = output.masks_queries_logits.sigmoid()
        logits = torch.einsum("bqc,bqhw->bchw", classes, masks)
        return F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )


def build_model(config: Any) -> Mask2Former:
    return Mask2Former(config)
