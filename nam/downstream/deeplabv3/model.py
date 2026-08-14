"""DeepLabV3 with a ResNet-50 backbone.

Source: https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation
Reference: Chen et al., Rethinking Atrous Convolution for Semantic Image
Segmentation, 2017.
"""

from typing import Any


def build_model(config: Any):
    """Build torchvision's official DeepLabV3-ResNet50 architecture."""
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50
    except ImportError as error:
        raise ImportError("DeepLabV3 requires `pip install torchvision`.") from error
    return deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=int(config.num_classes),
        aux_loss=bool(getattr(getattr(config, "model", config), "aux_loss", False)),
    )
