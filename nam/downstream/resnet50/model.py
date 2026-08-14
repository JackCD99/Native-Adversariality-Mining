"""Torchvision ResNet-50 classifier.

Source: https://github.com/pytorch/vision/tree/main/torchvision/models
Reference: He et al., Deep Residual Learning for Image Recognition, CVPR 2016.
"""

from typing import Any


def build_model(config: Any):
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as error:
        raise ImportError("ResNet-50 requires `pip install torchvision`.") from error
    weights = ResNet50_Weights.IMAGENET1K_V2 if bool(getattr(config, "imagenet_pretrained", True)) else None
    from torch import nn
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, int(config.num_classes))
    return model
