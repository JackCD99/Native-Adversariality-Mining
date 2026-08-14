"""timm ViT-S/16 classifier.

Source: https://github.com/huggingface/pytorch-image-models
Reference: Dosovitskiy et al., An Image is Worth 16x16 Words, ICLR 2021.
"""

from typing import Any


def build_model(config: Any):
    try:
        import timm
    except ImportError as error:
        raise ImportError("ViT-S/16 requires `pip install timm`.") from error
    return timm.create_model(
        str(getattr(config, "architecture", "vit_small_patch16_224")),
        pretrained=bool(getattr(config, "imagenet_pretrained", True)),
        num_classes=int(config.num_classes),
    )
