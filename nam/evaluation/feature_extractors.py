"""Modality-specific feature encoders for publication FID evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class NormalizedFeatureEncoder(nn.Module):
    """Resize, normalize, and expose a backbone's pooled feature vector."""

    def __init__(
        self,
        backbone: nn.Module,
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = int(image_size)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("A 2D FID encoder expects BCHW input tensors.")
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] != 3:
            raise ValueError(f"FID expects one or three image channels, received {images.shape[1]}.")
        images = images.float()
        if images.min() < 0.0:
            images = images.add(1.0).mul(0.5)
        images = images.clamp(0.0, 1.0)
        images = F.interpolate(
            images, (self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        return self.backbone((images - self.mean) / self.std)


class TorchXRayVisionFeatureEncoder(nn.Module):
    """Expose pooled DenseNet-121 features with TorchXRayVision preprocessing."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("The chest X-ray FID encoder expects BCHW tensors.")
        images = images.float()
        if images.shape[1] == 3:
            images = images.mean(1, keepdim=True)
        elif images.shape[1] != 1:
            raise ValueError(
                f"Chest X-ray FID expects one or three channels, received {images.shape[1]}."
            )
        if images.min() < 0.0:
            images = images.add(1.0).mul(0.5)
        images = F.interpolate(images.clamp(0.0, 1.0), (224, 224), mode="bilinear", align_corners=False)
        features = self.model.features(images.mul(2048.0).sub(1024.0))
        features = F.relu(features, inplace=False)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported feature checkpoint format: {path}")
    cleaned = {}
    for key, value in payload.items():
        key = str(key).removeprefix("module.").removeprefix("model.").removeprefix("backbone.")
        if key.startswith("fc.") or key.startswith("classifier."):
            continue
        if torch.is_tensor(value):
            cleaned[key] = value
    return cleaned


def _inception_v3() -> nn.Module:
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError as error:
        raise ImportError("Inception FID requires `pip install torchvision`.") from error
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval().requires_grad_(False)
    return NormalizedFeatureEncoder(
        model,
        image_size=299,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def _radimagenet_resnet50(checkpoint: str) -> nn.Module:
    try:
        from torchvision.models import resnet50
    except ImportError as error:
        raise ImportError("RadImageNet FID requires `pip install torchvision`.") from error
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"RadImageNet checkpoint not found: {path}. "
            "Follow pretrained_weights/fid/README.md before running medical FID."
        )
    model = resnet50(weights=None)
    state = _checkpoint_state(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    if loaded < 50:
        raise RuntimeError(
            f"The RadImageNet checkpoint matched only {loaded} tensors; "
            f"unexpected keys: {unexpected[:8]}."
        )
    model.fc = nn.Identity()
    model.eval().requires_grad_(False)
    return NormalizedFeatureEncoder(
        model,
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def _torchxrayvision_densenet121(weights: str) -> nn.Module:
    try:
        import torchxrayvision as xrv
    except ImportError as error:
        raise ImportError(
            "Chest X-ray FID requires `pip install torchxrayvision`."
        ) from error
    model = xrv.models.DenseNet(weights=weights)
    model.eval().requires_grad_(False)
    return TorchXRayVisionFeatureEncoder(model)


def build_feature_extractor(config: Any) -> nn.Module:
    """Build the encoder selected by ``fid.encoder``.

    Supported encoders follow Supplementary Section C.2: Inception-v3 for RGB,
    TorchXRayVision DenseNet-121 for chest radiographs, and RadImageNet-pretrained
    ResNet-50 for MRI, CT, and CTA images.
    """
    name = str(getattr(config, "encoder", "inception_v3")).lower().replace("-", "_")
    if name in {"inception", "inception_v3"}:
        return _inception_v3()
    if name in {"radimagenet", "radimagenet_resnet50"}:
        return _radimagenet_resnet50(str(getattr(config, "checkpoint", "")))
    if name in {"torchxrayvision", "torchxrayvision_densenet121", "xray_densenet121"}:
        return _torchxrayvision_densenet121(
            str(getattr(config, "weights", "densenet121-res224-all"))
        )
    raise KeyError(
        f"Unknown FID encoder '{name}'. Available: inception_v3, "
        "torchxrayvision_densenet121, radimagenet_resnet50."
    )
