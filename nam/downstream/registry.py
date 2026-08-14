"""Downstream model registry for Table I."""

from importlib import import_module
from typing import Any

from nam.downstream.base import DownstreamAdapter, DownstreamMetadata, TorchModuleAdapter
from nam.downstream.training import load_model_checkpoint


MODULES = {
    "nnunet": "nam.downstream.nnunet.model",
    "swinunet": "nam.downstream.swinunet.model",
    "swinunetr": "nam.downstream.swinunet.model",
    "samed": "nam.downstream.samed.model",
    "deeplabv3": "nam.downstream.deeplabv3.model",
    "mask2former": "nam.downstream.mask2former.model",
    "resnet50": "nam.downstream.resnet50.model",
    "vits16": "nam.downstream.vit_s16.model",
}

OFFICIAL_REPOSITORIES = {
    "nnunet": "https://github.com/MIC-DKFZ/nnUNet",
    "swinunet": "https://github.com/HuCaoFighting/Swin-Unet",
    "swinunetr": "https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR",
    "samed": "https://github.com/hitachinsk/SAMed",
    "deeplabv3": "https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation",
    "mask2former": "https://github.com/facebookresearch/Mask2Former",
    "resnet50": "https://github.com/pytorch/vision/tree/main/torchvision/models",
    "vits16": "https://github.com/huggingface/pytorch-image-models",
}


def build_downstream(config: Any) -> DownstreamAdapter:
    """Build one named downstream adapter."""
    name = str(config.name).lower().replace("-", "").replace("_", "")
    if name not in MODULES:
        raise KeyError(f"Unknown downstream model '{name}'. Available: {sorted(MODULES)}")
    canonical_name = "vit_s16" if name == "vits16" else name
    model = import_module(MODULES[name]).build_model(config)
    adapter = TorchModuleAdapter(model, config)
    adapter.metadata = DownstreamMetadata(
        name=canonical_name,
        official_repository=OFFICIAL_REPOSITORIES[name],
        spatial_dims=int(config.spatial_dims),
    )
    checkpoint = getattr(config, "checkpoint", None)
    if checkpoint:
        checkpoint = str(checkpoint).format(
            model=canonical_name,
            dataset=str(getattr(config, "dataset_name", "dataset")),
            generator=str(getattr(config, "generator_name", "generator")),
        )
        load_model_checkpoint(model, checkpoint)
    return adapter
