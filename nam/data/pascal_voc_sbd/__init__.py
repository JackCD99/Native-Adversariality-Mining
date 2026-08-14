"""PASCAL VOC and SBD semantic-segmentation dataset package."""

from nam.data.pascal_voc_sbd.dataset import build_dataloader, build_dataset

__all__ = ["build_dataset", "build_dataloader"]
