"""MMWHS modality-aware whole-heart axial segmentation dataset."""

from __future__ import annotations

from typing import Any

from nam.data.common import ManifestSegmentationDataset, build_nam_dataloader, config_value, package_root


CLASS_NAMES = ("background", "ascending aorta", "left atrium", "left ventricle", "myocardium", "pulmonary artery", "right atrium", "right ventricle")
RAW_LABEL_MAP = {0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7}


def _prompt(modality: str):
    def build(target, item) -> str:
        names = [CLASS_NAMES[index] for index in range(1, len(CLASS_NAMES)) if (target == index).any()]
        return f"a {modality} imaging of [{', '.join(names or CLASS_NAMES[1:])}]"
    return build


class MMWHSDataset(ManifestSegmentationDataset):
    def __init__(self, config: Any, split: str = "train", spatial_dims: int = 2) -> None:
        if spatial_dims != 2:
            raise ValueError("The paper evaluates MMWHS using axial 2D slices.")
        modality = str(config_value(config, "modality", "CT")).upper()
        if modality not in {"CT", "MRI"}:
            raise ValueError("MMWHS modality must be CT or MRI.")
        super().__init__(package_root(__file__, config), split, 2, modality, _prompt(modality),
                         config_value(config, "image_size", (256, 256)), config_value(config, "augment", True),
                         label_map=RAW_LABEL_MAP if config_value(config, "raw_labels", False) else None,
                         manifest_name=f"{modality.lower()}_{split}.list")

def build_dataset(config: Any, split: str = "train", spatial_dims: int = 2) -> MMWHSDataset:
    return MMWHSDataset(config, split, spatial_dims)


def build_dataloader(config: Any, split: str = "train", spatial_dims: int = 2):
    return build_nam_dataloader(build_dataset(config, split, spatial_dims), int(config_value(config, "batch_size", 8)), num_workers=int(config_value(config, "num_workers", 4)))
