"""Shared loader for evaluation-only cross-center polyp datasets."""

from __future__ import annotations

from typing import Any

from nam.data.common import ManifestSegmentationDataset, build_nam_dataloader, config_value, package_root


class EvaluationPolypDataset(ManifestSegmentationDataset):
    """No training split or stochastic augmentation is permitted."""

    def __init__(self, module_file: str, config: Any, split: str, spatial_dims: int) -> None:
        if split != "test":
            raise ValueError("EndoScene, ColonDB, and ETIS are evaluation-only in the paper.")
        if spatial_dims != 2:
            raise ValueError("Cross-center polyp benchmarks contain 2D RGB images.")
        super().__init__(package_root(module_file, config), "test", 2, "RGB", "a colonoscopy image of a polyp",
                         config_value(config, "image_size", (256, 256)), False, {0: 0, 1: 1, 255: 1})


def evaluation_loader(dataset, config):
    return build_nam_dataloader(dataset, int(config_value(config, "batch_size", 8)), shuffle=False, num_workers=int(config_value(config, "num_workers", 4)))
