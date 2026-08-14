"""ETIS-LaribPolypDB evaluation-only polyp dataset."""

from typing import Any
from nam.data.evaluation_polyp import EvaluationPolypDataset, evaluation_loader


class ETISDataset(EvaluationPolypDataset):
    def __init__(self, config: Any, split: str = "test", spatial_dims: int = 2) -> None:
        super().__init__(__file__, config, split, spatial_dims)


def build_dataset(config: Any, split: str = "test", spatial_dims: int = 2):
    return ETISDataset(config, split, spatial_dims)


def build_dataloader(config: Any, split: str = "test", spatial_dims: int = 2):
    return evaluation_loader(build_dataset(config, split, spatial_dims), config)
