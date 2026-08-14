"""Combined CVC-ClinicDB and Kvasir-SEG dataset."""

from nam.data.polyps.dataset import PolypsDataset, build_dataloader, build_dataset

__all__ = ["PolypsDataset", "build_dataset", "build_dataloader"]
