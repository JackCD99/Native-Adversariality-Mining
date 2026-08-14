"""Unified downstream-model interface for NAM and Table I evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class DownstreamMetadata:
    name: str
    official_repository: str
    spatial_dims: int


class DownstreamAdapter(ABC):
    """Bridge a task model to the NAM reward interface."""

    metadata: DownstreamMetadata

    def __init__(self, model: nn.Module, config: Any) -> None:
        self.model = model
        self.config = config

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        self.model.train()
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)

    @abstractmethod
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits in BC, BCHW, or BCDHW format."""

    def load_checkpoint(self, path: str) -> None:
        """Load either a raw state dict or a common nested checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict", "network_weights", "net"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        cleaned = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch for {self.metadata.name}: missing={missing}, "
                f"unexpected={unexpected}"
            )


class TorchModuleAdapter(DownstreamAdapter):
    """Default adapter for models whose forward pass returns logits."""

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        if isinstance(output, dict):
            for key in ("out", "logits", "pred"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError("The downstream bridge did not return a logits tensor.")
        return output
