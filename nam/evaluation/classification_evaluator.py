"""Held-out accuracy, balanced accuracy, and specificity evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from nam.downstream import build_downstream
from nam.downstream.classification_training import classification_metrics
from nam.downstream.training import build_loader


@torch.no_grad()
def evaluate_classification(config: Any) -> dict[str, float]:
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    adapter = build_downstream(config.downstream)
    adapter.model.to(device)
    adapter.freeze()
    loader = build_loader(
        config.dataset, "test", 2, int(config.evaluation.batch_size),
        int(config.runtime.num_workers), False,
    )
    result = classification_metrics(
        adapter.model, loader, device, int(config.downstream.num_classes)
    )
    result["samples"] = len(loader.dataset)
    output = Path(config.evaluation.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
