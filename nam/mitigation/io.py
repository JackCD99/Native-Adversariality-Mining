"""Stable metadata writer shared by mitigation runners."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

import torch

from nam.mitigation.base import MitigationCandidate


def save_manifest(candidates: Iterable[MitigationCandidate], destination: str | Path) -> Path:
    """Save auditable acceptance/fallback diagnostics for every output slot."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("sample_id", "score", "accepted", "trials", "metadata")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({
                "sample_id": candidate.sample_id,
                "score": candidate.score,
                "accepted": int(candidate.accepted),
                "trials": candidate.trials,
                "metadata": json.dumps(candidate.metadata, ensure_ascii=False),
            })
    return path


def save_tensor_pairs(candidates: Iterable[MitigationCandidate], destination: str | Path) -> Path:
    """Save lossless tensors without imposing image-format assumptions."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates):
        safe = re.sub(r"[^\w.-]+", "_", candidate.sample_id).strip("._") or "sample"
        torch.save({
            "image": candidate.image.detach().cpu(),
            "target": None if candidate.target is None else candidate.target.detach().cpu(),
            "metadata": candidate.metadata,
        }, root / f"{index:06d}_{safe}.pt")
    return root
