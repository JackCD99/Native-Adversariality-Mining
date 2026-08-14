"""Extract downstream bottleneck features and create a joint t-SNE figure."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from nam.config import apply_overrides, load_config
from nam.data import build_dataset, collate_medical_batch
from nam.downstream import build_downstream
from nam.evaluation.Vis.common import import_plotting, save_figure
from nam.utils.seed import seed_everything


def _resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    """Resolve named children and integer ModuleList indices from a dotted path."""
    current: Any = root
    for token in dotted_path.split("."):
        current = current[int(token)] if token.isdigit() else getattr(current, token)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Bottleneck path '{dotted_path}' did not resolve to a torch module.")
    return current


def _default_bottleneck(model: nn.Module) -> tuple[nn.Module, str]:
    """Select the final encoder output for the three official downstream families."""
    if hasattr(model, "image_encoder"):
        image_encoder = model.image_encoder
        if hasattr(image_encoder, "neck"):
            return image_encoder.neck, "image_encoder.neck"
        return image_encoder, "image_encoder"
    if hasattr(model, "encoder"):
        encoder = model.encoder
        if isinstance(encoder, nn.ModuleList):
            return encoder[-1], f"encoder.{len(encoder) - 1}"
        return encoder, "encoder"
    if hasattr(model, "layer4"):
        return model.layer4, "layer4"
    if hasattr(model, "blocks"):
        return model.blocks[-1], f"blocks.{len(model.blocks) - 1}"
    raise AttributeError("Set visualization.tsne.bottleneck_layer for this downstream model.")


def _select_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = next(iter(output.values()))
    if isinstance(output, (tuple, list)):
        output = output[-1]
    if not torch.is_tensor(output):
        raise TypeError("The bottleneck module did not return a tensor or tensor collection.")
    return output


def _global_pool(feature: torch.Tensor, input_batch: int, channel_last: bool) -> torch.Tensor:
    """Convert a spatial bottleneck map to one vector per input sample."""
    if feature.ndim < 2:
        raise ValueError("Bottleneck features must include batch and feature dimensions.")
    if feature.ndim == 2:
        pooled = feature
    elif channel_last:
        pooled = feature.mean(dim=tuple(range(1, feature.ndim - 1)))
    else:
        pooled = feature.mean(dim=tuple(range(2, feature.ndim)))
    # Slice-wise 3D models expose B*D encoder features; average slices back to B.
    if pooled.shape[0] != input_batch:
        if pooled.shape[0] % input_batch:
            raise ValueError("Captured feature batch cannot be mapped to the input batch.")
        pooled = pooled.reshape(input_batch, pooled.shape[0] // input_batch, -1).mean(1)
    return pooled


@torch.inference_mode()
def extract_bottleneck_features(config: Any, spatial_dims: int) -> dict[str, Any]:
    """Extract unchanged dataset samples through one frozen downstream checkpoint."""
    settings = getattr(getattr(config, "visualization", {}), "tsne", {})
    device_name = str(getattr(config.runtime, "device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    adapter = build_downstream(config.downstream)
    adapter.model.to(device)
    adapter.freeze()

    explicit_layer = str(settings.get("bottleneck_layer", ""))
    if explicit_layer:
        module, layer_name = _resolve_module(adapter.model, explicit_layer), explicit_layer
    else:
        module, layer_name = _default_bottleneck(adapter.model)
    configured_channel_last = settings.get("channel_last", None)
    channel_last = (
        bool(configured_channel_last)
        if configured_channel_last is not None
        else layer_name.startswith("blocks.")
    )
    captured: list[Any] = []

    def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captured.append(output)

    handle = module.register_forward_hook(capture)
    groups = settings.get("groups", [])
    if not groups:
        groups = [{"label": "Real", "dataset": "dataset", "split": "train"}]
        if hasattr(config, "synthetic_dataset"):
            groups.append({"label": "Synthetic", "dataset": "synthetic_dataset", "split": "train"})

    features: list[torch.Tensor] = []
    records: list[dict[str, str]] = []
    try:
        for group in groups:
            dataset_key = str(group.get("dataset", "dataset"))
            split = str(group.get("split", "train"))
            label = str(group.get("label", dataset_key))
            dataset = build_dataset(getattr(config, dataset_key), split, spatial_dims)
            limit = int(group.get("max_samples", settings.get("max_samples_per_group", 500)))
            if limit > 0:
                dataset = Subset(dataset, range(min(limit, len(dataset))))
            loader = DataLoader(
                dataset,
                batch_size=int(settings.get("batch_size", 8 if spatial_dims == 2 else 1)),
                shuffle=False,
                num_workers=int(getattr(config.runtime, "num_workers", 0)),
                collate_fn=collate_medical_batch,
                pin_memory=device.type == "cuda",
            )
            for batch in tqdm(loader, desc=f"Bottleneck features: {label}"):
                if batch.image is None:
                    raise ValueError("t-SNE extraction requires dataset items to provide 'image'.")
                captured.clear()
                images = batch.image.to(device, non_blocking=True)
                adapter.logits(images)
                if not captured:
                    raise RuntimeError(f"Bottleneck hook '{layer_name}' was not executed.")
                feature = _select_tensor(captured[-1])
                pooled = _global_pool(feature, images.shape[0], channel_last).float().cpu()
                features.append(pooled)
                metadata = batch.metadata.get("items", [{}] * len(batch.sample_id))
                for index, sample_id in enumerate(batch.sample_id):
                    item = metadata[index] if index < len(metadata) else {}
                    path = next((str(item[key]) for key in ("path", "image_path", "image", "source") if item.get(key)), "")
                    records.append({"id": str(sample_id), "path": path, "group": label})
    finally:
        handle.remove()

    matrix = torch.cat(features, dim=0) if features else torch.empty((0, 0))
    if matrix.shape[0] != len(records):
        raise RuntimeError("Feature and metadata counts differ.")
    return {"features": matrix, "records": records, "layer": layer_name, "model": adapter.metadata.name}


def run_tsne(config: Any, spatial_dims: int) -> dict[str, Any]:
    """Run deterministic joint t-SNE and save coordinates, features, and figures."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import normalize
    except ImportError as error:
        raise ImportError("Install visualization dependencies with: pip install -e '.[visualization]'") from error

    settings = getattr(getattr(config, "visualization", {}), "tsne", {})
    seed = int(settings.get("seed", getattr(config.runtime, "seed", 42)))
    seed_everything(seed, deterministic=True)
    extracted = extract_bottleneck_features(config, spatial_dims)
    tensor = extracted["features"]
    if tensor.shape[0] < 3:
        raise ValueError("t-SNE requires at least three extracted samples.")
    features = normalize(tensor.numpy(), norm="l2", axis=1)
    pca_dimensions = min(int(settings.get("pca_dimensions", 50)), features.shape[0] - 1, features.shape[1])
    if 2 <= pca_dimensions < features.shape[1]:
        features = PCA(n_components=pca_dimensions, random_state=seed).fit_transform(features)
    requested_perplexity = float(settings.get("perplexity", 30.0))
    perplexity = min(requested_perplexity, max(1.0, (features.shape[0] - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)

    output_dir = Path(str(settings.get("output_dir", "outputs/evaluation/Vis/tsne")))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"features": tensor, "records": extracted["records"], "layer": extracted["layer"], "model": extracted["model"]},
        output_dir / "bottleneck_features.pt",
    )
    coordinate_path = output_dir / "tsne_coordinates.csv"
    with coordinate_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "path", "group", "tsne_x", "tsne_y"])
        writer.writeheader()
        for record, point in zip(extracted["records"], embedding):
            writer.writerow({**record, "tsne_x": float(point[0]), "tsne_y": float(point[1])})

    plt = import_plotting()
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    group_names = list(dict.fromkeys(record["group"] for record in extracted["records"]))
    for group_name in group_names:
        indices = [index for index, record in enumerate(extracted["records"]) if record["group"] == group_name]
        axis.scatter(embedding[indices, 0], embedding[indices, 1], s=17, alpha=0.55, label=group_name, edgecolors="none")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(frameon=False)
    axis.grid(alpha=0.15)
    figure_paths = save_figure(figure, output_dir / "tsne.pdf")
    plt.close(figure)
    return {
        "samples": len(extracted["records"]),
        "feature_dimensions": int(tensor.shape[1]),
        "bottleneck_layer": extracted["layer"],
        "perplexity": perplexity,
        "coordinates": str(coordinate_path),
        "figures": figure_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize downstream bottleneck features with t-SNE.")
    parser.add_argument("--config", default="configs/table1_2d.yaml")
    parser.add_argument("--spatial-dims", type=int, choices=(2, 3), default=2)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(run_tsne(apply_overrides(load_config(arguments.config), arguments.set), arguments.spatial_dims))
