"""Base and NAM synthetic-set sampling specialized for SiameseDiff."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
from tqdm import tqdm

from nam.config import apply_overrides, load_config
from nam.models import AdversarialityMiner
from nam.objectives import reselect_noise
from nam.utils.monitoring import SamplingMonitor
from nam.utils.seed import seed_everything

_package = __package__ or "nam.diffusion.2D_M2I.siamesediff"
build_adapter = importlib.import_module(f"{_package}.model").build_adapter
build_siamesediff_loader = importlib.import_module(f"{_package}.utils.data").build_siamesediff_loader
_io = importlib.import_module(f"{_package}.utils.io")
prepare_output_directory, save_pair = _io.prepare_output_directory, _io.save_pair


def _load_miner(config: Any, device: torch.device) -> AdversarialityMiner:
    miner = AdversarialityMiner(
        spatial_dims=2,
        in_channels=int(getattr(config.miner, "in_channels", config.diffusion.noise_channels)),
        noise_channels=int(config.diffusion.noise_channels),
        base_channels=int(config.miner.base_channels),
        channel_multipliers=tuple(config.miner.channel_multipliers),
        downsample_levels=int(config.miner.downsample_levels),
        attention_heads=int(config.miner.attention_heads),
        head_channels=int(config.miner.head_channels),
    ).to(device)
    checkpoint_path = Path(config.siamesediff.checkpoints.nam_miner)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"NAM miner checkpoint was not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    miner.load_state_dict(checkpoint["miner"])
    return miner.eval()


@torch.no_grad()
def sample_dataset(config: Any, use_nam: bool = True) -> Path:
    """Generate one deterministic, condition-aligned synthetic training set."""
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    diffusion = build_adapter(config.diffusion)
    diffusion.model.to(device)
    diffusion.freeze()
    settings = config.siamesediff.sampling
    loader = build_siamesediff_loader(
        config.dataset,
        "train",
        int(settings.batch_size),
        int(config.runtime.num_workers),
        False,
    )
    miner = _load_miner(config, device) if use_nam else None
    method = "nam" if use_nam else "base"
    output = prepare_output_directory(settings.output_dir, config.experiment_name, method)
    monitor = SamplingMonitor(output, config, "siamesediff", method)
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))
    budget, written = int(settings.budget), 0
    for batch in tqdm(loader, desc=f"SiameseDiff {method.upper()} sampling"):
        if written >= budget:
            break
        batch = batch.to(device)
        condition = diffusion.prepare_condition(batch)
        probe = diffusion.sample_probe_noise(batch.target.shape[0], generator)
        selected = probe
        if miner is not None:
            score = diffusion.initial_score(probe, condition, float(settings.cfg_scale))
            delta_mean, delta_variance = miner(score.score)
            selected = reselect_noise(
                delta_mean,
                delta_variance,
                variance_bound=float(config.miner.variance_bound),
                generator=generator,
            ).sample
        images, targets = diffusion.sample(
            selected,
            condition,
            int(settings.ddim_steps),
            float(settings.cfg_scale),
        )
        available = min(images.shape[0], budget - written)
        monitor.log_batch(
            images[:available], targets[:available], probe[:available],
            selected[:available], batch.sample_id[:available],
        )
        for index, sample_id in enumerate(batch.sample_id[:available]):
            save_pair(
                output,
                sample_id,
                images[index],
                targets[index],
                {
                    "sample_id": sample_id,
                    "method": method,
                    "seed": int(config.runtime.seed),
                    "diffusion_checkpoint": str(config.diffusion.checkpoint),
                    "miner_checkpoint": str(config.siamesediff.checkpoints.nam_miner) if use_nam else None,
                    "ddim_steps": int(settings.ddim_steps),
                    "eta": 0.0,
                    "cfg_scale": float(settings.cfg_scale),
                },
            )
            written += 1
    if written != budget:
        raise RuntimeError(f"Requested {budget} samples, but generated {written}.")
    monitor.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Base or NAM data with SiameseDiff.")
    parser.add_argument("--config", default="configs/table1_2d.yaml")
    parser.add_argument("--method", choices=("base", "nam"), default="nam")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = apply_overrides(load_config(arguments.config), arguments.set)
    sample_dataset(configuration, use_nam=arguments.method == "nam")
