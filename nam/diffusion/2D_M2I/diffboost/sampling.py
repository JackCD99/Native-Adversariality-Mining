"""Fixed-budget Base and NAM synthetic-set sampling for DiffBoost."""

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

_package = __package__ or "nam.diffusion.2D_M2I.diffboost"
build_adapter = importlib.import_module(f"{_package}.model").build_adapter
build_diffboost_loader = importlib.import_module(f"{_package}.utils.data").build_diffboost_loader
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
    checkpoint_path = Path(config.diffboost.checkpoints.nam_miner)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"NAM miner checkpoint was not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    miner.load_state_dict(checkpoint["miner"])
    return miner.eval()


@torch.no_grad()
def sample_dataset(config: Any, use_nam: bool = True) -> Path:
    """Generate one deterministic, condition-aligned DiffBoost dataset."""
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    diffusion = build_adapter(config.diffusion)
    diffusion.model.to(device)
    diffusion.freeze()
    settings = config.diffboost.sampling
    loader = build_diffboost_loader(
        config.dataset, "train", int(settings.batch_size), int(config.runtime.num_workers), False
    )
    miner = _load_miner(config, device) if use_nam else None
    method = "nam" if use_nam else "base"
    output = prepare_output_directory(settings.output_dir, config.experiment_name, method)
    monitor = SamplingMonitor(output, config, "diffboost", method)
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))
    budget = int(getattr(settings, "budget", 0))
    written = 0
    for batch in tqdm(loader, desc=f"DiffBoost {method.upper()} sampling"):
        if budget > 0 and written >= budget:
            break
        batch = batch.to(device)
        condition = diffusion.prepare_condition(batch)
        probe = diffusion.sample_probe_noise(batch.target.shape[0], generator)
        selected = probe
        if miner is not None:
            score = diffusion.initial_score(probe, condition, float(settings.cfg_scale))
            delta_mean, delta_variance = miner(score.score)
            selected = reselect_noise(
                delta_mean, delta_variance,
                variance_bound=float(config.miner.variance_bound), generator=generator,
            ).sample
        images, targets = diffusion.sample(
            selected, condition, int(settings.ddim_steps), float(settings.cfg_scale)
        )
        monitor.log_batch(images, targets, probe, selected, batch.sample_id)
        available = images.shape[0] if budget <= 0 else min(images.shape[0], budget - written)
        for index in range(available):
            sample_id = batch.sample_id[index]
            save_pair(
                output, sample_id, images[index], targets[index],
                {
                    "sample_id": sample_id, "method": method, "seed": int(config.runtime.seed),
                    "diffusion_checkpoint": str(config.diffusion.checkpoint),
                    "miner_checkpoint": str(config.diffboost.checkpoints.nam_miner) if use_nam else None,
                    "ddim_steps": int(settings.ddim_steps), "eta": 0.0,
                    "cfg_scale": float(settings.cfg_scale),
                    "prompt": condition.extras["prompts"][index],
                    "augmentation_prompt": condition.extras["augmentation_prompts"][index],
                },
            )
            written += 1
    if budget > 0 and written != budget:
        raise RuntimeError(f"Requested {budget} samples, but the dataset supplied only {written} conditions.")
    monitor.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Base or NAM data with DiffBoost.")
    parser.add_argument("--config", default="configs/diffboost_2d.yaml")
    parser.add_argument("--method", choices=("base", "nam"), default="nam")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = apply_overrides(load_config(arguments.config), arguments.set)
    sample_dataset(configuration, use_nam=arguments.method == "nam")
