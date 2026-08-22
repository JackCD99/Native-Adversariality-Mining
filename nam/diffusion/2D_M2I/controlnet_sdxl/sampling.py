"""Fixed-budget Base and NAM sampling with ControlNet-SDXL."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nam.config import apply_overrides, load_config
from nam.models import AdversarialityMiner
from nam.objectives import reselect_noise
from nam.utils.monitoring import SamplingMonitor
from nam.utils.seed import build_sampling_generators, resolve_stage_seed, sampling_output_root, seed_everything

_package = __package__ or "nam.diffusion.2D_M2I.controlnet_sdxl"
build_adapter = importlib.import_module(f"{_package}.model").build_adapter
build_loader = importlib.import_module(f"{_package}.utils.data").build_loader
_io = importlib.import_module(f"{_package}.utils.io")
prepare_output_directory, save_pair = _io.prepare_output_directory, _io.save_pair


def _load_miner(config: Any, device: torch.device) -> AdversarialityMiner:
    miner = AdversarialityMiner(
        spatial_dims=2, in_channels=4, noise_channels=4,
        base_channels=int(config.miner.base_channels),
        channel_multipliers=tuple(config.miner.channel_multipliers),
        downsample_levels=int(config.miner.downsample_levels),
        attention_heads=int(config.miner.attention_heads),
        head_channels=int(config.miner.head_channels),
    ).to(device)
    path = Path(config.controlnet_sdxl.checkpoints.nam_miner)
    if not path.is_file():
        raise FileNotFoundError(f"ControlNet-SDXL NAM checkpoint was not found: {path}")
    miner.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["miner"])
    return miner.eval()


@torch.no_grad()
def sample_dataset(config: Any, use_nam: bool = True) -> Path:
    settings = config.controlnet_sdxl.sampling
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    sampling_seed = resolve_stage_seed(config, "sampling")
    seed_everything(sampling_seed, bool(config.runtime.deterministic))
    diffusion = build_adapter(config.diffusion)
    diffusion.model.to(device)
    diffusion.freeze()
    loader = build_loader(
        config.dataset, "train", int(settings.batch_size), int(config.runtime.num_workers), False
    )
    miner = _load_miner(config, device) if use_nam else None
    method = "nam" if use_nam else "base"
    output = prepare_output_directory(
        sampling_output_root(settings.output_dir, sampling_seed), config.experiment_name, method
    )
    monitor = SamplingMonitor(output, config, "controlnet_sdxl", method)
    probe_generator, reselection_generator = build_sampling_generators(device, sampling_seed)
    budget, written = int(settings.budget), 0
    for batch in tqdm(loader, desc=f"ControlNet-SDXL {method.upper()}"):
        if written >= budget:
            break
        batch = batch.to(device)
        condition = diffusion.prepare_condition(batch)
        probe = diffusion.sample_probe_noise(batch.target.shape[0], probe_generator)
        selected = probe
        if miner is not None:
            score = diffusion.initial_score(probe, condition, float(settings.cfg_scale))
            mean, variance = miner(score.score)
            selected = reselect_noise(
                mean, variance, float(config.miner.variance_bound), reselection_generator
            ).sample
        images, targets = diffusion.sample(
            selected, condition, int(settings.ddim_steps), float(settings.cfg_scale)
        )
        monitor.log_batch(images, targets, probe, selected, batch.sample_id[: images.shape[0]])
        available = min(images.shape[0], budget - written)
        for index in range(available):
            sample_id = f"{batch.sample_id[index]}-{written:06d}"
            save_pair(
                output, sample_id, images[index], targets[index],
                {
                    "sample_id": sample_id, "method": method,
                    "seed": sampling_seed,
                    "prompt": condition.extras["prompts"][index],
                    "base_model": str(config.diffusion.base_model),
                    "controlnet_checkpoint": str(config.diffusion.checkpoint),
                    "miner_checkpoint": str(config.controlnet_sdxl.checkpoints.nam_miner) if use_nam else None,
                    "ddim_steps": int(settings.ddim_steps), "eta": 0.0,
                },
            )
            written += 1
    if written != budget:
        raise RuntimeError(f"Requested {budget} samples, but only {written} conditions were available.")
    monitor.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample VOC/SBD pairs with ControlNet-SDXL.")
    parser.add_argument("--config", default="configs/controlnet_sdxl_voc.yaml")
    parser.add_argument("--method", choices=("base", "nam"), default="nam")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    sample_dataset(
        apply_overrides(load_config(arguments.config), arguments.set),
        use_nam=arguments.method == "nam",
    )
