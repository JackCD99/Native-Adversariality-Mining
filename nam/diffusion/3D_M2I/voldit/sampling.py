"""Fixed-budget deterministic Base/NAM sampling for VolDiT volumes."""

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
from nam.models import ResUNet3DMiner
from nam.objectives import reselect_noise
from nam.utils.monitoring import SamplingMonitor
from nam.utils.seed import build_sampling_generators, resolve_stage_seed, sampling_output_root, seed_everything

_package = __package__ or "nam.diffusion.3D_M2I.voldit"
build_adapter = importlib.import_module(f"{_package}.model").build_adapter
build_voldit_loader = importlib.import_module(f"{_package}.utils.data").build_voldit_loader
VolDiTConditionAugmentation = importlib.import_module(f"{_package}.utils.augmentations").VolDiTConditionAugmentation
_io = importlib.import_module(f"{_package}.utils.io")


def _load_miner(config: Any, device: torch.device) -> ResUNet3DMiner:
    miner = ResUNet3DMiner(
        in_channels=int(getattr(config.miner, "in_channels", config.diffusion.noise_channels)),
        noise_channels=int(config.diffusion.noise_channels), base_channels=int(config.miner.base_channels),
        channel_multipliers=tuple(config.miner.channel_multipliers), attention_heads=int(config.miner.attention_heads),
        head_channels=int(getattr(config.miner, "head_channels", 0)) or None,
    ).to(device)
    path = Path(config.voldit.checkpoints.nam_miner)
    if not path.is_file():
        raise FileNotFoundError(f"VolDiT NAM miner checkpoint was not found: {path}")
    miner.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["miner"])
    return miner.eval()


@torch.no_grad()
def sample_dataset(config: Any, use_nam: bool = True) -> Path:
    """Synthesize exactly the configured Table-I budget unless set to zero."""
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    sampling_seed = resolve_stage_seed(config, "sampling")
    seed_everything(sampling_seed, bool(config.runtime.deterministic))
    adapter = build_adapter(config.diffusion)
    adapter.model.to(device)
    settings = config.voldit.sampling
    loader = build_voldit_loader(config.dataset, "train", int(settings.batch_size), int(config.runtime.num_workers), False)
    miner = _load_miner(config, device) if use_nam else None
    method = "nam" if use_nam else "base"
    output = _io.prepare_output_directory(
        sampling_output_root(settings.output_dir, sampling_seed), config.experiment_name, method
    )
    monitor = SamplingMonitor(output, config, "voldit", method)
    probe_generator, reselection_generator = build_sampling_generators(device, sampling_seed)
    budget, written = int(settings.budget), 0
    augmentation = VolDiTConditionAugmentation(
        float(settings.flip_probability), tuple(settings.scale_range)
    ) if bool(settings.augment_conditions) else None
    for batch in tqdm(loader, desc=f"VolDiT {method.upper()} sampling"):
        if budget > 0 and written >= budget:
            break
        batch = (augmentation(batch) if augmentation is not None else batch).to(device)
        condition = adapter.prepare_condition(batch)
        probe = adapter.sample_probe_noise(batch.target.shape[0], probe_generator)
        selected = probe
        if miner is not None:
            score = adapter.initial_score(probe, condition, float(settings.cfg_scale))
            mean, variance = miner(score.score)
            selected = reselect_noise(mean, variance, float(config.miner.variance_bound), reselection_generator).sample
        images, targets = adapter.sample(selected, condition, int(settings.ddim_steps), float(settings.cfg_scale))
        monitor.log_batch(images, targets, probe, selected, batch.sample_id[: images.shape[0]])
        available = images.shape[0] if budget <= 0 else min(images.shape[0], budget - written)
        for index in range(available):
            sample_id = batch.sample_id[index]
            _io.save_volume_pair(output, sample_id, images[index], targets[index], {
                "sample_id": sample_id, "method": method, "seed": sampling_seed,
                "diffusion_checkpoint": str(config.diffusion.checkpoint),
                "miner_checkpoint": str(config.voldit.checkpoints.nam_miner) if use_nam else None,
                "ddim_steps": int(settings.ddim_steps), "eta": 0.0,
                "prompt": condition.extras["prompts"][index],
            })
            written += 1
    if budget > 0 and written != budget:
        raise RuntimeError(f"Requested {budget} volumes, but only {written} conditions were available.")
    monitor.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Base or NAM volumes with VolDiT.")
    parser.add_argument("--config", default="configs/voldit_3d.yaml")
    parser.add_argument("--method", choices=("base", "nam"), default="nam")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    sample_dataset(apply_overrides(load_config(arguments.config), arguments.set), arguments.method == "nam")
