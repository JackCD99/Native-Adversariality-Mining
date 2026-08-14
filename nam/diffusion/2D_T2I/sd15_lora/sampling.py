"""Fixed-budget Base and NAM sampling for class-conditional SD-v1.5 LoRA."""

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
from nam.models import AdversarialityMiner, reference_miner_configuration
from nam.objectives import reselect_noise
from nam.utils.monitoring import SamplingMonitor
from nam.utils.seed import seed_everything

_package = __package__ or "nam.diffusion.2D_T2I.sd15_lora"
build_adapter = importlib.import_module(f"{_package}.model").build_adapter
build_loader = importlib.import_module(f"{_package}.utils.data").build_loader
_io = importlib.import_module(f"{_package}.utils.io")
prepare_output_directory, save_sample = _io.prepare_output_directory, _io.save_sample


def _load_miner(config: Any, device: torch.device) -> AdversarialityMiner:
    defaults = reference_miner_configuration(4)
    miner = AdversarialityMiner(
        spatial_dims=2,
        in_channels=4,
        noise_channels=4,
        base_channels=int(getattr(config.miner, "base_channels", defaults["base_channels"])),
        channel_multipliers=tuple(
            getattr(config.miner, "channel_multipliers", defaults["channel_multipliers"])
        ),
        downsample_levels=int(
            getattr(config.miner, "downsample_levels", defaults["downsample_levels"])
        ),
        attention_heads=int(getattr(config.miner, "attention_heads", 4)),
        head_channels=int(getattr(config.miner, "head_channels", defaults["head_channels"])),
    ).to(device)
    path = Path(config.sd15_lora.checkpoints.nam_miner)
    if not path.is_file():
        raise FileNotFoundError(f"SD-v1.5 LoRA NAM checkpoint was not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    miner.load_state_dict(payload.get("miner", payload), strict=True)
    return miner.eval().requires_grad_(False)


@torch.no_grad()
def sample_dataset(config: Any, use_nam: bool = True) -> Path:
    settings = config.sd15_lora.sampling
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    diffusion = build_adapter(config.diffusion)
    diffusion.model.to(device)
    diffusion.freeze()
    loader = build_loader(
        config.dataset, "train", int(settings.batch_size), int(config.runtime.num_workers), False
    )
    miner = _load_miner(config, device) if use_nam else None
    method = "nam" if use_nam else "base"
    output = prepare_output_directory(settings.output_dir, config.experiment_name, method)
    monitor = SamplingMonitor(output, config, "sd15_lora", method)
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))
    budget, written = int(settings.budget), 0
    while written < budget:
        for batch in tqdm(loader, desc=f"SD-v1.5 LoRA {method.upper()}", leave=False):
            batch = batch.to(device)
            condition = diffusion.prepare_condition(batch)
            probe = diffusion.sample_probe_noise(batch.target.shape[0], generator)
            selected = probe
            if miner is not None:
                score = diffusion.initial_score(probe, condition, float(settings.cfg_scale))
                mean, variance = miner(score.score)
                selected = reselect_noise(
                    mean, variance, float(config.miner.variance_bound), generator
                ).sample
            images, targets = diffusion.sample(
                selected, condition, int(settings.ddim_steps), float(settings.cfg_scale)
            )
            available = min(images.shape[0], budget - written)
            monitor.log_batch(
                images[:available], targets[:available], probe[:available],
                selected[:available], batch.sample_id[:available],
            )
            for index in range(available):
                sample_id = f"{batch.sample_id[index]}-{written:06d}"
                save_sample(
                    output, sample_id, images[index], targets[index],
                    {
                        "sample_id": sample_id,
                        "method": method,
                        "seed": int(config.runtime.seed),
                        "prompt": condition.extras["prompts"][index],
                        "class_id": int(targets[index].item()),
                        "base_model": str(config.diffusion.base_model),
                        "lora_checkpoint": str(config.diffusion.checkpoint),
                        "miner_checkpoint": str(config.sd15_lora.checkpoints.nam_miner)
                        if use_nam else None,
                        "ddim_steps": int(settings.ddim_steps),
                        "eta": 0.0,
                    },
                )
                written += 1
            if written >= budget:
                break
        if len(loader) == 0:
            raise RuntimeError("The classification condition loader is empty.")
    monitor.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sd15_lora_pneumoniamnist.yaml")
    parser.add_argument("--method", choices=("base", "nam"), default="nam")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    sample_dataset(
        apply_overrides(load_config(arguments.config), arguments.set),
        use_nam=arguments.method == "nam",
    )
