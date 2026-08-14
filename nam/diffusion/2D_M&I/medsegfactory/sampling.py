"""Fixed-budget Base/NAM deterministic DDIM sampling for MedSegFactory."""

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
from nam.models import MedSegFactoryDualMiner
from nam.objectives import reselect_noise
from nam.utils.monitoring import SamplingMonitor
from nam.utils.seed import seed_everything

_package = __package__ or "nam.diffusion.2D_M&I.medsegfactory"
build_adapter = importlib.import_module(f"{_package}.model").build_adapter
build_loader = importlib.import_module(f"{_package}.utils.data").build_loader
_io = importlib.import_module(f"{_package}.utils.io")


def _miner(config: Any, device: torch.device) -> MedSegFactoryDualMiner:
    model = MedSegFactoryDualMiner().to(device)
    path = Path(config.medsegfactory.checkpoints.nam_miner)
    if not path.is_file():
        raise FileNotFoundError(f"MedSegFactory NAM checkpoint was not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["miner"], strict=True)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def sample_dataset(config: Any, use_nam: bool = True) -> Path:
    device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config.runtime.seed), bool(config.runtime.deterministic))
    settings = config.medsegfactory.sampling
    adapter = build_adapter(config.diffusion)
    adapter.model.to(device)
    adapter.freeze()
    loader = build_loader(
        config.dataset, "train", int(settings.batch_size),
        int(config.runtime.num_workers), False,
    )
    miner = _miner(config, device) if use_nam else None
    method = "nam" if use_nam else "base"
    output = _io.output_directory(settings.output_dir, config.experiment_name, method)
    monitor = SamplingMonitor(output, config, "medsegfactory", method)
    generator = torch.Generator(device=device).manual_seed(int(config.runtime.seed))
    budget, written = int(settings.budget), 0
    progress = tqdm(total=budget, desc=f"MedSegFactory {method.upper()} sampling")
    iterator = iter(loader)
    while written < budget:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError("The sampling condition loader is empty.") from error
        batch = batch.to(device)
        condition = adapter.prepare_condition(batch)
        probe = adapter.sample_probe_noise(batch.target.shape[0], generator)
        selected = probe
        if miner is not None:
            score = adapter.initial_score(probe, condition, float(settings.cfg_scale))
            (image_mean, image_variance), (mask_mean, mask_variance) = miner(score.score)
            image_noise = reselect_noise(
                image_mean, image_variance, float(config.miner.variance_bound), generator
            ).sample
            mask_noise = reselect_noise(
                mask_mean, mask_variance, float(config.miner.variance_bound), generator
            ).sample
            selected = torch.cat((image_noise, mask_noise), 1)
        images, targets = adapter.sample(
            selected, condition, int(settings.ddim_steps), float(settings.cfg_scale)
        )
        available = min(images.shape[0], budget - written)
        monitor.log_batch(
            images[:available], targets[:available], probe[:available], selected[:available],
            batch.sample_id[:available],
        )
        for index in range(available):
            sample_id = f"{batch.sample_id[index]}-{written:06d}"
            _io.save_pair(
                output, sample_id, images[index], targets[index],
                {
                    "id": sample_id,
                    "method": method,
                    "seed": int(config.runtime.seed),
                    "ddim_steps": int(settings.ddim_steps),
                    "eta": 0.0,
                    "image_prompt": condition.extras["image_prompts"][index],
                    "mask_prompt": condition.extras["mask_prompts"][index],
                },
            )
            written += 1
            progress.update(1)
    progress.close()
    monitor.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/medsegfactory_2d.yaml")
    parser.add_argument("--method", choices=("base", "nam"), default="nam")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    sample_dataset(
        apply_overrides(load_config(arguments.config), arguments.set),
        arguments.method == "nam",
    )
