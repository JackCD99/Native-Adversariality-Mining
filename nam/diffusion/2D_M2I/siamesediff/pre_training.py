"""Official SiameseDiff pre-training on any canonical 2D medical dataset.

This implements the upstream DHI/ControlNet Siamese objective through the
official ``shared_step`` implementation. The local loop removes Lightning and
DeepSpeed coupling while retaining its trainable parameter scope and AdamW
configuration. Validation FID checkpoint selection is exposed as a callback so
each dataset may supply the modality-appropriate feature extractor.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nam.config import apply_overrides, load_config
from nam.evaluation.metrics import frechet_distance
from nam.utils.distributed import finalize, initialize, reduce_mean
from nam.utils.seed import seed_everything
from nam.utils.monitoring import log_diffusion_diagnostics
from nam.utils.imports import import_factory

_package = __package__ or "nam.diffusion.2D_M2I.siamesediff"
_augmentations = importlib.import_module(f"{_package}.utils.augmentations")
_data = importlib.import_module(f"{_package}.utils.data")
_runtime = importlib.import_module(f"{_package}.utils.runtime")
PairedAugmentation = _augmentations.PairedAugmentation
build_siamesediff_loader = _data.build_siamesediff_loader
to_official_batch = _data.to_official_batch
import_official_siamesediff = _runtime.import_official_siamesediff
load_official_checkpoint = _runtime.load_official_checkpoint
resolve_project_path = _runtime.resolve_project_path


class _SharedStepModule(torch.nn.Module):
    """Make the upstream Lightning ``shared_step`` visible to PyTorch DDP."""

    def __init__(self, official_model: torch.nn.Module) -> None:
        super().__init__()
        self.official_model = official_model

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.official_model.shared_step(batch)


def _features(output: Any) -> torch.Tensor:
    """Normalize common feature-extractor outputs to a BxF matrix."""
    if isinstance(output, dict):
        output = output.get("features", output.get("pooler_output", output.get("logits")))
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.flatten(2).mean(-1) if output.ndim > 2 else output


class _ValidationFID:
    """Generate the validation conditions and compute modality-specific FID."""

    def __init__(self, config: Any, device: torch.device) -> None:
        factory = import_factory(config.fid.feature_factory, getattr(config.fid, "project_dir", None))
        self.extractor = factory(config=config.fid).to(device).eval()
        self.loader = build_siamesediff_loader(
            config.dataset,
            "val",
            int(config.siamesediff.pre_training.validation_batch_size),
            int(config.runtime.num_workers),
            False,
        )
        self.augmentation = PairedAugmentation(int(config.siamesediff.pre_training.resolution))
        self.prompt = str(config.diffusion.default_prompt)
        self.steps = int(config.siamesediff.pre_training.validation_ddim_steps)
        self.cfg_scale = float(config.siamesediff.pre_training.validation_cfg_scale)
        self.device = device
        self.seed = int(config.runtime.seed)
        self.real_features: torch.Tensor | None = None

    @torch.no_grad()
    def __call__(self, model: torch.nn.Module, step: int) -> float:
        was_training = model.training
        model.eval()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        real, synthetic = [], []
        for canonical in self.loader:
            canonical = canonical.to(self.device)
            official = to_official_batch(
                canonical, self.augmentation, self.prompt, 0.0, training=False
            )
            generated = model.log_images(
                official,
                N=official["jpg"].shape[0],
                ddim_steps=self.steps,
                ddim_eta=0.0,
                unconditional_guidance_scale=self.cfg_scale,
                plot_diffusion_rows=False,
                plot_progressive_rows=False,
                plot_denoise_rows=False,
            )
            key = f"samples_cfg_scale_{self.cfg_scale:.2f}_mask"
            if key not in generated:
                raise KeyError(f"Official SiameseDiff did not return expected sample key '{key}'.")
            real_images = official["jpg"].permute(0, 3, 1, 2).contiguous()
            real.append(_features(self.extractor(real_images)).float().cpu())
            synthetic.append(_features(self.extractor(generated[key])).float().cpu())
        if was_training:
            model.train()
        real_matrix = torch.cat(real).numpy()
        synthetic_matrix = torch.cat(synthetic).numpy()
        return frechet_distance(real_matrix, synthetic_matrix)


def _model_and_optimizer(config: Any, device: torch.device) -> tuple[torch.nn.Module, torch.optim.AdamW, Any]:
    settings = config.siamesediff.pre_training
    diffusion = config.diffusion
    api = import_official_siamesediff(diffusion.project_dir)
    model = api.create_model(str(resolve_project_path(diffusion.project_dir, diffusion.config))).cpu()
    initialization = resolve_project_path(diffusion.project_dir, settings.initialization_checkpoint)
    if not initialization.is_file():
        raise FileNotFoundError(f"Missing SiameseDiff initialization checkpoint: {initialization}")
    load_official_checkpoint(model, initialization, api.load_state_dict, strict=True)
    model.learning_rate = float(settings.learning_rate)
    model.sd_locked = bool(settings.sd_locked)
    model.only_mid_control = bool(settings.only_mid_control)
    # Upstream ``p_losses`` reads Lightning's trainer/global_step properties.
    # A manual property keeps the official loss intact without requiring Trainer.
    from types import SimpleNamespace
    state = SimpleNamespace(max_steps=int(settings.max_iterations), global_step=0)
    model._trainer = state
    model._manual_global_step = 0
    if not isinstance(getattr(type(model), "global_step", None), property):
        type(model).global_step = property(
            lambda instance: getattr(instance, "_manual_global_step", 0)
        )
    model.to(device)
    parameters = list(model.control_model.parameters())
    if not model.sd_locked:
        parameters += list(model.model.diffusion_model.output_blocks.parameters())
        parameters += list(model.model.diffusion_model.out.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(settings.learning_rate))
    return model, optimizer, state


def train_pretrained_model(
    config: Any,
    fid_callback: Callable[[torch.nn.Module, int], float] | None = None,
) -> Path:
    """Train an official SiameseDiff checkpoint using the configured dataset interface."""
    settings = config.siamesediff.pre_training
    context = initialize(config.runtime.device)
    seed_everything(int(config.runtime.seed) + context.rank, bool(config.runtime.deterministic))
    loader = build_siamesediff_loader(
        config.dataset,
        "train",
        int(settings.batch_size_per_gpu),
        int(config.runtime.num_workers),
        True,
    )
    model, optimizer, trainer_state = _model_and_optimizer(config, context.device)
    if fid_callback is None and context.is_main:
        fid_callback = _ValidationFID(config, context.device)
    raw_model = model
    training_module: torch.nn.Module = _SharedStepModule(model)
    if context.world_size > 1:
        training_module = DistributedDataParallel(
            training_module,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=True,
        )
    augmentation = PairedAugmentation(
        int(settings.resolution),
        float(getattr(settings, "horizontal_flip_probability", 0.0)),
        tuple(getattr(settings, "scale_range", (1.0, 1.0))),
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(settings.output_dir) / f"{config.experiment_name}-{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    writer = None
    if context.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, ensure_ascii=False)
        writer = SummaryWriter(run_dir / "tensorboard")
    iterator = iter(loader)
    best_fid = float("inf")
    progress = tqdm(range(1, int(settings.max_iterations) + 1), desc="SiameseDiff pre-training") if context.is_main else range(1, int(settings.max_iterations) + 1)

    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(context.device)
        official = to_official_batch(
            batch,
            augmentation,
            str(config.diffusion.default_prompt),
            float(settings.prompt_dropout),
            training=True,
        )
        official = {
            name: value.to(context.device, non_blocking=True) if torch.is_tensor(value) else value
            for name, value in official.items()
        }
        trainer_state.global_step = step - 1
        raw_model._manual_global_step = step - 1
        loss, metrics = training_module(official)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        reduced_loss = reduce_mean(loss, context)
        reduced_metrics = {
            name: reduce_mean(value.mean(), context)
            for name, value in metrics.items()
            if torch.is_tensor(value)
        }
        if context.is_main:
            writer.add_scalar("train/loss", reduced_loss.item(), step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
            for name, value in reduced_metrics.items():
                writer.add_scalar(name, value.item(), step)
            progress.set_postfix(loss=f"{reduced_loss.item():.4f}")
            log_diffusion_diagnostics(
                writer,
                raw_model,
                optimizer,
                step,
                {"image": official["jpg"].permute(0, 3, 1, 2), "condition": official["hint"].permute(0, 3, 1, 2)},
            )
        evaluate = step % int(settings.fid_every) == 0 or step == int(settings.max_iterations)
        if evaluate and context.is_main:
            payload = {"global_step": step, "state_dict": raw_model.state_dict()}
            torch.save(payload, checkpoint_dir / "last.ckpt")
            stable_root = Path(config.siamesediff.checkpoints.diffusion_root)
            stable_root.mkdir(parents=True, exist_ok=True)
            torch.save(payload, stable_root / "last.ckpt")
            fid = float("inf") if fid_callback is None else float(fid_callback(raw_model, step))
            if fid_callback is not None:
                writer.add_scalar("validation/fid", fid, step)
                if fid < best_fid:
                    best_fid = fid
                    payload["validation_fid"] = fid
                    torch.save(payload, checkpoint_dir / "best_fid.ckpt")
                    torch.save(payload, stable_root / "best_fid.ckpt")
        if context.world_size > 1:
            torch.distributed.barrier()
    if writer is not None:
        writer.close()
    finalize(context)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train SiameseDiff on a canonical 2D dataset.")
    parser.add_argument("--config", default="configs/table1_2d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
