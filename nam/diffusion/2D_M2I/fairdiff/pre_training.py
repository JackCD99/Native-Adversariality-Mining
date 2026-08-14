"""Official FairDiff ControlNet fine-tuning for canonical 2D datasets.

The public recipe initializes ``cldm_v15.yaml`` from
``control_sd15_seg.pth``, sets ``sd_locked=False``, and optimizes the official
``shared_step`` objective with AdamW at 1e-5 and batch size four. The paper
additionally selects checkpoints by validation FID every 500 iterations.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Support both ``python -m ...`` and direct ``python pre_training.py`` runs.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nam.config import apply_overrides, load_config
from nam.evaluation.metrics import frechet_distance
from nam.utils.distributed import finalize, initialize, reduce_mean
from nam.utils.imports import import_factory
from nam.utils.seed import seed_everything
from nam.utils.monitoring import log_diffusion_diagnostics

_package = __package__ or "nam.diffusion.2D_M2I.fairdiff"
_runtime = importlib.import_module(f"{_package}.utils.runtime")
_utils = importlib.import_module(f"{_package}.utils")
build_official_model = _runtime.build_official_model
import_official_fairdiff = _runtime.import_official_fairdiff
FairDiffAugmentation = _utils.FairDiffAugmentation
build_fairdiff_loader = _utils.build_fairdiff_loader
prepare_fairdiff_condition = _utils.prepare_fairdiff_condition
save_checkpoint = _utils.save_checkpoint
to_official_batch = _utils.to_official_batch


def _features(output: Any) -> torch.Tensor:
    """Normalize common feature-extractor outputs to a BxF matrix."""
    if isinstance(output, dict):
        output = output.get("features", output.get("pooler_output", output.get("logits")))
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.flatten(2).mean(-1) if output.ndim > 2 else output


class _ValidationFID:
    """Compute the paper's validation FID with the modality-specific encoder."""

    def __init__(self, config: Any, device: torch.device) -> None:
        factory = import_factory(config.fid.feature_factory, getattr(config.fid, "project_dir", None))
        self.extractor = factory(config=config.fid).to(device).eval()
        settings = config.fairdiff.pre_training
        self.loader = build_fairdiff_loader(
            config.dataset, "val", int(settings.validation_batch_size),
            int(config.runtime.num_workers), False,
        )
        self.augmentation = FairDiffAugmentation(int(config.diffusion.resolution))
        self.diffusion = config.diffusion
        self.steps = int(settings.validation_ddim_steps)
        self.cfg_scale = float(settings.validation_cfg_scale)
        self.max_batches = int(settings.validation_batches)
        self.device = device
        self.seed = int(config.runtime.seed)
        self.sampler_class = import_official_fairdiff(config.diffusion.project_dir).sampler_class

    @torch.no_grad()
    def __call__(self, model: torch.nn.Module, step: int) -> float:
        was_training = model.training
        model.eval()
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        real_features, synthetic_features = [], []
        for batch_index, batch in enumerate(self.loader):
            if batch_index >= self.max_batches:
                break
            batch = batch.to(self.device)
            official = to_official_batch(
                batch, self.augmentation, str(self.diffusion.default_prompt),
                int(self.diffusion.num_classes), str(self.diffusion.mask_encoding), training=True,
            )
            control, _, prompts = prepare_fairdiff_condition(
                batch, int(self.diffusion.resolution), str(self.diffusion.default_prompt),
                int(self.diffusion.num_classes), str(self.diffusion.mask_encoding),
            )
            text = model.get_learned_conditioning(
                [f"{prompt}, {self.diffusion.positive_prompt}" for prompt in prompts]
            )
            negative = model.get_learned_conditioning(
                [str(self.diffusion.negative_prompt)] * len(prompts)
            )
            condition = {"c_concat": [control], "c_crossattn": [text]}
            unconditional = {"c_concat": [control], "c_crossattn": [negative]}
            noise = torch.randn(
                (len(prompts), int(self.diffusion.noise_channels), *self.diffusion.noise_size),
                device=self.device, generator=generator,
            )
            sampler = self.sampler_class(model)
            latent, _ = sampler.sample(
                self.steps, len(prompts),
                (int(self.diffusion.noise_channels), *tuple(self.diffusion.noise_size)),
                condition, x_T=noise, verbose=False, eta=0.0,
                unconditional_guidance_scale=self.cfg_scale,
                unconditional_conditioning=unconditional,
            )
            synthetic = model.decode_first_stage(latent).clamp(-1.0, 1.0)
            real = official["jpg"].permute(0, 3, 1, 2).contiguous()
            real_features.append(_features(self.extractor(real)).float().cpu())
            synthetic_features.append(_features(self.extractor(synthetic)).float().cpu())
        if was_training:
            model.train()
        if not real_features:
            raise RuntimeError("FairDiff validation loader produced no samples for FID.")
        return frechet_distance(
            torch.cat(real_features).numpy(), torch.cat(synthetic_features).numpy()
        )


class _SharedStepModule(torch.nn.Module):
    """Expose the official Lightning loss through standard PyTorch DDP."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.official_model = model

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        result = self.official_model.shared_step(batch)
        return result if isinstance(result, tuple) else (result, {"train/loss": result})


def _load_training_model(config: Any, device: torch.device) -> torch.nn.Module:
    settings = config.fairdiff.pre_training
    resume = str(getattr(settings, "resume_checkpoint", "")).strip()
    initialization = resume or str(settings.initialization_checkpoint)
    if not initialization:
        raise ValueError(
            "fairdiff.pre_training.initialization_checkpoint must point to the official "
            "control_sd15_seg.pth initialization. See fairdiff/README.md."
        )
    model_config = SimpleNamespace(**dict(config.diffusion))
    model_config.checkpoint = initialization
    model = build_official_model(model_config, load_checkpoint=True)
    model.learning_rate = float(settings.learning_rate)
    model.sd_locked = bool(settings.sd_locked)
    model.only_mid_control = bool(settings.only_mid_control)
    model.control_scales = [float(config.diffusion.control_strength)] * 13
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.control_model.parameters():
        parameter.requires_grad_(True)
    # This exactly mirrors official ControlLDM.configure_optimizers().
    if not model.sd_locked:
        for parameter in model.model.diffusion_model.output_blocks.parameters():
            parameter.requires_grad_(True)
        for parameter in model.model.diffusion_model.out.parameters():
            parameter.requires_grad_(True)
    return model.to(device)


@torch.no_grad()
def _validation_loss(
    model: torch.nn.Module,
    loader: Any,
    augmentation: Any,
    config: Any,
    device: torch.device,
    max_batches: int,
) -> torch.Tensor:
    model.eval()
    values = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = batch.to(device)
        official = to_official_batch(
            batch, augmentation, str(config.diffusion.default_prompt),
            int(config.diffusion.num_classes), str(config.diffusion.mask_encoding), training=True,
        )
        result = model.shared_step(official)
        values.append(result[0] if isinstance(result, tuple) else result)
    model.train()
    return torch.stack(values).mean() if values else torch.tensor(float("inf"), device=device)


def train_pretrained_model(config: Any, fid_callback: Any = None) -> Path | None:
    """Fine-tune official FairDiff and publish stable last/best checkpoints."""
    settings = config.fairdiff.pre_training
    context = initialize(config.runtime.device)
    seed_everything(int(config.runtime.seed) + context.rank, bool(config.runtime.deterministic))
    model = _load_training_model(config, context.device)
    if fid_callback is None and context.is_main:
        fid_callback = _ValidationFID(config, context.device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(settings.learning_rate))
    start_step = 0
    best_validation = float("inf")
    resume = str(getattr(settings, "resume_checkpoint", "")).strip()
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload.get("step", 0))
            best_validation = float(payload.get("best_validation_fid", float("inf")))
    train_loader = build_fairdiff_loader(
        config.dataset, "train", int(settings.batch_size_per_gpu),
        int(config.runtime.num_workers), True,
    )
    validation_loader = build_fairdiff_loader(
        config.dataset, "val", int(settings.validation_batch_size),
        int(config.runtime.num_workers), False,
    )
    augmentation = FairDiffAugmentation(int(config.diffusion.resolution))
    configured_iterations = int(getattr(settings, "max_iterations", 0))
    max_iterations = configured_iterations or int(settings.max_epochs) * len(train_loader)
    raw_model = model
    training_module: torch.nn.Module = _SharedStepModule(model)
    if context.world_size > 1:
        training_module = DistributedDataParallel(
            training_module, device_ids=[context.local_rank],
            output_device=context.local_rank, find_unused_parameters=True,
        )
    run_dir = Path(settings.output_dir) / (
        f"{config.experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    writer = None
    if context.is_main:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, ensure_ascii=False)
        writer = SummaryWriter(run_dir / "tensorboard")
    iterator = iter(train_loader)
    progress = (
        tqdm(range(start_step + 1, max_iterations + 1), desc="FairDiff fine-tuning")
        if context.is_main else range(start_step + 1, max_iterations + 1)
    )
    raw_model.train()
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(context.device)
        official = to_official_batch(
            batch, augmentation, str(config.diffusion.default_prompt),
            int(config.diffusion.num_classes), str(config.diffusion.mask_encoding), training=True,
        )
        loss, metrics = training_module(official)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(settings.gradient_clip))
        optimizer.step()
        reduced_loss = reduce_mean(loss, context)
        reduced_metrics = {
            name: reduce_mean(value.mean(), context)
            for name, value in metrics.items() if torch.is_tensor(value)
        }
        if context.is_main:
            writer.add_scalar("train/loss", reduced_loss.item(), step)
            writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
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
        evaluate = step % int(settings.validation_every) == 0 or step == max_iterations
        if evaluate and context.is_main:
            validation = _validation_loss(
                raw_model, validation_loader, augmentation, config,
                context.device, int(settings.validation_batches),
            )
            writer.add_scalar("validation/loss", float(validation.item()), step)
            value = float(fid_callback(raw_model, step))
            writer.add_scalar("validation/fid", value, step)
            state = {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "best_validation_fid": min(best_validation, value),
                "config": dict(config),
                "initialization_checkpoint": str(settings.initialization_checkpoint),
                "official_commit": "3a0a67ad1f1a3be719b6d529178eeb217a2868a0",
            }
            stable_root = Path(config.fairdiff.checkpoints.diffusion_root)
            save_checkpoint(raw_model, stable_root / "last.ckpt", state)
            save_checkpoint(raw_model, run_dir / "checkpoints" / "last.ckpt", state)
            if value < best_validation:
                best_validation = value
                state["best_validation_fid"] = value
                save_checkpoint(raw_model, stable_root / "best.ckpt", state)
        if evaluate and context.world_size > 1:
            torch.distributed.barrier()
    if writer is not None:
        writer.close()
    finalize(context)
    return run_dir if context.is_main else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FairDiff on a canonical 2D dataset.")
    parser.add_argument("--config", default="configs/fairdiff_2d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
