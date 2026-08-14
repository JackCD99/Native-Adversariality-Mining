"""Official DiffBoost ControlNet fine-tuning for canonical 2D datasets.

The official recipe initializes from a RadImageNet ControlNet checkpoint,
freezes Stable Diffusion, and updates only ControlNet with AdamW at 1e-6.
This loop removes the PyTorch-Lightning runner while preserving the upstream
``shared_step`` objective and its image/text/edge input contract.
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nam.config import apply_overrides, load_config
from nam.evaluation.metrics import frechet_distance
from nam.utils.imports import import_factory
from nam.utils.distributed import finalize, initialize, reduce_mean
from nam.utils.seed import seed_everything
from nam.utils.monitoring import log_diffusion_diagnostics

_package = __package__ or "nam.diffusion.2D_M2I.diffboost"
_runtime = importlib.import_module(f"{_package}.utils.runtime")
_utils = importlib.import_module(f"{_package}.utils")
build_official_model = _runtime.build_official_model
DiffBoostAugmentation = _utils.DiffBoostAugmentation
build_diffboost_loader = _utils.build_diffboost_loader
save_checkpoint = _utils.save_checkpoint
to_official_batch = _utils.to_official_batch
prepare_diffboost_condition = _utils.prepare_diffboost_condition
import_official_diffboost = _runtime.import_official_diffboost


def _features(output: Any) -> torch.Tensor:
    """Normalize common feature-extractor outputs to a BxF matrix."""
    if isinstance(output, dict):
        output = output.get("features", output.get("pooler_output", output.get("logits")))
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.flatten(2).mean(-1) if output.ndim > 2 else output


class _ValidationFID:
    """Compute paper checkpoint-selection FID with the configured modality encoder."""

    def __init__(self, config: Any, device: torch.device) -> None:
        factory = import_factory(config.fid.feature_factory, getattr(config.fid, "project_dir", None))
        self.extractor = factory(config=config.fid).to(device).eval()
        self.loader = build_diffboost_loader(
            config.dataset, "val", int(config.diffboost.pre_training.validation_batch_size),
            int(config.runtime.num_workers), False,
        )
        self.augmentation = DiffBoostAugmentation(int(config.diffusion.resolution), enabled=False)
        self.diffusion = config.diffusion
        self.steps = int(config.diffboost.pre_training.validation_ddim_steps)
        self.cfg_scale = float(config.diffboost.pre_training.validation_cfg_scale)
        self.max_batches = int(config.diffboost.pre_training.validation_batches)
        self.device = device
        self.seed = int(config.runtime.seed)
        self.sampler_class = import_official_diffboost(config.diffusion.project_dir).sampler_class

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
                batch, self.augmentation, str(self.diffusion.default_prompt), training=True,
                condition_mode=str(self.diffusion.condition_mode),
                num_classes=int(self.diffusion.num_classes),
            )
            control, _, prompts, augmentation_prompts = prepare_diffboost_condition(
                batch, int(self.diffusion.resolution), str(self.diffusion.default_prompt),
                str(self.diffusion.augmentation_prompt), str(self.diffusion.condition_mode),
                int(self.diffusion.num_classes),
            )
            w_base, w_augmentation, w_sketch = [float(value) for value in self.diffusion.prompt_weights]
            text = (
                w_base * model.get_learned_conditioning(
                    [f"{prompt},{self.diffusion.positive_prompt}" for prompt in prompts]
                )
                + w_augmentation * model.get_learned_conditioning(augmentation_prompts)
                + w_sketch * model.get_learned_conditioning(["gray, sketch"] * len(prompts))
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
                condition,
                x_T=noise, verbose=False, eta=0.0,
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
            raise RuntimeError("DiffBoost validation loader produced no samples for FID.")
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
        if isinstance(result, tuple):
            return result
        return result, {"train/loss": result}


def _load_training_model(config: Any, device: torch.device) -> torch.nn.Module:
    settings = config.diffboost.pre_training
    resume = str(getattr(settings, "resume_checkpoint", "")).strip()
    initialization = resume or str(settings.initialization_checkpoint)
    if not initialization:
        raise ValueError(
            "diffboost.pre_training.initialization_checkpoint must point to a licensed "
            "RadImageNet/ControlNet initialization. See the DiffBoost guide."
        )
    model_config = SimpleNamespace(**dict(config.diffusion))
    model_config.checkpoint = initialization
    model = build_official_model(model_config, load_checkpoint=True)
    model.learning_rate = float(settings.learning_rate)
    model.sd_locked = bool(settings.sd_locked)
    model.only_mid_control = bool(settings.only_mid_control)
    model.control_scales = [float(config.diffusion.control_strength)] * 13
    # Explicit trainability prevents accidental updates to CLIP, VAE, or the
    # Stable-Diffusion backbone when using a newer Lightning/PyTorch version.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.control_model.parameters():
        parameter.requires_grad_(True)
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
    default_prompt: str,
    device: torch.device,
    max_batches: int,
    condition_mode: str,
    num_classes: int,
) -> torch.Tensor:
    model.eval()
    values = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = batch.to(device)
        official = to_official_batch(
            batch, augmentation, default_prompt, training=True,
            condition_mode=condition_mode,
            num_classes=num_classes,
        )
        loss = model.shared_step(official)
        values.append(loss[0] if isinstance(loss, tuple) else loss)
    model.train()
    return torch.stack(values).mean() if values else torch.tensor(float("inf"), device=device)


def train_pretrained_model(config: Any, fid_callback: Any = None) -> Path | None:
    """Fine-tune official DiffBoost and publish stable last/best checkpoints."""
    settings = config.diffboost.pre_training
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
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        best_validation = float(
            payload.get("best_validation_fid", payload.get("best_validation_loss", float("inf")))
        )
    train_loader = build_diffboost_loader(
        config.dataset, "train", int(settings.batch_size_per_gpu),
        int(config.runtime.num_workers), True,
    )
    validation_loader = build_diffboost_loader(
        config.dataset, "val", int(settings.validation_batch_size),
        int(config.runtime.num_workers), False,
    )
    train_augmentation = DiffBoostAugmentation(
        int(config.diffusion.resolution),
        float(settings.rotation_degrees), tuple(settings.scale_range), enabled=True,
    )
    validation_augmentation = DiffBoostAugmentation(
        int(config.diffusion.resolution), enabled=False
    )
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
        tqdm(range(start_step + 1, int(settings.max_iterations) + 1), desc="DiffBoost fine-tuning")
        if context.is_main else range(start_step + 1, int(settings.max_iterations) + 1)
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
            batch, train_augmentation, str(config.diffusion.default_prompt), training=True,
            condition_mode=str(config.diffusion.condition_mode),
            num_classes=int(config.diffusion.num_classes),
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
        evaluate = step % int(settings.validation_every) == 0 or step == int(settings.max_iterations)
        if evaluate and context.is_main:
            validation = _validation_loss(
                raw_model, validation_loader, validation_augmentation,
                str(config.diffusion.default_prompt), context.device,
                int(settings.validation_batches), str(config.diffusion.condition_mode),
                int(config.diffusion.num_classes),
            )
            writer.add_scalar("validation/loss", float(validation.item()), step)
            value = float(fid_callback(raw_model, step))
            writer.add_scalar("validation/fid", value, step)
            state = {
                "step": step, "optimizer": optimizer.state_dict(),
                "best_validation_fid": min(best_validation, value), "config": dict(config),
                "initialization_checkpoint": str(settings.initialization_checkpoint),
            }
            stable_root = Path(config.diffboost.checkpoints.diffusion_root)
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
    parser = argparse.ArgumentParser(description="Fine-tune DiffBoost on a canonical 2D dataset.")
    parser.add_argument("--config", default="configs/diffboost_2d.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_pretrained_model(apply_overrides(load_config(arguments.config), arguments.set))
