"""Shared optimization engine for all 2D and 3D NAM miners."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from nam.data import build_dataset, collate_medical_batch
from nam.diffusion import build_diffusion
from nam.downstream import build_downstream
from nam.models import (
    AdversarialityMiner,
    JoDiffusionMiner,
    MedSegFactoryDualMiner,
    ResUNet3DMiner,
    reference_miner_configuration,
)
from nam.objectives import adversariality_reward, normalized_kl, reselect_noise
from nam.utils.distributed import DistributedContext, finalize, initialize, reduce_mean
from nam.utils.monitoring import ExperimentMonitor, logging_interval
from nam.utils.seed import seed_everything


class MinerTrainer:
    """Optimize a method-specific miner while freezing the generator and anchor."""

    def __init__(self, config: Any, spatial_dims: int) -> None:
        self.config = config
        self.spatial_dims = spatial_dims
        self.method = str(config.diffusion.name).lower()
        self.context: DistributedContext = initialize(config.runtime.device)
        seed_everything(
            int(config.runtime.seed) + self.context.rank,
            bool(getattr(config.runtime, "deterministic", False)),
        )

        self.diffusion = build_diffusion(config.diffusion)
        self.diffusion.model.to(self.context.device)
        self.diffusion.freeze()
        if self.diffusion.metadata.spatial_dims != spatial_dims:
            raise ValueError(
                f"{self.method} exposes {self.diffusion.metadata.spatial_dims}D noise, "
                f"but the selected entry point is {spatial_dims}D."
            )
        self.anchor = build_downstream(config.anchor)
        self.anchor.model.to(self.context.device)
        self.anchor.freeze()
        self.settings = self._settings()
        self.miner = self._build_miner().to(self.context.device)
        if self.context.world_size > 1:
            self.miner = DistributedDataParallel(
                self.miner,
                device_ids=[self.context.local_rank],
                output_device=self.context.local_rank,
            )
        self.optimizer = torch.optim.AdamW(
            self.miner.parameters(),
            lr=float(self.settings.learning_rate),
            betas=tuple(getattr(self.settings, "betas", (0.9, 0.999))),
            weight_decay=float(self.settings.weight_decay),
        )
        self.loader = self._build_loader()
        self.run_dir = self._run_directory()
        self.monitor = ExperimentMonitor(self.run_dir, config, enabled=self.context.is_main)
        if self.context.is_main:
            self.monitor.describe_model("miner", self.raw_miner)
            self.monitor.describe_model("frozen_generator", self.diffusion.model)
            self.monitor.describe_model("frozen_anchor", self.anchor.model)
        self.start_step = self._resume()

    def _log_interval(self, name: str, default: int) -> int:
        value = int(getattr(self.settings, name, logging_interval(self.config, name, default)))
        return max(value, 1)

    @property
    def raw_miner(self) -> torch.nn.Module:
        return self.miner.module if hasattr(self.miner, "module") else self.miner

    def _settings(self) -> Any:
        method_section = getattr(self.config, self.method, None)
        if method_section is not None and hasattr(method_section, "nam_training"):
            return method_section.nam_training
        return self.config.training

    def _miner_checkpoint(self) -> str:
        """Resolve one trained-miner path shared by sampling and mitigation."""
        method_section = getattr(self.config, self.method, None)
        if method_section is not None and hasattr(method_section, "checkpoints"):
            return str(getattr(method_section.checkpoints, "nam_miner", ""))
        return str(getattr(self.config.miner, "checkpoint", ""))

    def _build_miner(self) -> torch.nn.Module:
        if self.method == "jodiffusion":
            return JoDiffusionMiner()
        if self.method == "medsegfactory":
            return MedSegFactoryDualMiner()
        defaults = reference_miner_configuration(int(self.diffusion.metadata.noise_channels))
        common = {
            "in_channels": int(
                getattr(self.config.miner, "in_channels", self.diffusion.metadata.noise_channels)
            ),
            "noise_channels": int(self.diffusion.metadata.noise_channels),
            "base_channels": int(getattr(self.config.miner, "base_channels", defaults["base_channels"])),
            "channel_multipliers": tuple(
                getattr(self.config.miner, "channel_multipliers", defaults["channel_multipliers"])
            ),
            "attention_heads": int(getattr(self.config.miner, "attention_heads", 4)),
        }
        if self.spatial_dims == 3:
            return ResUNet3DMiner(
                **common,
                head_channels=int(getattr(self.config.miner, "head_channels", 0)) or None,
            )
        miner_defaults = reference_miner_configuration(common["noise_channels"])
        common["base_channels"] = int(
            getattr(self.config.miner, "base_channels", miner_defaults["base_channels"])
        )
        return AdversarialityMiner(
            spatial_dims=2,
            downsample_levels=int(self.config.miner.downsample_levels),
            head_channels=int(
                getattr(self.config.miner, "head_channels", miner_defaults["head_channels"])
            ),
            **common,
        )

    def _build_loader(self) -> DataLoader:
        dataset = build_dataset(self.config.dataset, "train", self.spatial_dims)
        sampler = None
        if self.context.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.context.world_size,
                rank=self.context.rank,
                shuffle=True,
                drop_last=True,
            )
        return DataLoader(
            dataset,
            batch_size=int(self.settings.batch_size_per_gpu),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(self.config.runtime.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=int(self.config.runtime.num_workers) > 0,
            collate_fn=collate_medical_batch,
        )

    def _checkpoint_root(self) -> Path:
        method_section = getattr(self.config, self.method, None)
        if method_section is not None and hasattr(method_section, "checkpoints"):
            root = getattr(method_section.checkpoints, "nam_root", None)
            if root:
                return Path(root)
        stable = str(getattr(self.settings, "stable_checkpoint", "")).strip()
        if stable:
            return Path(stable).parent
        return Path(self.config.runtime.output_dir) / "checkpoints" / self.method / "nam"

    def _run_directory(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self._checkpoint_root() / "runs" / f"{self.config.experiment_name}-{timestamp}"

    def _resume(self) -> int:
        resume_from = str(getattr(self.settings, "resume_from", "")).strip()
        if not resume_from:
            return 1
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        self.raw_miner.load_state_dict(checkpoint["miner"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint.get("step", 0)) + 1

    def _select_noise(self, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        bound = float(self.config.miner.variance_bound)
        if self.method == "medsegfactory":
            (image_mean, image_variance), (mask_mean, mask_variance) = self.miner(score)
            image = reselect_noise(image_mean, image_variance, bound)
            mask = reselect_noise(mask_mean, mask_variance, bound)
            selected = torch.cat((image.sample, mask.sample), dim=1)
            kl = normalized_kl(image.kl_per_sample, image.sample) + normalized_kl(
                mask.kl_per_sample, mask.sample
            )
            diagnostics = {
                "score": score,
                "selected_noise": selected,
                "image_mean": image.mean,
                "image_variance": image.variance,
                "mask_mean": mask.mean,
                "mask_variance": mask.variance,
            }
            return selected, kl, diagnostics

        delta_mean, delta_variance = self.miner(score)
        selection = reselect_noise(delta_mean, delta_variance, bound)
        kl = selection.kl_per_sample
        if self.method == "jodiffusion":
            kl = normalized_kl(kl, selection.sample)
        diagnostics = {
            "score": score,
            "selected_noise": selection.sample,
            "mean": selection.mean,
            "variance": selection.variance,
        }
        return selection.sample, kl, diagnostics

    @staticmethod
    def _unpack_rollout(result: Any, fallback_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, fallback_target

    def _diagnostics(
        self,
        step: int,
        probe: torch.Tensor,
        selected: torch.Tensor,
        condition: Any,
        target: torch.Tensor,
        synthetic: torch.Tensor,
        logits: torch.Tensor,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        if not self.context.is_main:
            return
        with torch.no_grad():
            base_result = self.diffusion.truncated_rollout(
                probe,
                condition,
                int(self.settings.truncated_steps),
                float(self.settings.cfg_scale),
            )
            base_image, base_target = self._unpack_rollout(base_result, target)
            base_logits = self.anchor.logits(base_image)
            reward_name = str(getattr(self.settings, "reward", "lce"))
            base_adv = adversariality_reward(
                reward_name,
                base_logits,
                base_target,
                int(getattr(self.config.dataset, "ignore_index", -100)),
            )
            nam_adv = adversariality_reward(
                reward_name,
                logits,
                target,
                int(getattr(self.config.dataset, "ignore_index", -100)),
            )
        self.monitor.log_metrics(
            "diagnostics",
            {
                "base_adversariality": base_adv.mean(),
                "nam_adversariality": nam_adv.mean(),
                "adversariality_gain": nam_adv.mean() - base_adv.mean(),
                "noise_l1_shift": (selected - probe).abs().mean(),
                "image_l1_shift": (synthetic - base_image).abs().mean(),
            },
            step,
        )
        tensors = {
            **tensors,
            "probe_noise": probe,
            "noise_shift": selected - probe,
            "base_adversariality": base_adv,
            "nam_adversariality": nam_adv,
        }
        self.monitor.log_nam_comparison(
            step,
            base_image,
            synthetic,
            target,
            base_logits,
            logits,
            tensors,
            max_items=2 if self.spatial_dims == 3 else 4,
        )
        self.monitor.flush()

    def train(self) -> Path | None:
        maximum = int(self.settings.max_iterations)
        iterator = iter(self.loader)
        epoch = 0
        progress = (
            tqdm(range(self.start_step, maximum + 1), desc=f"{self.method} NAM")
            if self.context.is_main
            else range(self.start_step, maximum + 1)
        )
        image_every = self._log_interval("sample_every", 100)
        histogram_every = self._log_interval("histogram_every", 100)
        self.miner.train()
        for step in progress:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if isinstance(self.loader.sampler, DistributedSampler):
                    self.loader.sampler.set_epoch(epoch)
                iterator = iter(self.loader)
                batch = next(iterator)
            batch = batch.to(self.context.device)
            condition = self.diffusion.prepare_condition(batch)
            probe = self.diffusion.sample_probe_noise(batch.target.shape[0])
            with torch.no_grad():
                score = self.diffusion.initial_score(
                    probe, condition, float(self.settings.cfg_scale)
                ).score.detach()
            selected, kl, tensors = self._select_noise(score)
            result = self.diffusion.truncated_rollout(
                selected,
                condition,
                int(self.settings.truncated_steps),
                float(self.settings.cfg_scale),
            )
            synthetic, target = self._unpack_rollout(result, condition.target)
            logits = self.anchor.logits(synthetic)
            adversariality = adversariality_reward(
                str(getattr(self.settings, "reward", "lce")),
                logits,
                target,
                int(getattr(self.config.dataset, "ignore_index", -100)),
            )
            capped = adversariality.clamp(max=float(self.settings.kappa_up))
            loss = -(capped.mean() - float(self.settings.beta) * kl.mean())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.miner.parameters(), float(self.settings.gradient_clip)
            )
            self.optimizer.step()

            metrics = {
                "loss": loss,
                "objective": -loss,
                "adversariality": adversariality.mean(),
                "capped_adversariality": capped.mean(),
                "cap_fraction": (adversariality >= float(self.settings.kappa_up)).float().mean(),
                "kl": kl.mean(),
                "gradient_norm": torch.as_tensor(gradient_norm, device=self.context.device),
            }
            metrics = {name: reduce_mean(value, self.context) for name, value in metrics.items()}
            if self.context.is_main:
                self.monitor.log_metrics("train", metrics, step)
                self.monitor.log_optimizer(self.optimizer, step)
                progress.set_postfix(
                    loss=f"{metrics['loss'].item():.3f}",
                    adv=f"{metrics['adversariality'].item():.3f}",
                    kl=f"{metrics['kl'].item():.2e}",
                )
                if step == 1 or step % histogram_every == 0:
                    self.monitor.log_histograms("nam/distributions", tensors, step)
            visualize = step == 1 or step % image_every == 0 or step == maximum
            if visualize:
                self._diagnostics(
                    step, probe, selected, condition, target, synthetic, logits, tensors
                )
            if visualize and self.context.world_size > 1:
                torch.distributed.barrier()
            should_save = step % int(self.settings.save_every) == 0 or step == maximum
            if should_save:
                self._save(step)
            if should_save and self.context.world_size > 1:
                torch.distributed.barrier()

        self.monitor.close()
        finalize(self.context)
        return self.run_dir if self.context.is_main else None

    def _save(self, step: int) -> None:
        if not self.context.is_main:
            return
        checkpoint = {
            "step": step,
            "miner": self.raw_miner.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "diffusion": self.diffusion.metadata.__dict__,
            "diffusion_checkpoint": str(getattr(self.config.diffusion, "checkpoint", "")),
            "downstream_checkpoint": str(getattr(self.config.anchor, "checkpoint", "")),
            "config": dict(self.config),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        destination = self.run_dir / "checkpoints"
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination / f"nam_step_{step:07d}.pt")
        torch.save(checkpoint, destination / "nam_latest.pt")
        root = self._checkpoint_root()
        root.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, root / "nam_latest.pt")
        stable_checkpoint = str(getattr(self.settings, "stable_checkpoint", "")).strip()
        if stable_checkpoint:
            stable_path = Path(stable_checkpoint)
            stable_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, stable_path)


def run_miner_training(config: Any, spatial_dims: int) -> Path | None:
    """Run the configured method through the shared NAM optimization engine."""
    return MinerTrainer(config, spatial_dims).train()
