"""Concrete HAT/QSF backend for every existing NAM diffusion adapter.

LSRS and ASG additionally require method-native trajectory/attention hooks and
therefore use subclasses of this bridge inside a diffusion method package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from nam.data import NAMBatch, build_dataset, collate_medical_batch
from nam.diffusion.registry import build_diffusion
from nam.downstream import build_downstream
from nam.mitigation.base import MitigationCandidate, NAMMitigationBackend
from nam.objectives import class_balanced_cross_entropy, reselect_noise
from nam.utils.imports import import_factory


def build_miner(config: Any) -> torch.nn.Module:
    """Construct a configured NAM miner architecture."""
    if getattr(config, "factory", None):
        return import_factory(config.factory, getattr(config, "project_dir", None))(config=config)
    from nam.models import (
        AdversarialityMiner,
        JoDiffusionMiner,
        MedSegFactoryDualMiner,
        ResUNet3DMiner,
        reference_miner_configuration,
    )
    architecture = str(config.architecture).lower()
    if architecture == "jodiffusion":
        return JoDiffusionMiner()
    if architecture == "medsegfactory":
        return MedSegFactoryDualMiner()
    if architecture == "resunet3d":
        return ResUNet3DMiner(
            int(config.in_channels), int(config.noise_channels),
            int(getattr(config, "base_channels", 82)),
            tuple(getattr(config, "channel_multipliers", (1, 2, 4, 4))),
            int(getattr(config, "attention_heads", 4)),
            int(getattr(config, "head_channels", 0)) or None,
        )
    if architecture == "resunet2d":
        defaults = reference_miner_configuration(int(config.noise_channels))
        return AdversarialityMiner(
            2, int(config.in_channels), int(config.noise_channels),
            int(getattr(config, "base_channels", defaults["base_channels"])),
            tuple(getattr(config, "channel_multipliers", defaults["channel_multipliers"])),
            int(getattr(config, "downsample_levels", defaults["downsample_levels"])),
            int(getattr(config, "attention_heads", 4)),
            int(getattr(config, "head_channels", defaults["head_channels"])),
        )
    raise KeyError("Unknown miner architecture: resunet2d, resunet3d, jodiffusion, medsegfactory.")


def _single_item(batch: NAMBatch, index: int) -> NAMBatch:
    """Slice a canonical batch while preserving nested conditions and metadata."""
    def select(value: Any) -> Any:
        if torch.is_tensor(value):
            return value[index : index + 1]
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        if isinstance(value, list):
            return value[index : index + 1]
        if isinstance(value, tuple):
            return value[index : index + 1]
        return value
    image = None if batch.image is None else batch.image[index : index + 1]
    return NAMBatch(image, batch.target[index : index + 1], select(batch.condition),
                    [batch.sample_id[index]], select(batch.metadata))


class AdapterMitigationBackend(NAMMitigationBackend):
    """Run frozen NAM reselection and anchor scoring from configured paths."""

    def __init__(self, config: Any) -> None:
        device_name = str(config.runtime.device)
        self._device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.config = config
        self.diffusion = build_diffusion(config.diffusion)
        self.diffusion.model.to(self._device); self.diffusion.freeze()
        self.anchor = build_downstream(config.anchor)
        self.anchor.model.to(self._device); self.anchor.freeze()
        self.miner = build_miner(config.miner).to(self._device).eval()
        checkpoint = Path(config.miner.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"NAM miner checkpoint was not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.miner.load_state_dict(payload.get("miner", payload), strict=True)
        self.miner.requires_grad_(False)
        dataset = build_dataset(config.dataset, str(getattr(config.dataset, "split", "train")),
                                int(self.diffusion.metadata.spatial_dims))
        self.loader = DataLoader(dataset, batch_size=int(getattr(config.backend, "batch_size", 1)),
            shuffle=False, num_workers=int(config.runtime.num_workers), pin_memory=True,
            collate_fn=collate_medical_batch)
        self.cfg_scale = float(config.sampling.cfg_scale)
        self.ddim_steps = int(config.sampling.ddim_steps)
        self.variance_bound = float(getattr(config.miner, "variance_bound", 0.95))
        # Some adapters sample random auxiliary conditions in prepare_condition.
        # Reuse that exact object so NAM scoring and final synthesis are aligned.
        self._prepared_conditions: dict[int, Any] = {}

    @property
    def device(self) -> torch.device:
        return self._device

    def conditions(self) -> Iterable[NAMBatch]:
        budget, emitted = int(self.config.sampling.budget), 0
        if len(self.loader) == 0:
            raise RuntimeError("The mitigation condition loader is empty.")
        while emitted < budget:
            for batch in self.loader:
                batch = batch.to(self.device)
                for index in range(batch.target.shape[0]):
                    if emitted >= budget:
                        return
                    yield _single_item(batch, index)
                    emitted += 1

    @torch.no_grad()
    def select_nam_noise(self, condition: NAMBatch, generator=None) -> torch.Tensor:
        prepared = self.diffusion.prepare_condition(condition)
        self._prepared_conditions[id(condition)] = prepared
        probe = self.diffusion.sample_probe_noise(condition.target.shape[0], generator)
        score = self.diffusion.initial_score(probe, prepared, self.cfg_scale)
        prediction = self.miner(score.score)
        if (isinstance(prediction, (tuple, list)) and len(prediction) == 2
                and isinstance(prediction[0], (tuple, list))):
            (image_mean, image_variance), (mask_mean, mask_variance) = prediction
            image = reselect_noise(image_mean, image_variance, self.variance_bound, generator).sample
            mask = reselect_noise(mask_mean, mask_variance, self.variance_bound, generator).sample
            return torch.cat((image, mask), 1)
        mean, variance = prediction
        return reselect_noise(mean, variance, self.variance_bound, generator).sample

    @torch.no_grad()
    def sample_from_noise(self, selected_noise, condition: NAMBatch, cache_timesteps=()):
        if cache_timesteps:
            raise NotImplementedError(
                "Cached LSRS trajectories must be implemented by the selected diffusion method."
            )
        prepared = self._prepared_conditions.pop(id(condition), None)
        if prepared is None:
            prepared = self.diffusion.prepare_condition(condition)
        image, target = self.diffusion.sample(selected_noise, prepared, self.ddim_steps, self.cfg_scale)
        return MitigationCandidate(image, target, condition, condition.sample_id[0])

    @torch.no_grad()
    def adversariality(self, candidate: MitigationCandidate) -> torch.Tensor:
        if candidate.target is None:
            raise ValueError("Segmentation HAT requires a generated target.")
        return class_balanced_cross_entropy(
            self.anchor.logits(candidate.image), candidate.target,
            int(getattr(self.config.dataset, "ignore_index", -100)),
        )


def build_backend(config: Any) -> AdapterMitigationBackend:
    """Configured backend factory used by the generic runner."""
    return AdapterMitigationBackend(config)
