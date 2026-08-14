"""Medical VQA-score filtering (QSF), Appendix Algorithm 3.

The probability of the teacher-forced answer ``Yes`` follows VQAScore
(ECCV 2024), while the medical scorer uses Google's instruction-tuned
MedGemma model selected by the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

import torch
from torch.nn import functional as F

from nam.mitigation.base import (
    MitigationCandidate,
    NAMMitigationBackend,
    generate_nam_candidate,
    scalar,
)


QSF_QUESTION = (
    "Is this synthetic {modality} image for {target} visually realistic, "
    "medically plausible, and consistent with its conditioning mask or class "
    "prompt? Answer only Yes or No."
)


class QualityScorer(Protocol):
    """Callable medical-quality scorer returning one probability per image."""

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor | None, prompts: list[str]
    ) -> torch.Tensor: ...


def _image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float().cpu()
    if image.ndim != 3:
        raise ValueError("QSF expects each generated image in CHW layout.")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.min() < 0:
        image = (image + 1.0) / 2.0
    return image.clamp(0, 1).mul(255).round().byte()


def render_mask_contour(
    image: torch.Tensor,
    target: torch.Tensor | None,
    color: tuple[int, int, int] = (255, 64, 64),
) -> torch.Tensor:
    """Render the paper's mask-contour overlay without dataset-specific colors."""
    rendered = _image_to_uint8(image)
    if target is None:
        return rendered
    labels = target.detach().cpu()
    while labels.ndim > 2:
        if labels.shape[0] != 1:
            raise ValueError("QSF contour rendering expects one label map per image.")
        labels = labels[0]
    labels = labels.long()[None, None].float()
    horizontal = labels[..., 1:] != labels[..., :-1]
    vertical = labels[..., 1:, :] != labels[..., :-1, :]
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    boundary[..., 1:] |= horizontal
    boundary[..., :-1] |= horizontal
    boundary[..., 1:, :] |= vertical
    boundary[..., :-1, :] |= vertical
    boundary = F.max_pool2d(boundary.float(), 3, stride=1, padding=1)[0, 0].bool()
    for channel, value in enumerate(color):
        rendered[channel, boundary] = value
    return rendered


class MedGemmaYesScorer:
    """Teacher-forced ``Yes`` likelihood for a frozen MedGemma checkpoint.

    Weight access is gated by the upstream Gemma license. The model is loaded
    lazily so HAT/LSRS/ASG do not require Transformers or MedGemma weights.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/medgemma-4b-it",
        device: str = "cuda",
        dtype: str = "bfloat16",
        answer: str = "Yes",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if not hasattr(torch, dtype):
            raise ValueError(f"Unsupported PyTorch dtype for QSF: {dtype}")
        self.dtype = getattr(torch, dtype)
        self.answer = answer
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise ImportError("QSF requires a recent Transformers release with MedGemma support.") from error
        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype if self.device.type == "cuda" else torch.float32,
        ).to(self.device).eval()
        self._model.requires_grad_(False)

    @torch.no_grad()
    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor | None, prompts: list[str]
    ) -> torch.Tensor:
        """Compute sequence probabilities, not free-form generated answers."""
        self._load()
        from PIL import Image

        values: list[torch.Tensor] = []
        for index, prompt in enumerate(prompts):
            target = None if targets is None else targets[index]
            overlay = render_mask_contour(images[index], target)
            pil = Image.fromarray(overlay.permute(1, 2, 0).numpy())
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ]}]
            prefix = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prefix_inputs = self._processor(text=prefix, images=pil, return_tensors="pt")
            full_inputs = self._processor(text=prefix + self.answer, images=pil, return_tensors="pt")
            full_inputs = {
                key: value.to(
                    device=self.device,
                    dtype=(self.dtype if self.device.type == "cuda" else torch.float32),
                ) if value.is_floating_point() else value.to(self.device)
                for key, value in full_inputs.items()
            }
            prefix_length = int(prefix_inputs["input_ids"].shape[1])
            prefix_ids = prefix_inputs["input_ids"].to(self.device)
            if (full_inputs["input_ids"].shape[1] <= prefix_length or not torch.equal(
                full_inputs["input_ids"][:, :prefix_length], prefix_ids
            )):
                raise RuntimeError(
                    "MedGemma tokenization did not preserve the prompt prefix; "
                    "teacher-forced QSF would otherwise score the wrong tokens."
                )
            output = self._model(**full_inputs)
            logits = output.logits[:, prefix_length - 1 : -1]
            labels = full_inputs["input_ids"][:, prefix_length:]
            token_log = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            values.append(token_log.sum().exp().float().cpu())
        return torch.stack(values)


class VQAScoreFiltering:
    """Reject candidates below the medical VQA ``Yes`` probability threshold."""

    name = "qsf"

    def __init__(
        self,
        scorer: QualityScorer,
        threshold: float = 0.8,
        maximum_trials: int = 5,
        question_template: str = QSF_QUESTION,
        prompt_builder: Callable[[Any], tuple[str, str]] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("QSF threshold must lie in [0, 1].")
        if maximum_trials < 1:
            raise ValueError("QSF maximum_trials must be positive.")
        self.scorer, self.threshold = scorer, float(threshold)
        self.maximum_trials, self.question_template = int(maximum_trials), question_template
        self.prompt_builder = prompt_builder

    def _prompt(self, condition: Any) -> str:
        if self.prompt_builder is not None:
            modality, target = self.prompt_builder(condition)
        elif isinstance(condition, dict):
            modality = str(condition.get("modality", "medical"))
            target = str(condition.get("target_name", condition.get("prompt", "the target anatomy")))
        else:
            modality, target = "medical", str(condition)
        return self.question_template.format(modality=modality, target=target)

    @torch.no_grad()
    def _score(self, candidate: MitigationCandidate) -> float:
        image = candidate.image if candidate.image.ndim == 4 else candidate.image.unsqueeze(0)
        target = candidate.target
        if target is not None and target.ndim == image.ndim - 2:
            target = target.unsqueeze(0)
        return scalar(self.scorer(image, target, [self._prompt(candidate.condition)]))

    @torch.no_grad()
    def run_one(
        self,
        backend: NAMMitigationBackend,
        condition: Any,
        generator: torch.Generator | None = None,
    ) -> MitigationCandidate:
        candidates: list[MitigationCandidate] = []
        for trial in range(1, self.maximum_trials + 1):
            candidate = generate_nam_candidate(backend, condition, generator)
            candidate.score, candidate.trials = self._score(candidate), trial
            candidates.append(candidate)
            if candidate.score >= self.threshold:
                candidate.metadata.update({"strategy": self.name, "threshold": self.threshold})
                return candidate
        fallback = max(candidates, key=lambda item: float(item.score))
        fallback.accepted = False
        fallback.metadata.update({"strategy": self.name, "threshold": self.threshold, "fallback": True})
        return fallback

    def run(
        self,
        backend: NAMMitigationBackend,
        conditions: Iterable[Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> list[MitigationCandidate]:
        return [self.run_one(backend, condition, generator) for condition in (
            backend.conditions() if conditions is None else conditions
        )]
