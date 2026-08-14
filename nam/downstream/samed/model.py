"""Minimal self-contained SAMed architecture.

The implementation preserves SAM ViT-B image encoding, LoRA updates on every
query/value projection, prompt-free mask tokens, and the SAM two-way decoder.
It removes interactive-prompt and automatic-mask-generation utilities that are
not used by the semantic-segmentation experiments.

Official sources:
    https://github.com/hitachinsk/SAMed
    https://github.com/facebookresearch/segment-anything

References: Zhang and Liu, arXiv 2023; Kirillov et al., ICCV 2023.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Configurable feed-forward network used by SAM encoders and decoders."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(left, right) for left, right in zip(dimensions[:-1], dimensions[1:])]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            inputs = F.relu(layer(inputs)) if index < len(self.layers) - 1 else layer(inputs)
        return inputs


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm used by the original SAM convolutional neck."""

    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(1, keepdim=True)
        variance = (inputs - mean).square().mean(1, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(variance + self.epsilon)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


def _relative_position(query_size: int, key_size: int, embedding: torch.Tensor) -> torch.Tensor:
    maximum = 2 * max(query_size, key_size) - 1
    if embedding.shape[0] != maximum:
        embedding = F.interpolate(
            embedding.transpose(0, 1).unsqueeze(0), size=maximum, mode="linear", align_corners=False
        ).squeeze(0).transpose(0, 1)
    query_coordinates = torch.arange(query_size, device=embedding.device)[:, None] * max(key_size / query_size, 1.0)
    key_coordinates = torch.arange(key_size, device=embedding.device)[None, :] * max(query_size / key_size, 1.0)
    coordinates = (query_coordinates - key_coordinates) + (key_size - 1) * max(query_size / key_size, 1.0)
    return embedding[coordinates.long()]


class Attention(nn.Module):
    """SAM image attention with optional decomposed relative position bias."""

    def __init__(
        self, channels: int, heads: int, qkv_bias: bool, use_relative_position: bool, input_size: tuple[int, int]
    ) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.projection = nn.Linear(channels, channels)
        self.relative_height = nn.Parameter(torch.zeros(2 * input_size[0] - 1, channels // heads)) if use_relative_position else None
        self.relative_width = nn.Parameter(torch.zeros(2 * input_size[1] - 1, channels // heads)) if use_relative_position else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, height * width, 3, self.heads, channels // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        if self.relative_height is not None and self.relative_width is not None:
            query_grid = query.reshape(batch, self.heads, height, width, -1)
            relative_h = _relative_position(height, height, self.relative_height)
            relative_w = _relative_position(width, width, self.relative_width)
            height_bias = torch.einsum("bhqwc,qkc->bhqwk", query_grid, relative_h)
            width_bias = torch.einsum("bhqwc,wkc->bhqwk", query_grid, relative_w)
            attention = attention.reshape(batch, self.heads, height, width, height, width)
            attention = attention + height_bias[..., :, None] + width_bias[..., None, :]
            attention = attention.reshape(batch, self.heads, height * width, height * width)
        output = (attention.softmax(dim=-1) @ value).transpose(1, 2).reshape(batch, height, width, channels)
        return self.projection(output)


def _partition(inputs: torch.Tensor, window: int) -> tuple[torch.Tensor, tuple[int, int]]:
    batch, height, width, channels = inputs.shape
    padded_h, padded_w = math.ceil(height / window) * window, math.ceil(width / window) * window
    if (padded_h, padded_w) != (height, width):
        inputs = F.pad(inputs.permute(0, 3, 1, 2), (0, padded_w - width, 0, padded_h - height)).permute(0, 2, 3, 1)
    windows = inputs.reshape(batch, padded_h // window, window, padded_w // window, window, channels)
    return windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, window, window, channels), (padded_h, padded_w)


def _unpartition(windows: torch.Tensor, window: int, padded: tuple[int, int], original: tuple[int, int]) -> torch.Tensor:
    padded_h, padded_w = padded
    batch = windows.shape[0] // ((padded_h // window) * (padded_w // window))
    inputs = windows.reshape(batch, padded_h // window, padded_w // window, window, window, -1)
    inputs = inputs.permute(0, 1, 3, 2, 4, 5).reshape(batch, padded_h, padded_w, -1)
    return inputs[:, : original[0], : original[1]]


class ImageBlock(nn.Module):
    """SAM ViT block with window or global attention."""

    def __init__(self, channels: int, heads: int, mlp_ratio: float, window: int, grid_size: int) -> None:
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        attention_size = (window, window) if window else (grid_size, grid_size)
        self.attention = Attention(channels, heads, True, True, attention_size)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)), nn.GELU(), nn.Linear(int(channels * mlp_ratio), channels)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(inputs)
        if self.window:
            original = tuple(hidden.shape[1:3])
            hidden, padded = _partition(hidden, self.window)
            hidden = _unpartition(self.attention(hidden), self.window, padded, original)
        else:
            hidden = self.attention(hidden)
        hidden = inputs + hidden
        return hidden + self.mlp(self.norm2(hidden))


class LoRAQKV(nn.Module):
    """Frozen SAM QKV projection plus trainable low-rank query/value updates."""

    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base = base
        channels = base.in_features
        self.query_down, self.query_up = nn.Linear(channels, rank, bias=False), nn.Linear(rank, channels, bias=False)
        self.value_down, self.value_up = nn.Linear(channels, rank, bias=False), nn.Linear(rank, channels, bias=False)
        nn.init.kaiming_uniform_(self.query_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.query_up.weight)
        nn.init.zeros_(self.value_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        qkv = self.base(inputs)
        channels = self.base.in_features
        query_update = self.query_up(self.query_down(inputs))
        value_update = self.value_up(self.value_down(inputs))
        return torch.cat(
            [qkv[..., :channels] + query_update, qkv[..., channels : 2 * channels], qkv[..., 2 * channels :] + value_update],
            dim=-1,
        )


class ImageEncoderViT(nn.Module):
    """SAM image encoder with ViT-B defaults."""

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        channels: int = 768,
        depth: int = 12,
        heads: int = 12,
        output_channels: int = 256,
        window_size: int = 14,
        global_attention_indexes: Sequence[int] = (2, 5, 8, 11),
    ) -> None:
        super().__init__()
        self.image_size = image_size
        grid_size = image_size // patch_size
        self.patch_embedding = nn.Conv2d(3, channels, patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, grid_size, grid_size, channels))
        self.blocks = nn.ModuleList(
            [
                ImageBlock(channels, heads, 4.0, 0 if index in global_attention_indexes else window_size, grid_size)
                for index in range(depth)
            ]
        )
        self.neck = nn.Sequential(
            nn.Conv2d(channels, output_channels, 1, bias=False),
            LayerNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            LayerNorm2d(output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_embedding(inputs).permute(0, 2, 3, 1)
        position = self.position_embedding
        if position.shape[1:3] != hidden.shape[1:3]:
            position = F.interpolate(position.permute(0, 3, 1, 2), size=hidden.shape[1:3], mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        hidden = hidden + position
        for block in self.blocks:
            hidden = block(hidden)
        return self.neck(hidden.permute(0, 3, 1, 2))


class TokenAttention(nn.Module):
    """Batch-first cross-attention used in SAM's two-way transformer."""

    def __init__(self, channels: int, heads: int, downsample: int = 1) -> None:
        super().__init__()
        internal = channels // downsample
        self.heads, self.internal = heads, internal
        self.query = nn.Linear(channels, internal)
        self.key = nn.Linear(channels, internal)
        self.value = nn.Linear(channels, internal)
        self.output = nn.Linear(internal, channels)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch = query.shape[0]
        q = self.query(query).reshape(batch, -1, self.heads, self.internal // self.heads).transpose(1, 2)
        k = self.key(key).reshape(batch, -1, self.heads, self.internal // self.heads).transpose(1, 2)
        v = self.value(value).reshape(batch, -1, self.heads, self.internal // self.heads).transpose(1, 2)
        output = (torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1]), dim=-1) @ v)
        return self.output(output.transpose(1, 2).reshape(batch, -1, self.internal))


class TwoWayBlock(nn.Module):
    """Token self-attention and bidirectional token/image cross-attention."""

    def __init__(self, channels: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.self_attention = TokenAttention(channels, heads)
        self.token_to_image = TokenAttention(channels, heads, 2)
        self.image_to_token = TokenAttention(channels, heads, 2)
        self.mlp = MLP(channels, mlp_dim, channels, 2)
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(4)])

    def forward(self, tokens: torch.Tensor, image: torch.Tensor, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.norms[0](tokens + self.self_attention(tokens, tokens, tokens))
        tokens = self.norms[1](tokens + self.token_to_image(tokens, image + position, image))
        tokens = self.norms[2](tokens + self.mlp(tokens))
        image = self.norms[3](image + self.image_to_token(image + position, tokens, tokens))
        return tokens, image


class MaskDecoder(nn.Module):
    """Prompt-free SAM mask decoder with one learned token per semantic class."""

    def __init__(self, num_classes: int, channels: int = 256, depth: int = 2, heads: int = 8) -> None:
        super().__init__()
        self.mask_tokens = nn.Embedding(num_classes, channels)
        self.no_prompt = nn.Embedding(1, channels)
        self.blocks = nn.ModuleList([TwoWayBlock(channels, heads, 2048) for _ in range(depth)])
        self.final_attention = TokenAttention(channels, heads, 2)
        self.final_norm = nn.LayerNorm(channels)
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(channels, 64, 2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.GELU(),
        )
        self.hypernetworks = nn.ModuleList([MLP(channels, channels, 32, 3) for _ in range(num_classes)])

    @staticmethod
    def _position(height: int, width: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        frequencies = torch.arange(channels // 4, device=device, dtype=dtype) + 1
        encoding = torch.cat(
            [torch.sin(math.pi * x[..., None] * frequencies), torch.cos(math.pi * x[..., None] * frequencies),
             torch.sin(math.pi * y[..., None] * frequencies), torch.cos(math.pi * y[..., None] * frequencies)], dim=-1
        )
        return encoding[..., :channels].reshape(1, height * width, channels)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = embeddings.shape
        image = embeddings.flatten(2).transpose(1, 2)
        position = self._position(height, width, channels, embeddings.device, embeddings.dtype).expand(batch, -1, -1)
        tokens = torch.cat([self.no_prompt.weight, self.mask_tokens.weight], dim=0).unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            tokens, image = block(tokens, image, position)
        tokens = self.final_norm(tokens + self.final_attention(tokens, image + position, image))[:, 1:]
        upscaled = self.upscale(image.transpose(1, 2).reshape(batch, channels, height, width))
        weights = torch.stack([network(tokens[:, index]) for index, network in enumerate(self.hypernetworks)], dim=1)
        return torch.einsum("bkc,bchw->bkhw", weights, upscaled)


class SAMed(nn.Module):
    """Semantic SAM with LoRA-tuned image attention and slice-wise 3D support."""

    def __init__(self, num_classes: int, rank: int = 4, **encoder_options: Any) -> None:
        super().__init__()
        self.image_encoder = ImageEncoderViT(**encoder_options)
        decoder_channels = int(encoder_options.get("output_channels", 256))
        self.mask_decoder = MaskDecoder(num_classes, decoder_channels)
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)
        for block in self.image_encoder.blocks:
            block.attention.qkv = LoRAQKV(block.attention.qkv, rank)

    def _low_resolution_2d(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        target_size = self.image_encoder.image_size
        resized = F.interpolate(inputs, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return self.mask_decoder(self.image_encoder(resized))

    def forward_low_resolution(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the native SAM decoder logits used by the official loss."""
        if inputs.ndim == 4:
            return self._low_resolution_2d(inputs)
        batch, channels, depth, height, width = inputs.shape
        slices = inputs.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
        logits = self._low_resolution_2d(slices)
        return logits.reshape(batch, depth, logits.shape[1], *logits.shape[2:]).permute(0, 2, 1, 3, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            return F.interpolate(
                self.forward_low_resolution(inputs), size=inputs.shape[2:], mode="bilinear", align_corners=False
            )
        if inputs.ndim != 5:
            raise ValueError("SAMed expects BCHW images or BCDHW volumes.")
        batch, channels, depth, height, width = inputs.shape
        logits = self.forward_low_resolution(inputs)
        return F.interpolate(logits, size=(depth, height, width), mode="trilinear", align_corners=False)


def build_model(config: Any) -> SAMed:
    """Build SAMed ViT-B with configurable encoder dimensions."""
    options = getattr(config, "model", config)
    model = SAMed(
        num_classes=int(config.num_classes),
        rank=int(getattr(options, "rank", 4)),
        image_size=int(getattr(options, "image_size", 512)),
        patch_size=int(getattr(options, "patch_size", 16)),
        channels=int(getattr(options, "channels", 768)),
        depth=int(getattr(options, "depth", 12)),
        heads=int(getattr(options, "heads", 12)),
        output_channels=int(getattr(options, "output_channels", 256)),
        window_size=int(getattr(options, "window_size", 14)),
        global_attention_indexes=tuple(getattr(options, "global_attention_indexes", (2, 5, 8, 11))),
    )
    checkpoint = getattr(options, "sam_checkpoint", None)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        converted = {}
        for key, value in state.items():
            if not key.startswith("image_encoder."):
                continue
            key = key.replace("image_encoder.patch_embed.proj", "image_encoder.patch_embedding")
            key = key.replace(".attn.", ".attention.")
            key = key.replace(".attention.qkv.", ".attention.qkv.base.")
            key = key.replace(".attention.proj.", ".attention.projection.")
            key = key.replace(".attention.rel_pos_h", ".attention.relative_height")
            key = key.replace(".attention.rel_pos_w", ".attention.relative_width")
            key = key.replace(".mlp.lin1.", ".mlp.0.")
            key = key.replace(".mlp.lin2.", ".mlp.2.")
            key = key.replace("image_encoder.pos_embed", "image_encoder.position_embedding")
            converted[key] = value
        expected = model.state_dict()
        compatible = {}
        for key, value in converted.items():
            if key not in expected:
                continue
            if value.shape != expected[key].shape and key.endswith("position_embedding"):
                value = F.interpolate(
                    value.permute(0, 3, 1, 2),
                    size=expected[key].shape[1:3],
                    mode="bicubic",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            elif value.shape != expected[key].shape and (
                key.endswith("relative_height") or key.endswith("relative_width")
            ):
                value = F.interpolate(
                    value.transpose(0, 1).unsqueeze(0),
                    size=expected[key].shape[0],
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1)
            if value.shape == expected[key].shape:
                compatible[key] = value
        model.load_state_dict(compatible, strict=False)
    return model
