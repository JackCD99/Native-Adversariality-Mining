"""Minimal Swin-Unet (2D) and SwinUNETR (3D) architectures.

This file retains the defining official components: non-overlapping patch
embedding, shifted-window self-attention with relative-position bias, patch
merging, hierarchical transformer stages, and skip-connected decoders. The 2D
decoder is transformer-based as in Swin-Unet; the 3D decoder uses residual
convolutional UNETR blocks as in SwinUNETR.

Official sources:
    https://github.com/HuCaoFighting/Swin-Unet
    https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR

References: Cao et al., ECCVW 2022; Hatamizadeh et al., WACV 2022.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _tuple(value: int | Sequence[int], dimensions: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * dimensions
    result = tuple(value)
    if len(result) != dimensions:
        raise ValueError(f"Expected {dimensions} values, received {result}.")
    return result


def _channels_first(inputs: torch.Tensor) -> torch.Tensor:
    order = (0, inputs.ndim - 1, *range(1, inputs.ndim - 1))
    return inputs.permute(order).contiguous()


def _channels_last(inputs: torch.Tensor) -> torch.Tensor:
    order = (0, *range(2, inputs.ndim), 1)
    return inputs.permute(order).contiguous()


def window_partition(inputs: torch.Tensor, window_size: tuple[int, ...]) -> torch.Tensor:
    """Partition a channels-last tensor into flattened local windows."""
    batch, *spatial, channels = inputs.shape
    view_shape: list[int] = [batch]
    for size, window in zip(spatial, window_size):
        view_shape.extend([size // window, window])
    view_shape.append(channels)
    inputs = inputs.reshape(view_shape)
    dimensions = len(window_size)
    grid_axes = [1 + 2 * index for index in range(dimensions)]
    window_axes = [2 + 2 * index for index in range(dimensions)]
    order = [0, *grid_axes, *window_axes, 1 + 2 * dimensions]
    windows = inputs.permute(order).contiguous()
    return windows.reshape(-1, math.prod(window_size), channels)


def window_reverse(
    windows: torch.Tensor, window_size: tuple[int, ...], spatial: tuple[int, ...]
) -> torch.Tensor:
    """Reverse ``window_partition`` into a channels-last tensor."""
    windows_per_sample = math.prod(size // window for size, window in zip(spatial, window_size))
    batch = windows.shape[0] // windows_per_sample
    channels = windows.shape[-1]
    grid = [size // window for size, window in zip(spatial, window_size)]
    shaped = windows.reshape(batch, *grid, *window_size, channels)
    dimensions = len(window_size)
    order = [0]
    for index in range(dimensions):
        order.extend([1 + index, 1 + dimensions + index])
    order.append(1 + 2 * dimensions)
    return shaped.permute(order).contiguous().reshape(batch, *spatial, channels)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return inputs
        keep = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random = keep + torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
        return inputs * random.floor() / keep


class MLP(nn.Module):
    """Swin feed-forward network with GELU activation."""

    def __init__(self, channels: int, ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(channels * ratio)
        self.layers = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class WindowAttention(nn.Module):
    """Multi-head window attention with learned relative-position bias."""

    def __init__(
        self,
        channels: int,
        window_size: tuple[int, ...],
        heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Swin channels must be divisible by the number of heads.")
        self.channels = channels
        self.window_size = window_size
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        table_size = math.prod(2 * size - 1 for size in window_size)
        self.relative_position_bias = nn.Parameter(torch.zeros(table_size, heads))
        coordinates = torch.stack(
            torch.meshgrid(*[torch.arange(size) for size in window_size], indexing="ij")
        ).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        for axis, size in enumerate(window_size):
            relative[..., axis] += size - 1
        multipliers = []
        for axis in range(len(window_size)):
            multipliers.append(math.prod(2 * size - 1 for size in window_size[axis + 1 :]))
        index = sum(relative[..., axis] * multipliers[axis] for axis in range(len(window_size)))
        self.register_buffer("relative_position_index", index.long(), persistent=False)
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(channels, channels)
        self.projection_dropout = nn.Dropout(projection_dropout)
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, tokens, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_windows, tokens, 3, self.heads, channels // self.heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        bias = self.relative_position_bias[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0)
        if mask is not None:
            window_count = mask.shape[0]
            attention = attention.reshape(
                batch_windows // window_count, window_count, self.heads, tokens, tokens
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.reshape(batch_windows, self.heads, tokens, tokens)
        attention = self.attention_dropout(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.projection_dropout(self.projection(output))


def shifted_window_mask(
    padded_spatial: tuple[int, ...], window_size: tuple[int, ...], shift_size: tuple[int, ...], device: torch.device
) -> torch.Tensor | None:
    """Build the official Swin cyclic-shift attention mask."""
    if not any(shift_size):
        return None
    mask = torch.zeros((1, *padded_spatial, 1), device=device)
    slices_per_axis = []
    for window, shift in zip(window_size, shift_size):
        slices_per_axis.append(
            (slice(0, -window), slice(-window, -shift), slice(-shift, None))
        )
    counter = 0
    for regions in itertools.product(*slices_per_axis):
        mask[(slice(None), *regions, slice(None))] = counter
        counter += 1
    windows = window_partition(mask, window_size).squeeze(-1)
    difference = windows.unsqueeze(1) - windows.unsqueeze(2)
    return difference.masked_fill(difference != 0, -100.0).masked_fill(difference == 0, 0.0)


class SwinBlock(nn.Module):
    """Pre-normalized shifted-window transformer block."""

    def __init__(
        self,
        channels: int,
        heads: int,
        window_size: tuple[int, ...],
        shifted: bool,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(size // 2 if shifted else 0 for size in window_size)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = WindowAttention(channels, window_size, heads, projection_dropout=dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(channels, mlp_ratio, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = inputs
        hidden = self.norm1(inputs)
        spatial = tuple(hidden.shape[1:-1])
        padding = tuple((window - size % window) % window for size, window in zip(spatial, self.window_size))
        channel_first = _channels_first(hidden)
        pad_values = []
        for amount in reversed(padding):
            pad_values.extend([0, amount])
        channel_first = F.pad(channel_first, pad_values)
        hidden = _channels_last(channel_first)
        padded_spatial = tuple(hidden.shape[1:-1])
        if any(self.shift_size):
            hidden = torch.roll(hidden, shifts=tuple(-value for value in self.shift_size), dims=tuple(range(1, hidden.ndim - 1)))
        mask = shifted_window_mask(padded_spatial, self.window_size, self.shift_size, hidden.device)
        windows = window_partition(hidden, self.window_size)
        windows = self.attention(windows, mask)
        hidden = window_reverse(windows, self.window_size, padded_spatial)
        if any(self.shift_size):
            hidden = torch.roll(hidden, shifts=self.shift_size, dims=tuple(range(1, hidden.ndim - 1)))
        crop = (slice(None), *(slice(0, size) for size in spatial), slice(None))
        hidden = hidden[crop]
        hidden = shortcut + self.drop_path(hidden)
        return hidden + self.drop_path(self.mlp(self.norm2(hidden)))


class SwinStage(nn.Module):
    """Alternating regular and shifted Swin blocks."""

    def __init__(
        self,
        channels: int,
        depth: int,
        heads: int,
        window_size: tuple[int, ...],
        mlp_ratio: float,
        dropout: float,
        drop_paths: Sequence[float],
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    channels, heads, window_size, index % 2 == 1, mlp_ratio, dropout, drop_paths[index]
                )
                for index in range(depth)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


class PatchMerging(nn.Module):
    """Concatenate 2^D neighboring tokens and linearly reduce to 2C."""

    def __init__(self, dimensions: int, channels: int) -> None:
        super().__init__()
        self.dimensions = dimensions
        factor = 2**dimensions
        self.norm = nn.LayerNorm(factor * channels)
        self.reduction = nn.Linear(factor * channels, 2 * channels, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial = tuple(inputs.shape[1:-1])
        padding = tuple(size % 2 for size in spatial)
        if any(padding):
            channel_first = _channels_first(inputs)
            values = []
            for amount in reversed(padding):
                values.extend([0, amount])
            inputs = _channels_last(F.pad(channel_first, values))
        pieces = []
        for offsets in itertools.product((0, 1), repeat=self.dimensions):
            index = (slice(None), *(slice(offset, None, 2) for offset in offsets), slice(None))
            pieces.append(inputs[index])
        return self.reduction(self.norm(torch.cat(pieces, dim=-1)))


class PatchExpanding(nn.Module):
    """Swin-Unet token upsampling by linear expansion and pixel rearrangement."""

    def __init__(self, channels: int, scale: int = 2, output_channels: int | None = None) -> None:
        super().__init__()
        if output_channels is None and scale == 2 and channels % 2:
            raise ValueError("PatchExpanding requires an even channel count.")
        self.scale = scale
        output_channels = output_channels or (channels // 2 if scale == 2 else channels)
        self.output_channels = output_channels
        self.expansion = nn.Linear(channels, output_channels * scale * scale, bias=False)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = inputs.shape
        hidden = self.expansion(inputs).reshape(
            batch, height, width, self.scale, self.scale, self.output_channels
        )
        hidden = hidden.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, height * self.scale, width * self.scale, self.output_channels
        )
        return self.norm(hidden)


class ResidualConvBlock(nn.Module):
    """Instance-normalized residual block used by the SwinUNETR decoder."""

    def __init__(self, dimensions: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        convolution = nn.Conv2d if dimensions == 2 else nn.Conv3d
        normalization = nn.InstanceNorm2d if dimensions == 2 else nn.InstanceNorm3d
        self.body = nn.Sequential(
            convolution(in_channels, out_channels, 3, padding=1, bias=False),
            normalization(out_channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
            convolution(out_channels, out_channels, 3, padding=1, bias=False),
            normalization(out_channels, affine=True),
        )
        self.skip = convolution(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.activation = nn.LeakyReLU(1e-2, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class SwinEncoder(nn.Module):
    """Hierarchical Swin Transformer encoder shared by both variants."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        feature_size: int,
        patch_size: tuple[int, ...],
        depths: Sequence[int],
        heads: Sequence[int],
        window_size: tuple[int, ...],
        mlp_ratio: float,
        dropout: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        convolution = nn.Conv2d if dimensions == 2 else nn.Conv3d
        self.patch_embedding = convolution(
            in_channels, feature_size, kernel_size=patch_size, stride=patch_size
        )
        self.position_dropout = nn.Dropout(dropout)
        total_blocks = sum(depths)
        drop_paths = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        offset = 0
        for stage, (depth, head_count) in enumerate(zip(depths, heads)):
            channels = feature_size * (2**stage)
            self.stages.append(
                SwinStage(
                    channels,
                    depth,
                    head_count,
                    window_size,
                    mlp_ratio,
                    dropout,
                    drop_paths[offset : offset + depth],
                )
            )
            offset += depth
            if stage < len(depths) - 1:
                self.mergers.append(PatchMerging(dimensions, channels))
        self.norms = nn.ModuleList(
            [nn.LayerNorm(feature_size * (2**stage)) for stage in range(len(depths))]
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.position_dropout(_channels_last(self.patch_embedding(inputs)))
        outputs = []
        for stage, layer in enumerate(self.stages):
            hidden = layer(hidden)
            outputs.append(_channels_first(self.norms[stage](hidden)))
            if stage < len(self.mergers):
                hidden = self.mergers[stage](hidden)
        return outputs


class SwinUnet(nn.Module):
    """Two-dimensional symmetric Swin Transformer U-Net."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feature_size: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        patch_size: int = 4,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_channels = in_channels
        self.encoder = SwinEncoder(
            2,
            3,
            feature_size,
            (patch_size, patch_size),
            depths,
            heads,
            (window_size, window_size),
            4.0,
            0.0,
            drop_path_rate,
        )
        self.up = nn.ModuleList()
        self.concat_projection = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for stage in range(len(depths) - 1, 0, -1):
            channels = feature_size * (2**stage)
            skip_channels = channels // 2
            self.up.append(PatchExpanding(channels))
            self.concat_projection.append(nn.Linear(skip_channels * 2, skip_channels))
            decoder_depth = depths[stage - 1]
            self.decoder_stages.append(
                SwinStage(
                    skip_channels,
                    decoder_depth,
                    heads[stage - 1],
                    (window_size, window_size),
                    4.0,
                    0.0,
                    [0.0] * decoder_depth,
                )
            )
        self.final_up = PatchExpanding(
            feature_size, scale=patch_size, output_channels=feature_size
        )
        self.output = nn.Conv2d(feature_size, num_classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_size = inputs.shape[2:]
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        features = self.encoder(inputs)
        hidden = _channels_last(features[-1])
        for upsample, projection, decoder, skip_channels_first in zip(
            self.up, self.concat_projection, self.decoder_stages, reversed(features[:-1])
        ):
            hidden = upsample(hidden)
            skip = _channels_last(skip_channels_first)
            if hidden.shape[1:3] != skip.shape[1:3]:
                hidden = _channels_last(
                    F.interpolate(_channels_first(hidden), size=skip.shape[1:3], mode="bilinear", align_corners=False)
                )
            hidden = decoder(projection(torch.cat([hidden, skip], dim=-1)))
        hidden = _channels_first(self.final_up(hidden))
        logits = self.output(hidden)
        return F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)


class SwinUNETR(nn.Module):
    """Three-dimensional Swin encoder with residual UNETR decoder."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 4,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = SwinEncoder(
            3,
            in_channels,
            feature_size,
            (2, 2, 2),
            depths,
            heads,
            (window_size, window_size, window_size),
            4.0,
            0.0,
            drop_path_rate,
        )
        self.input_encoder = ResidualConvBlock(3, in_channels, feature_size)
        channels = [feature_size * (2**stage) for stage in range(len(depths))]
        self.skip_encoders = nn.ModuleList(
            [ResidualConvBlock(3, channel, channel) for channel in channels[:-1]]
        )
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for stage in range(len(channels) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(channels[stage], channels[stage - 1], 2, stride=2))
            self.decoder.append(
                ResidualConvBlock(3, channels[stage - 1] * 2, channels[stage - 1])
            )
        self.final_up = nn.ConvTranspose3d(feature_size, feature_size, 2, stride=2)
        self.final_decoder = ResidualConvBlock(3, feature_size * 2, feature_size)
        self.output = nn.Conv3d(feature_size, num_classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_size = inputs.shape[2:]
        transformer_features = self.encoder(inputs)
        skips = [block(feature) for block, feature in zip(self.skip_encoders, transformer_features[:-1])]
        hidden = transformer_features[-1]
        for upsample, decoder, skip in zip(self.up, self.decoder, reversed(skips)):
            hidden = upsample(hidden)
            if hidden.shape[2:] != skip.shape[2:]:
                hidden = F.interpolate(hidden, size=skip.shape[2:], mode="trilinear", align_corners=False)
            hidden = decoder(torch.cat([hidden, skip], dim=1))
        hidden = self.final_up(hidden)
        input_skip = self.input_encoder(inputs)
        if hidden.shape[2:] != input_skip.shape[2:]:
            hidden = F.interpolate(hidden, size=input_skip.shape[2:], mode="trilinear", align_corners=False)
        logits = self.output(self.final_decoder(torch.cat([hidden, input_skip], dim=1)))
        return F.interpolate(logits, size=original_size, mode="trilinear", align_corners=False)


def build_model(config: Any) -> nn.Module:
    """Build the official 2D or 3D Swin segmentation variant."""
    model_config = getattr(config, "model", config)
    spatial_dims = int(config.spatial_dims)
    if spatial_dims == 2:
        return SwinUnet(
            in_channels=int(getattr(config, "in_channels", 1)),
            num_classes=int(config.num_classes),
            feature_size=int(getattr(model_config, "feature_size", 96)),
            depths=tuple(getattr(model_config, "depths", (2, 2, 6, 2))),
            heads=tuple(getattr(model_config, "heads", (3, 6, 12, 24))),
            window_size=int(getattr(model_config, "window_size", 7)),
            patch_size=int(getattr(model_config, "patch_size", 4)),
            drop_path_rate=float(getattr(model_config, "drop_path_rate", 0.2)),
        )
    return SwinUNETR(
        in_channels=int(getattr(config, "in_channels", 1)),
        num_classes=int(config.num_classes),
        feature_size=int(getattr(model_config, "feature_size", 48)),
        depths=tuple(getattr(model_config, "depths", (2, 2, 2, 2))),
        heads=tuple(getattr(model_config, "heads", (3, 6, 12, 24))),
        window_size=int(getattr(model_config, "window_size", 4)),
        drop_path_rate=float(getattr(model_config, "drop_path_rate", 0.0)),
    )
