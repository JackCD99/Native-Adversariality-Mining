"""Self-configuring PlainConvUNet used by nnU-Net v2.

The implementation retains the architectural rules used by the experiments
while removing nnU-Net's planning, preprocessing, cascade, and export systems.
It follows the official nnU-Net v2 network contract and dynamic-network-
architectures PlainConvUNet design:

https://github.com/MIC-DKFZ/nnUNet
https://github.com/MIC-DKFZ/dynamic-network-architectures

Reference: Isensee et al., Nature Methods 2021.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _expand(value: int | Sequence[int], stages: int) -> list[int]:
    if isinstance(value, int):
        return [value] * stages
    values = list(value)
    if len(values) != stages:
        raise ValueError(f"Expected {stages} stage values, received {len(values)}.")
    return values


def _spatial_tuple(value: int | Sequence[int], spatial_dims: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * spatial_dims
    result = tuple(value)
    if len(result) != spatial_dims:
        raise ValueError(f"Expected a {spatial_dims}D kernel/stride, received {result}.")
    return result


class ConvNormNonlinearity(nn.Module):
    """nnU-Net convolution block: convolution, InstanceNorm, LeakyReLU."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int],
    ) -> None:
        super().__init__()
        convolution = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        normalization = nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d
        padding = tuple(kernel // 2 for kernel in kernel_size)
        self.block = nn.Sequential(
            convolution(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            normalization(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class StackedConvBlocks(nn.Module):
    """A configurable stack of nnU-Net convolution blocks."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        first_stride: Sequence[int],
        block_count: int,
    ) -> None:
        super().__init__()
        blocks = []
        for index in range(block_count):
            blocks.append(
                ConvNormNonlinearity(
                    spatial_dims,
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    first_stride if index == 0 else (1,) * spatial_dims,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


class NNUNet(nn.Module):
    """Plan-configurable 2D/3D PlainConvUNet with deep supervision."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_classes: int,
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[int | Sequence[int]],
        strides: Sequence[int | Sequence[int]],
        encoder_blocks: int | Sequence[int] = 2,
        decoder_blocks: int | Sequence[int] = 2,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("nnU-Net supports 2D and 3D configurations.")
        stages = len(features_per_stage)
        if len(kernel_sizes) != stages or len(strides) != stages:
            raise ValueError("features_per_stage, kernel_sizes, and strides must align.")
        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.strides = [_spatial_tuple(value, spatial_dims) for value in strides]
        kernels = [_spatial_tuple(value, spatial_dims) for value in kernel_sizes]
        encoder_counts = _expand(encoder_blocks, stages)
        decoder_counts = _expand(decoder_blocks, stages - 1)

        self.encoder = nn.ModuleList()
        previous = in_channels
        for stage, features in enumerate(features_per_stage):
            self.encoder.append(
                StackedConvBlocks(
                    spatial_dims,
                    previous,
                    features,
                    kernels[stage],
                    self.strides[stage],
                    encoder_counts[stage],
                )
            )
            previous = features

        transposed = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        convolution = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.upsampling = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.segmentation_heads = nn.ModuleList()
        for stage in range(stages - 2, -1, -1):
            lower_features = features_per_stage[stage + 1]
            skip_features = features_per_stage[stage]
            stride = self.strides[stage + 1]
            self.upsampling.append(
                transposed(lower_features, skip_features, kernel_size=stride, stride=stride)
            )
            self.decoder.append(
                StackedConvBlocks(
                    spatial_dims,
                    skip_features * 2,
                    skip_features,
                    kernels[stage],
                    (1,) * spatial_dims,
                    decoder_counts[stages - 2 - stage],
                )
            )
            self.segmentation_heads.append(convolution(skip_features, num_classes, 1))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        skips = []
        hidden = inputs
        for block in self.encoder:
            hidden = block(hidden)
            skips.append(hidden)

        outputs = []
        for upsample, decoder, head, skip in zip(
            self.upsampling, self.decoder, self.segmentation_heads, reversed(skips[:-1])
        ):
            hidden = upsample(hidden)
            if hidden.shape[2:] != skip.shape[2:]:
                hidden = F.interpolate(hidden, size=skip.shape[2:], mode="nearest")
            hidden = decoder(torch.cat([hidden, skip], dim=1))
            outputs.append(head(hidden))
        outputs.reverse()
        return outputs if self.deep_supervision and self.training else outputs[0]


def default_features(spatial_dims: int, stages: int) -> list[int]:
    """Apply nnU-Net's feature-doubling rule with dimension-specific caps."""
    base, cap = (32, 512) if spatial_dims == 2 else (32, 320)
    return [min(base * (2**stage), cap) for stage in range(stages)]


def build_model(config: Any) -> NNUNet:
    """Build an nnU-Net directly from the experiment configuration."""
    spatial_dims = int(config.spatial_dims)
    model_config = getattr(config, "model", config)
    stages = int(getattr(model_config, "stages", 6 if spatial_dims == 2 else 5))
    features = list(getattr(model_config, "features_per_stage", default_features(spatial_dims, stages)))
    kernels = list(getattr(model_config, "kernel_sizes", [[3] * spatial_dims] * stages))
    default_strides = [[1] * spatial_dims] + [[2] * spatial_dims] * (stages - 1)
    strides = list(getattr(model_config, "strides", default_strides))
    return NNUNet(
        spatial_dims=spatial_dims,
        in_channels=int(getattr(config, "in_channels", 1)),
        num_classes=int(config.num_classes),
        features_per_stage=features,
        kernel_sizes=kernels,
        strides=strides,
        encoder_blocks=getattr(model_config, "encoder_blocks", 2),
        decoder_blocks=getattr(model_config, "decoder_blocks", 2),
        deep_supervision=bool(getattr(model_config, "deep_supervision", True)),
    )
