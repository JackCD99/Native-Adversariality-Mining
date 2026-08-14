"""Dimension-agnostic adversariality miner used by NAM."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    """Residual block shared by the 2D and 3D miners."""

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.conv1 = conv(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = conv(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.skip = conv(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        return F.silu(hidden + self.skip(inputs))


class SpatialSelfAttention(nn.Module):
    """Memory-aware self-attention at the miner bottleneck."""

    def __init__(self, spatial_dims: int, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Attention channels must be divisible by the head count.")
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = conv(channels, channels * 3, 1)
        self.projection = conv(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, *spatial = inputs.shape
        locations = int(torch.tensor(spatial).prod().item())
        qkv = self.qkv(self.norm(inputs)).reshape(
            batch, 3, self.heads, self.head_dim, locations
        )
        query, key, value = qkv.unbind(dim=1)
        attention = torch.matmul(query.transpose(-1, -2), key) * self.scale
        attention = attention.softmax(dim=-1)
        hidden = torch.matmul(attention, value.transpose(-1, -2))
        hidden = hidden.transpose(-1, -2).reshape(batch, channels, *spatial)
        return inputs + self.projection(hidden)


class AdversarialityMiner(nn.Module):
    """Four-level Res-UNet predicting diagonal Gaussian offsets.

    The independent mean and variance heads are zero initialized, making the
    initial reselection distribution exactly the standard Gaussian prior.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        noise_channels: int,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4, 4),
        downsample_levels: int | None = None,
        attention_heads: int = 4,
        head_channels: int | None = None,
        final_head_bias: bool = True,
    ) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("The miner supports 2D or 3D tensors only.")
        if len(channel_multipliers) != 4:
            raise ValueError("NAM miners use exactly four resolution blocks.")
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        pool = nn.AvgPool2d if spatial_dims == 2 else nn.AvgPool3d
        mode = "bilinear" if spatial_dims == 2 else "trilinear"
        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.spatial_dims = spatial_dims
        self.downsample_levels = downsample_levels or (2 if spatial_dims == 2 else 3)
        self.mode = mode

        self.encoder = nn.ModuleList()
        previous = in_channels
        for current in channels:
            self.encoder.append(ResBlock(spatial_dims, previous, current))
            previous = current
        self.pool = pool(2)
        self.middle = SpatialSelfAttention(spatial_dims, channels[-1], attention_heads)

        self.decoder = nn.ModuleList()
        for level in range(2, -1, -1):
            self.decoder.append(
                ResBlock(spatial_dims, channels[level + 1] + channels[level], channels[level])
            )
        self.refine = ResBlock(spatial_dims, channels[0], channels[0])

        head_channels = int(head_channels or max(16, channels[0] // 2))
        self.mean_head = nn.Sequential(
            conv(channels[0], head_channels, 3, padding=1),
            nn.SiLU(),
            conv(head_channels, noise_channels, 3, padding=1, bias=final_head_bias),
        )
        self.variance_head = nn.Sequential(
            conv(channels[0], head_channels, 3, padding=1),
            nn.SiLU(),
            conv(head_channels, noise_channels, 3, padding=1, bias=final_head_bias),
        )
        for head in (self.mean_head, self.variance_head):
            nn.init.zeros_(head[-1].weight)
            if head[-1].bias is not None:
                nn.init.zeros_(head[-1].bias)

    def forward(self, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features: list[torch.Tensor] = []
        hidden = score
        for level, block in enumerate(self.encoder):
            hidden = block(hidden)
            features.append(hidden)
            if level < len(self.encoder) - 1 and level < self.downsample_levels:
                hidden = self.pool(hidden)
        hidden = self.middle(hidden)
        for decoder, skip in zip(self.decoder, reversed(features[:-1])):
            hidden = F.interpolate(hidden, size=skip.shape[2:], mode=self.mode, align_corners=False)
            hidden = decoder(torch.cat([hidden, skip], dim=1))
        hidden = self.refine(hidden)
        return self.mean_head(hidden), self.variance_head(hidden)


class ResUNet3DMiner(AdversarialityMiner):
    """Three-dimensional ResUNet miner for volumetric diffusion backbones.

    VolDiT and MAISI both operate on five-dimensional latent tensors. The
    volumetric configuration downsamples only the first two encoder levels for
    latent diffusion models, keeping the bottleneck memory tractable while
    preserving anisotropic depth information.
    """

    def __init__(
        self,
        in_channels: int,
        noise_channels: int,
        base_channels: int = 82,
        channel_multipliers: Sequence[int] = (1, 2, 4, 4),
        attention_heads: int = 4,
        head_channels: int | None = None,
    ) -> None:
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            noise_channels=noise_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            downsample_levels=2,
            attention_heads=attention_heads,
            head_channels=head_channels or {4: 42, 8: 38}.get(int(noise_channels), 41),
        )

    def forward(self, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if score.ndim != 5:
            raise ValueError(
                "ResUNet3DMiner expects BxCxDxHxW volumetric score tensors."
            )
        return super().forward(score)


class JoDiffusionMiner(AdversarialityMiner):
    """Single 8-channel miner for JoDiffusion's shared image-mask noise."""

    def __init__(self) -> None:
        super().__init__(
            spatial_dims=2, in_channels=8, noise_channels=8,
            base_channels=79, channel_multipliers=(1, 2, 4, 4),
            downsample_levels=2, attention_heads=4, head_channels=93,
            final_head_bias=False,
        )


class MedSegFactoryDualMiner(nn.Module):
    """Two identical miners producing branch-specific noises from joint scores.

    Both branches observe ``[S_img, S_mask]``. Separate parameter sets produce
    the image and mask Gaussian offsets, exactly matching the dual-stream
    adaptation used by the joint image-mask pipeline.
    """

    def __init__(self) -> None:
        super().__init__()
        kwargs = dict(
            spatial_dims=2, in_channels=8, noise_channels=4,
            base_channels=79, channel_multipliers=(1, 2, 4, 4),
            downsample_levels=2, attention_heads=4, head_channels=96,
        )
        self.image_miner = AdversarialityMiner(**kwargs)
        self.mask_miner = AdversarialityMiner(**kwargs)

    def forward(
        self, joint_score: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        if joint_score.ndim != 4 or joint_score.shape[1] != 8:
            raise ValueError("MedSegFactoryDualMiner expects Bx8xHxW joint scores.")
        return self.image_miner(joint_score), self.mask_miner(joint_score)


def reference_miner_configuration(noise_channels: int) -> dict[str, int | tuple[int, ...]]:
    """Return the released 2D ResUNet dimensions.

    Pixel-space SegDiff and four-channel latent pipelines share the same
    four-level topology but use channel-dependent output heads. These dimensions
    keep their parameter counts within rounding distance of the reported
    8.387/8.388 million parameters for one miner.
    """
    channels = int(noise_channels)
    return {
        "base_channels": 79,
        "channel_multipliers": (1, 2, 4, 4),
        "downsample_levels": 2,
        "head_channels": {1: 103, 3: 100, 4: 98}.get(channels, 96),
    }
