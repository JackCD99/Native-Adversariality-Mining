"""Minimal distributed helpers shared by 2D and 3D training."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize(device_name: str = "cuda") -> DistributedContext:
    """Initialize NCCL when launched by torchrun; otherwise use one process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed NAM training currently requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    return DistributedContext(device, rank, world_size, local_rank)


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Return a detached cross-rank mean."""
    result = value.detach().clone()
    if context.world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= context.world_size
    return result


def finalize(context: DistributedContext) -> None:
    """Release the distributed process group."""
    if context.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
