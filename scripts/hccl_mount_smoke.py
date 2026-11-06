# SPDX-License-Identifier: Apache-2.0
"""Two-or-more-rank HCCL smoke for the mountable communication MOD."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.distributed.communication_provider import (
    load_communication_provider,
    run_collective,
)


class HostOwnedHcclCommunicator:
    """Minimal stand-in for the communicator whose lifecycle vLLM owns."""

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        output = input_.clone()
        dist.all_reduce(output)
        return output

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        outputs = [torch.empty_like(input_) for _ in range(dist.get_world_size())]
        dist.all_gather(outputs, input_)
        return torch.cat(outputs, dim=dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if dim != 0:
            raise ValueError("the smoke communicator supports dim=0 only")
        output = torch.empty(
            input_.shape[0] // dist.get_world_size(),
            dtype=input_.dtype,
            device=input_.device,
        )
        dist.reduce_scatter_tensor(output, input_)
        return output

    def broadcast(self, input_: torch.Tensor, src: int) -> torch.Tensor:
        output = input_.clone()
        dist.broadcast(output, src=src)
        return output


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"npu:{local_rank}")
    communicator = HostOwnedHcclCommunicator()
    provider = load_communication_provider(
        "vllm_hust_unified_comm.provider.UnifiedCommunicationProvider",
        communicator,
        "tp:hccl-mount-smoke",
    )

    value = torch.tensor([rank + 1.0], device=device)
    reduced = run_collective(provider, "all_reduce", communicator.all_reduce, value)
    expected_sum = world_size * (world_size + 1) / 2
    assert reduced.item() == expected_sum

    gathered = run_collective(provider, "all_gather", communicator.all_gather, value, 0)
    assert gathered.cpu().tolist() == [float(index + 1) for index in range(world_size)]

    scatter_input = torch.full((world_size,), rank + 1.0, device=device)
    scattered = run_collective(
        provider,
        "reduce_scatter",
        communicator.reduce_scatter,
        scatter_input,
        0,
    )
    assert scattered.item() == expected_sum

    broadcast_input = torch.tensor([37.0 if rank == 0 else -1.0], device=device)
    broadcast = run_collective(
        provider,
        "broadcast",
        communicator.broadcast,
        broadcast_input,
        0,
    )
    assert broadcast.item() == 37.0

    provider.destroy()
    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "backend": dist.get_backend(),
                    "collectives": [
                        "all_reduce",
                        "all_gather",
                        "reduce_scatter",
                        "broadcast",
                    ],
                    "provider": type(provider).__name__,
                    "status": "PASS",
                    "world_size": world_size,
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
