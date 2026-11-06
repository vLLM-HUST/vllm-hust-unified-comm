# SPDX-License-Identifier: Apache-2.0
"""Mountable adapter for the vLLM-HUST communication-provider seam."""

from typing import Any

try:
    from vllm.distributed.communication_provider import CommunicationProviderBase
except ModuleNotFoundError as exc:
    if exc.name not in {
        "vllm",
        "vllm.distributed",
        "vllm.distributed.communication_provider",
    }:
        raise
    raise RuntimeError(
        "Unified Communication requires the vLLM-HUST communication-provider "
        "host contract"
    ) from exc


class UnifiedCommunicationProvider(CommunicationProviderBase):
    """Delegate collectives to the host-owned device communicator.

    The MOD never creates or destroys a process group. Mounting changes the
    dispatch owner only; disabling the environment claim restores the native
    host path on the next process start.
    """

    def __init__(self, device_communicator: Any, group_name: str) -> None:
        super().__init__(device_communicator=device_communicator, group_name=group_name)
        self._active = True

    def _communicator(self) -> Any:
        if not self._active:
            raise RuntimeError("Unified Communication provider is already destroyed")
        return self.device_communicator

    def all_reduce(self, input_: Any) -> Any:
        return self._communicator().all_reduce(input_)

    def all_gather(self, input_: Any, dim: int) -> Any:
        return self._communicator().all_gather(input_, dim)

    def reduce_scatter(self, input_: Any, dim: int) -> Any:
        return self._communicator().reduce_scatter(input_, dim)

    def broadcast(self, input_: Any, src: int) -> Any:
        return self._communicator().broadcast(input_, src)

    def destroy(self) -> None:
        # The host owns the communicator and process group lifecycle.
        self._active = False


__all__ = ["UnifiedCommunicationProvider"]
