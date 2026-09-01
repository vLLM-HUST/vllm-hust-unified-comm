# SPDX-License-Identifier: Apache-2.0
"""Communication decision policies extracted from the legacy implementation."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import (
    CommAlgorithm,
    CommContext,
    CommDecision,
    CommPattern,
    TransferProtocol,
)


class CommStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def decide(self, context: CommContext) -> CommDecision: ...


class DefaultStrategy:
    SMALL_MSG_THRESHOLD = 256 * 1024
    LARGE_MSG_THRESHOLD = 64 * 1024 * 1024
    BUCKET_SIZE = 25 * 1024 * 1024

    @property
    def name(self) -> str:
        return "default"

    def decide(self, context: CommContext) -> CommDecision:
        if context.pattern is CommPattern.KV_TRANSFER:
            return self._kv_transfer(context)
        if context.pattern is CommPattern.EC_DISPATCH:
            return CommDecision(
                algorithm=CommAlgorithm.AUTO,
                protocol=TransferProtocol.COLLECTIVE,
                overlap_compute=context.urgency == "high",
            )
        if context.pattern is CommPattern.WEIGHT_SYNC:
            return self._weight_sync(context)
        return self._collective(context)

    def _collective(self, context: CommContext) -> CommDecision:
        if context.tensor_size_bytes < self.SMALL_MSG_THRESHOLD:
            algorithm = CommAlgorithm.DIRECT
        elif context.is_intra_node and context.topology.has_high_bandwidth_fabric:
            algorithm = CommAlgorithm.TREE
        elif (
            context.tensor_size_bytes > self.LARGE_MSG_THRESHOLD
            and not context.is_intra_node
        ):
            return CommDecision(
                algorithm=CommAlgorithm.BUCKET,
                protocol=TransferProtocol.COLLECTIVE,
                num_chunks=max(1, context.tensor_size_bytes // self.BUCKET_SIZE),
                overlap_compute=True,
            )
        else:
            algorithm = CommAlgorithm.RING
        return CommDecision(algorithm, TransferProtocol.COLLECTIVE)

    @staticmethod
    def _kv_transfer(context: CommContext) -> CommDecision:
        if context.topology.has_rdma and not context.is_intra_node:
            return CommDecision(
                CommAlgorithm.RDMA_BASED,
                TransferProtocol.RDMA,
                use_high_priority_stream=True,
            )
        if context.is_intra_node:
            return CommDecision(CommAlgorithm.SHM_BASED, TransferProtocol.SHM)
        return CommDecision(
            CommAlgorithm.P2P_BASED,
            TransferProtocol.P2P,
            use_high_priority_stream=True,
        )

    def _weight_sync(self, context: CommContext) -> CommDecision:
        if context.tensor_size_bytes > self.LARGE_MSG_THRESHOLD:
            return CommDecision(
                CommAlgorithm.BUCKET,
                TransferProtocol.COLLECTIVE,
                num_chunks=max(1, context.tensor_size_bytes // self.BUCKET_SIZE),
            )
        return CommDecision(CommAlgorithm.DIRECT, TransferProtocol.COLLECTIVE)


class ConfigDrivenStrategy:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "config_driven"

    def decide(self, context: CommContext) -> CommDecision:
        config = self._config.get(context.pattern.name.lower(), {})
        algorithm = CommAlgorithm.__members__.get(
            str(config.get("algorithm", "auto")).upper(), CommAlgorithm.AUTO
        )
        protocol = TransferProtocol.__members__.get(
            str(config.get("protocol", "collective")).upper(),
            TransferProtocol.COLLECTIVE,
        )
        return CommDecision(
            algorithm=algorithm,
            protocol=protocol,
            use_high_priority_stream=bool(config.get("high_priority", False)),
            num_chunks=int(config.get("num_chunks", 1)),
            overlap_compute=bool(config.get("overlap_compute", False)),
            compression=config.get("compression"),
            backend_hint=config.get("backend"),
        )


__all__ = ["CommStrategy", "ConfigDrivenStrategy", "DefaultStrategy"]
