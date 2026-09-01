# SPDX-License-Identifier: Apache-2.0
"""Host-independent communication contracts extracted from legacy PR #42."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


class TransferProtocol(Enum):
    COLLECTIVE = auto()
    P2P = auto()
    RDMA = auto()
    SHM = auto()
    STORE = auto()


class TransferType(Enum):
    KV_CACHE = auto()
    EC = auto()
    WEIGHT = auto()
    ACTIVATION = auto()
    CUSTOM = auto()


class CommPattern(Enum):
    ALL_REDUCE = auto()
    ALL_GATHER = auto()
    REDUCE_SCATTER = auto()
    BROADCAST = auto()
    ALL_TO_ALL = auto()
    P2P_SEND_RECV = auto()
    KV_TRANSFER = auto()
    WEIGHT_SYNC = auto()
    EC_DISPATCH = auto()


class CommAlgorithm(Enum):
    RING = auto()
    TREE = auto()
    RECURSIVE_HALVING = auto()
    DIRECT = auto()
    BUCKET = auto()
    COLLECTIVE_BASED = auto()
    P2P_BASED = auto()
    RDMA_BASED = auto()
    SHM_BASED = auto()
    AUTO = auto()


@dataclass(frozen=True)
class TopologyInfo:
    num_nodes: int = 1
    accelerators_per_node: int = 8
    intra_node_bandwidth_gbps: float = 600.0
    inter_node_bandwidth_gbps: float = 100.0
    has_high_bandwidth_fabric: bool = False
    has_rdma: bool = False
    device_type: str = "cuda"
    nic_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommContext:
    pattern: CommPattern
    tensor_size_bytes: int
    world_size: int
    topology: TopologyInfo
    is_intra_node: bool = True
    dtype: str = "float16"
    urgency: str = "normal"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tensor_size_bytes < 0 or self.world_size <= 0:
            raise ValueError("communication size/world size is invalid")


@dataclass(frozen=True)
class CommDecision:
    algorithm: CommAlgorithm
    protocol: TransferProtocol
    use_high_priority_stream: bool = False
    num_chunks: int = 1
    overlap_compute: bool = False
    compression: str | None = None
    backend_hint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CommBackendProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def device_type(self) -> str: ...

    def is_available(self) -> bool: ...


class CommBackendRegistry:
    """Explicit registry that rejects accidental backend replacement."""

    def __init__(self) -> None:
        self._backends: dict[str, CommBackendProvider] = {}

    def register(self, backend: CommBackendProvider) -> None:
        if not isinstance(backend, CommBackendProvider):
            raise TypeError("backend does not implement CommBackendProvider")
        if backend.name in self._backends:
            raise ValueError(f"backend already registered: {backend.name}")
        self._backends[backend.name] = backend

    def get(self, name: str) -> CommBackendProvider:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise KeyError(f"backend is not registered: {name}") from exc

    def select(self, device_type: str) -> CommBackendProvider:
        for backend in self._backends.values():
            if backend.device_type == device_type and backend.is_available():
                return backend
        raise LookupError(f"no available backend for device type: {device_type}")

    def list_backends(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))


__all__ = [
    "CommAlgorithm",
    "CommBackendProvider",
    "CommBackendRegistry",
    "CommContext",
    "CommDecision",
    "CommPattern",
    "TopologyInfo",
    "TransferProtocol",
    "TransferType",
]
