"""Unified communication contracts, policies, and inert activation metadata."""

from .contracts import (
    CommAlgorithm,
    CommBackendProvider,
    CommBackendRegistry,
    CommContext,
    CommDecision,
    CommPattern,
    TopologyInfo,
    TransferProtocol,
    TransferType,
)
from .strategy import CommStrategy, ConfigDrivenStrategy, DefaultStrategy


class VllmHustUnifiedCommContractProposal:
    """Metadata-only proposal; this class performs no runtime activation."""


__all__ = [
    "CommAlgorithm",
    "CommBackendProvider",
    "CommBackendRegistry",
    "CommContext",
    "CommDecision",
    "CommPattern",
    "CommStrategy",
    "ConfigDrivenStrategy",
    "DefaultStrategy",
    "TopologyInfo",
    "TransferProtocol",
    "TransferType",
    "VllmHustUnifiedCommContractProposal",
]
