import pytest

from vllm_hust_unified_comm import (
    CommAlgorithm,
    CommBackendRegistry,
    CommContext,
    CommPattern,
    DefaultStrategy,
    TopologyInfo,
    TransferProtocol,
)


class FakeBackend:
    name = "fake"
    device_type = "test"

    def is_available(self) -> bool:
        return True


def test_registry_rejects_backend_collision() -> None:
    registry = CommBackendRegistry()
    registry.register(FakeBackend())
    assert registry.select("test").name == "fake"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeBackend())


def test_default_strategy_selects_rdma_for_cross_node_kv() -> None:
    decision = DefaultStrategy().decide(
        CommContext(
            pattern=CommPattern.KV_TRANSFER,
            tensor_size_bytes=1024,
            world_size=2,
            topology=TopologyInfo(num_nodes=2, has_rdma=True),
            is_intra_node=False,
        )
    )
    assert decision.algorithm is CommAlgorithm.RDMA_BASED
    assert decision.protocol is TransferProtocol.RDMA


def test_default_strategy_uses_shared_memory_intra_node() -> None:
    decision = DefaultStrategy().decide(
        CommContext(
            pattern=CommPattern.KV_TRANSFER,
            tensor_size_bytes=1024,
            world_size=2,
            topology=TopologyInfo(),
        )
    )
    assert decision.protocol is TransferProtocol.SHM
