# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


def _install_host_contract(monkeypatch: pytest.MonkeyPatch) -> type:
    module = ModuleType("vllm.distributed.communication_provider")

    class CommunicationProviderBase:
        def __init__(self, device_communicator, group_name):
            self.device_communicator = device_communicator
            self.group_name = group_name

    module.CommunicationProviderBase = CommunicationProviderBase
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed", ModuleType("vllm.distributed"))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sys.modules.pop("vllm_hust_unified_comm.provider", None)
    return CommunicationProviderBase


def test_provider_mounts_all_collectives_without_owning_host_lifecycle(monkeypatch):
    host_base = _install_host_contract(monkeypatch)
    from vllm_hust_unified_comm.provider import UnifiedCommunicationProvider

    communicator = Mock()
    communicator.all_reduce.return_value = "reduced"
    communicator.all_gather.return_value = "gathered"
    communicator.reduce_scatter.return_value = "scattered"
    communicator.broadcast.return_value = "broadcast"
    provider = UnifiedCommunicationProvider(communicator, "tp:0")

    assert isinstance(provider, host_base)
    assert provider.group_name == "tp:0"
    assert provider.all_reduce("x") == "reduced"
    assert provider.all_gather("x", 1) == "gathered"
    assert provider.reduce_scatter("x", 0) == "scattered"
    assert provider.broadcast("x", 2) == "broadcast"
    communicator.all_reduce.assert_called_once_with("x")
    communicator.all_gather.assert_called_once_with("x", 1)
    communicator.reduce_scatter.assert_called_once_with("x", 0)
    communicator.broadcast.assert_called_once_with("x", 2)

    provider.destroy()
    communicator.destroy.assert_not_called()
    with pytest.raises(RuntimeError, match="already destroyed"):
        provider.all_reduce("x")


def test_provider_propagates_collective_failure(monkeypatch):
    _install_host_contract(monkeypatch)
    from vllm_hust_unified_comm.provider import UnifiedCommunicationProvider

    communicator = Mock()
    communicator.all_reduce.side_effect = RuntimeError("collective failed")
    provider = UnifiedCommunicationProvider(communicator, "tp:0")

    with pytest.raises(RuntimeError, match="collective failed"):
        provider.all_reduce("x")
