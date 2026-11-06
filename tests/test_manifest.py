from pathlib import Path

from vllm_hust_ext.manifest import activation_blocker, load_manifest

import vllm_hust_unified_comm


def test_descriptor_exposes_an_exclusive_mountable_provider() -> None:
    manifest = load_manifest(
        Path(vllm_hust_unified_comm.__file__).with_name("vllm-hust-extension-v0.3.json")
    )
    assert manifest.bundle_id == "org.vllm-hust.unified-communication"
    assert manifest.schema_version == "0.3-experimental"
    assert activation_blocker(manifest) is None
    assert manifest.activation.environment == (
        (
            "VLLM_HUST_COMMUNICATION_PROVIDER",
            "vllm_hust_unified_comm.provider.UnifiedCommunicationProvider",
        ),
    )
    assert manifest.resource_claims[0].resource == "vllm.communication.provider"
    assert manifest.resource_claims[0].mode == "exclusive"
