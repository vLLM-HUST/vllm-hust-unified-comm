# Unified Communication for vLLM-HUST

Owner-maintained migration carrier for the device-neutral communication
runtime. The target is an independent library plus a narrow vLLM Host Provider,
not an import-time scheduler plugin.

**Status: device-neutral contracts, registry, and decision policies are
installable and tested. Host collective delegation and the Ascend HCCL adapter
are merged in [vllm-hust#74](https://github.com/vLLM-HUST/vllm-hust/pull/74)
(`8ec6ac0`) and [vllm-ascend-hust#71](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/71)
(`591d40b`). The strategy and registry remain `import_only`. A default-off
provider carrier is now mountable for validation through the host seam;
production qualification still requires model-serving, performance-regression,
and CUDA/NCCL evidence.**

The responsible maintainer is Ma Chuanhu (@machuanhu), advised by Xiong Wang
(王雄). Shuhao Zhang provided integration assistance; that assistance does not
transfer project ownership or the advising relationship. Source extraction
preserves original authorship, license, tests, constraints, and evidence.

See [MAINTAINERS.md](MAINTAINERS.md) and [PROVENANCE.md](PROVENANCE.md).

## Extension framework

Extension ID: `org.vllm-hust.unified-communication`

This repository follows the vLLM-HUST Extension Template. Installation alone
is inert. Extension Manager activation claims the exclusive
`vllm.communication.provider` resource and configures:

```text
VLLM_HUST_COMMUNICATION_PROVIDER=vllm_hust_unified_comm.provider.UnifiedCommunicationProvider
```

The provider reuses the host-owned communicator and process group. It does not
create or destroy communication groups, and collective exceptions remain
visible. Disable the bundle (or remove the environment variable) and restart
the vLLM processes to roll back to the native path.

```bash
python -m pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.unified-communication
vllm-hust-ext extension check org.vllm-hust.unified-communication
pytest -q
```

For a direct mount smoke against a compatible `vllm-hust` checkout:

```bash
python -m pip install .
export VLLM_HUST_COMMUNICATION_PROVIDER=vllm_hust_unified_comm.provider.UnifiedCommunicationProvider
# Start the ordinary vLLM command. Unset the variable and restart to roll back.
```

On an idle Ascend node, the focused collective mount smoke can be run with:

```bash
VLLM_HUST_COMMUNICATION_PROVIDER=vllm_hust_unified_comm.provider.UnifiedCommunicationProvider \
  python -m torch.distributed.run --nproc-per-node 2 scripts/hccl_mount_smoke.py
```

This validates provider loading and four HCCL collectives. It is not a model
serving or performance qualification.

A subsequent two-device serving smoke loaded Qwen3-0.6B with tensor parallel
size 2, logged `UnifiedCommunicationProvider` on both HCCL workers, and returned
HTTP 200 with generated tokens from both Chat Completions and Completions. See
[`evidence/model-serving-smoke-20261107.json`](evidence/model-serving-smoke-20261107.json).
This is correctness evidence, not a throughput or latency claim.

The validated serving matrix is vLLM `0.23.0` plus the communication-provider
host seam and lazy adapter compatibility fix, vLLM Ascend `0.23.0.post1`, Python
3.12, CANN 9.1, and Ascend 910B2. When running from source, prepend the host and
MOD paths to the existing `PYTHONPATH`; do not replace the existing CANN Python
paths. Newer host/adapter combinations require their own compatibility check.

The static Manifest 0.3 descriptor lives inside the Python distribution under
`src/`. Installation alone changes no vLLM behavior.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
