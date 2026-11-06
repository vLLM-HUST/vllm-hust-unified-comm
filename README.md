# Unified Communication for vLLM-HUST

Organization-maintained migration carrier for the device-neutral communication
runtime. The target is an independent library plus a narrow vLLM Host Provider,
not an import-time scheduler plugin.

**Status: device-neutral contracts, registry, and decision policies are
installable and tested. Host collective delegation and the Ascend HCCL adapter
are merged in [vllm-hust#74](https://github.com/vLLM-HUST/vllm-hust/pull/74)
(`8ec6ac0`) and [vllm-ascend-hust#71](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/71)
(`591d40b`). The extension remains `import_only` until multi-device hardware
validation is recorded.**

Technical ownership belongs to @ShuhaoZhangTony on behalf of vLLM-HUST. Source
extraction preserves original authorship, license, tests, constraints, and
evidence before activation is considered.

See [MAINTAINERS.md](MAINTAINERS.md) and [PROVENANCE.md](PROVENANCE.md).

## Extension framework

Extension ID: `org.vllm-hust.unified-communication`

This repository follows the vLLM-HUST Extension Template. The current package
is deliberately `import_only`: the host-independent library can be imported and
tested, but Extension Manager must refuse enablement until backend packages
and multi-device hardware compatibility evidence land.

```bash
python -m pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.unified-communication
vllm-hust-ext extension check org.vllm-hust.unified-communication
pytest -q
```

The static Manifest 0.2 descriptor lives inside the Python distribution under
`src/`. Installation alone changes no vLLM behavior.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
