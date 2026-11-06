# Unified communication host contract

The extracted registry and decision policies are independent of torch, NCCL,
HCCL, and vLLM. The initial runtime contract is implemented by
[vllm-hust#74](https://github.com/vLLM-HUST/vllm-hust/pull/74) and its Ascend
binding by
[vllm-ascend-hust#71](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/71).
It provides:

1. An explicit provider import path, with optional selection by an out-of-tree
   platform and an environment override.
2. Reuse of the host-owned device communicator and existing process groups.
3. Per-collective delegation for all-reduce, all-gather, reduce-scatter, and
   broadcast. `NotImplemented` requests the native fallback; provider failures
   are not silently hidden.
4. Provider teardown before the native communicator and process groups.

GroupCoordinator remains lifecycle owner. The extension must not create a second
default process group, destroy host-owned communicators, or silently replace an
existing backend. Backend registration and topology-aware policy remain in this
package; direct NCCL/HCCL C-API implementations still require separate native
packages and hardware-specific validation.
