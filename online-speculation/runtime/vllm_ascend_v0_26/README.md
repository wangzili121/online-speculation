# DeepSeek-V4 Orthrus overlay for vLLM-Ascend 0.26

This guarded overlay adds a dedicated `method="orthrus"` proposer and model
path to the pinned vLLM/vLLM-Ascend runtime. It does not replace the native
DeepSeek-V4 verifier and does not reinterpret DSpark as Orthrus.

Before editing, the patcher also identifies the audited
`v0.26.0rc1-monolithic` or `v0.26-nightly-split` DSV4 private API family. An
unknown family fails the dry-run; the runtime lock then pins the exact image,
commits, CANN, and driver used for the accepted deployment.

The overlay enables:

1. vLLM speculative-config recognition for the v2 Orthrus model;
2. an Ascend Orthrus proposer using a 32-position non-causal draft block;
3. binding of target-owned embedding/backbone/LM-head state after model load;
4. metadata over target-owned SWA, C4 compressed/indexer, and C128 caches;
5. generic cache-sharing names for the non-standard hybrid layers;
6. the vLLM probability hand-off needed by standard rejection sampling.

The overlay also guards the exact vLLM proposer probability API and the two
audited vLLM-Ascend DSV4 layouts before changing source files. DSA context
parallel and FlashComm1 remain disabled for Orthrus until their full-head
collective semantics have a dedicated implementation and NPU correctness gate.

`apply.sh` requires a runtime lock, dry-runs every edit, rejects unknown source
anchors, applies idempotently, installs the local package without dependencies,
and imports the registered model/proposer as a final smoke test.

```bash
RUNTIME_LOCK=runtime/vllm_ascend_v0_26/runtime-lock.local.json \
  bash runtime/vllm_ascend_v0_26/apply.sh
```

The implementation is version-specific. Generate a new overlay and rerun the
full correctness matrix before changing the image, vLLM, vLLM-Ascend, CANN, or
driver revision.
