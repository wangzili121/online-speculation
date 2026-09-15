# Orthrus DSV4 head v2 contract

`orthrus-dsv4-head-v2` is the only accepted DSV4 Orthrus artifact format. A v1
SWA-only artifact is rejected.

## Fixed semantics

- `format_version`: 2
- `history_backend`: `shared_target_hybrid`
- `block_size`: 32
- `num_speculative_tokens`: 31
- architecture: `OrthrusDSV4DraftModel`
- exactly one `compress_ratio` for each target layer
- the same mask/noise token ID and vocabulary as the pinned target
- `pard_token` equal to `mask_token_id` for the vLLM 0.26 runtime alias
- target `config.json` and safetensors index SHA256 identities

The production model is expected to have 43 layers. The tooling derives the
count from the pinned target so small synthetic fixtures can exercise the same
contract; `--production` enforces the real DSV4 constraints and TP=8.
For DSV4-Flash-0731, export resolves the parallel placeholder from
`mask_token_id` first and then `dspark_noise_token_id`; an explicit
`--mask-token-id` may be used only when the pinned target omits both.

## Owned tensors

Every layer owns diffusion `wq_a`, Q norm, `wq_b`, `wkv`, KV norm, `wo_a`,
`wo_b`, and attention sink tensors. C4 layers additionally own diffusion-side
`indexer.wq_b` and `indexer.weights_proj` tensors.

Canonical shard names use:

```text
layers.<layer>.attn.<tensor>
```

The head does not contain the embedding, mHC parameters, model norms, MoE/MLP,
LM head, target compressor, indexer compressor, or any target history cache.
Those objects are bound from the live verifier and remain target-owned.
Only completed C4/C128 groups are readable from compressed KV; an incomplete
tail remains target-owned compressor state and is never treated as a cache row.

## BF16 exchange artifact

External training delivers a complete v2 manifest, config, safetensors index,
and shards with `artifact_role=trained`. Its base identity and tensor schema
must pass preflight with `--allow-bf16`. It is an exchange format, not the
production artifact.

## Ascend W8A8 artifact

The trained BF16 head is independently calibrated with representative data.
The ModelSlim adapter emits head-only W8A8 shards and a quantization description
covering each quantized linear weight and its scale/offset tensors. The sealed
manifest records:

- `weight_format=ascend-w8a8`;
- `independently_calibrated=true`;
- `shares_target_scale_storage=false`;
- calibration dataset/method/tool/runtime provenance and SHA256;
- exact tensor schema and shard SHA256 values.

Copying target scales is allowed only for an `initialized-plumbing` artifact.
It is explicitly marked `target-copy-initialization` and rejected by
`--require-trained`.

## Replacement

Head replacement is atomic at the directory/release level. Point `HEAD` to a
fully preflighted directory and restart the vLLM service. In-process hot swap is
not part of the contract.
