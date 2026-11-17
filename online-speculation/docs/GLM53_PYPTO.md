# GLM-5.3-Flash Orthrus on PyPTO Serving

The inference path uses the resident GLM-5.3-Flash W8A8 backbone. The draft adds BF16 MLA projections, reads the target's paged latent and pooled index caches, and uses request-local KDA scratch rows. The PyPTO draft and verifier use the resident vocabulary shards for token embedding. The Serving worker selects greedy tokens from the returned logits, commits the anchor and the accepted prefix, and restores convolution and recurrent state from the last committed row.

The initial configuration is TP8/EP8, one active request, and five guesses per six-position block. Stochastic requests and requests with host constraints use normal GLM decode. The draft head is initialized from the checkpoint using its weight scales. Measure performance with a trained head once end-to-end correctness has been established.

## Build and check

- PyPTO Serving `39f36c0`, PyPTO Lib `79be1fd`, PyPTO `ff2e0a4`, simpler `31f9e72`, PTO-ISA `f83d4b7`, PTOAS `0.72`. [GLM Serving ABI](https://github.com/hw-native-sys/pypto-serving/pull/314)
- Apply `runtime/pypto_serving_glm53/draft_lib.patch` to the fixed Lib revision and `runtime/pypto_serving_glm53/serving.patch` to the fixed Serving revision. Set `PYPTO_LIB_ROOT` to the Lib checkout and make Serving, Lib and this project's `src` visible on `PYTHONPATH`.
- Compile the full set of TP8 programs without initializing device workers.

```bash
python scripts/validate_glm53_pypto_verify.py --compile-only \
  --mask-token-id 0 --artifact-dir /path/to/compile-cache --output compile.json
```

Run the acceptance check on an idle eight-device node. The placeholder ID in this example is for the initialized head smoke test.

```bash
python scripts/validate_glm53_pypto_verify.py \
  --checkpoint /data/disk1/wangzili/models/Eco-Tech/GLM-5.3-Flash-w8a8 \
  --mask-token-id 0 --artifact-dir /path/to/compile-cache --output tp8.json
```

The script checks ordinary decode against forced zero, partial and full acceptance across a cache page, then checks the initialized Orthrus draft. The corresponding HTTP configuration is `--tensor-parallel-size 8 --expert-parallel-size 8 --block-size 640 --max-num-seqs 1 --speculative-config '{"method":"orthrus","num_speculative_tokens":5,"mask_token_id":0}'`.

## Validation status

The complete TP8 program set passed compilation, runner ABI checks and artifact reload. Single-card Ascend tests covered empty history, the 640-token page boundary, and history beyond the 2,048-token Indexer selection limit. C8 cache and other input bytes remained unchanged. The complete BF16 draft attention sublayer had maximum relative L2 error of 0.1244% with synthetic weights and 0.0746% with the final MLA layer initialized from the W8A8 checkpoint. The native KDA test checked convolution and recurrent rollback against its golden. Full-model eight-device generation and HTTP acceptance remain to be run.

For a 1,280-token context, preflight reports 38.46 GiB per rank for staged target weights, 6.33 GiB for resident caches and 0.58 GiB for draft MLA. These allocations are separate from the vocabulary head, runtime rings and temporary kernel memory.

## Online update components

The device KL loss and FP32 logit gradient passed for all 154,880 vocabulary entries on one device. Isolated Ascend tests also passed for the vocabulary shard, final RMSNorm and mHC mean, KDA recurrence, causal convolution and SiLU, KDA Q/K L2 normalization, mHC pre/post mixing, coefficient projection and Sinkhorn derivatives, expert SwiGLU, joint history/draft softmax, MLA projections and AdamW. A separate training dispatch updates the actual MLA output weight with an FP32 master copy and emits its BF16 view. The frozen INT8 linear VJP uses the checkpoint's row scales and passes the activation gradient through the quantization boundary with a straight-through estimator.

`validate_glm53_pypto_shards.py` tests global KL normalization across vocabulary shards, including padding and reuse of the same communication windows across training steps. `validate_glm53_pypto_suffix.py` chains the KL gradient, the resident vocabulary projection, an FP32 TP sum and final RMSNorm/mHC mean in one submission. Two Ascend devices passed both tests using the TP8 shard geometry (19,360 real entries per rank). The chained gradient had a maximum absolute error of `5.13e-8` against the double-precision reference using the deployed checkpoint's LM head and final RMSNorm with synthetic replay inputs. The eight-shard suffix compiled for all 154,880 vocabulary entries.

The acceptance harness includes a replay check that compares teacher logits from known continuation tokens with the normal target predictions. It runs in a private request slot. The outstanding work includes replaying the final draft MLA layer (backbone layer 43, zero-based) with its frozen suffix, the remaining attention, KDA projection, mHC and routed-expert derivatives, and publishing the updated head to Serving at a request boundary.
