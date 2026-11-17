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

Set `"device_greedy":true` in `--speculative-config` or use `--device-greedy` in the acceptance script to evaluate device-side top-1. The draft and verifier write logits to an NPU buffer. PyPTO scans the local vocabulary in parallel and uses the GLM TP reduction with its lowest-global-ID tie break. The host reads the compact token IDs from one rank and matches the accepted prefix. An active ID of -1 reports a nonfinite score on any shard. Distillation still gets full resident logits through its own buffers.

The standalone regression exercises the reduction without loading the model. It uses synthetic logits, changes the active query count over consecutive calls, and covers ties, padding and nonfinite inputs.

```bash
python scripts/validate_glm53_pypto_greedy.py --devices 0 1 --output greedy-tp2.json
```

The TP2 regression passed on Ascend. A two-device Serving dispatch smoke used synthetic draft and verify programs and checked accepted prefixes of lengths 0, 2 and 5. In a separate six-row benchmark on the same two devices, the final parallel top-1 dispatch and compact readback took 3.06 ms median, versus 10.37 ms for the initial serial device scan. Each number uses ten samples after two warm-ups and excludes model forward. The TP8 model artifacts and device-greedy auxiliary program passed compilation and the runner ABI check.

## Validation status

The complete TP8 program set passed compilation, runner ABI checks and artifact reload. Single-card Ascend tests covered empty history, the 640-token page boundary, and history beyond the 2,048-token Indexer selection limit. C8 cache and other input bytes remained unchanged. The complete BF16 draft attention sublayer had maximum relative L2 error of 0.1244% with synthetic weights and 0.0746% with the final MLA layer initialized from the W8A8 checkpoint. The native KDA test checked convolution and recurrent rollback against its golden. Full-model eight-device generation and HTTP acceptance remain to be run.

For a 1,280-token context, preflight reports 38.46 GiB per rank for staged target weights, 6.33 GiB for resident caches and 0.58 GiB for draft MLA. These allocations are separate from the vocabulary head, runtime rings and temporary kernel memory.

The pinned GLM Serving runner allocates the full device weight stacks before startup. The same checkpoint requires 76.38 GiB of target weights per TP4/EP4 rank before caches and runtime buffers. The streamed validation below moves individual shards through a single NPU. It exercises native kernels with real weights while the TP8 Serving acceptance retains its original topology.

## Online update components

The device KL loss and FP32 logit gradient passed for all 154,880 vocabulary entries on one device. Isolated Ascend tests also passed for the vocabulary shard, final RMSNorm and mHC mean, KDA recurrence, causal convolution and SiLU, KDA Q/K L2 normalization, mHC pre/post mixing, coefficient projection and Sinkhorn derivatives, expert SwiGLU, joint history/draft softmax, MLA projections and AdamW. A separate training dispatch updates the actual MLA output weight with an FP32 master copy and emits its BF16 view. The frozen INT8 linear VJP uses the checkpoint's row scales and passes the activation gradient through the quantization boundary with a straight-through estimator.

`validate_glm53_pypto_shards.py` tests global KL normalization across vocabulary shards, including padding and reuse of the same communication windows across training steps. `validate_glm53_pypto_suffix.py` chains the KL gradient, the resident vocabulary projection, an FP32 TP sum and final RMSNorm/mHC mean in one submission. Two Ascend devices passed both tests using the TP8 shard geometry (19,360 real entries per rank). The chained gradient had a maximum absolute error of `5.13e-8` against the double-precision reference using the deployed checkpoint's LM head and final RMSNorm with synthetic replay inputs. The eight-shard suffix compiled for all 154,880 vocabulary entries.

`validate_glm53_pypto_loss.py --mla-latent-backward --history-capacity 2176` checks MLA replay and its adjoint in one device submission. Given the latent queries, selected history, draft block and masks, the program recomputes the joint probabilities, differentiates both value bases, and combines the draft latent's key and value gradient contributions. The single-device maximum absolute error was `7.21e-7` against FP64 autograd for synthetic inputs with the deployed selection capacity. The target history stays read-only.

The inference acceptance harness checks teacher logits against clean target predictions. With `--resident-distillation`, it also binds the draft and teacher replay to separate device output banks and feeds those banks and the resident LM head to the distillation program. This option compiles with the TP8 Serving runner. A two-device smoke exercised the same dispatch and buffer bindings for three consecutive rounds with synthetic forwards. The two real-model replay rounds require eight idle devices. The reference readback is confined to the acceptance harness.

On a single Ascend device, the W8A8 expert replay and input/route adjoints passed with layer 44 expert 0 checkpoint weights. The maximum absolute errors were `7.19e-8` for the input gradient and `8.96e-5` for the route gradient. The selected sigmoid router adjoint also passed with the layer 44 weights. The deployed rank-0 KDA shard passed the output projection and gated RMSNorm tests, including near-zero activations. Its QKV, forget, beta and output-gate projection adjoints also passed, with a maximum input gradient error of `3.13e-7`.

Run the complete local KDA forward replay and input adjoint on one device by streaming the eight real TP8 shards in turn. The test uses nonzero initial convolution and recurrent states and six active loss rows. It passed on 111 with a largest input gradient error of `1.21e-7` against an FP64 reference.

```bash
python scripts/validate_glm53_pypto_streamed_kda.py \
  --checkpoint /data/disk1/wangzili/models/Eco-Tech/GLM-5.3-Flash-w8a8 \
  --device 0 --output streamed-kda.json
```

The shared expert test consumes the same NZ-packed W8A8 weights used by Serving. It unpacks the tiles on device and replays the checkpoint's BF16 rounding before differentiating SwiGLU. The layer 44 rank-0 shard matched the FP64 STE reference with a maximum input gradient error of `1.39e-7`.

The expert replay uses the native router for selection and quantization. With layer 44 checkpoint weights, the selected IDs, normalized route weights, integer expert inputs and row scales matched the reference. The recovered sigmoid scores differed by at most `5.87e-8`. The mHC post-mix adjoint also matches the BF16 coefficient rounding used in the forward.

The layer 43 MLA input adjoint reduces the normalized query and latent gradients across ranks before backpropagating through the replicated weights. Each rank retains its own query-up projection gradient. On two devices using real TP8 shards, the four replicated parameter gradients matched bit-for-bit across ranks and all six outputs matched the FP64 reference with maximum absolute error `2.49e-6`. The absorbed-query and draft key-basis adjoints also passed on real layer-43 weights.

The local MLA backward now starts at the FP32 attention output gradient and runs the output projection, shared history/draft softmax and both absorbed query paths in one PyPTO submission. With layer 43 checkpoint weights, synthetic activations and 2,176 history slots, all five gradients matched the FP64 reference within `7.62e-8`.

```bash
python scripts/validate_glm53_pypto_loss.py --mla-input-projections \
  --checkpoint "$MODEL" --device 0 --peer-device 1 --output mla-projections.json
python scripts/validate_glm53_pypto_loss.py --mla-local-backward --history-capacity 2176 \
  --checkpoint "$MODEL" --device 0 --output mla-local.json
python scripts/validate_glm53_pypto_loss.py --shared-expert-backward \
  --checkpoint "$MODEL" --device 0 --output moe-shared.json
python scripts/validate_glm53_pypto_head_publish.py \
  --checkpoint "$MODEL" --devices 0 1 --output mla-publish.json
```

The last check initializes the FP32 master weight and zero moments from two resident BF16 MLA shards. It stages AdamW with fixture gradients, verifies that staging preserves the live weights, and publishes the update. The FP32 master matched the reference within `2.98e-8`. Indexer weights and neighboring layers remained byte-identical.

Apply `runtime/pypto_serving_glm53/online_replay_lib.patch` after `draft_lib.patch` and compile into a fresh artifact directory. For layers 43 and 44, the draft replay records the four mHC residual streams and the BF16 sublayer output at both the attention and MoE boundaries. Normal inference leaves this tape untouched. All four offsets passed an Ascend copy test with byte-identical values and untouched neighboring rows.

The mHC replay uses the native coefficient calculation and backpropagates through stream collapse, post-mixing and the Sinkhorn-normalized residual coefficients. With layers 43 and 44 checkpoint parameters and fixture sublayer VJPs, the largest deviation from FP64 autograd was `1.61e-8`. The complete MoE VJP replays the native router, differentiates the selected W8A8 experts and the NZ shared expert, and applies the frozen RMSNorm adjoint. Layer 43 and 44 two-device tests used real weights for eight expert IDs and covered six active rows. The largest input-gradient error was `2.07e-7` against the FP64 STE reference. The TP8 MoE graph also compiled with 36 routed experts per rank.

```bash
python scripts/validate_glm53_pypto_mhc_replay.py \
  --checkpoint "$MODEL" --devices 0 --layer 44 --output mhc-replay.json
python scripts/validate_glm53_pypto_ep.py \
  --checkpoint "$MODEL" --devices 0 1 --layer 44 --full-moe --output moe-adjoint.json
python scripts/validate_glm53_pypto_verify.py \
  --compile-only --mask-token-id 0 --device-greedy --frozen-suffix \
  --artifact-dir /path/to/compile-cache --output frozen-suffix-abi.json
```

`--frozen-suffix` adds the training programs to the same Serving worker and propagates the KL gradient through layer 44 MoE/KDA and layer 43 MoE to the final trainable MLA output. KDA rebuilds its causal tape from the committed convolution and recurrent states in private buffers, uses an FP32 sum over rank-local input gradients, and differentiates the input RMSNorm. Two real TP8 shards streamed through one Ascend device passed the resident KDA forward and backward, with a maximum absolute input-gradient error of `8.58e-8` against FP64. The complete suffix passed TP8 compilation and the Serving ABI check.

```bash
python scripts/validate_glm53_pypto_streamed_kda.py \
  --checkpoint "$MODEL" --device 0 --resident-adjoint --output kda-adjoint.json
```

### Update the final draft MLA

`--resident-update LR` extends replay through the final trainable MLA and publishes a device AdamW update to the live Serving head. One session retains its optimizer state across requests. It rebuilds the final layer's BF16 projections, pooled Indexer selection and selected C8 history with the same kernels as the draft forward. It differentiates the shared history/block softmax, the absorbed key/value paths and the input projections. The trainable set is Q-A, its RMSNorm, Q-B, KV-A, its RMSNorm, the draft key/value bases and the output projection. Top-K selection and the two Indexer projections remain fixed. Model activations, gradients and weights stay on the NPU. The host passes page/segment metadata and checks the compact finite-gradient flags before publishing.

```bash
python scripts/validate_glm53_pypto_resident_step.py \
  --checkpoint "$MODEL" --devices 0 1 --output resident-step.json
python scripts/validate_glm53_pypto_verify.py \
  --checkpoint "$MODEL" --mask-token-id 0 --device-greedy --resident-update 0.0001 \
  --artifact-dir /path/to/compile-cache --output resident-tp8.json
```

The first command passed on two Ascend devices with real head shards, a paged fixture and synthetic upstream gradients. It checks two consecutive updates, each trainable field, replay after publication and byte-identical Indexer and target cache. Separately, the MLA activation replay passed at anchors 0, 17, 639, 640, 642 and 2,055, and the native local MLA adjoint matched FP64 within `7.62e-8`. The TP8 Serving acceptance uses native draft and teacher logits for the loss, checks that the update changes draft logits with the teacher unchanged, and reruns greedy verification with the updated head. Full-model TP8 execution is pending.

### Automatic update after an HTTP request

Apply `runtime/pypto_serving_glm53/online_http_serving.patch` after `serving.patch`. The Lib checkout also needs `online_replay_lib.patch` as described above. Set `online_learning_rate` in the Orthrus speculative configuration to opt in.

```bash
pypto-serving --model "$MODEL" --served-model-name glm53-orthrus \
  --devices 0,1,2,3,4,5,6,7 --tp 8 --ep 8 \
  --max-num-seqs 1 --max-model-len 1280 --max-num-batched-tokens 256 \
  --block-size 640 --host 127.0.0.1 --port 8000 \
  --speculative-config '{"method":"orthrus","num_speculative_tokens":5,"mask_token_id":0,"device_greedy":true,"online_learning_rate":0.0001}'
```

A normally completed request with at least six generated tokens supplies one replay. The worker releases the request cache, replays its prompt and the first six scheduler-confirmed output tokens, then runs one update before acknowledging the release. Cancellation, failure, preemption and shorter completions skip the update. The first eligible request initializes optimizer state. Subsequent requests on the same worker reuse that state. The update runs on the same device lane, so its time contributes to terminal request latency. The CLI option, request/worker lifecycle and TP8 Serving compilation have been checked. Full-model requests against the HTTP endpoint still need an eight-device run.

On an idle service, send the following request twice. Each response should contain eight generated tokens and the worker log should report `GLM_ORTHRUS_ONLINE_UPDATE step=1` followed by `step=2`. A third request capped at three tokens should leave the update count unchanged.

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-orthrus","messages":[{"role":"user","content":"Count from one to twenty."}],"max_tokens":8,"temperature":0,"ignore_eos":true}'
```

### Profile the update components on fewer devices

```bash
python scripts/validate_glm53_pypto_resident_step.py \
  --checkpoint "$MODEL" --devices 0 1 --profile-steps 8 --output mla-profile.json
python scripts/validate_glm53_pypto_streamed_kda.py \
  --checkpoint "$MODEL" --device 0 --tp-ranks 0 --resident-adjoint \
  --profile-steps 8 --output kda-profile.json
python scripts/validate_glm53_pypto_ep.py \
  --checkpoint "$MODEL" --devices 0 1 --layer 44 --full-moe \
  --profile-steps 8 --output moe-profile.json
```

The reports time synchronous program dispatch, including host overhead. The MLA total includes metadata upload and the finite-gradient check. On the single-card KDA fixture, causal replay took 12.61 ms median, while the temporal and projection VJPs took 1.98 and 1.92 ms. These component timings use short fixtures with resident weight shards. Full-model and HTTP profiling use the TP8 acceptance environment.


The final-layer MLA update stops at the frozen input, packs replay metadata into one transfer per rank, and uses wider vector tiles for AdamW, gradient validation and weight publication. On the same physical device with the same fixture, the median of ten measured steps fell from 40.64 ms to 22.30 ms after two warm-up steps. The five input-projection parameter gradients matched FP64 within `1.84e-6`, and the AdamW master weights matched within `2.98e-8`.

| Synchronous MLA dispatch | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Input projection backward | 9.21 | 6.74 |
| AdamW | 9.12 | 2.73 |
| Finite-gradient scan | 3.21 | 1.50 |
| Publish updated weights | 3.37 | 1.46 |

In the single-card MoE fixture with eight resident experts, routing took 3.00 ms, routed-expert backward took 8.20 ms, and the shared-expert and RMSNorm backward took 5.79 ms.
