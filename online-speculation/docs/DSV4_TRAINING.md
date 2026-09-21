# DSV4 Orthrus head training

## The short version

Serving and training use two views of the same DSV4 model:

1. vLLM-Ascend serves `Eco-Tech/DeepSeek-V4-Flash-0731-w8a8`.
2. Training loads a dequantized BF16 DSV4, freezes its parameters, and updates
   only the Orthrus diffusion head.
3. The trained BF16 head is calibrated separately to W8A8 and then replaces the
   head used by the service.

`resume` restores the exact optimizer, RNG, data cursor, and step. `continue`
loads an earlier head on new completed traffic, but deliberately starts a fresh
optimizer and schedule. Continuation runs beside the service, never inside the
latency-sensitive vLLM process.

## Why not train the W8A8 checkpoint directly?

W8A8 in ModelSlim and vLLM-Ascend is an inference/export format. Its kernels do
not expose the input-gradient path needed to backpropagate through the frozen
shared mHC, MoE/MLP, norm, and LM head. "Frozen" means their parameter gradients
are disabled; gradients still have to pass through their operations to reach an
earlier diffusion-head layer.

Quantization-aware training does not normally update integer tensors directly.
PyTorch QAT keeps floating-point weights and activations, inserts fake-quantize
operations, and approximates rounding gradients with a straight-through
estimator. Ascend AMCT likewise documents QAT starting from a float network.

For DSV4 specifically, MindSpeed-LLM says the released mixed-precision weights
must be dequantized to BF16 before training. ModelSlim and vLLM-Ascend document
W8A8 as quantization and inference deployment paths, not a differentiable DSV4
training backend.

References:

- <https://docs.pytorch.org/blog/quantization-aware-training/>
- <https://github.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash/README.md>
- <https://github.com/Ascend/msmodelslim/blob/master/docs/en/feature_guide/quick_quantization_v1/usage.md>
- <https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/quantization.md>
- <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/devaids/amct/atlasamct_16_0058.html>

## What is trained

The trainer covers the full DSV4 hybrid layout:

- SWA, C4, and C128 each have a trainable diffusion-attention twin.
- C4 also trains its diffusion-side query/indexer projections.
- K is 32: one anchor token plus 31 predicted positions.
- The target embedding, mHC, MoE/MLP, target compressor, final norm, and LM head
  are shared and frozen.
- The default loss is full-vocabulary forward KL plus the C4 indexer loss.

The exported artifact is `orthrus-dsv4-head-v2` in BF16 exchange format. It must
still pass independent ModelSlim calibration before W8A8 serving.

## CPU offload

Raw 284B BF16 weights occupy about 568 GB before activations and runtime state.
The checked-in CPU-offload path loads the full BF16 base in host RAM, puts a
bounded number of layers on each NPU, leaves the remaining layers on CPU, and
moves activations through them sequentially. This is very slow but is suitable
for proving the complete forward/backward/checkpoint/export path when eight
NPUs cannot hold the BF16 base.

It still requires enough host RAM. It is not the final high-throughput topology;
formal training will need a MindSpeed/Megatron sharded adapter or more nodes.

The architecture smoke test does not need the full checkpoint:

```bash
python3 scripts/train_dsv4_orthrus_head.py smoke \
  --npu-devices 0,1 \
  --precision bf16
```

## Offline paper data

The paper uses the Nemotron post-training dataset with a 1:1:1 chat/math/code
mixture, K=32, sequence length 2048, 256 anchors per sequence, two epochs,
learning rate `2e-4`, 5% warmup, cosine decay, and global batch 128.

Prepare a revision-pinned local snapshot as follows:

```bash
python3 scripts/prepare_dsv4_orthrus_data.py paper \
  --source /data/disk/data/Nemotron-Post-Training-Dataset-v2 \
  --tokenizer /data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --tokenizer-mode deepseek_v4 \
  --examples-per-domain 200000 \
  --offline \
  --output /data/disk/data/orthrus-dsv4-paper
```

The current public dataset may contain fewer than 200,000 unique code rows.
The script refuses silent oversampling: lower `--examples-per-domain` or pass
`--allow-repeat` explicitly. It writes exact source counts, fingerprints,
rejections, split metadata, and SHA256 values to `manifest.json`.

DSV4 checkpoints do not ship a Jinja chat template. `deepseek_v4` mode reuses
the pinned vLLM image's built-in DSV4 message encoder, exactly as the serving
flag `--tokenizer-mode deepseek_v4` does. `auto` selects this mode when tokenizer
metadata has no template; `hf` requires an explicit `--chat-template`.

Reference: <https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml>

Packing is boundary-aware and assistant-only by default. Every row carries
`anchor_ranges`, so a K=32 block cannot cross a conversation boundary or train
on prompt tokens. This follows the public Orthrus trainer's required
`assistant_mask` behavior. `--all-token-anchors` exists only for explicit
ablation/debugging.

Reference: <https://github.com/chiennv2000/orthrus/blob/main/src/utils/data_utils.py>

## Online continuation data

Capture completed requests as JSONL. Token IDs are preferred because they
preserve the exact service boundary:

```json
{"request_id":"r1","status":"completed","prompt_token_ids":[1,2],"output_token_ids":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]}
```

Then prepare a disjoint continuation snapshot:

```bash
python3 scripts/prepare_dsv4_orthrus_data.py online \
  --input /data/disk/data/traffic/completed.jsonl \
  --output /data/disk/data/orthrus-online-2026-09-22
```

Failed or unfinished requests and responses too short to form a complete block
are rejected and counted. Online `anchor_ranges` begin at the last prompt token,
so all 31 prediction targets lie inside the accepted model response. Text-form
`messages` are also supported when `--tokenizer` is supplied and the rendered
full conversation preserves the rendered prompt as an exact token prefix.

## Run training

The wrapper defaults to a one-update CPU-offload smoke. Paths are explicit so it
can be copied to an offline host without editing repository files:

```bash
BF16_BASE=/data/disk/models/DeepSeek-V4-Flash-0731-bf16 \
SERVING_TARGET=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
TRAIN_DATA=/data/disk/data/orthrus-dsv4-paper/train.tokens.jsonl \
VALIDATION_DATA=/data/disk/data/orthrus-dsv4-paper/validation.tokens.jsonl \
MASK_TOKEN_ID=128799 \
OUTPUT_CHECKPOINT=/data/disk/checkpoints/orthrus-dsv4-smoke.pt \
EXPORT_HEAD=/data/disk/models/orthrus-dsv4-head-v2-bf16-smoke \
bash scripts/run_dsv4_orthrus_training.sh offline smoke
```

After the smoke passes, replace `smoke` with `paper`. Because eight NPUs are
used as one model-parallel replica rather than eight data-parallel replicas,
the wrapper uses gradient accumulation 128 to preserve global batch 128. This
full CPU-offloaded run is expected to be extremely slow.

For asynchronous continuation, point `TRAIN_DATA` at the new online snapshot,
set `INPUT_CHECKPOINT`, choose a new `OUTPUT_CHECKPOINT` and `EXPORT_HEAD`, then:

```bash
bash scripts/run_dsv4_orthrus_training.sh continue smoke
```

Training reports the same held-out validation loss before and after the update.
That is only a training gate: the candidate head must then be calibrated to
W8A8 and pass acceptance-rate, exactness, goodput, and service-restart tests
before publication.
