# DSV4 Orthrus head training

## Status and boundary

The DSV4 inference path and the training path are intentionally separate.

- vLLM-Ascend serves the W8A8 target and a calibrated W8A8 Orthrus head.
- The trainer loads a differentiable BF16 DSV4 target, freezes every target
  parameter, and updates only the tensors in `orthrus-dsv4-head-v2`.
- `resume` restores the exact optimizer, RNG, data cursor, and update number on
  the same dataset and refuses a different BF16 base identity.
- `continue` starts a fresh schedule from an earlier head using newly completed
  production requests. This is asynchronous online continuation, not autograd
  inside the latency-sensitive vLLM process.

The generic small-model implementation in `parallel/online.py` can reuse live
verifier rows and replay only a suffix. The DSV4 implementation currently
recomputes the frozen teacher on the BF16 sidecar. It does not yet capture and
replay DSV4 verifier boundaries directly from vLLM.

## What is trained

`dsv4_training.py` implements the complete hybrid layout used by inference:

- SWA reads the frozen target's clean sliding history.
- C4 reads the target compressor output and trains a diffusion-side query and
  indexer while keeping the target compressor frozen.
- C128 reads all complete target compressed-history entries.
- Every block contains one anchor and 31 mask positions. Local block attention
  is bidirectional, while clean history is read-only.
- The shared embedding, mHC, MoE/MLP, final norm, and LM head remain frozen but
  stay in the differentiable path from the head to the loss.
- The default objective is full-vocabulary forward KL. Cross entropy is also
  available for comparison. C4 indexer scores receive a separate teacher loss.

The exporter writes one BF16 shard per layer using the exact v2 tensor names.
The result is marked `artifact_role=trained` and still requires independent
ModelSlim calibration before serving.

## Required weights

Do not train from either of these serving artifacts:

- `Eco-Tech/DeepSeek-V4-Flash-0731-w8a8`
- the official mixed FP8/FP4 inference checkpoint

Quantized inference kernels do not provide the backward path needed through
the frozen shared backbone. Prepare a dequantized BF16 checkpoint first. The
MindSpeed-LLM DeepSeek-V4 recipe provides FP8-to-BF16 and HF-to-MCore conversion
as well as pretraining/continuation support:

<https://github.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/deepseek4_flash>

The raw 284B BF16 weights alone require about 568 GB. They do not fit in
8 x 64 GB before activations, communication buffers, and optimizer state are
counted. The checked-in trainer can place a bounded number of layers on each
NPU and keep the rest on host memory. This is useful for correctness and slow
short-run validation on a host with enough RAM, but it is not a production
throughput solution. Production training needs a MindSpeed/Megatron sharded
adapter, more nodes, or deliberate CPU/NVMe offload engineering.

## Architecture smoke test

The smoke test creates a tiny three-layer DSV4 with one SWA, one C4, and one
C128 layer. It runs a real optimizer step and verifies that the head changes,
the target stays bitwise unchanged, and no target gradients remain.

```bash
python scripts/train_dsv4_orthrus_head.py smoke \
  --device npu:0 \
  --precision bf16
```

The same smoke test can cross two or more NPUs and exercise differentiable
layer transfers:

```bash
python scripts/train_dsv4_orthrus_head.py smoke \
  --npu-devices 0,1 \
  --precision bf16
```

For a memory-constrained full-model validation, load the BF16 target in host
memory and move only a few consecutive head/backbone layers to every NPU:

```bash
python scripts/train_dsv4_orthrus_head.py train \
  --bf16-base /data/disk/models/DeepSeek-V4-Flash-0731-bf16 \
  --data /data/disk/data/orthrus/train.tokens.jsonl \
  --mask-token-id 128799 \
  --npu-devices 0,1,2,3,4,5,6,7 \
  --layers-per-npu 1 \
  --steps 1 \
  --output-checkpoint /data/disk/checkpoints/orthrus-dsv4-offload-smoke.pt
```

This mode still needs enough host RAM for the complete BF16 base and moves
activations between devices one layer at a time. Treat it as a deployment gate,
not as the final training topology. Increase `--layers-per-npu` only after
measuring target-layer, head-gradient, optimizer-state, and activation memory
on the exact checkpoint.

## Initial training

The JSONL input contains one `input_ids` list per record. Sequences shorter
than `--sequence-length` are skipped. Training and serving paths are separate
because the serving target identity is embedded in the exported manifest.

```bash
python scripts/train_dsv4_orthrus_head.py train \
  --bf16-base /data/disk/models/DeepSeek-V4-Flash-0731-bf16 \
  --serving-target /data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --data /data/disk/data/orthrus/train.tokens.jsonl \
  --mask-token-id 128799 \
  --device npu:0 \
  --sequence-length 4096 \
  --anchors-per-sequence 16 \
  --steps 1000 \
  --output-checkpoint /data/disk/checkpoints/orthrus-dsv4-step-1000.pt \
  --export-head /data/disk/models/orthrus-dsv4-head-v2-bf16-step-1000
```

## Resume and continual updates

Exact interruption recovery keeps the saved schedule and requires the same
tokenized data fingerprint:

```bash
python scripts/train_dsv4_orthrus_head.py resume \
  --bf16-base /data/disk/models/DeepSeek-V4-Flash-0731-bf16 \
  --data /data/disk/data/orthrus/train.tokens.jsonl \
  --mask-token-id 128799 \
  --sequence-length 4096 \
  --checkpoint /data/disk/checkpoints/orthrus-dsv4-step-500.pt \
  --output-checkpoint /data/disk/checkpoints/orthrus-dsv4-step-1000.pt
```

For asynchronous online continuation, first convert newly completed request
and response token streams to a disjoint JSONL file. `continue` loads the prior
head but intentionally starts a fresh optimizer and data schedule:

```bash
python scripts/train_dsv4_orthrus_head.py continue \
  --bf16-base /data/disk/models/DeepSeek-V4-Flash-0731-bf16 \
  --serving-target /data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --data /data/disk/data/orthrus/traffic-2026-09-20.tokens.jsonl \
  --mask-token-id 128799 \
  --checkpoint /data/disk/checkpoints/orthrus-dsv4-step-1000.pt \
  --steps 50 \
  --output-checkpoint /data/disk/checkpoints/orthrus-dsv4-online-0050.pt \
  --export-head /data/disk/models/orthrus-dsv4-head-v2-bf16-online-0050
```

The candidate must then pass held-out acceptance/goodput evaluation, independent
W8A8 calibration, strict preflight, and service restart. Online continuation
never publishes directly into the live process.
