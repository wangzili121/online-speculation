# DeepSeek-V4 Flash Orthrus deployment

This deployment targets `Eco-Tech/DeepSeek-V4-Flash-0731-w8a8` on one
8 x 64 GiB Ascend A2 node with
`quay.io/ascend/vllm-ascend:nightly-releases-v0.26.0rc`.

It is an inference service. Training runs in a separate BF16 process: the
service consumes a versioned diffusion head, independently converts it to
Ascend W8A8, and loads new weights after a restart. An initialized head proves
the integration only; production performance is accepted only with a trained
head. The checked-in training and continuation path is documented in
[DSV4_TRAINING.md](DSV4_TRAINING.md).

The vLLM path is the fixed-head inference subset of Chang Bo's
`online-speculation`; online updates remain outside the serving process. The
exact algorithmic boundary and upstream PR comparison are recorded in
[DSV4_CHANGBO_SCOPE.md](DSV4_CHANGBO_SCOPE.md).

## Offline one-command deployment

On an exclusively reserved eight-card A2 node, preload the pinned image and
place the exact Eco-Tech target and calibrated trained head under one models
root. Then, from this repository's `online-speculation/` directory:

```bash
ACK_EXCLUSIVE_NODE=1 ORTHRUS_PROBABILISTIC_VALIDATED=1 \
MODELS_HOST=/data/disk/models \
MODEL_HOST=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
HEAD_HOST=/data/disk/models/orthrus-dsv4-head-v2-w8a8 \
PROFILE=prod-128k PORT=8901 \
scripts/deploy_dsv4_offline.sh
```

`ORTHRUS_PROBABILISTIC_VALIDATED=1` is an explicit operator assertion, **not**
an automatic test: set it only after multi-request sampling validation on
this exact runtime/head. The script checks the v2 head contract, local image,
eight-card reservation acknowledgment, paths, and free port, then pins and
patches the runtime, starts the API service, and waits for one completion.
The model/head paths may be changed, but both must be descendants of
`MODELS_HOST` so the container sees them. It never pulls packages or images.
The default `EXPECTED_IMAGE_DIGEST` is the digest verified on the online v0.26
test host and recorded in the offline deployment script. An internal image with
a different digest is rejected until its vLLM,
vLLM-Ascend, CANN, driver, and patch layout are revalidated and the operator
explicitly supplies the newly approved digest.
It refuses an existing container rather than replacing it. Logs and the
runtime lock remain in this repository after a failed service smoke.

For an initialized head used only for integration testing, set
`REQUIRE_TRAINED_HEAD=0 ORTHRUS_DRAFT_SAMPLE_METHOD=greedy`; this does not
meet the production acceptance gate. For the target-only rollback path, set
`MODE=target` with a new `CONTAINER` and `PORT` after stopping the owned
Orthrus service. An unhealthy Orthrus process is not killed automatically
because the script cannot safely infer ownership of an existing process.

## Implemented execution path

`method="orthrus"` has its own proposer and model registration. Target-only and
the image's native DSpark path remain separate deployment/performance
references; DSpark does not define Orthrus generation semantics.

The draft owns only per-layer diffusion attention/head parameters. It calls the
live target's embedding, mHC, normalization, MoE/MLP, final norm, and LM head.
It does not allocate another DSV4 backbone or another target history cache.

For every 32-position draft block (anchor plus 31 proposals):

1. SWA layers gather the committed target SWA history.
2. C4 layers read target compressed KV and indexer-key caches, then execute the
   diffusion-side query/indexer without updating either cache.
3. C128 layers gather target compressed KV history.
4. The 32 positions attend bidirectionally inside their local block.
5. History, local block, and attention-sink output/LSE are merged with an exact
   log-sum-exp reduction.
6. The regular vLLM verifier commits only the accepted prefix; the draft path
   never writes target KV.

The first implementation intentionally gathers paged history and uses Ascend
FIA because that operation returns the LSE needed for exact segment merging.
The C4 QLI bridge treats returned indices as original-token coordinates and
maps only completed four-token groups to C4 rows; incomplete C4/C128 groups
remain in the target compressor state cache. That coordinate contract and
FIA LSE layout are explicit A2 bring-up gates; a mismatch must fail correctness
testing instead of being hidden by a fallback.

The initial correctness backend is TP-only. Although the official target-only
A2 recipe can enable DSA context parallel and FlashComm1, its attention
implementation changes the query to a full-head layout, performs an all-to-all
head restoration, and uses a different output-projection path. Orthrus rejects
`enable_dsa_cp=true` and `enable_flashcomm1=true` until that CP-specific path
passes the same cache, token-equality, and rejection tests. The supplied target
and Orthrus launchers therefore use the same non-CP settings for paired
correctness and performance measurements; overriding those flags is not a
supported production configuration in this revision.

## Artifact layout

```text
/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8
/data/disk/models/orthrus-dsv4-head-v2-bf16
/data/disk/models/orthrus-dsv4-head-v2-w8a8
/data/disk/packages/dsv4-orthrus
```

The head contract is documented in [DSV4_HEAD_CONTRACT.md](DSV4_HEAD_CONTRACT.md).

## 1. Pin and patch the runtime

Create the container on the internal host. The script resolves the mutable tag
to a digest, records vLLM/vLLM-Ascend commits and CANN/driver versions, applies
the guarded runtime overlay, and installs this package without network access.

```bash
CONTAINER=dsv4-orthrus-v026 \
REPO_HOST=/data/disk/repos/online-speculation \
MODELS_HOST=/data/disk/models \
NPU_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_PULL=never \
scripts/create_dsv4_container.sh

docker exec -it -w /workspace/online-speculation \
  dsv4-orthrus-v026 bash
```

On a shared bring-up host, expose only the cards reserved for this experiment,
for example `NPU_DEVICES=0` or `NPU_DEVICES=0,1`. The script does not use
Docker privileged mode by default because that would expose other users'
devices despite the explicit device list. Set `DOCKER_PRIVILEGED=1` only if
the host's Ascend runtime demonstrably requires it and the node is exclusively
reserved. Physical device IDs are remapped to contiguous container IDs, so
`NPU_DEVICES=5` is used as `npu:0` inside the container and
`NPU_DEVICES=0,1,4,5` becomes container devices `0,1,2,3`.

`IMAGE_PULL=auto` is the default: an image already present in the internal
Docker store is used without network access, and a missing image is pulled.
Use `IMAGE_PULL=never` on the disconnected production host to fail immediately
if the expected image was not preloaded; use `always` only while refreshing an
online test host. In every mode the script resolves and records the immutable
repository digest before creating the container.

For an already-running container:

```bash
export ORTHRUS_IMAGE_REF=quay.io/ascend/vllm-ascend:nightly-releases-v0.26.0rc
export ORTHRUS_IMAGE_DIGEST=sha256:<resolved-digest>
python scripts/capture_dsv4_runtime_lock.py \
  --image-ref "$ORTHRUS_IMAGE_REF" \
  --image-digest "$ORTHRUS_IMAGE_DIGEST" \
  --output runtime/vllm_ascend_v0_26/runtime-lock.local.json

RUNTIME_LOCK=runtime/vllm_ascend_v0_26/runtime-lock.local.json \
  bash runtime/vllm_ascend_v0_26/apply.sh
```

The patcher first dry-runs all vLLM and vLLM-Ascend edits and refuses unknown
source layouts. Before patching it also rechecks the package commits, CANN,
driver, image reference, and immutable image digest against the lock. It is
version-specific, guarded, and idempotent.

## 2. Prepare the head

Create an initialized artifact from the target only to verify model loading:

```bash
python scripts/export_dsv4_orthrus_head.py \
  /data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  /data/disk/models/orthrus-dsv4-head-v2-w8a8
```

The initialized artifact records `scale_origin=target-copy-initialization` and
cannot pass the trained-head production gate.

Train or continually adapt the BF16 head as described in
[DSV4_TRAINING.md](DSV4_TRAINING.md). Then run the project-specific ModelSlim
adapter and seal its independently calibrated output:

```bash
ORTHRUS_MODELSLIM_ADAPTER=/workspace/tools/quantize_orthrus_dsv4_head.py \
MODEL=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
BF16_HEAD=/data/disk/models/orthrus-dsv4-head-v2-bf16 \
OUTPUT_HEAD=/data/disk/models/orthrus-dsv4-head-v2-w8a8 \
CALIBRATION_DATA=/data/disk/calibration/dsv4.jsonl \
scripts/quantize_dsv4_orthrus_head.sh
```

Stock ModelSlim has no head-only Orthrus architecture entry, so the adapter is
an explicit integration boundary. It must emit the quantized shards,
`quant_model_description.json`, and a calibration manifest consumed by the
finalizer. Target-model scales are never accepted as trained-head calibration.

Run strict validation before every service start:

```bash
python scripts/preflight_dsv4_orthrus.py \
  --model /data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --head /data/disk/models/orthrus-dsv4-head-v2-w8a8 \
  --tensor-parallel-size 8 \
  --production \
  --require-trained
```

## 3. Bring-up order

Run target-only first, then Orthrus on the same eight-card node. Both scripts
default to the `prod-128k` profile. Set `PROFILE=prod-1m` for the 1M profile.
All launchers default to eager for the first correctness pass.

```bash
PROFILE=prod-128k PORT=8900 scripts/serve_dsv4_target.sh

PROFILE=prod-128k PORT=8901 REQUIRE_TRAINED_HEAD=1 \
  scripts/serve_dsv4_orthrus.sh

python scripts/healthcheck_dsv4.py \
  --base-url http://127.0.0.1:8901 \
  --model dsv4-orthrus \
  --wait-seconds 1800
```

The native DSpark comparison is optional and has its own launcher. It is never
used as an Orthrus fallback or implementation shortcut:

```bash
PROFILE=prod-128k PORT=8902 scripts/serve_dsv4_dspark.sh
```

All three launchers deliberately leave DSA context parallel and FlashComm1
disabled so their measurements are comparable. The target-only service may be
benchmarked separately with the vendor's CP settings, but that result is not a
paired Orthrus baseline.

Orthrus defaults to full eager execution, including the target, for initial
correctness. Target graph execution is gated and cannot be selected by
accident:

```bash
ORTHRUS_EXECUTION_MODE=graph ORTHRUS_GRAPH_VALIDATED=1 \
  REQUIRE_TRAINED_HEAD=1 scripts/serve_dsv4_orthrus.sh
```

Use `TARGET_EXECUTION_MODE=graph` for its paired target baseline and
`DSPARK_EXECUTION_MODE=graph` for the optional DSpark baseline. Never compare
an eager service with a graph service for the production goodput gate.

Set that validation flag only after the same head/runtime combination passes
the complete eager-versus-graph matrix. The draft itself remains eager until a
separate exact graph implementation is validated.

The launcher defaults `ORTHRUS_DRAFT_SAMPLE_METHOD=greedy`. This still supports
temperature sampling losslessly: standard rejection sampling corrects the
greedy proposal against the target distribution, without retaining a full
draft-vocabulary probability tensor. `probabilistic` is available as an
explicit experiment, but is not the production default until the upstream
parallel-draft RNG path is proven to preserve per-request seed isolation.

## 4. Correctness gate

The two full services normally run sequentially because each occupies the
eight-card node. Snapshot target-only, stop it, start Orthrus, then replay:

```bash
python scripts/validate_dsv4_service.py snapshot \
  --base-url http://127.0.0.1:8900 \
  --model dsv4-target \
  --cases configs/dsv4/correctness.json \
  --include-long \
  --output results/dsv4/target-correctness.json

python scripts/validate_dsv4_service.py snapshot \
  --base-url http://127.0.0.1:8901 \
  --model dsv4-orthrus \
  --cases configs/dsv4/correctness.json \
  --include-long \
  --output results/dsv4/orthrus-correctness.json

python scripts/validate_dsv4_service.py compare \
  --baseline results/dsv4/target-correctness.json \
  --candidate results/dsv4/orthrus-correctness.json \
  --output results/dsv4/correctness.json
```

The gate checks greedy text/token equality, repeated-request cache stability,
fixed-seed reproducibility within each service, and empirical first-token TV
across services. Cross-service same-seed equality is diagnostic only because
valid rejection sampling may consume random values in a different order. The
default cases cross 127/128/129-token boundaries; `--include-long` adds roughly
32K and 125K-token prompts. Also exercise consecutive rejections and service
restart with `ORTHRUS_VERIFY_CACHE_CHECKSUM=1` during bring-up.

Use fewer cards only for implementation checks:

- no NPU: schema, shard hash, patch dry-run, attention math, and sampling tests;
- one card: dummy construction and synthetic SWA/C4/C128 cache tests;
- two to four cards: TP sharding, collectives, W8A8 loading, scheduling, rollback;
- eight cards: real target plus Orthrus end-to-end acceptance and performance.

Less-card tests cannot establish that the complete DSV4 model fits or performs.

The no-NPU contract self-test has no Torch or safetensors dependency:

```bash
python3 scripts/selftest_dsv4_contract.py
python3 scripts/selftest_dsv4_evaluation.py
```

The second command checks the greedy/cache/distribution comparison semantics
and the paired 5% goodput gate using generated result fixtures.

Run the one-card kernel/cache test and head construction test inside the pinned
container:

```bash
python scripts/smoke_dsv4_hybrid_npu.py

python scripts/smoke_construct_dsv4_orthrus.py \
  --model /data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --head /data/disk/models/orthrus-dsv4-head-v2-w8a8
```

The construction test accepts `torchrun` for TP sharding checks without loading
the target backbone. It does a checkpoint-loader round trip at TP1; TP2/TP4
verify distributed construction and local shard allocation:

```bash
torchrun --standalone --nproc-per-node=4 \
  scripts/smoke_construct_dsv4_orthrus.py \
  --model /data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --head /data/disk/models/orthrus-dsv4-head-v2-w8a8 \
  --tensor-parallel-size 4
```

## 5. Performance gate

Run the same matrix against each service:

```bash
python scripts/benchmark_dsv4_matrix.py \
  --base-url http://127.0.0.1:8900 --model dsv4-target \
  --tokenizer /data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --profile prod-128k --label target \
  --matrix benchmarks/dsv4/matrix.json \
  --output-dir results/dsv4/target

python scripts/benchmark_dsv4_matrix.py \
  --base-url http://127.0.0.1:8901 --model dsv4-orthrus \
  --tokenizer /data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --profile prod-128k --label orthrus \
  --matrix benchmarks/dsv4/matrix.json \
  --output-dir results/dsv4/orthrus

python scripts/evaluate_dsv4_gate.py \
  --target-dir results/dsv4/target \
  --orthrus-dir results/dsv4/orthrus \
  --correctness results/dsv4/correctness.json \
  --output results/dsv4/acceptance.json
```

Random tokens measure engine limits only. Repeat the paired benchmark with the
representative internal workload before making an enablement decision. Capture
TTFT, TPOT, output throughput, request goodput, acceptance length/rejection
rate, draft/verify time, and NPU memory. The matrix runner snapshots vLLM's
spec-decode Prometheus counters around each case and adds their delta and mean
acceptance length to the benchmark JSON; service profiling supplies the
draft/verify timing split. Orthrus remains off by default unless
every required production slice is correct and representative-workload
goodput improves by at least 5%.

For a bounded torch-NPU trace, restart the service with an output directory,
then drive the existing profiler against a representative workload manifest:

```bash
ORTHRUS_TORCH_PROFILE_DIR=/data/disk/profiles/dsv4-orthrus \
  REQUIRE_TRAINED_HEAD=1 scripts/serve_dsv4_orthrus.sh

python scripts/profile_vllm_orthrus.py \
  --endpoint http://127.0.0.1:8901 \
  --served-model dsv4-orthrus \
  --manifest /data/disk/workloads/representative.jsonl \
  --profile-seconds 10 \
  --output results/dsv4/profile-window.json

python scripts/analyze_orthrus_profile.py \
  --profile /data/disk/profiles/dsv4-orthrus \
  --output results/dsv4/profile-analysis.json
```

Keep stack and memory tracing disabled for the timing run; collect NPU memory
separately with `npu-smi` so profiler instrumentation does not distort TPOT.

## 6. Offline release and rollback

Create the complete hash manifest and code bundle after acceptance:

```bash
RUNTIME_LOCK=runtime/vllm_ascend_v0_26/runtime-lock.local.json \
MODEL=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
HEAD=/data/disk/models/orthrus-dsv4-head-v2-w8a8 \
OUTPUT_DIR=/data/disk/packages/dsv4-orthrus \
scripts/package_dsv4_offline.sh
```

Production manifests hash all model and head shards; the fast
`--skip-weight-shards` option is development-only. The target-only launcher is
always the rollback path when the head is missing, incompatible, incorrect, or
below the goodput threshold.
