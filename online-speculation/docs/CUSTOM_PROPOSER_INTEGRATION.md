# Orthrus plugin integration

## Repository boundary

The clean integration has three deliberately different ownership areas:

| Repository | Change | Reason |
| --- | --- | --- |
| vLLM | None | Upstream already parses and imports `custom_class`, schedules candidate blocks, and performs rejection sampling. |
| vLLM-Ascend | Generic model-aware `custom_class` lifecycle | A neural proposer needs the live NPU runner, model loading, attention metadata, and cache planning; upstream's token-only callback cannot provide them. |
| online-speculation | All Orthrus and DSV4 code | Head validation, SWA/C4/C128 attention, target-backbone sharing, K=32 drafting, quantization, training, and deployment are algorithm/model-specific. |

This prevents an Orthrus method enum, DSV4 architecture, or checkpoint format
from becoming a permanent vLLM-Ascend maintenance burden. Existing token-only
custom proposers remain unchanged.

## Framework extension

The plugin class declares `npu_model_runner_api_version = 1`. vLLM-Ascend then
calls `bind_runtime()` with its existing `NPUModelRunner`, device, and config.
The plugin delegates to `AscendOrthrusProposer`, so it uses the same padded
batch, rejection, distributed, and cache lifecycle as built-in model
proposers.

Only two generic model hooks are needed:

1. A draft model may implement `bind_shared_backbone(target_model)` after load
   to reuse target embedding, mHC, norms, MoE/MLP, and LM head.
2. Any attention wrapper may set `kv_sharing_target_layer_name`; cache planning
   records the alias and allocates no duplicate history cache.

These hooks do not mention Orthrus or DeepSeek-V4.

## Plugin configuration

The public method remains upstream `custom_class`; the head path is separate
from the Python class path:

```bash
--speculative-config '{
  "method":"custom_class",
  "model":"blockspec.vllm_ascend.dsv4_custom_proposer.OrthrusDSV4CustomProposer",
  "num_speculative_tokens":31,
  "parallel_drafting":true,
  "draft_sample_method":"probabilistic",
  "rejection_sample_method":"standard",
  "enforce_eager":true
}' \
--additional-config '{
  "enable_cpu_binding":true,
  "custom_proposer_config":{
    "orthrus":{"head_model":"/data/disk/models/orthrus-dsv4-head-v2-w8a8"}
  }
}'
```

`scripts/serve_dsv4_orthrus_plugin.sh` builds this configuration and runs the
strict head preflight. The existing `serve_dsv4_orthrus.sh` plus guarded v0.26
runtime overlay remains the pinned deployment fallback until the generic
vLLM-Ascend extension is available in the target image.

## Execution boundary

The plugin creates an internal draft `ModelConfig` for the head without asking
vLLM to reinterpret the Orthrus architecture as DFlash or DSpark. It reuses the
existing Ascend block-draft implementation internally, while the public method
stays `custom_class`. This is an implementation detail: DSpark is neither the
configured algorithm nor a semantic fallback.

The first supported mode is eager TP with the target and draft using the same
TP size. Graph support, online weight publication, and trained-head performance
acceptance remain separate gates.

The launcher defaults to `MODEL_LOAD_THREADS=128` for an exclusive production
node. Use `MODEL_LOAD_THREADS=8` or `16` on a shared bring-up host to avoid
saturating host CPU and storage while eight TP workers load the target.
