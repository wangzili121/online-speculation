# vLLM-Ascend 0.23 Orthrus compatibility

This overlay is baseline plumbing, not an optimization. It is intentionally
locked to vLLM and vLLM-Ascend 0.23.0 and fixes three missing capabilities:

1. load a `custom_class` proposer and bind it lazily to the live target runner;
2. pass Orthrus proposal probabilities into exact rejection sampling;
3. route K32 verification batches (31 proposals plus one anchor) through the existing prefill attention path
   when the fused decode kernel's 16-token limit is exceeded.

The proposer refuses a second full draft model. The formal B1 baseline must
implement `orthrus_propose` on the target model and physically share the
embedding, MLP, norms and LM head.

Run `./apply.sh /vllm-workspace/vllm-ascend` inside the pinned container. The
script first performs a dry run, so source drift fails without partial edits.

`Dockerfile.msprof` adds only the pinned MS Service Profiler dependencies used
for the independent service-level profile pass. Performance trials continue to
use the unmodified official base image; the profiler image is not used for
throughput numbers.
