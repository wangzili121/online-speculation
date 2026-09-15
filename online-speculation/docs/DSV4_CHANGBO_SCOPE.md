# Chang Bo online-speculation versus Orthrus

The original Orthrus design supplies two attention views over one backbone:
an autoregressive target and a bidirectional masked-block draft. The draft
reads real target history KV; the verifier corrects its proposed block. The
anchor/mask alignment and exact rejection rule are the inference foundation.

Chang Bo's `online-speculation` retains that foundation and adds serving-side
learning mechanisms: tensorized parallel block sampling, verifier-feedback
collection, frozen-prefix/decoder-suffix replay, FP32 draft-attention master
weights, complete-vocabulary TV supervision, update publication at request
boundaries, a cold-start path from copied AR attention, and a speed gate for
promoting an updated draft. These are real additions, not an alias for the
original Orthrus checkpoint. See `src/blockspec/parallel/online.py`,
`parallel/feedback.py`, `parallel/generation.py`, and `docs/ALGORITHM.md`.

The DSV4-Flash v0.26 service currently implements the **fixed-head inference
subset** of Chang Bo's design. It preserves K=32 anchor-plus-mask proposals,
bidirectional local attention, target-history sharing, and target correction;
the DSV4-specific extension is SWA/C4/C128 cache sharing and C4 indexer
querying. It consumes a separately trained and calibrated v2 head. It does
**not** train or publish a new head online, replay verifier feedback, or run
the cold-start/speed-gate learner in the serving process. Those remain a
separate integration milestone, consistent with external head training.

Greedy proposal is useful for wiring tests and still permits exact target
sampling through standard rejection, but it is not the trained parallel
distribution `q` used by Chang Bo's learner. Production deployment of a
trained head must use `ORTHRUS_DRAFT_SAMPLE_METHOD=probabilistic` and only
after concurrent-request seed isolation and exactness tests pass. An
initialized head only tests the transport and model path, never acceptance
or throughput benefit.

Compared with vLLM's Orthrus PRs, this code is narrower in deployment target
but broader in DSV4-specific cache contracts. [PR #46007](https://github.com/vllm-project/vllm/pull/46007)
remains open and has GPU/Qwen tests and benchmarks. [PR #53753](https://github.com/vllm-project/vllm/pull/53753)
was closed unmerged on 2026-09-06. Its last reported A10G/Qwen3-1.7B
100-prompt run completed, accepted 52.62% of proposals, but ran at 0.648x
target-only output throughput. Its author found that merely aliasing the
target's KV cache suppressed the local diffusion KV writes; restoring block
visibility raised acceptance but did not pay for a full extra forward pass.
Our DSV4 path instead keeps local block KV in a separate workspace and reads
target history without writing target cache, so it must pass the same block
visibility and checksum tests before any throughput comparison. Neither PR
proves that this Ascend/DSV4 path is production ready. The release gate is
eight-card model loading, cache-invariant and concurrent rejection tests,
target-only token equality, trained-head sampling tests, and representative-
workload goodput. No superiority claim is warranted without paired results.
