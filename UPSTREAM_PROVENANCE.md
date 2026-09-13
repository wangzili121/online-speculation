# Upstream provenance

This repository was initialized from a clean source snapshot of Changbo's
`inference_scaling` repository. The upstream Git history was intentionally not
imported so that this private repository can maintain its own experiment and
integration history.

- Upstream: `git@github.com:xchang1121/inference_scaling.git`
- Branch: `main`
- Commit: `1c3f65d38a7f9176ce9d9897f43515ed52654ce2`
- Commit date: `2026-09-11T23:54:47+08:00`
- Imported: `2026-09-14`
- Snapshot scope: all files tracked by the upstream commit
- Excluded: upstream `.git`, model weights, caches, and generated experiment artifacts

The initial serving target is the `online-speculation` implementation and the
public `chiennv/Orthrus-Qwen3-8B` checkpoint. Infrastructure work is compared
against the vLLM-Ascend baseline built from this snapshot; ordinary AR decoding
is retained only as a correctness oracle.
