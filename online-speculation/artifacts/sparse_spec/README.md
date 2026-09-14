# SparseSpec port experiment artifacts

This directory contains the small, versioned part of the SparseSpec port
experiments:

- `sparse-draft-microbenchmark-v2.json`: raw NPU page-sparse attention timings.
- `*/manifest.json`: runtime, workload hash, source and feature configuration.
- `*/summary.json`: end-to-end latency and throughput summaries.

Per-request `result.json`, service logs and profiler traces remain outside Git.
The complete server archive is under:

```text
/data/disk/wangzili/online-speculation/artifacts/tuning-sparse
/data/disk/wangzili/online-speculation/artifacts/tuning/paged-segmented-formal-64-out128-m94
```

The upstream source inspected for the port was:

```text
repository: https://github.com/sspec-project/SparseSpec
commit: 5c9616c9af109ecec0fdf193221d3f5be6f05922
license: Apache-2.0
```
