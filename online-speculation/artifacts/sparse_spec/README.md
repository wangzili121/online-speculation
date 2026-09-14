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

The following two server runs are retained only as invalid-run evidence and
must not be used for comparisons. Both host-network containers used port 8000,
so their clients reached the same service:

```text
pillar-recent-r075-max24k-t24k-16-out1024-20260914
pillar-recent-r100-max40k-t24k-16-out1024-20260914
```

Use the `samecard-repeat` manifests and summaries for the valid 75% versus
dense comparison.

The upstream source inspected for the port was:

```text
repository: https://github.com/sspec-project/SparseSpec
commit: 5c9616c9af109ecec0fdf193221d3f5be6f05922
license: Apache-2.0
```
