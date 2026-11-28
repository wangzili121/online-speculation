# Orthrus DSV4.1 A3 company build

From the online-speculation checkout on an ARM64 Docker host, run:

```bash
bash online-speculation/runtime/pypto_dsv41_a3/company/build.sh
```

The build reuses a compatible local PyPTO image or pulls the pinned A3 CANN
base from SWR. The public PyPTO checkout, submodules and build dependencies
are obtained during the build. Lib and Serving come from `sources/`. Their
upstream commits are in `sources/commits.env`. The Serving copy contains
the HTTP adapter and model package selector. Use the bundled Lib for the
model, Draft, Verify and online learner.

After the build prints `Image ready`, tag the built image with your EKS
repository address and push from the authenticated Docker host. The deployment
can run `bash /opt/orthrus/start-serving.sh`. Supply `CHECKPOINT`, `HEAD_DIR`
and `CACHE_DIR` as container paths, and use the 16 allocated device IDs via
`TASK_DEVICE` or `DEVICES`. The service listens on port 8000. An A3 run must
still verify device initialization, collectives and inference.

For the study scripts inside the image, pass
`--lib-root /opt/orthrus-a3/pypto-lib` and
`--serving-root /opt/orthrus-a3/DeepSeek-V4.1-Flash-A3`. The scripts are under
`/opt/orthrus-a3/study/online-speculation/scripts`.

## Update the snapshots

Run this after advancing the corresponding Lib and Serving commits:

```bash
bash online-speculation/runtime/pypto_dsv41_a3/company/sync-sources.sh \
  /path/to/pypto-lib /path/to/DeepSeek-V4.1-Flash-A3
git diff --stat
```

Review the snapshots and commit them with the matching experiment changes.
The sync copies committed source using Git archives and records both commit IDs.
