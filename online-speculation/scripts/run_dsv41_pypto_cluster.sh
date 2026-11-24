#!/usr/bin/env bash
# Apply the pinned integration on top of the prebuilt offline image's writable layer.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patches="$root/runtime/pypto_dsv41_a3"
: "${PYPTO_LIB_ROOT:?Use the paired A3 offline image}"
: "${PYPTO_SERVING_ROOT:?Use the paired A3 offline image}"

# Verify the pinned base without requiring a pristine worktree: refreshed
# images intentionally carry the reviewed expert-capture patch uncommitted.
python - "$PYPTO_LIB_ROOT" <<'PY'
from pathlib import Path
import subprocess
import sys

from pypto_serving_a3.source import A3_COMMIT, A3_TREE

root = Path(sys.argv[1])
git = lambda *args: subprocess.check_output(
    ["git", "-C", str(root), *args], text=True,
).strip()
commit, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
if commit != A3_COMMIT and tree != A3_TREE:
    raise SystemExit(
        f"Unverified A3 base source: expected {A3_COMMIT}/{A3_TREE}, "
        f"found {commit}/{tree}"
    )
PY
if ! grep -q 'state\["expert_outputs"\]' \
    "$PYPTO_LIB_ROOT/models/deepseek_v4_1_flash_a3/prefill_layer.py"; then
  git -C "$PYPTO_LIB_ROOT" apply --check "$patches/lib-online-expert-capture.patch"
  git -C "$PYPTO_LIB_ROOT" apply "$patches/lib-online-expert-capture.patch"
fi
if ! grep -q 'ffn_collapsed=collapsed' \
    "$PYPTO_LIB_ROOT/models/deepseek_v4_1_flash_a3/prefill_layer.py"; then
  git -C "$PYPTO_LIB_ROOT" apply --check "$patches/lib-online-state-capture.patch"
  git -C "$PYPTO_LIB_ROOT" apply "$patches/lib-online-state-capture.patch"
fi
if [[ -n "$(git -C "$PYPTO_SERVING_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo 'Use a fresh offline image with the paired Serving adapter.' >&2
  exit 2
fi

for patch in lib-rank-weights.patch lib-rank-cache.patch lib-cluster-execution.patch \
    lib-pypto72-compat.patch lib-cluster-retry.patch lib-cluster-prefill-trace.patch \
    lib-deterministic-reduction.patch lib-cluster-online.patch; do
  git -C "$PYPTO_LIB_ROOT" apply --check "$patches/$patch"
  git -C "$PYPTO_LIB_ROOT" apply "$patches/$patch"
done
git -C "$PYPTO_LIB_ROOT" add -u
git -C "$PYPTO_LIB_ROOT" -c user.name=Integration -c user.email=integration@localhost \
  -c commit.gpgsign=false commit -qm 'Run the resident DSV4.1 network across two nodes'
export ORTHRUS_CLUSTER_LIB_TREE="$(git -C "$PYPTO_LIB_ROOT" rev-parse 'HEAD^{tree}')"
git -C "$PYPTO_SERVING_ROOT" apply --check "$patches/serving-cluster.patch"
git -C "$PYPTO_SERVING_ROOT" apply "$patches/serving-cluster.patch"
git -C "$PYPTO_SERVING_ROOT" apply --check "$patches/serving-online.patch"
git -C "$PYPTO_SERVING_ROOT" apply "$patches/serving-online.patch"
python -c 'from pypto_serving_a3.source import KernelSource; KernelSource.resolve(cluster=True)'

if [[ "${1:-}" == --prepare-only ]]; then
  exec python - <<'PY'
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace

import blockspec.pypto_dsv41_runtime as draft_runtime
from pypto_serving_a3.session import NativeSession
from pypto_serving_a3.source import KernelSource

source = KernelSource.resolve(cluster=True)
source.require_generation()
from inspect import getsource, signature
assert "transport_factory" in signature(source.module("prefill_fwd").prefill_fwd).parameters
assert "cluster" in signature(NativeSession).parameters

# Target-only runs precede deferred draft loading. They must neither call nor
# clean up an online learner before the first draft has bound resident state.
execute_source = getsource(NativeSession._execute)
assert "self.online is not None and self.online.bound and phase" in execute_source

@dataclass
class Handle:
    request_id: str = "target-only"
    dp: int = 0
    slot: int = 0
    generation: int = 1

class Unbound:
    bound = False
    def finish(self, *args, **kwargs):
        raise AssertionError("unbound target cleanup called online learner")

released = []
session = NativeSession.__new__(NativeSession)
session.online, session.closed, session.cluster, session.ledger = Unbound(), False, None, None
session.native = SimpleNamespace(release_request=lambda *owner: released.append(owner))
session.clear_request(Handle())
assert released == [(0, 0, 1)]

session.online.bound = True
session.closed = True
session.clear_request(Handle())
assert released == [(0, 0, 1)]
session.closed = False

class Binding:
    bound = False
    def bind(self, draft, native, cluster):
        self.bound = draft is marker and native is session.native and cluster is None

marker, binding = object(), Binding()
session.online = binding
session.checkpoint = session.orthrus_head = "fixture"
session.native = SimpleNamespace(runtime=object())
session.config = SimpleNamespace(topology={})
session.directory = Path("/tmp")
session.source = SimpleNamespace(module=lambda name: None)
session.local_ranks = (0,)
original = draft_runtime.ResidentDraftAttention
draft_runtime.ResidentDraftAttention = lambda *args, **kwargs: marker
try:
    session._load_draft()
finally:
    draft_runtime.ResidentDraftAttention = original
assert session.draft is marker and binding.bound

print(json.dumps(dict(
    passed=True,
    stage="offline-overlay",
    tree=os.environ["ORTHRUS_CLUSTER_LIB_TREE"],
    lifecycle={"target_unbound": True, "draft_bind": True, "unbound_cleanup": True,
               "closed_cleanup": True},
)))
PY
fi

if [[ "${1:-}" == --compile-only ]]; then
  shift
  exec python "$root/scripts/compile_dsv41_pypto_cluster.py" "$@"
fi

: "${TARGET_WEIGHTS:?Mount the paired W8A8 model}"
if [[ "${PYPTO_REUSE_PREVIOUS_HEAD:-1}" == 1 && -z "${DRAFT_HEAD:-}" && -n "${PYPTO_PREVIOUS_HEAD:-}" ]]; then
  if python - "$TARGET_WEIGHTS" "$PYPTO_PREVIOUS_HEAD" <<'PY'
from pathlib import Path
import sys
from blockspec.pypto_dsv41_draft import read_draft_artifact

try:
    _, manifest, _ = read_draft_artifact(Path(sys.argv[1]), Path(sys.argv[2]))
    if manifest["artifact_role"] != "initialized-plumbing" or manifest["mask_token_id"] != 129279:
        raise ValueError("Expected the default target-initialized head")
except (OSError, ValueError) as error:
    print(f"Previous initialized head could not be reused: {error}", file=sys.stderr)
    raise SystemExit(1) from None
PY
  then
    DRAFT_HEAD="$PYPTO_PREVIOUS_HEAD"
    echo 'Reusing the validated initialized head from the previous run.'
  fi
fi
if [[ -z "${DRAFT_HEAD:-}" ]]; then
  DRAFT_HEAD=/results/initialized-head
  echo 'Preparing a target-initialized BF16 Orthrus head for the inference smoke test.'
  python "$root/scripts/export_dsv41_orthrus_head.py" \
    --model "$TARGET_WEIGHTS" --output "$DRAFT_HEAD" --from-target --mask-token-id 129279
fi

exec python "$root/scripts/run_dsv41_pypto_cluster.py" "$@" \
  --model "$TARGET_WEIGHTS" --head "$DRAFT_HEAD" --output /results/acceptance
