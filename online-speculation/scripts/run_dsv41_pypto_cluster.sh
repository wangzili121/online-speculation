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
    lib-deterministic-reduction.patch lib-cluster-online.patch lib-workspace-write-only.patch \
    lib-orthrus-output.patch; do
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
git -C "$PYPTO_SERVING_ROOT" apply --check "$patches/serving-orthrus-fast.patch"
git -C "$PYPTO_SERVING_ROOT" apply "$patches/serving-orthrus-fast.patch"
python -c 'from pypto_serving_a3.source import KernelSource; KernelSource.resolve(cluster=True)'

if [[ "${1:-}" == --prepare-only ]]; then
  exec python - <<'PY'
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch

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
assert "self.online is not None and self.online.bound" in execute_source
assert 'if phase != "accept":' in execute_source

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

# The accept path commits target state but returns no logits and sends no
# greedy-output message. Exercise the patched Serving function itself.
class CommitProbe:
    def __init__(self):
        self.weights = SimpleNamespace(counters=lambda: {})
        self.epoch = 0

    def commit_candidates(self, chunk, **kwargs):
        self.epoch += 1
        return dict(completed_layers=list(range(40)), logits=[], timings={})

def unexpected_exchange(*args):
    raise AssertionError("accept communicated vocabulary output")

session.native = CommitProbe()
session.cluster = SimpleNamespace(link=SimpleNamespace(exchange=unexpected_exchange))
session.calls = 0
session._prepare = lambda *args: {"contract": {"input_contract_sha256": "accept-probe"}}
receipts = []
session._record = receipts.append
assert session._execute("accept", {"DP_engines": [{"active": 1, "dp_rank": 0}]}, (), (Handle(),)) is None
assert session.native.epoch == session.calls == 1 and session.predecessor == "accept-probe"
assert receipts[-1]["output_head_skipped"] and receipts[-1]["greedy_rows"] == []

class FeedbackProbe:
    bound = True

    def feedback_pending(self, request_id):
        return request_id == "feedback"

    def verified(self, request_id, prefix, candidates, delivered, logits, *, defer):
        assert (request_id, prefix, candidates, delivered, defer) == ("feedback", (6, 0), (1,), (1, 2), True)
        assert logits.shape == (2, 3)
        return "prepared"

    def synchronize(self, update, exchange):
        assert update == "prepared"

logits = torch.tensor([[0., 2., 0.], [0., 0., 3.], [1., 0., 0.], [0., 4., 0.]])
session.native.verify_candidates = lambda *args, **kwargs: dict(
    completed_layers=list(range(40)), timings={}, logits=[{"fresh": logits}] * 4,
)
session.cluster.link.exchange = lambda tag, value: [] if tag == "greedy-output/verify" else unexpected_exchange(tag)
session.online = FeedbackProbe()
session.config = SimpleNamespace(tp=4)
session.ledger = SimpleNamespace(get=lambda handle: SimpleNamespace(history=[6]))
session.local_ranks = (0, 1, 2, 3)
handles = (Handle(request_id="feedback"), Handle(request_id="ordinary", slot=1))
inputs = {"DP_engines": [{"active": 4, "dp_rank": 0, "requests": [
    dict(request_slot=0, generation=1, token_ids=[0, 1]),
    dict(request_slot=1, generation=1, token_ids=[0, 2]),
]}]}
assert [row.tolist() for row in session._execute("verify", inputs, (), handles)] == [[1, 2], [0, 1]]

runtime_type = source.module("prefill_resident").ResidentNetworkRuntime
runtime = runtime_type.__new__(runtime_type)
replay_root, scratch = torch.zeros(1), torch.zeros(1)
key = lambda value: (0, value.untyped_storage()._cdata)
runtime.values = {key(value): value for value in (replay_root, scratch)}
runtime._retire_values = lambda keys: [runtime.values.pop(item) for item in keys]
runtime.replay_values = set()
runtime._closed = False
runtime.retain_replay_values([(0, replay_root)])
runtime.release_values()
assert tuple(runtime.values) == (key(replay_root),)
runtime.release_replay_values()
assert not runtime.values and not runtime.replay_values

print(json.dumps(dict(
    passed=True,
    stage="offline-overlay",
    tree=os.environ["ORTHRUS_CLUSTER_LIB_TREE"],
    lifecycle={"target_unbound": True, "draft_bind": True, "unbound_cleanup": True,
               "closed_cleanup": True, "replay_survives_scratch_recycle": True},
    accept_without_logits=True, verifier_feedback_only_for_owner=True,
)))
PY
fi

if [[ "${1:-}" == --compile-only ]]; then
  shift
  exec python "$root/scripts/compile_dsv41_pypto_cluster.py" "$@"
fi

check_online_abi() {
  echo 'Checking resident online kernel signatures before loading model weights.'
  bash "$patches/docker/cache-pto-isa.sh"
  PYTHONPATH="$PYPTO_LIB_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python "$root/scripts/validate_dsv41_pypto_online_programs.py" \
    --compile-only --resident-abi --output /results/resident-online-abi.json
  python "$root/scripts/validate_dsv41_pypto_expert_input_vjp.py" \
    --production --simulate --output /results/resident-expert-vjp.json
}
if [[ "${1:-}" == --online-abi-only ]]; then
  check_online_abi
  exit
fi
for option in "$@"; do
  if [[ "$option" == --online-study ]]; then
    check_online_abi
    break
  fi
done

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
