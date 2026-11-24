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
    lib-orthrus-output.patch lib-orthrus-controls.patch lib-orthrus-network-hotpath.patch \
    lib-orthrus-verify-rows.patch lib-orthrus-verify-parallel.patch lib-orthrus-verify-output.patch \
    lib-orthrus-teacher-rows.patch lib-orthrus-student-rows.patch; do
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
git -C "$PYPTO_SERVING_ROOT" apply --check "$patches/serving-orthrus-metadata.patch"
git -C "$PYPTO_SERVING_ROOT" apply "$patches/serving-orthrus-metadata.patch"
git -C "$PYPTO_SERVING_ROOT" apply --check "$patches/serving-orthrus-output.patch"
git -C "$PYPTO_SERVING_ROOT" apply "$patches/serving-orthrus-output.patch"
git -C "$PYPTO_SERVING_ROOT" apply --check "$patches/serving-orthrus-verify-output.patch"
git -C "$PYPTO_SERVING_ROOT" apply "$patches/serving-orthrus-verify-output.patch"
git -C "$PYPTO_SERVING_ROOT" apply --check "$patches/serving-orthrus-teacher-rows.patch"
git -C "$PYPTO_SERVING_ROOT" apply "$patches/serving-orthrus-teacher-rows.patch"
git -C "$PYPTO_SERVING_ROOT" apply --check "$patches/serving-orthrus-student-rows.patch"
git -C "$PYPTO_SERVING_ROOT" apply "$patches/serving-orthrus-student-rows.patch"
python -c 'from pypto_serving_a3.source import KernelSource; KernelSource.resolve(cluster=True)'

if [[ "${1:-}" == --prepare-only ]]; then
  exec python - <<'PY'
from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
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
assert receipts[-1]["commit_tokens_per_dp"] == [1]

class FeedbackProbe:
    bound = True
    needed = 2

    def feedback_pending(self, request_id):
        return request_id == "feedback"

    def feedback_rows_needed(self, request_id):
        assert request_id == "feedback"
        return self.needed

    def verified(self, request_id, prefix, candidates, delivered, logits, *, defer):
        assert (request_id, prefix, defer) == ("feedback", (6, 0), True)
        assert delivered == self.delivered
        assert torch.equal(logits, self.expected_logits)
        return "prepared"

    def synchronize(self, update, exchange):
        assert update == "prepared"

logits = torch.tensor([[0., 2., 0.], [0., 0., 3.], [1., 0., 0.], [0., 4., 0.], [0., 0., 5.]])
def verify_probe(chunk, *, greedy_shard, logit_ranges, **kwargs):
    if greedy_shard is None:
        assert logit_ranges is None
        return dict(completed_layers=list(range(40)), timings={}, logits=[{"fresh": logits}] * 8)
    greedy = greedy_shard(0, logits, 5)
    assert torch.equal(greedy_shard(4, logits, 5), greedy)
    assert logit_ranges(0, greedy) == [(2, 2 + len(session.online.expected_logits))]
    assert not logit_ranges(1, greedy)
    return dict(completed_layers=list(range(40)), timings={}, logits=[{}] * 8,
                greedy_tokens={0: greedy, 1: greedy},
                selected_logits={(0, begin): logits[begin:end] for begin, end in logit_ranges(0, greedy)})

session.native.verify_candidates = verify_probe
session.compact_verifier = False
session.draft = SimpleNamespace(greedy_shard=lambda rank, value, active: value[:active].argmax(-1))
session.cluster.link.exchange = lambda tag, value: [] if tag == "greedy-output/verify" else unexpected_exchange(tag)
session.online = FeedbackProbe()
session.config = SimpleNamespace(tp=4)
session.ledger = SimpleNamespace(get=lambda handle: SimpleNamespace(history=[6]))
session.local_ranks = (0, 1, 2, 3)
handles = (Handle(request_id="ordinary"), Handle(request_id="feedback", slot=1))
feedback_request = dict(request_slot=1, generation=1, token_ids=[0, 0, 1])
inputs = {"DP_engines": [{"active": 5, "dp_rank": 0, "requests": [
    dict(request_slot=0, generation=1, token_ids=[0, 1]), feedback_request,
]}]}
session.online.delivered = (0, 1, 2)
session.online.expected_logits = logits[2:5]
assert [row.tolist() for row in session._execute("verify", inputs, (), handles)] == [[1, 2], [0, 1, 2]]
session.compact_verifier = True
session.local_ranks = tuple(range(8))
handles += (Handle(request_id="greedy-1", dp=1), Handle(request_id="greedy-2", dp=1, slot=1))
inputs["DP_engines"].append({"active": 5, "dp_rank": 1, "requests": inputs["DP_engines"][0]["requests"]})
for candidates, delivered in (((0, 1), (0, 1, 2)), ((0, 2), (0, 1)), ((2, 1), (0,))):
    feedback_request["token_ids"] = [0, *candidates]
    session.online.delivered = delivered
    session.online.expected_logits = logits[2:2 + min(2, len(delivered))]
    assert [row.tolist() for row in session._execute("verify", inputs, (), handles)] == [[1, 2], [0, 1, 2]] * 2
    assert receipts[-1]["teacher_logit_rows"] == min(2, len(delivered))

# Compact serialized evidence must not remove static data from the native call.
utility = source.module("utils")
with TemporaryDirectory() as directory:
    session.directory = Path(directory)
    session.source = source
    session.checkpoint, session.identity = "checkpoint", "identity"
    session.native_config = {"capacity": 16}
    session.cache = {"cache_ABI": [{"hash_constants": {"token_map": torch.arange(256)}}] * 2}
    session.host = SimpleNamespace(lookup=lambda *args, **options: {})
    session._draft_metadata = None
    payload = {"layout": {"views": {}}, "metadata": {"swa": 7}}
    payloads = (payload,) * 2
    draft_prepared = NativeSession._prepare(session, {"DP_engines": []}, payloads, "draft")
    prepared = NativeSession._prepare(session, {"DP_engines": []}, payloads, "verify")
    assert prepared["metadata_sources"] is draft_prepared["metadata_sources"]
    assert session._draft_metadata is None
    evidence = prepared["metadata_sources"][0]
    assert utility.file_sha(evidence["path"]) == evidence["sha256"]
    assert prepared["metadata_sources"][1]["path"] == evidence["path"]
    assert torch.load(evidence["path"], weights_only=True) == ({"metadata": {"swa": 7}},) * 2
    assert set(prepared["cache"]["cache_ABI"][0]) == {"layout", "metadata", "hash_constants"}

# TP groups are host-local and token rows preserve shard and request order.
network = source.module("network")
shards = [{}, {}, {"fresh": torch.arange(12).reshape(4, 3)},
          {"fresh": torch.arange(12, 24).reshape(4, 3)},
          {"fresh": torch.zeros(4, 3)}, {"fresh": torch.zeros(4, 3)}]
ranks = {2: 0, 3: 1, 4: 2, 5: 3}
reads = []
executor = SimpleNamespace(
    rank_devices=ranks, map_ranks=lambda fn: [fn(rank) for rank in ranks],
    resident=SimpleNamespace(read_output=lambda value, device: (reads.append((device, value.numel())), value)[1]),
)
records = [{"positions": positions} for positions in ((), (), (4, 5, 6), (4, 5, 6), (), ())]
outputs, greedy, selected = network._read_speculative_vocab(shards, records, "fresh", executor, 2)
expected = torch.cat((shards[2]["fresh"][:3], shards[3]["fresh"][:3]), dim=-1)
assert not greedy and not selected and reads == [(0, 9), (1, 9)] and torch.equal(outputs[2], expected)
assert outputs[2] is outputs[3] and outputs[4].shape == outputs[5].shape == (0, 6)

# One local DP group captures teacher feedback, while another only returns IDs.
# They share the same projection path but may independently choose how to read it.
session.online = None
session.draft = SimpleNamespace(online_bank=None)
session.native.propose_candidates = lambda *args, **kwargs: dict(
    completed_layers=list(range(40)), timings={}, logits=[], greedy_tokens={0: torch.arange(32)},
    vocabulary_layout="device_greedy",
)
session.cluster.link.exchange = lambda tag, value: [] if tag == "greedy-output/draft" else unexpected_exchange(tag)
draft_input = {"DP_engines": [{"active": 32, "dp_rank": 0, "requests": [
    dict(request_slot=0, generation=1, token_ids=[0] * 32),
]}]}
assert session._execute("draft", draft_input, (), (Handle(),)).tolist() == [list(range(31))]
assert receipts[-1]["vocabulary_layout"] == "device_greedy"

student = torch.arange(64 * 17).reshape(64, 17).float()
student[::2, 0] += 100
proposals = student.argmax(dim=-1)
captured = []
capture_owner = (0, 1, 1)

def select_capture(requests, _exchange):
    assert requests[capture_owner] == ("student", (6, 0))
    return {capture_owner: requests[capture_owner]}

def take_capture(*owner):
    assert owner == capture_owner
    return SimpleNamespace(with_student_logits=lambda logits: logits)

def draft_probe(_chunk, _draft, *, logit_ranges, **kwargs):
    base = dict(completed_layers=list(range(40)), timings={})
    if logit_ranges is None:
        return dict(base, logits=[{"fresh": student}] * 4)
    assert logit_ranges(0, proposals) == [(32, 39)]
    assert not logit_ranges(1, proposals)
    return dict(base, logits=[{}] * 4, greedy_tokens={0: proposals},
                selected_logits={(0, 32): student[32:39]})

session.native.propose_candidates = draft_probe
session.native.take_draft_replay = take_capture
session.online = SimpleNamespace(
    bound=True, capture_due=lambda: True, select_capture=select_capture,
    max_feedback_rows=7, plan="last-1", register_replay=captured.append,
)
session.draft = SimpleNamespace(online_bank=None, arm_online_capture=lambda *_: None)
draft_input["DP_engines"][0]["requests"].append(
    dict(request_slot=1, generation=1, token_ids=[0] * 32),
)
draft_input["DP_engines"][0]["active"] = 64
draft_handles = (Handle(), Handle(request_id="student", slot=1))
for compact in (False, True):
    session.compact_verifier = compact
    proposed = session._execute("draft", draft_input, (), draft_handles)
    assert proposed.tolist() == [proposals[:31].tolist(), proposals[32:63].tolist()]
    assert torch.equal(captured[-1], student[32:39 if compact else 63])
    assert receipts[-1]["student_logit_rows"] == (7 if compact else None)

records = [{"positions": range(32 if rank < 4 else 64)} for rank in range(6)]
shards = [{}, {}] + [{"fresh": torch.arange(64 * 3).reshape(64, 3) + 200 * rank}
                    for rank in ranks]
def reduce_rank(rank, values, active):
    best, ids = values[:active].max(dim=-1)
    return torch.stack((best, ids, torch.ones_like(best)), dim=-1).float()

def reduce_non_teacher_rank(rank, values, active):
    if rank // 2 != 1:
        return reduce_rank(rank, values, active)

reads.clear()
outputs, greedy, selected = network._read_speculative_vocab(
    shards, records, "fresh", executor, 2, reduce_non_teacher_rank,
)
assert reads == [(0, 96), (1, 96)] and set(outputs) == {2, 3} and not selected
assert torch.equal(greedy[2], torch.cat((shards[4]["fresh"], shards[5]["fresh"]), dim=-1).argmax(-1))

reads.clear()
outputs, greedy, selected = network._read_speculative_vocab(
    shards, records, "fresh", executor, 2, reduce_rank,
    lambda dp, tokens: [(5, 6), (32, 35)] if dp == 2 else [],
)
assert not outputs and set(greedy) == {1, 2}
assert set(selected) == {(2, 5), (2, 32)}
assert reads == [(2, 3), (2, 9), (3, 3), (3, 9)]
full = torch.cat((shards[4]["fresh"], shards[5]["fresh"]), dim=-1)
assert torch.equal(selected[2, 5], full[5:6])
assert torch.equal(selected[2, 32], full[32:35])

runtime_type = source.module("prefill_resident").ResidentNetworkRuntime
# Only adjacent intervals may share a D2D copy. Restore on failure as well.
from ctypes import memmove

cache = torch.arange(20, dtype=torch.uint8)
values = {cache.data_ptr(): cache}
copies = []
def alloc_tensor(shape, dtype):
    tensor = torch.empty(shape, dtype=dtype)
    values[tensor.data_ptr()] = tensor
    return SimpleNamespace(data_ptr=tensor.data_ptr(), nbytes=tensor.nbytes)
def device_copy(dst, src, count, rank):
    copies.append(count)
    memmove(dst, src, count)

snapshot = SimpleNamespace(
    _closed=False, _select=lambda rank: None, _device_copy=device_copy, _record_transfer=lambda *args: None,
    pinned={(0, cache.untyped_storage()._cdata): {"readonly": False, "name": "raw/cache",
                                                 "allocation": SimpleNamespace(data_ptr=cache.data_ptr(),
                                                                               nbytes=cache.nbytes)}},
    workers=[SimpleNamespace(alloc_tensor=alloc_tensor, free_tensor=lambda ptr: values.pop(ptr.data_ptr))],
)
try:
    with runtime_type.preserve_cache_ranges(snapshot, 0, [(cache, [(1, 2), (3, 1), (10, 4)])]):
        cache[1:4].fill_(1)
        cache[10:14].fill_(1)
        cache[16].fill_(90)
        raise RuntimeError("synthetic verification failure")
except RuntimeError as error:
    assert str(error) == "synthetic verification failure"
assert copies == [3, 4, 3, 4]
assert cache[1:4].tolist() == [1, 2, 3] and cache[10:14].tolist() == [10, 11, 12, 13]
assert int(cache[16]) == 90

# Drain every live snapshot if model execution aborts and the rank queues
# have already rejected future stages.
from contextlib import contextmanager

events = []
@contextmanager
def record_snapshot(rank, ranges):
    events.append(("save", rank))
    try:
        yield
    finally:
        events.append(("restore", rank))

def map_snapshot_ranks(function):
    assert not disabled
    return [function(rank) for rank in range(3)]

disabled = False
snapshots = SimpleNamespace(preserve_cache_ranges=record_snapshot, map_ranks=map_snapshot_ranks)
with runtime_type.preserve_rank_cache_ranges(snapshots, {0: (), 2: ()}):
    assert events == [("save", 0), ("save", 2)]
assert events == [("save", 0), ("save", 2), ("restore", 0), ("restore", 2)]
events.clear()
try:
    with runtime_type.preserve_rank_cache_ranges(snapshots, {0: (), 2: ()}):
        disabled = True
        raise RuntimeError("verify dispatch failed")
except RuntimeError as error:
    assert str(error) == "verify dispatch failed"
assert events == [("save", 0), ("save", 2), ("restore", 2), ("restore", 0)]

# A verifier crossing two nonconsecutive pages only backs up written SWA rows.
# Compression and token history remain protected across all their live pages.
provider_type = source.module("state").NetworkCacheProvider
swa, compressed = "model.layers.1.self_attn.swa_cache", "model.layers.2.self_attn.long_kv_cache"
views = {
    swa: dict(storage_key="swa", dtype="torch.bfloat16", shape=[8, 128, 1, 512],
              stride=[65600, 512, 512, 1], storage_offset_bytes=128),
    compressed: dict(storage_key="long", dtype="torch.bfloat16", shape=[8, 128, 1, 512],
                     stride=[65536, 512, 512, 1], storage_offset_bytes=0),
}
layout = dict(views=views, raw={key: {"storage_offset_bytes": 0} for key in ("swa", "long")})
provider = SimpleNamespace(local_ranks=(0,), layouts={0: {"fresh": layout}})
metadata = {"groups": [dict(
    names=[swa, compressed], live_block_ids=[2, 5, 7],
    block_table=torch.tensor([[2, 7, 5]]), slot_mapping=torch.tensor([[7, 127], [5, 0], [5, 1]]),
)]}
ranges, history = provider_type.verification_ranges(provider, 0, metadata)
assert ranges["swa"] == [(5 * 131200 + 128, 2048), (7 * 131200 + 128 + 127 * 1024, 1024)]
assert ranges["long"] == [(page * 131072, 131072) for page in (2, 5, 7)]
assert history == (2, 5, 7)

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
    accept_without_logits=True, verifier_feedback_only_for_owner=True, compact_speculative_metadata=True,
    draft_student_rows_only_for_owner=True,
    host_vocabulary_shard_order=True, compact_greedy_with_full_feedback=True,
    verify_copy_coalescing=True, verify_swa_written_rows=True, parallel_verifier_rollback=True,
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
