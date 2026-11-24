#!/usr/bin/env python3
"""Read existing DSV4.1 cluster receipts and summarize dispatch cost on one node."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

from verify_dsv41_pypto_acceptance import TokenDivergence, check_generated_prefix


PHASES = ("prefill", "decode", "draft", "verify", "accept")
PROFILE_ARMS = ("ar", "fixed_before", "update", "update_extra", "fixed_after")
COUNTERS = (
    "verify_backup_d2d_bytes", "verify_restore_d2d_bytes",
    "device_draft_weight_upload_calls", "device_draft_weight_upload_seconds",
)


def read_receipts(directory):
    receipt_paths = sorted(directory.glob("native/session-*/serving-dispatches.jsonl"))
    if len(receipt_paths) != 1:
        raise ValueError(f"Expected one shared native session in {directory}, found {len(receipt_paths)}")
    rows = []
    with receipt_paths[0].open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break  # The Serving process may still be writing the final receipt.
            rows.append(json.loads(line))
    return receipt_paths[0].parent, rows


def profile_comparisons(arms):
    baseline = arms["ar"]["requests"]
    comparisons = {}
    for name in ("fixed_before", "fixed_after"):
        probes = arms[name]["requests"]
        if len(baseline) != len(probes):
            raise ValueError(f"{name} has {len(probes)} probes; AR has {len(baseline)}")
        matched = 0
        divergence = None
        for a, b in zip(baseline, probes):
            if (a["prompt_ids"], a["workload_id"]) != (b["prompt_ids"], b["workload_id"]):
                raise ValueError(f"{name} uses different probes from AR")
            try:
                check_generated_prefix(b["index"], a["token_ids"], b["token_ids"], phase=name)
                if len(b["token_ids"]) < len(a["token_ids"]):
                    offset = len(b["token_ids"])
                    raise TokenDivergence(b["index"], offset, name, a["token_ids"][offset], None)
            except TokenDivergence as error:
                divergence = error.details
                break
            matched += len(a["token_ids"])
        comparisons[name] = dict(matched_tokens=matched, divergence=divergence)
    return comparisons


def profile_head_versions(receipt_path, requests):
    fixed = {
        row["request_id"]: row["phase"] for row in requests
        if row["phase"] in ("fixed_before", "fixed_after")
    }
    versions = {phase: set() for phase in fixed.values()}
    with receipt_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["phase"] != "draft":
                continue
            for owner in row["owners"]:
                phase = fixed.get(owner["request_id"])
                if phase is not None:
                    versions[phase].add(row.get("draft_head_version") or 0)
    return {phase: sorted(values) for phase, values in versions.items()}


def recover_online_profile(directory):
    """Summarize saved probes and local update costs even if final acceptance failed."""
    arms = {}
    for name in PROFILE_ARMS:
        path = directory / f"online-profile-{name}.json"
        if name != "update_extra" or path.exists():
            arms[name] = json.loads(path.read_text())
    with (directory / "online-profile.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    hot = [row for row in records if not (row.get("error") or row["compiled"] or row["first_dispatch"])]
    local_hot = [row for row in hot if row["owner_is_local"]]
    requests = [row for arm in arms.values() for row in arm["timing"]]
    session, = directory.glob("native/session-*/serving-dispatches.jsonl")
    return dict(
        greedy_comparison=profile_comparisons(arms),
        probe_head_versions=profile_head_versions(session, requests),
        probes={
            name: dict(
                draft_rounds=arms[name]["speculation"].get("draft_rounds", 0),
                proposed=arms[name]["speculation"].get("proposed_candidates", 0),
                accepted=arms[name]["speculation"].get("accepted_candidates", 0),
                decode_tok_s=round(arms[name]["decode_tokens_per_second"], 3),
            ) for name in ("ar", "fixed_before", "fixed_after")
        },
        updates=[dict(
            update=row["update"], dp=row["owner_dp"], teacher_rows=row["teacher_rows"],
            seconds=round(row["update_seconds"], 3), publication=round(row["publication_seconds"], 3),
            compile_seconds=round(sum(item["seconds"] for item in row["compiled"]), 3),
            first_dispatches=len(row["first_dispatch"]),
        ) for row in records if not row.get("error")],
        hot_update_median_seconds=round(median(row["update_seconds"] for row in hot), 3) if hot else None,
        hot_local_owner_stage_seconds={
            name: round(median(row["stage_seconds"].get(name, 0.0) for row in local_hot), 4)
            for name in sorted({name for row in local_hot for name in row["stage_seconds"]})
        },
    )


def online_kernels(directory):
    """Match resident update calls with already enabled Simpler runtime spans."""
    dispatch_path = directory / "online-profile-dispatches.jsonl"
    with dispatch_path.open() as stream:
        calls = sorted((json.loads(line) for line in stream), key=lambda row: row["start_ns"])
    timelines = defaultdict(lambda: ([], []))
    for index, row in enumerate(calls):
        keys = [(field, row[field]) for field in ("device", "tid") if field in row]
        for key in keys or [None]:
            starts, indices = timelines[key]
            starts.append(row["start_ns"])
            indices.append(index)
    traces = [Counter() for _ in calls]
    pending = {}

    def containing(timestamp, duration=0, *, device=None, tid=None):
        keys = [("device", device)] if device is not None else []
        if tid is not None:
            keys.append(("tid", tid))
        for key in (*keys, None):
            starts, indices = timelines.get(key, ([], []))
            position = bisect_right(starts, timestamp) - 1
            if position >= 0 and timestamp + duration <= calls[indices[position]]["end_ns"]:
                return indices[position]
        return None

    with (directory.parent / "cluster-accept.log").open() as stream:
        for line in stream:
            if "[STRACE]" in line:
                _, _, fields = line.partition("[STRACE]")
                span = dict(part.split("=", 1) for part in fields.split() if "=" in part)
                name = span.get("name")
                if name not in ("chip.run", "chip.run.runner_run.device_wall"):
                    continue
                identity = tuple(span.get(key) for key in ("pid", "tid", "inv", "hid"))
                if name == "chip.run.runner_run.device_wall":
                    pending[identity] = (int(span["dur"]), int(span["dev_id"]) if "dev_id" in span else None)
                else:
                    host_ns = int(span["dur"])
                    device_ns, device = pending.pop(identity, (None, None))
                    tid = int(span["tid"]) if "tid" in span else None
                    index = containing(int(span["ts"]), host_ns, device=device, tid=tid)
                    if index is not None:
                        traces[index]["host_run_ns"] += host_ns
                        traces[index]["host_runs"] += 1
                        if device_ns is not None:
                            traces[index]["device_ns"] += device_ns
                            traces[index]["device_runs"] += 1
            elif "Callable image: device=" in line and "[mono_ns=" in line:
                timestamp = int(line.split("[mono_ns=", 1)[1].split("]", 1)[0])
                device = int(line.split("Callable image: device=", 1)[1].split()[0])
                index = containing(timestamp, device=device)
                if index is not None:
                    upload = line.split("upload_bytes=", 1)[1].split()[0]
                    traces[index]["callable_upload_bytes"] += int(upload)
                    traces[index]["callable_uploads"] += 1

    groups = defaultdict(Counter)
    for call, trace in zip(calls, traces):
        item = groups[call["update"], call["operator"], call["rank"]]
        item.update(call["counters"])
        item.update(trace)
        item["calls"] += 1
        item["first_dispatches"] += call["first"]
        item["calls_with_device_span"] += bool(trace["device_runs"])
        item["dispatch_ns"] += call["end_ns"] - call["start_ns"]

    phases = {}
    phase_by_request = {}
    for name in PROFILE_ARMS:
        path = directory / f"online-profile-{name}.json"
        if path.is_file():
            for row in json.loads(path.read_text())["timing"]:
                phase_by_request[row["request_id"]] = row["phase"]
    _, receipts = read_receipts(directory)
    dispatches = defaultdict(list)
    native = defaultdict(list)
    for row in receipts:
        if row["phase"] not in PHASES:
            continue
        for owner in row["owners"]:
            phase = phase_by_request.get(owner["request_id"])
            if phase is not None:
                key = phase, row["phase"]
                dispatches[key].append(row["seconds"])
                native[key].append(row["timings"]["chunk_seconds"])
    for (phase, operation), values in sorted(dispatches.items()):
        phases.setdefault(phase, {})[operation] = dict(
            calls=len(values), wall_seconds=sum(values), wall_median_seconds=median(values),
            native_chunk_seconds=sum(native[phase, operation]),
        )

    return dict(
        calls=len(calls), calls_with_device_span=sum(bool(trace["device_runs"]) for trace in traces),
        operators=[dict(update=update, operator=operator, rank=rank, **counts)
                   for (update, operator, rank), counts in sorted(groups.items())],
        serving_phases=phases,
    )


def online_kernel_summary(report):
    last_update = max((row["update"] for row in report["operators"]), default=None)
    by_operator = defaultdict(Counter)
    for row in report["operators"]:
        if row["update"] == last_update:
            by_operator[row["operator"]].update({
                key: row.get(key, 0) for key in ("dispatch_ns", "device_ns", "device_runs", "callable_upload_bytes")
            })
    top = sorted(by_operator.items(), key=lambda entry: entry[1]["dispatch_ns"], reverse=True)[:5]
    return dict(
        stage="online_kernel_summary", calls=report["calls"], calls_with_device_span=report["calls_with_device_span"],
        update=last_update, top_operators=[
            dict(operator=name, dispatch_ms=round(row["dispatch_ns"] / 1e6, 3),
                 device_ms=round(row["device_ns"] / 1e6, 3) if row["device_runs"] else None,
                 callable_upload_bytes=row["callable_upload_bytes"])
            for name, row in top
        ],
    )


def summarize(directory):
    timings = defaultdict(list)
    native_timings = defaultdict(list)
    totals = {}
    _, rows = read_receipts(directory)
    for row in rows:
        totals.update({key: row["runtime"][key] for key in COUNTERS if key in row.get("runtime", {})})
        phase = row["phase"]
        if phase not in PHASES:
            continue
        modes = {owner["request_id"].split("-", 1)[0] for owner in row["owners"]}
        if len(modes) != 1 or not modes.issubset(("target", "orthrus")):
            raise ValueError(f"Unexpected request ownership for {phase}: {sorted(modes)}")
        key = modes.pop(), phase
        timings[key].append(row["seconds"])
        native_timings[key].append(row["timings"]["chunk_seconds"])

    def sample(values):
        return dict(
            calls=len(values),
            first_ms=round(values[0] * 1000, 3),
            subsequent_median_ms=round(median(values[1:]) * 1000, 3) if len(values) > 1 else None,
        )

    dispatches = {mode: {} for mode in ("target", "orthrus")}
    for mode, phase in timings:
        dispatches[mode][phase] = dict(
            dispatch=sample(timings[mode, phase]),
            native_chunk=sample(native_timings[mode, phase]),
        )
    path = directory / "acceptance.json"
    status = json.loads(path.read_text()) if path.is_file() else {}
    return dict(
        passed=status.get("passed"), node=status.get("node"), error=status.get("error"),
        divergence=status.get("divergence"),
        dispatches=dispatches, node_local_counters=totals,
    )


def verifier_logits(directory):
    """Compare the saved native rows after proving both saw the same causal prefix."""
    import torch

    session, receipts = read_receipts(directory)
    failure = json.loads((directory / "acceptance.json").read_text())["divergence"]
    if failure["phase"] != "verify/accept" or failure["generated_index"] < 1:
        raise ValueError("The report does not contain a verifier divergence")
    index, offset = failure["request"], failure["generated_index"]
    result = json.loads((directory / "target/result.json").read_text())
    baseline, = (request for request in result["requests"] if request["index"] == index)
    prefix = baseline["prompt_ids"] + baseline["token_ids"][:offset]
    dp = baseline["dp"]
    manifest = json.loads((session / "run-manifest.json").read_text())
    ranks = [row["rank"] for row in manifest["ranks"] if row["dp_rank"] == dp]
    if not ranks:
        raise ValueError(f"This node does not own DP {dp}. Read the report on the other node.")
    rank = min(ranks)
    decode = [row for row in receipts if row["phase"] == "decode"]
    matching = [(step, row) for step, row in enumerate(decode, 1)
                if any(owner["request_id"] == f"target-request-{index}" for owner in row["owners"])]
    step, target_row = matching[offset - 1]
    verify_call = failure["native"]["verify"]["epoch"] + 1

    def read_row(controls, forward):
        path = session / controls / "input-contract.json"
        contract = json.loads(path.read_text())
        request, = contract["DP_engines"][dp]["requests"]
        position = len(prefix) - 1
        source_row = request["position_ids"].index(position)
        if request["history_tokens"] + request["token_ids"][:source_row + 1] != prefix:
            raise ValueError(f"Saved {forward} differs from the target input prefix")
        path = session / forward / "actual-layer39-output" / f"rank-{rank}-fresh-network-logits.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        row = list(map(int, saved["positions"])).index(position)
        trace = session / forward / "verifier-activations.json"
        stages = tensors = None
        if trace.is_file():
            evidence = json.loads(trace.read_text())
            if evidence["position"] == position:
                stages = evidence["stages"]
                tensor_path = trace.parent / evidence.get("tensors", "verifier-tensors.pt")
                if tensor_path.is_file():
                    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        return saved["logits"][row].float(), stages, tensors

    target, target_stages, target_tensors = read_row(
        f"controls-{target_row['native_epoch']}", f"decode/step-{step}")
    verify, verify_stages, verify_tensors = read_row(
        f"controls-verify-{verify_call}", f"verify-{verify_call}")
    if (int(target.argmax()), int(verify.argmax())) != (failure["target_token"], failure["orthrus_token"]):
        raise ValueError("Saved argmax does not match the recorded divergence")
    difference = (target - verify).abs()
    report = dict(
        request=index, generated_index=offset, prefix_tokens=len(prefix),
        max_abs=float(difference.max()), mean_abs=float(difference.mean()),
        target_top5=dict(zip(target.topk(5).indices.tolist(), target.topk(5).values.tolist(), strict=True)),
        verify_top5=dict(zip(verify.topk(5).indices.tolist(), verify.topk(5).values.tolist(), strict=True)),
    )
    if target_stages is not None and verify_stages is not None:
        def identity(row):
            return row["stage"], row["rank"], row["case"]

        verify_by_key = {identity(row): row for row in verify_stages}
        first = None
        compared = 0
        for left in target_stages:
            key = identity(left)
            right = verify_by_key.get(key)
            if right is None:
                first = dict(stage=left["stage"], rank=left["rank"], case=left["case"],
                             reason="missing_from_verify")
                break
            compared += 1
            if left["sha256"] == right["sha256"]:
                continue
            first = dict(
                stage=left["stage"], rank=left["rank"], case=left["case"],
                layer=left.get("layer"), role=left.get("role"),
                target=dict(shape=left["shape"], dtype=left["dtype"],
                            max_abs=left["max_abs"], mean=left["mean"]),
                verify=dict(shape=right["shape"], dtype=right["dtype"],
                            max_abs=right["max_abs"], mean=right["mean"]),
            )
            tensor_key = f"{left['stage']}/rank-{left['rank']}/case-{left['case']}"
            if target_tensors is not None and verify_tensors is not None:
                a, b = target_tensors[tensor_key], verify_tensors[tensor_key]
                if a.shape == b.shape and a.dtype == b.dtype:
                    delta = (a.float() - b.float()).abs()
                    flat = delta.flatten()
                    count = min(5, flat.numel())
                    values, indices = flat.topk(count)
                    first["difference"] = dict(
                        byte_exact=bool(torch.equal(a, b)),
                        mismatched=int(torch.count_nonzero(a != b)),
                        elements=a.numel(),
                        max_abs=float(delta.max()) if delta.numel() else 0.0,
                        mean_abs=float(delta.mean()) if delta.numel() else 0.0,
                        largest_flat_indices=indices.tolist(),
                        largest_abs=values.tolist(),
                    )
            break
        extra = sorted(set(verify_by_key) - {identity(row) for row in target_stages})
        report["activation_trace"] = dict(
            target_stages=len(target_stages), verify_stages=len(verify_stages),
            compared_before_difference=compared, extra_verify_stages=len(extra),
            first_difference=first,
        )
    return report


def failure_diagnosis(directory):
    """Return the most specific diagnosis already captured by one acceptance run."""
    status = json.loads((directory / "acceptance.json").read_text())
    failure = status.get("divergence")
    if not failure:
        return dict(passed=status.get("passed"), node=status.get("node"), error=status.get("error"))
    phase = failure.get("phase")
    if phase == "verify/accept":
        report = verifier_logits(directory)
        report["phase"] = phase
        return report
    native = failure.get("native", {})
    report = dict(
        phase=phase,
        request=failure.get("request"),
        generated_index=failure.get("generated_index"),
        target_token=failure.get("target_token"),
        orthrus_token=failure.get("orthrus_token"),
    )
    if phase == "prefill":
        report.update(
            trace_position=native.get("trace_position"),
            target_topk=native.get("target"),
            orthrus_topk=native.get("orthrus"),
            first_activation_difference=native.get("first_activation_difference"),
            trace_error=failure.get("native_trace_error"),
        )
    else:
        report["native"] = native
    return report


def _saved_path(directory, value):
    """Map an in-container evidence path back into a mounted acceptance directory."""
    path = Path(value)
    if path.is_file():
        return path
    for marker in ("native", "target-native", "orthrus-native"):
        if marker in path.parts:
            candidate = directory.joinpath(*path.parts[path.parts.index(marker):])
            if candidate.is_file():
                return candidate
    raise ValueError(f"Cannot locate saved evidence {value} below {directory}")


def prefill_reduction(directory):
    """Reconstruct the first differing TP Attention reduction from saved rows."""
    import torch

    status = json.loads((directory / "acceptance.json").read_text())
    failure = status.get("divergence", {})
    first = failure.get("native", {}).get("first_activation_difference", {})
    target_stage = first.get("target") or {}
    if failure.get("phase") != "prefill" or target_stage.get("stage", "").split("/")[-1] != "attention-scattered":
        raise ValueError("The saved failure is not a prefill Attention reduction divergence")
    layer, rank = target_stage["layer"], target_stage["rank"]
    session, rows = read_receipts(directory)
    traces = {}
    request = failure["request"]
    for mode in ("target", "orthrus"):
        request_id = f"{mode}-request-{request}"
        matches = [row for row in rows if row["phase"] == "prefill"
                   and any(owner["request_id"] == request_id for owner in row["owners"])]
        if not matches:
            raise ValueError(f"No saved prefill receipt for {request_id}")
        evidence_path = _saved_path(directory, matches[-1]["prefill_activation_trace"])
        evidence = json.loads(evidence_path.read_text())
        traces[mode] = torch.load(evidence_path.parent / evidence["tensors"], map_location="cpu",
                                  weights_only=True)
    manifest = json.loads((session / "run-manifest.json").read_text())
    topology = manifest.get("topology", {})
    tp = topology.get("tp", 4)
    ranks = list(range(rank // tp * tp, rank // tp * tp + tp))

    def key(name, member):
        return f"layer-{layer}/{name}/rank-{member}/case-fresh"

    contributors = {
        mode: [traces[mode][key("wo-b", member)].float() for member in ranks]
        for mode in traces
    }
    reference = {mode: torch.stack(values).sum(0).to(torch.bfloat16)
                 for mode, values in contributors.items()}
    outputs = {mode: traces[mode][key("attention-scattered", rank)] for mode in traces}

    def difference(left, right):
        delta = (left.float() - right.float()).abs()
        values, indices = delta.flatten().topk(min(5, delta.numel()))
        return dict(
            byte_exact=bool(torch.equal(left, right)),
            mismatched=int(torch.count_nonzero(left != right)),
            elements=left.numel(),
            max_abs=float(delta.max()) if delta.numel() else 0.0,
            mean_abs=float(delta.mean()) if delta.numel() else 0.0,
            largest_flat_indices=indices.tolist(),
            largest_abs=values.tolist(),
        )

    contributor_differences = {
        str(member): difference(contributors["target"][index], contributors["orthrus"][index])
        for index, member in enumerate(ranks)
    }
    report = dict(
        phase="prefill", layer=layer, rank=rank, tp_ranks=ranks,
        contributor_differences=contributor_differences,
        fixed_order_reference=difference(reference["target"], reference["orthrus"]),
        observed_output=difference(outputs["target"], outputs["orthrus"]),
        target_vs_fixed_order=difference(outputs["target"], reference["target"]),
        orthrus_vs_fixed_order=difference(outputs["orthrus"], reference["orthrus"]),
    )
    if all(item["byte_exact"] for item in contributor_differences.values()):
        report["conclusion"] = (
            "All saved Wo-B contributors are byte-identical; the divergence is introduced by "
            "reduce_pack or the TP BF16 collective, not Attention, weights, KV, or Orthrus."
        )
    else:
        report["conclusion"] = "At least one TP Wo-B contributor already differs."
    return report


def _gsm8k_serving_phases(directory):
    """Recover phase deltas from the original receipts, including older runs."""
    _, receipts = read_receipts(directory)
    arms = {arm: defaultdict(Counter) for arm in ("ar", "initial", "online", "frozen")}
    counter_names = ("output_d2h_bytes", "verify_backup_d2d_bytes", "verify_restore_d2d_bytes")
    previous = Counter()
    for row in receipts:
        runtime = row.get("runtime") or {}
        deltas = {name: runtime[name] - previous[name] for name in counter_names if name in runtime}
        for name, delta in deltas.items():
            previous[name] += delta
        owners = [row["owner"]] if row["phase"] == "release" else row["owners"]
        labels = {item["request_id"].split("-", 3)[2] for item in owners}
        if len(labels) != 1:
            continue
        name, = labels
        if name not in arms:
            continue
        phase = arms[name][row["phase"]]
        phase.update(deltas)
        phase["calls"] += 1
        vocabulary = row.get("vocabulary_layout")
        if vocabulary is not None:
            phase[f"{vocabulary}_calls"] += 1
        for name in ("teacher_logit_rows", "student_logit_rows"):
            if row.get(name) is not None:
                phase[name] += row[name]
        timings = row.get("timings", {})
        phase["native_seconds"] += timings.get("chunk_seconds", 0.0)
        capacity = row.get("execution_capacity")
        if capacity is not None:
            phase[f"capacity{capacity}_calls"] += 1
            phase[f"capacity{capacity}_native_seconds"] += timings.get("chunk_seconds", 0.0)
        commit_tokens = row.get("commit_tokens_per_dp")
        if commit_tokens:
            largest = max(commit_tokens)
            phase["max_commit_tokens_per_dp"] = max(phase["max_commit_tokens_per_dp"], largest)
            phase["accept_calls_fitting_capacity16"] += int(largest <= 16)
        for key in ("verify_snapshot_seconds", "verify_restore_seconds"):
            if key in timings:
                phase[key] += timings[key]
    return arms


def compare_gsm8k_reports(baseline, candidate):
    """Compare completed identical workloads and recover local timings on older builds."""
    def load(path, filename):
        return json.loads((path / filename).read_text())

    before, after = (load(path, "acceptance.json") for path in (baseline, candidate))
    fields = ("node", "topology", "physical_to_runtime", "checkpoint", "head_sha256")
    if not before["passed"] or not after["passed"] or not all(before[key] == after[key] for key in fields):
        raise ValueError("GSM8K comparison requires successful runs on the same host, cards, topology and weights")
    if before["peer"]["physical_to_runtime"] != after["peer"]["physical_to_runtime"]:
        raise ValueError("GSM8K comparison requires the same peer cards")
    left, right = (load(path, "gsm8k-online-report.json") for path in (baseline, candidate))
    fields = ("training_requests", "heldout_requests", "max_concurrent_requests", "max_new_tokens", "updates_policy")
    if not left["complete"] or not right["complete"] or not all(left[key] == right[key] for key in fields):
        raise ValueError("GSM8K comparison requires the same completed workload and update policy")

    receipts = [_gsm8k_serving_phases(path) for path in (baseline, candidate)]
    comparisons = {}
    for name in ("ar", "initial", "online", "frozen"):
        old, new = (load(path, f"gsm8k-online-{name}.json") for path in (baseline, candidate))
        if len(old["requests"]) != len(new["requests"]):
            raise ValueError(f"{name} has a different number of requests")
        for a, b in zip(old["requests"], new["requests"]):
            if (a["workload_id"], a["prompt_ids"], a["dp"]) != (b["workload_id"], b["prompt_ids"], b["dp"]):
                raise ValueError(f"{name} used different prompts or DP assignments")
            check_generated_prefix(b["index"], a["token_ids"], b["token_ids"], phase=name)
            if len(a["token_ids"]) != len(b["token_ids"]):
                raise ValueError(f"{name} did not finish with the same token counts")

        a, b = left["arms"][name], right["arms"][name]
        stats = {
            key: [a[key], b[key]]
            for key in ("wall_tok_s", "decode_tok_s", "accepted_per_round", "acceptance_rate")
        }
        if name == "online":
            for key in ("updates", "publications", "update_seconds", "publication_seconds",
                        "first_update_and_publish_seconds", "wall_tok_s_excluding_all_updates_estimate"):
                stats[key] = [a[key], b[key]]
        stats["wall_throughput_ratio"] = b["wall_tok_s"] / a["wall_tok_s"]
        stats["local_serving"] = [rows[name] for rows in receipts]
        comparisons[name] = stats
    return dict(node=after["node"], build_ids=[before["build_id"], after["build_id"]],
                compared_greedy=True, order=["baseline", "candidate"], arms=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="The acceptance directory printed by cluster-accept")
    parser.add_argument("--verifier-logits", action="store_true",
                        help="Compare the saved target and verification logits on the request's host")
    parser.add_argument("--failure-diagnosis", action="store_true",
                        help="Diagnose whichever acceptance phase failed using evidence from the same run")
    parser.add_argument("--prefill-reduction", action="store_true",
                        help="Reconstruct the first differing TP Attention reduction from saved tensors")
    parser.add_argument("--online-kernels", action="store_true",
                        help="Attribute online operator dispatches to the saved runtime trace")
    parser.add_argument("--online-profile", action="store_true",
                        help="Recover the saved greedy probes and update timings without running the model")
    parser.add_argument("--compare-gsm8k", type=Path, metavar="BASELINE",
                        help="Compare this completed GSM8K report with the baseline acceptance directory")
    parser.add_argument("--output", type=Path, help="Save the full online-kernel or GSM8K comparison report")
    args = parser.parse_args()
    if sum((args.verifier_logits, args.failure_diagnosis, args.prefill_reduction,
            args.online_kernels, args.online_profile, args.compare_gsm8k is not None)) > 1:
        parser.error("Choose only one diagnostic mode")
    if args.output is not None and not (args.online_kernels or args.compare_gsm8k is not None):
        parser.error("--output requires --online-kernels or --compare-gsm8k")
    try:
        if args.prefill_reduction:
            summary = prefill_reduction(args.directory)
        elif args.failure_diagnosis:
            summary = failure_diagnosis(args.directory)
        elif args.verifier_logits:
            summary = verifier_logits(args.directory)
        elif args.online_profile:
            summary = recover_online_profile(args.directory)
        elif args.compare_gsm8k is not None:
            summary = compare_gsm8k_reports(args.compare_gsm8k, args.directory)
            if args.output is not None:
                args.output.write_text(json.dumps(summary, indent=2) + "\n")
        elif args.online_kernels:
            summary = online_kernels(args.directory)
            if args.output is not None:
                args.output.write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(online_kernel_summary(summary)))
                return
        else:
            summary = summarize(args.directory)
    except (OSError, ImportError, IndexError, KeyError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    compact = args.verifier_logits or args.failure_diagnosis or args.prefill_reduction or args.online_profile
    print(json.dumps(summary, indent=None if compact else 2))


if __name__ == "__main__":
    main()
