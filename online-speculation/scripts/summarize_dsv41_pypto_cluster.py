#!/usr/bin/env python3
"""Read existing DSV4.1 cluster receipts and summarize dispatch cost on one node."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


PHASES = ("prefill", "decode", "draft", "verify", "accept")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="The acceptance directory printed by cluster-accept")
    parser.add_argument("--verifier-logits", action="store_true",
                        help="Compare the saved target and verification logits on the request's host")
    parser.add_argument("--failure-diagnosis", action="store_true",
                        help="Diagnose whichever acceptance phase failed using evidence from the same run")
    parser.add_argument("--prefill-reduction", action="store_true",
                        help="Reconstruct the first differing TP Attention reduction from saved tensors")
    args = parser.parse_args()
    if sum((args.verifier_logits, args.failure_diagnosis, args.prefill_reduction)) > 1:
        parser.error("Choose only one diagnostic mode")
    try:
        if args.prefill_reduction:
            summary = prefill_reduction(args.directory)
        elif args.failure_diagnosis:
            summary = failure_diagnosis(args.directory)
        elif args.verifier_logits:
            summary = verifier_logits(args.directory)
        else:
            summary = summarize(args.directory)
    except (OSError, ImportError, IndexError, KeyError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    compact = args.verifier_logits or args.failure_diagnosis or args.prefill_reduction
    print(json.dumps(summary, indent=None if compact else 2))


if __name__ == "__main__":
    main()
