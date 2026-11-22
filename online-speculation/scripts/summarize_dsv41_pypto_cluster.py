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
        stages = None
        if trace.is_file():
            evidence = json.loads(trace.read_text())
            if evidence["position"] == position:
                stages = evidence["stages"]
        return saved["logits"][row].float(), stages

    target, target_stages = read_row(f"controls-{target_row['native_epoch']}", f"decode/step-{step}")
    verify, verify_stages = read_row(f"controls-verify-{verify_call}", f"verify-{verify_call}")
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
        keys = ("hidden_sha256", "premix_sha256")
        report["first_activation_difference"] = next(
            (dict(stage=a["stage"], target_rank=a["rank"], verify_rank=b["rank"],
                  tensors=[name for name in keys if a[name] != b[name]])
             for a, b in zip(target_stages, verify_stages, strict=True)
             if a["stage"] == b["stage"] and any(a[name] != b[name] for name in keys)), None,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="The acceptance directory printed by cluster-accept")
    parser.add_argument("--verifier-logits", action="store_true",
                        help="Compare the saved target and verification logits on the request's host")
    args = parser.parse_args()
    try:
        summary = verifier_logits(args.directory) if args.verifier_logits else summarize(args.directory)
    except (OSError, ImportError, IndexError, KeyError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(summary, indent=None if args.verifier_logits else 2))


if __name__ == "__main__":
    main()
