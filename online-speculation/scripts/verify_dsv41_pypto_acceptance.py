#!/usr/bin/env python3
"""Check greedy parity and prove that the paired A3 speculative path ran."""

import argparse
from collections import Counter
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def request_parity(target, draft):
    if target["status"] != "executed" or draft["status"] != "executed":
        raise ValueError("Both offline runs must finish successfully")
    if target["sampling"] != draft["sampling"] or target["sampling"]["temperature"] != 0:
        raise ValueError("Both offline runs must use the same greedy sampling settings")
    if target["devices"] != draft["devices"]:
        raise ValueError("Offline runs used different device mappings")
    if not target["requests"] or len(target["requests"]) != len(draft["requests"]):
        raise ValueError("Both offline runs must contain the same nonempty request set")
    tokens = 0
    for a, b in zip(target["requests"], draft["requests"], strict=True):
        for field in ("index", "prompt_ids", "finish_reason", "usage"):
            if a[field] != b[field]:
                raise ValueError(f"Request {a['index']} differs in {field}")
        if a["token_ids"] != b["token_ids"]:
            offset = next((i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y),
                          min(len(a["token_ids"]), len(b["token_ids"])))
            raise ValueError(f"Request {a['index']} diverged at generated token index {offset}")
        tokens += len(a["token_ids"])
    if tokens == 0:
        raise ValueError("No generated tokens were compared")
    return tokens


def audit_session(directory, *, speculative):
    receipts = list(directory.rglob("serving-dispatches.jsonl"))
    if len(receipts) != 1:
        raise ValueError(f"Expected exactly one native session receipt in {directory}, found {len(receipts)}")
    session = receipts[0].parent
    manifest = read_json(session / "run-manifest.json")
    cleanup = read_json(session / "cleanup.json")
    if bool(manifest["orthrus_head"]) != speculative:
        raise ValueError(f"Unexpected Orthrus head configuration in {session}")
    if not cleanup["closed"] or not cleanup["runtime_closed"] or cleanup["errors"]:
        raise ValueError(f"Native runtime cleanup failed in {session}")

    phases = Counter()
    sequence = ("draft", "verify", "accept")
    owners, epoch, position = None, None, 0
    backup = restored = 0
    with receipts[0].open() as records:
        for line in records:
            row = json.loads(line)
            phase = row["phase"]
            phases[phase] += 1
            if phase not in sequence and position:
                raise ValueError(f"Incomplete speculative round before {phase}")
            if phase not in sequence:
                continue
            if not speculative or sequence[position] != phase:
                raise ValueError(f"Unexpected {phase} dispatch in {session}")
            if row["completed_layers"] != list(range(40)) or not row["TP_logits_equal"]:
                raise ValueError(f"Incomplete layers or inconsistent TP logits in {phase}")
            group = sorted((item["dp"], item["slot"], item["generation"]) for item in row["owners"])
            if not group or len(group) != len(set(group)):
                raise ValueError(f"Missing or duplicated {phase} owners")
            if phase == "draft":
                owners, epoch = group, row["native_epoch"]
            elif group != owners or row["native_epoch"] != epoch + int(phase == "accept"):
                raise ValueError(f"{phase} changed request ownership or advanced the wrong epoch")
            if phase == "verify":
                runtime = row["runtime"]
                next_backup = runtime["verify_backup_d2d_bytes"]
                restored = runtime["verify_restore_d2d_bytes"]
                if next_backup <= backup or next_backup != restored:
                    raise ValueError(f"Verifier snapshot and restore disagree in {session}")
                backup = next_backup
            position = (position + 1) % len(sequence)
    if position or not phases["prefill"]:
        raise ValueError(f"Missing prefill or incomplete speculative round in {session}")
    if speculative and not phases["accept"]:
        raise ValueError("No Orthrus round executed. Greedy parity alone cannot validate the speculative path.")
    if not speculative and not phases["decode"]:
        raise ValueError("Baseline did not execute a target decode step")
    return manifest["checkpoint"], dict(phases), restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Directory with target, orthrus and their -native subdirectories")
    args = parser.parse_args()
    try:
        target, draft = (read_json(args.results / mode / "result.json") for mode in ("target", "orthrus"))
        tokens = request_parity(target, draft)
        a, target_phases, _ = audit_session(args.results / "target-native", speculative=False)
        b, draft_phases, restored = audit_session(args.results / "orthrus-native", speculative=True)
        if a != b:
            raise ValueError("Native sessions used different checkpoint identities")
    except (OSError, KeyError, ValueError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        raise SystemExit(1) from None
    print(json.dumps(dict(passed=True, compared_tokens=tokens, target_phases=target_phases,
                          orthrus_phases=draft_phases, verifier_restored_bytes=restored)), flush=True)


if __name__ == "__main__":
    main()
