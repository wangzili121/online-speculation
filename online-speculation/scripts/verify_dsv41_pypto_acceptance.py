#!/usr/bin/env python3
"""Check greedy parity and prove that the paired A3 speculative path ran."""

import argparse
from collections import Counter
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


class TokenDivergence(ValueError):
    def __init__(self, index, offset, phase, expected, actual):
        super().__init__(f"Request {index} diverged at generated token index {offset} during {phase} "
                         f"(target={expected}, orthrus={actual})")
        self.details = dict(request=index, generated_index=offset, phase=phase,
                            target_token=expected, orthrus_token=actual)


def check_generated_prefix(index, expected, actual, *, phase):
    """Fail at the first available token difference, including an excess token."""
    for offset, observed in enumerate(actual):
        reference = expected[offset] if offset < len(expected) else None
        if observed != reference:
            raise TokenDivergence(index, offset, phase, reference, observed)


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
        fields = ("index", "prompt_ids", "finish_reason", "usage")
        if "workload_id" in a or "workload_id" in b:
            fields += ("workload_id",)
        for field in fields:
            if a[field] != b[field]:
                raise ValueError(f"Request {a['index']} differs in {field}")
        check_generated_prefix(a["index"], a["token_ids"], b["token_ids"], phase="final comparison")
        if len(b["token_ids"]) < len(a["token_ids"]):
            offset = len(b["token_ids"])
            raise TokenDivergence(a["index"], offset, "final comparison", a["token_ids"][offset], None)
        tokens += len(a["token_ids"])
    if tokens == 0:
        raise ValueError("No generated tokens were compared")
    return tokens


def audit_online_update(local, peer, *, publish_every, require_update):
    """Validate the completed optimizer and publication steps on both hosts."""
    counters = ("updates", "publications")
    if not local.get("bound") or not peer.get("bound"):
        raise ValueError("Both hosts must bind the online learner")
    if any(local[key] != peer[key] for key in counters):
        raise ValueError("Hosts disagree on online update or publication counts")
    if require_update and not local["updates"]:
        raise ValueError("Online capture completed without applying an update")
    if local["publications"] != local["updates"] // publish_every:
        raise ValueError("Online draft publication did not follow the configured cadence")


def audit_session(directory, *, speculative, node_local=False, request_prefix=None):
    receipts = list(directory.rglob("serving-dispatches.jsonl"))
    if len(receipts) != 1:
        raise ValueError(f"Expected exactly one native session receipt in {directory}, found {len(receipts)}")
    session = receipts[0].parent
    manifest = read_json(session / "run-manifest.json")
    cleanup = read_json(session / "cleanup.json")
    if (request_prefix is None and bool(manifest["orthrus_head"]) != speculative
            or request_prefix is not None and not manifest.get("deferred_draft")):
        raise ValueError(f"Unexpected Orthrus head configuration in {session}")
    if not cleanup["closed"] or not cleanup["runtime_closed"] or cleanup["errors"]:
        raise ValueError(f"Native runtime cleanup failed in {session}")
    local_dps = {rank["dp_rank"] for rank in manifest["ranks"]} if node_local else None

    phases = Counter()
    sequence = ("draft", "verify", "accept")
    owners, epoch, position = None, None, 0
    backup = restored = 0
    with receipts[0].open() as records:
        for line in records:
            row = json.loads(line)
            phase = row["phase"]
            if request_prefix is not None:
                owners_on_row = [row["owner"]] if phase == "release" else row["owners"]
                belongs = [item["request_id"].startswith(request_prefix) for item in owners_on_row]
                if not any(belongs):
                    continue
                if not all(belongs) or (not speculative and row["draft_resident"]):
                    raise ValueError("Mismatched request or resident draft during target-only acceptance")
                if phase in sequence and not row["draft_resident"]:
                    raise ValueError("Speculation ran without the resident draft head")
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
                owns_request = local_dps is None or any(dp in local_dps for dp, _, _ in group)
                moved_expected_bytes = next_backup > backup if owns_request else next_backup == backup
                if not moved_expected_bytes or next_backup != restored:
                    raise ValueError(f"Verifier snapshot and restore disagree in {session}")
                backup = next_backup
            position = (position + 1) % len(sequence)
    if position or not phases["prefill"]:
        raise ValueError(f"Missing prefill or incomplete speculative round in {session}")
    if speculative and not phases["accept"]:
        raise ValueError("No Orthrus round executed. Greedy parity alone cannot validate the speculative path.")
    if not speculative and not phases["decode"]:
        raise ValueError("Baseline did not execute a target decode step")
    if node_local and not phases["release"]:
        raise ValueError(f"Missing request release in {session}")
    return manifest["checkpoint"], dict(phases), restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Directory with target, orthrus and their -native subdirectories")
    args = parser.parse_args()
    try:
        target, draft = (read_json(args.results / mode / "result.json") for mode in ("target", "orthrus"))
        tokens = request_parity(target, draft)
        shared = args.results / "native"
        if shared.is_dir():
            a, target_phases, _ = audit_session(shared, speculative=False, node_local=True,
                                                 request_prefix="target-request-")
            b, draft_phases, restored = audit_session(shared, speculative=True, node_local=True,
                                                       request_prefix="orthrus-request-")
        else:
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
