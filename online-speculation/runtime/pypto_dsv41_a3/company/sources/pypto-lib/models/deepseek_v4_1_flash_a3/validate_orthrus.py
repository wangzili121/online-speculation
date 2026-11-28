# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare multi-row A3 target verification against serial Decode."""

import torch

from . import config as C


def validate_resident_verifier(session, prompt, *, width=8):
    """Compare packed verification against serial Decode in one resident session."""
    def prefill(request_id):
        handle = session.admit(request_id, 0, len(prompt))
        for offset in range(0, len(prompt), session.capacities[0]):
            result = session.advance_batch("prefill", [(handle, prompt[offset:offset + session.capacities[0]])])
        return handle, _greedy(result[handle])

    def decode(handle, token):
        return _greedy(session.advance_batch("decode", [(handle, (token,))])[handle])

    handle, anchor = prefill("serial")
    serial = [anchor]
    for _ in range(width + 1):
        serial.append(decode(handle, serial[-1]))
    session.release(handle)

    reports = []
    for accepted in range(width):
        handle, first = prefill("verify")
        if first != anchor:
            raise ValueError("Prefill changed after recycling the resident request slot")
        candidates = list(serial[1:width])
        if accepted < len(candidates):
            candidates[accepted] = (candidates[accepted] + 1) % C.FLASH.vocab_size
        reply = session.verify_batch([(handle, anchor, candidates)])[handle]
        expected = tuple(serial[1:accepted + 2])
        if reply.tokens != expected or reply.accepted != accepted:
            raise ValueError(f"Verify diverged from serial Decode at acceptance length {accepted}")
        following = decode(handle, reply.tokens[-1])
        if following != serial[accepted + 2]:
            raise ValueError(f"Committed cache diverged from serial Decode after acceptance length {accepted}")
        reports.append(dict(accepted=accepted, emitted=len(reply.tokens), continuation_token=following))
        session.release(handle)

    if len(prompt) + width + 3 <= session.context:
        handle, pending = prefill("chained")
        cursor = 0
        for step, accepted in enumerate((0, 1, 0)):
            kept = serial[cursor + 1:cursor + accepted + 1]
            wrong = (serial[cursor + accepted + 1] + 1) % C.FLASH.vocab_size
            candidates = [*kept, *([wrong] * (width - 1 - accepted))]
            teacher = {handle: None}
            reply = session.verify_batch([(handle, pending, candidates)], teacher=teacher)[handle]
            want = tuple(serial[cursor + 1:cursor + accepted + 2])
            if reply.accepted != accepted or reply.tokens != want:
                raise ValueError(f"Chained verification diverged at step {step}")
            if teacher[handle].shape[0] != len(want) or tuple(teacher[handle].argmax(dim=1).tolist()) != want:
                raise ValueError(f"Teacher rows differ from the committed continuation at step {step}")
            pending = reply.tokens[-1]
            cursor += len(want)
            reports.append(dict(chain_step=step, accepted=accepted, emitted=len(want), teacher_rows=len(want)))
        if decode(handle, pending) != serial[cursor + 1]:
            raise ValueError("Committed cache diverged after consecutive rejected blocks")
        session.release(handle)
    return reports


def _greedy(logits):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite serial reference logits")
    return int(logits.argmax())


def main():
    """Run the verifier gate with previously compiled A3 model artifacts."""
    import argparse
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from tokenizers import Tokenizer

    from .driver import load_manifest
    from .engram import HostEngram
    from .packed import open_packed_session
    from .weights import Checkpoint, ResidentWeights

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="Comma-separated local runtime IDs in rank order")
    parser.add_argument("--width", type=int, choices=(8, 16), default=8)
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")]
    if len(devices) != C.EP_SIZE or len(set(devices)) != C.EP_SIZE or min(devices) < 0:
        parser.error("Expected sixteen distinct local runtime IDs")
    prefill_dir, decode_dir = args.graphs / "prefill", args.graphs / "decode"
    prefill = load_manifest(prefill_dir, mode="prefill")
    decode = load_manifest(decode_dir, mode="decode")
    context = decode["history_capacity"]
    if prefill["history_capacity"] != context or args.width > decode["capacity"]:
        parser.error("Graph manifests must share a context and cover the requested verification width")
    if context < C.BLOCK_SIZE + args.width + 1:
        parser.error("The gate requires context to cross a physical page boundary")
    checkpoint = Checkpoint(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    prompt = tokenizer.encode("We can verify the mathematical result by working through each step. " * 128).ids
    prompt = prompt[:C.BLOCK_SIZE - 1]
    options = SimpleNamespace(
        runtime_dir=prefill_dir, decode_runtime_dir=decode_dir, profile_all_steps=None,
        capacity=prefill["capacity"], decode_capacity=decode["capacity"],
        context_capacity=context, max_new_tokens=args.width + 2,
    )
    memory = SimpleNamespace(phase="prepare", sample=lambda: None)
    report = {}
    engram = HostEngram(checkpoint, tokenizer)
    with open_packed_session(options, devices, ResidentWeights(checkpoint), engram, memory, report,
                             request_slots=1, audit_logits=True) as session:
        cases = validate_resident_verifier(session, prompt, width=args.width)
    print(json.dumps(dict(passed=True, width=args.width, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
