#!/usr/bin/env python3
"""同 worker、head 和历史下，对照原生与 Vector GDN 的完整 K32 起草。"""

import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import shutil
import statistics
import time

from compare_qwen38_pypto_draft import tensor_digest
from validate_qwen38_pypto_neural import resident_checksum, history_checksum, audit_round, audit_verifier, audit_slot_reuse
from summarize_qwen38_pypto_profile import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "head-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--draft-group-size", type=int, choices=(1, 4), default=4)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--profile-gdn", type=int, default=0, help="separate first-GDN-layer profiles, group-size=1 only")
    parser.add_argument("--audit-boundaries", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() or args.device < 0 or args.rounds < 2 or args.profile_gdn < 0:
        parser.error("require new output, assigned device and 2+ rounds")
    if args.draft_group_size == 4 and (args.capture or args.profile_gdn):
        parser.error("per-layer capture/profile requires group-size=1")
    args.output_dir.mkdir(parents=True)
    logging.getLogger("simpler").setLevel(logging.WARNING)
    import torch
    import pypto
    from blockspec.pypto_qwen38.baseline import native_module, runtime_identity
    from blockspec.pypto_qwen38.draft import make_engine, metadata
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.inspection import capture_layers, save_history, head_digest
    from blockspec.pypto_qwen38.programs import make_programs
    from blockspec.pypto_qwen38.target import NativeTarget
    from blockspec.pypto_qwen38.greedy import verify_transaction

    report = dict(passed=False, scope="same_worker_real_27b_gdn_comparison", rounds=[],
                  runtime=runtime_identity(pypto), group_size=args.draft_group_size, profiles={})
    engine = request = None
    try:
        torch.set_num_threads(4)
        head = HeadSource(args.model, args.head_dir)
        engine = make_engine(args.source, args.model, args.device, head, attention_backend="paged",
                             group_size=args.draft_group_size, head_transfer="packed")
        draft = engine.runner.draft
        key = "gdn" if args.draft_group_size == 1 else "group"
        original_compile, programs = draft.compile, {}

        def compile_both(samples, config):
            original = original_compile(samples, config)
            programs["native"] = draft.programs[key]
            program = make_programs(args.source, attention_backend="paged", group_size=args.draft_group_size,
                                    gdn_backend="vector")[key]
            programs["vector"] = program.compile(*metadata(head, samples)[key], config=config)
            return [*original, programs["vector"]]

        draft.compile = compile_both
        start = time.monotonic()
        engine.start()
        native, schedule = engine.runner.native, native_module(args.source, "schedule")
        report.update(startup_seconds=time.monotonic() - start, head_sha256=head_digest(head),
                      head_manifest=json.loads((args.head_dir / "manifest.json").read_text()), head_storage=draft.storage)
        request = schedule.RequestState(engine.tokenizer.encode("Hello", add_special_tokens=False), 40, frozenset())
        native.admit(request)
        for position in range(0, len(request.prompt_ids), 16):
            tokens = tuple(request.prompt_ids[position:position + 16])
            anchor = native.run_batch([schedule.BatchSlot(request, tokens, position, "prefill",
                                      position + len(tokens) == len(request.prompt_ids))])[0]
        request.generated.append(anchor)
        request.prefill_offset = len(request.prompt_ids)
        target, expected = NativeTarget(native, request, schedule.BatchSlot), []
        snapshot = target.snapshot()
        token = anchor
        for _ in range(35):
            token = target.forward([token])[0]
            expected.append(token)
        target.restore(snapshot)
        before = resident_checksum(native)
        report["initial_history_sha256"] = history_checksum(native, request)
        if args.capture:
            save_history(native, request, args.output_dir / "capture")

        def propose(backend):
            draft.programs[key], draft.gdn_backend = programs[backend], backend
            return draft.propose(request, anchor, 32)

        for backend in programs:
            propose(backend)
        signatures, logits = {}, {}
        for step in range(args.rounds):
            for backend in (list(programs) if step % 2 == 0 else list(reversed(programs))):
                candidates = propose(backend)
                signature = dict(candidates=candidates, logits_sha256=tensor_digest(draft.host_logits[:31]))
                if backend in signatures and signature != signatures[backend]:
                    raise AssertionError("repeated GDN draft was not deterministic")
                signatures[backend], logits[backend] = signature, draft.host_logits[:31].float().clone()
                row = dict(backend=backend, round=step, seconds=draft.last_seconds, phases=dict(draft.last_timings))
                report["rounds"].append(row)
                print(json.dumps(row), flush=True)
        report["signatures"] = signatures
        error = logits["vector"] - logits["native"]
        report["logits_comparison"] = dict(max_abs=float(error.abs().max()),
            relative_l2=float(error.norm() / logits["native"].norm().clamp_min(1e-12)),
            argmax_agreement=int((logits["vector"].argmax(-1) == logits["native"].argmax(-1)).sum()), rows=31)
        if args.capture:
            for backend in programs:
                draft.programs[key], draft.gdn_backend = programs[backend], backend
                with capture_layers(draft, args.output_dir / "capture" / backend):
                    propose(backend)
        for backend in programs:
            profiles = []
            for sample in range(args.profile_gdn):
                original_run = draft._run

                def collect(name, arguments, original_run=original_run, backend=backend, sample=sample, profiles=profiles):
                    if name != "gdn" or arguments[-2] != 0:
                        return original_run(name, arguments)
                    config = native.config
                    native.config = replace(config, enable_chip_swimlane=1)
                    try:
                        result = original_run(name, arguments)
                    finally:
                        native.config = config
                    paths = list(Path(programs[backend].output_dir).rglob("merged_swimlane*.json"))
                    path = max(paths, key=lambda p: p.stat().st_mtime_ns)
                    data = path.read_bytes()
                    saved = args.output_dir / f"layer0-{backend}-{sample}.json"
                    shutil.copyfile(path, saved)
                    profiles.append(dict(path=str(saved), sha256=hashlib.sha256(data).hexdigest(),
                                         **summarize(json.loads(data))))
                    return result

                draft._run = collect
                try:
                    propose(backend)
                finally:
                    draft._run = original_run
                if tensor_digest(draft.host_logits[:31]) != signatures[backend]["logits_sha256"]:
                    raise AssertionError("profiling changed draft logits")
            report["profiles"][backend] = profiles
        report["target_cache_unchanged"] = before == resident_checksum(native)
        if not report["target_cache_unchanged"]:
            raise AssertionError("GDN changed target history")
        report["medians"] = {b: statistics.median(r["seconds"] for r in report["rounds"] if r["backend"] == b) for b in programs}
        for signature in signatures.values():
            target.restore(snapshot)
            corrected = verify_transaction(target, anchor, signature["candidates"], budget=32)
            if list(corrected.tokens) != expected[:len(corrected.tokens)]:
                raise AssertionError("GDN candidate correction differs from target-only")
            actual = history_checksum(native, request)
            target.restore(snapshot)
            target.forward((anchor, *signature["candidates"])[:corrected.committed_inputs])
            if actual != history_checksum(native, request):
                raise AssertionError("accepted prefix does not match replay")
        report["verification_and_replay_passed"] = True
        report["verifier_cases"] = audit_verifier(target, anchor, expected, snapshot)
        if args.audit_boundaries:
            native.release(request)
            request = None
            draft.programs[key], draft.gdn_backend = programs["vector"], "vector"
            report["boundary_cases"] = []
            for past in (15, 17, 96, 97, 224, 225):
                seed = engine.tokenizer.encode(f"Case {past}. Please explain how computers work. ", add_special_tokens=False)
                ids = (seed * ((past + len(seed) - 1) // len(seed)))[:past]
                audit = dict(past=past, width=32, passed=False)
                report["boundary_cases"].append(audit)
                audit_round(engine, args.source, 32, audit, prompt_ids=ids)
                audit["requests_released"] = not bool(native.requests)
                if not audit["requests_released"]:
                    raise AssertionError("request state leaked after GDN boundary check")
            report["slot_reuse"] = audit_slot_reuse(engine, args.source)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            try:
                if request is not None and engine is not None:
                    engine.runner.native.release(request)
            finally:
                if engine is not None:
                    engine.stop()
        except BaseException as error:
            report.update(passed=False, cleanup_error=repr(error))
            raise
        finally:
            (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
