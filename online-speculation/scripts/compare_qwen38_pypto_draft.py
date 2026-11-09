#!/usr/bin/env python3
"""Compare both Attention backends in one real 27B worker, with identical heads/history."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import statistics
import time


def tensor_digest(value):
    import torch
    return hashlib.sha256(memoryview(value.contiguous().view(torch.uint8).numpy())).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--candidate-backend", choices=("fused", "paged"), default="fused")
    parser.add_argument("--baseline-backend", choices=("native", "paged"), default="native")
    parser.add_argument("--draft-group-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--head-transfer", choices=("individual", "packed"), default="individual")
    parser.add_argument("--audit-boundaries", action="store_true", help="audit real-model page and length-bucket boundaries")
    parser.add_argument("--audit-verifier", action="store_true", help="audit full acceptance, late rejection and output limits with target-derived candidates")
    parser.add_argument("--reload-reference", type=Path, help="require identical head, history and logits after restart")
    args = parser.parse_args()
    if args.output_dir.exists() or args.device < 0 or args.rounds < 2:
        parser.error("require a new output directory, assigned device and 2+ rounds")
    if args.draft_group_size == 4 and (args.candidate_backend != "paged" or args.capture):
        parser.error("grouped comparison requires paged attention without per-layer capture")
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
    from validate_qwen38_pypto_neural import resident_checksum, history_checksum, audit_round, audit_verifier

    candidate_name = args.candidate_backend if args.draft_group_size == 1 else "paged-group4"
    if candidate_name == args.baseline_backend:
        parser.error("baseline and candidate must differ")
    report = dict(passed=False, scope="same_worker_same_head_paired_draft", rounds=[], candidate_backend=candidate_name,
                  model=str(args.model), head_dir=str(args.head_dir), runtime=runtime_identity(pypto))
    engine = request = primary = extra = None
    try:
        torch.set_num_threads(4)
        head = HeadSource(args.model, args.head_dir)
        engine = make_engine(args.source, args.model, args.device, head, attention_backend=args.baseline_backend,
                             head_transfer=args.head_transfer)
        draft = engine.runner.draft
        original_compile = draft.compile
        programs = {}

        def compile_both(samples, config):
            result = original_compile(samples, config)
            base_key = "full_layer" if args.baseline_backend == "paged" else "full"
            programs[args.baseline_backend] = draft.programs[base_key]
            key = "group" if args.draft_group_size == 4 else "full_layer" if args.candidate_backend == "paged" else "full"
            programs[candidate_name] = make_programs(args.source, attention_backend=args.candidate_backend,
                                                   group_size=args.draft_group_size)[key].compile(
                *metadata(head, samples)[key], config=config)
            return [*result, programs[candidate_name]]

        draft.compile = compile_both
        start = time.monotonic()
        engine.start()
        report.update(startup_seconds=time.monotonic() - start, source_tp=head.tp,
                      head_sha256=head_digest(head), head_storage=draft.storage)
        native = engine.runner.native
        primary = draft.weights
        if args.draft_group_size == 4:
            from blockspec.pypto_qwen38.weights import DraftWeights
            extra = DraftWeights(head, "auto", "packed", group_size=4)
            extra.start(native.worker)
            draft.group_scratch = [draft._allocate((32, 5120)), draft._allocate((32, 5120))]
            report["group_head_storage"] = extra.storage
        schedule = native_module(args.source, "schedule")
        request = schedule.RequestState(engine.tokenizer.encode("Hello", add_special_tokens=False), 40, frozenset())
        native.admit(request)
        for position in range(0, len(request.prompt_ids), 16):
            tokens = tuple(request.prompt_ids[position:position + 16])
            anchor = native.run_batch([schedule.BatchSlot(request, tokens, position, "prefill",
                                      position + len(tokens) == len(request.prompt_ids))])[0]
        request.generated.append(anchor)
        request.prefill_offset = len(request.prompt_ids)
        target = NativeTarget(native, request, schedule.BatchSlot)
        snapshot = target.snapshot()
        expected, token = [], anchor
        for _ in range(35):
            token = target.forward([token])[0]
            expected.append(token)
        target.restore(snapshot)
        before = resident_checksum(native)
        report.update(anchor=anchor, initial_history_sha256=history_checksum(native, request),
                      head_manifest=json.loads((args.head_dir / "manifest.json").read_text()))
        if args.capture:
            save_history(native, request, args.output_dir / "capture")
        signatures = {}

        def select(backend):
            grouped = backend == "paged-group4"
            attention_backend = "paged" if grouped else backend
            key = "group" if grouped else "full_layer" if attention_backend == "paged" else "full"
            draft.programs[key] = programs[backend]
            draft.attention_backend, draft.group_size = attention_backend, 4 if grouped else 1
            draft.weights = extra if grouped else primary

        def propose(backend):
            select(backend)
            return draft.propose(request, anchor, 32)

        for backend in programs:
            propose(backend)
        for index in range(args.rounds):
            for backend in (list(programs) if index % 2 == 0 else list(reversed(programs))):
                start = time.perf_counter()
                candidate = propose(backend)
                elapsed = time.perf_counter() - start
                signature = dict(candidates=candidate, logits_sha256=tensor_digest(draft.read_logits(31)))
                if backend in signatures and signature != signatures[backend]:
                    raise AssertionError(f"{backend}: repeated draft changed candidates/logits")
                signatures[backend] = signature
                row = dict(round=index, backend=backend, seconds=elapsed, phases=dict(draft.last_timings))
                if row["phases"]["compute_dispatches"] != (18 if backend == "paged-group4" else 66 if backend == "paged" else 82):
                    raise AssertionError("unexpected K32 compute dispatch count")
                report["rounds"].append(row)
                print(json.dumps(row), flush=True)
        report["signatures"] = signatures
        if args.draft_group_size == 4 and signatures[args.baseline_backend] != signatures[candidate_name]:
            raise AssertionError("grouping changed draft logits or candidates")
        if args.capture:
            for backend in programs:
                with capture_layers(draft, args.output_dir / "capture" / backend):
                    propose(backend)
                if tensor_digest(draft.read_logits(31)) != signatures[backend]["logits_sha256"]:
                    raise AssertionError("capturing changed draft logits")
        report["target_cache_unchanged"] = before == resident_checksum(native)
        if not report["target_cache_unchanged"]:
            raise AssertionError("draft backend changed the target history")
        report["medians"] = {backend: statistics.median(row["seconds"] for row in report["rounds"]
                                                       if row["backend"] == backend) for backend in programs}
        report["paired_speedup"] = report["medians"][args.baseline_backend] / report["medians"][candidate_name]
        for backend, signature in signatures.items():
            target.restore(snapshot)
            result = verify_transaction(target, anchor, signature["candidates"], budget=32)
            if list(result.tokens) != expected[:len(result.tokens)]:
                raise AssertionError(f"{backend}: corrected output differs from target-only")
            actual = history_checksum(native, request)
            target.restore(snapshot)
            target.forward((anchor, *signature["candidates"])[:result.committed_inputs])
            if actual != history_checksum(native, request):
                raise AssertionError(f"{backend}: accepted history differs from replay")
        report["verification_and_replay_passed"] = True
        if args.audit_verifier:
            report["target_derived_verifier_cases"] = audit_verifier(target, anchor, expected, snapshot)
        if args.audit_boundaries:
            native.release(request)
            request = None
            select(candidate_name)
            report["boundary_cases"] = []
            prefix_hits = native.allow_prefix_hits
            native.allow_prefix_hits = False
            try:
                for past in (15, 17, 96, 97, 224, 225):
                    seed = engine.tokenizer.encode(f"Case {past}. Please explain how computers work. ", add_special_tokens=False)
                    ids = (seed * ((past + len(seed) - 1) // len(seed)))[:past]
                    audit = dict(past=past, width=32, passed=False)
                    report["boundary_cases"].append(audit)
                    audit_round(engine, args.source, 32, audit, prompt_ids=ids)
                    audit["requests_released"] = not bool(native.requests)
                    if not audit["requests_released"]:
                        raise AssertionError("request boundary leaked target state")
            finally:
                native.allow_prefix_hits = prefix_hits
        if args.reload_reference:
            reference = json.loads(args.reload_reference.read_text())
            for field in ("head_sha256", "initial_history_sha256", "signatures"):
                if report[field] != reference[field]:
                    raise AssertionError(f"restart changed {field}")
            report["restart_head_history_logits_exact"] = True
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
                    try:
                        if extra is not None:
                            extra.close()
                    finally:
                        if primary is not None:
                            engine.runner.draft.weights = primary
                        engine.stop()
        except BaseException as error:
            report.update(passed=False, cleanup_error=repr(error))
            raise
        finally:
            (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
