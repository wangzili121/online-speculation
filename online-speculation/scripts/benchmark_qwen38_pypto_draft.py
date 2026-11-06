"""Measure K32 draft overhead on one fixed prefix; audit cache and greedy verification."""

import argparse
import hashlib
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import time


@contextmanager
def profile(draft):
    stats = {}
    originals = []

    def install(obj, name, label):
        original = getattr(obj, name)
        originals.append((obj, name, original))

        def timed(*args, **kwargs):
            start = time.monotonic()
            try:
                return original(*args, **kwargs)
            finally:
                item = stats.setdefault(label, {"calls": 0, "seconds": 0.0})
                item["calls"] += 1
                item["seconds"] += time.monotonic() - start

        setattr(obj, name, timed)

    for obj, name, label in [
        (draft.head, "banks", "head_prepare"),
        (draft.native.worker, "alloc_tensor", "allocate_upload"),
        (draft.native.worker, "free_tensor", "free"),
        (draft, "_run", "run"),
        (draft, "_upload", "upload"),
        (draft.native.worker, "copy_from", "readback"),
        (draft.native.worker, "copy_to", "host_to_device"),
    ]:
        install(obj, name, label)
    try:
        yield stats
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--head-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--head-storage", choices=("auto", "resident", "stream"))
    p.add_argument("--head-transfer", choices=("individual", "packed"))
    p.add_argument("--attention-backend", choices=("native", "fused", "paged"), default="native")
    p.add_argument("--save-logits", action="store_true")
    p.add_argument("--compare-transfer", action="store_true")
    a = p.parse_args()
    if a.output.exists() or a.rounds < 1 or a.device < 0:
        p.error("require a new output, assigned device and positive round count")
    logging.getLogger("simpler").setLevel(logging.WARNING)
    # 与功能验收共用缓存检查，计时只包住 propose，不包含这些审计。
    from validate_qwen38_pypto_neural import resident_checksum, history_checksum
    from blockspec.pypto_qwen38.baseline import native_module, runtime_identity
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.target import NativeTarget
    from blockspec.pypto_qwen38.greedy import verify_transaction
    import pypto
    import torch

    report = {"passed": False, "rounds": [], "runtime": runtime_identity(pypto)}
    engine = primary = extra = None
    request = None
    try:
        source = HeadSource(a.model, a.head_dir)
        kw = {} if a.head_storage is None else {"head_storage": a.head_storage}
        if a.head_transfer is not None:
            kw["head_transfer"] = a.head_transfer
        report["attention_backend"] = a.attention_backend
        engine = make_engine(a.source, a.model, a.device, source, attention_backend=a.attention_backend, **kw)
        start = time.monotonic()
        engine.start()
        report["startup_seconds"] = time.monotonic() - start
        native, draft = engine.runner.native, engine.runner.draft
        primary = draft.weights
        strategies = {getattr(primary, "transfer", "individual"): primary}
        if a.compare_transfer:
            from blockspec.pypto_qwen38.weights import DraftWeights
            if draft.storage != "stream":
                raise ValueError("transfer comparison requires streaming head storage")
            other = "packed" if primary.transfer == "individual" else "individual"
            extra = DraftWeights(source, "stream", other)
            extra.start(native.worker)
            strategies[other] = extra
        schedule = native_module(a.source, "schedule")
        request = schedule.RequestState(
            engine.tokenizer.encode("Hello", add_special_tokens=False), 40, frozenset()
        )
        native.admit(request)
        for position in range(0, len(request.prompt_ids), 16):
            tokens = tuple(request.prompt_ids[position : position + 16])
            anchor = native.run_batch(
                [
                    schedule.BatchSlot(
                        request,
                        tokens,
                        position,
                        "prefill",
                        position + len(tokens) == len(request.prompt_ids),
                    )
                ]
            )[0]
        request.generated.append(anchor)
        request.prefill_offset = len(request.prompt_ids)
        target = NativeTarget(native, request, schedule.BatchSlot)
        initial = target.snapshot()
        expected = []
        token = anchor
        for _ in range(35):
            token = target.forward([token])[0]
            expected.append(token)
        target.restore(initial)
        before = resident_checksum(native)
        report["initial_history_sha256"] = history_checksum(native, request)
        sequence = [(n, name) for n in range(a.rounds)
                    for name in (list(strategies) if n % 2 == 0 else list(reversed(strategies)))]
        for n, name in sequence:
            draft.weights = strategies[name]
            with profile(draft) as stats:
                start = time.monotonic()
                candidate = draft.propose(request, anchor, 32)
                elapsed = time.monotonic() - start
            if "candidates" in report:
                assert candidate == report["candidates"]
            report["candidates"] = candidate
            row = {"round": n, "transfer": name, "seconds": elapsed, "phases": stats}
            if hasattr(draft, "last_timings"):
                row["internal_timings"] = draft.last_timings
            logits = torch.empty((31, 248320), dtype=torch.bfloat16)
            native.worker.copy_from(
                logits.data_ptr(),
                draft.logits.data_ptr,
                logits.numel() * logits.element_size(),
            )
            row["logits_sha256"] = hashlib.sha256(
                memoryview(logits.view(torch.uint8).numpy())
            ).hexdigest()
            if report["rounds"]:
                assert row["logits_sha256"] == report["rounds"][0]["logits_sha256"]
            if not report["rounds"] and a.save_logits:
                from safetensors.torch import save_file

                save_file(
                    {"logits": logits}, str(a.output.with_suffix(".logits.safetensors"))
                )
            del logits
            report["rounds"].append(row)
            print(json.dumps(row), flush=True)
            a.output.with_suffix(".progress.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        report["target_cache_unchanged"] = before == resident_checksum(native)
        assert report["target_cache_unchanged"]
        start = time.monotonic()
        corrected = verify_transaction(target, anchor, candidate, budget=32)
        report["verify_seconds"] = time.monotonic() - start
        assert list(corrected.tokens) == expected[: len(corrected.tokens)]
        actual = history_checksum(native, request)
        target.restore(initial)
        target.forward((anchor, *candidate)[: corrected.committed_inputs])
        report["accepted_history_exact"] = actual == history_checksum(native, request)
        assert report["accepted_history_exact"]
        report["accepted"] = corrected.accepted
        report["output_tokens"] = list(corrected.tokens)
        report["storage"] = getattr(draft, "storage", None)
        report["preparation_seconds"] = getattr(source, "preparation_seconds", None)
        native.release(request)
        request = None
        report["passed"] = True
    except BaseException as e:
        report["error"] = repr(e)
        raise
    finally:
        try:
            try:
                if extra is not None:
                    extra.close()
            finally:
                try:
                    if primary is not None:
                        engine.runner.draft.weights = primary
                    if request is not None:
                        engine.runner.native.release(request)
                finally:
                    if engine is not None:
                        engine.stop()
        except BaseException as e:
            report["cleanup_error"] = repr(e)
            report["passed"] = False
            raise
        finally:
            a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
