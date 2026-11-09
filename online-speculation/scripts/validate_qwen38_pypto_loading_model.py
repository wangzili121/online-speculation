#!/usr/bin/env python3
"""真实主模型全部 bank 与 CPU 基线比对，随后验证 K32 事务及调度输出。"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from validate_qwen38_pypto_flow import collect
from validate_qwen38_pypto_neural import audit_round


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--head-dir", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-json", type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error("new output required")
    import torch
    import pypto
    from blockspec.pypto_qwen38.baseline import runtime_identity, native_module
    from blockspec.pypto_qwen38.draft import make_engine
    from blockspec.pypto_qwen38.head import HeadSource
    torch.set_num_threads(4)
    report = dict(passed=False, npu_execution=True, runtime=runtime_identity(pypto), banks=[])
    engine = None
    try:
        head = HeadSource(a.model, a.head_dir)
        engine = make_engine(a.source, a.model, a.device, head, attention_backend="paged", gdn_backend="vector",
                             group_size=4, head_transfer="packed")
        start = time.perf_counter()
        engine.start()
        report["startup_seconds"] = time.perf_counter() - start
        native = engine.runner.native
        assert native.weight_backend == "pypto" and head.preparation_backend == "pypto"
        returned = torch.empty(32 << 20, dtype=torch.uint8)
        reference = native_module(a.source, "weight").TextWeights(a.model)
        for (name, expected), actual in zip(reference.banks(), native.banks, strict=True):
            assert tuple(expected.shape) == tuple(actual.shape) and expected.dtype == actual.dtype, name
            raw = expected.view(torch.uint8).reshape(-1)
            digest = hashlib.sha256()
            for offset in range(0, raw.numel(), returned.numel()):
                count = min(returned.numel(), raw.numel() - offset)
                native.worker.copy_from(returned.data_ptr(), actual.data_ptr, count, src_offset=offset)
                assert torch.equal(raw[offset:offset + count], returned[:count]), (name, offset)
                digest.update(memoryview(returned[:count].numpy()))
            report["banks"].append(dict(name=name, shape=list(actual.shape), dtype=str(actual.dtype),
                                        byte_exact=True, sha256=digest.hexdigest()))
            print("target bank byte-exact", name, flush=True)
        report["block"] = {}
        audit_round(engine, a.source, 32, report["block"])
        logits = engine.runner.draft.read_logits(31)
        sha = hashlib.sha256(memoryview(logits.view(torch.uint8).numpy())).hexdigest()
        report["logits_sha256"] = sha
        if a.reference_json:
            old = json.loads(a.reference_json.read_text())["signatures"]["device"]
            assert sha == old["logits_sha256"] and report["block"]["candidates"] == old["tokens"]
            report["prior_loader_logits_and_tokens_byte_exact"] = True
        runner = engine.runner
        engine.scheduler.runner = native
        baseline = collect(engine, 4, 1200)
        deadline = time.monotonic() + 30
        while engine.scheduler.counts()["active"]:
            if time.monotonic() > deadline:
                raise TimeoutError("baseline request was not released")
            time.sleep(.01)
        engine.scheduler.runner = runner
        actual = collect(engine, 4, 1200)
        assert baseline == actual, (baseline, actual)
        report.update(passed=True, baseline=baseline, actual=actual)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if engine is not None:
                engine.stop()
        except BaseException as error:
            report.update(passed=False, cleanup_error=repr(error))
            raise
        finally:
            a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
