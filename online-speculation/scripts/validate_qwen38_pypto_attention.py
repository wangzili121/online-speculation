#!/usr/bin/env python3
"""Compile/test the standalone Orthrus Attention without loading 27B weights."""

import argparse
import json
import logging
from pathlib import Path
import statistics
import shutil
import hashlib
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="assigned logical NPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="run functional simulator, not NPU timing")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--query-tile", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--backend", choices=("fused", "staged", "paged", "both", "all"), default="fused")
    parser.add_argument("--profile", action="store_true", help="collect a separate final-case chip swimlane, outside timing")
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--tune-paged", action="store_true", help="compare paged query tiles 16/32 in the same worker")
    args = parser.parse_args()
    if args.device < 0 or args.repeats < 3 or args.output.exists() or args.profile_repeats < 1:
        parser.error("require an assigned device, 3+ repeats and a new output")
    if args.tune_paged and (args.backend != "all" or args.query_tile != 16):
        parser.error("tune-paged requires backend=all and query-tile=16")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 在 worker 初始化前关闭运行时逐调用 tracing，避免干扰正式计时。
    logging.getLogger("simpler").setLevel(logging.WARNING)
    import torch
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, ExecutionMode
    from blockspec.pypto_qwen38.attention import AttentionShape, attention_golden
    from blockspec.pypto_qwen38.attention_programs import make_attention
    from blockspec.pypto_qwen38.paged_attention_programs import make_paged_attention
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from summarize_qwen38_pypto_profile import summarize

    shape = AttentionShape(query_tile=args.query_tile)
    report = dict(passed=False, scope="standalone_paged_draft_attention", npu_execution=False,
                  full_model=False, shape=vars(shape), runtime_log_level="WARNING", cases=[])
    worker, handles = None, []
    try:
        torch.set_num_threads(4)
        report["runtime"] = runtime_identity(pypto)
        mode = ExecutionMode.SIM if args.simulate else ExecutionMode.ONBOARD
        config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=mode,
                           distributed_config=DistributedConfig(
            device_ids=[args.device], num_sub_workers=0))
        if config.execution_mode != mode:
            raise RuntimeError("requested execution mode was overridden")
        report["platform"] = config.platform
        generator = torch.Generator().manual_seed(173)
        rand = lambda dims: (torch.randn(dims, generator=generator) * .5).bfloat16().share_memory_()
        q, gate = [rand((32, shape.query_width)) for _ in range(2)]
        k, v = [rand((32, shape.kv_width)) for _ in range(2)]
        kp, vp = [rand((2, shape.kv_heads, 35, 16, shape.head_dim)) for _ in range(2)]
        table = torch.randperm(35, generator=generator)[:32].int().share_memory_()
        bias = torch.empty((1, shape.context), dtype=torch.float32).share_memory_()
        output = torch.empty_like(q).share_memory_()
        tensors = [q, k, v, gate, kp, vp, table, bias, output]
        backends = (["staged", "fused", "paged"] if args.backend == "all" else
                    ["staged", "fused"] if args.backend == "both" else [args.backend])
        if args.tune_paged:
            backends.append("paged32")
        programs = {}
        configurations = {}
        for backend in backends:
            print(f"compiling Attention {backend}", flush=True)
            tile = 32 if backend == "paged32" else args.query_tile
            candidate_shape = AttentionShape(query_tile=tile)
            configurations[backend] = dict(query_tile=tile)
            factory = (make_paged_attention(candidate_shape, physical_pages=kp.shape[2], pool_layers=2) if backend.startswith("paged") else
                       make_attention(shape, backend=backend, physical_pages=kp.shape[2], pool_layers=2))
            programs[backend] = factory[1].compile(*tensors, 0, 32, 0, config=config)
        report["compiled"] = True
        report["artifact_directories"] = {name: str(program.output_dir) for name, program in programs.items()}
        if args.compile_only:
            report.update(passed=True, scope="attention_compile_only")
            return
        worker = DistributedWorker(list(programs.values()), config=config, inherited_host_tensors=tensors)
        report["npu_execution"] = not args.simulate
        report["simulator_execution"] = args.simulate
        for tensor in tensors:
            handles.append(worker.alloc_tensor(tuple(tensor.shape), tensor.dtype))
        before = [torch.empty_like(tensor).share_memory_() for tensor in tensors[:-1]]
        original_pools = [pool.clone() for pool in (kp, vp)]
        original_block = [tensor.clone() for tensor in (k, v)]
        # 包含空历史、非整页边界、部分有效块及上下文末端。
        cases = ((0, 32), (1, 2), (15, 32), (16, 31), (17, 7), (127, 32),
                 (95, 32), (96, 32), (97, 32), (223, 32), (224, 32), (225, 32), (480, 32))
        for case_index, (past, width) in enumerate(cases):
            cache_layer = case_index % 2
            pages = (past + 15) // 16
            table.copy_(torch.randperm(35, generator=generator)[:32].int())
            table[pages:] = -1
            bias.fill_(-10000.)
            bias[:, :past + width] = 0
            # 无效页、历史末页尾部和 inactive draft 行用 NaN，检测误读。
            for pool, original in zip((kp, vp), original_pools, strict=True):
                pool.fill_(float("nan"))
                for logical in range(pages):
                    physical = table[logical].item()
                    count = min(16, past - logical * 16)
                    pool[cache_layer, :, physical, :count].copy_(original[cache_layer, :, physical, :count])
            for tensor, original in zip((k, v), original_block, strict=True):
                tensor.copy_(original)
                tensor[width:].fill_(float("nan"))
            expected = attention_golden(shape, q, k, v, gate, kp[cache_layer], vp[cache_layer], table, past, width)
            native = attention_golden(shape, q, k, v, gate, kp[cache_layer], vp[cache_layer], table, past, width,
                                      native_rounding=True)
            for tensor, handle in zip(tensors[:-1], handles[:-1], strict=True):
                worker.copy_to(handle.data_ptr, tensor.data_ptr(), tensor.numel() * tensor.element_size())
            order = backends if case_index % 2 == 0 else list(reversed(backends))
            samples = {backend: [] for backend in backends}
            for backend in order:
                for _ in range(5):
                    worker.run(programs[backend], *handles, past, width, cache_layer, config=config)
            # 交错计时，避免先后运行整组样本时的温度/频率变化偏向某个实现。
            for repeat in range(args.repeats):
                for backend in (order if repeat % 2 == 0 else list(reversed(order))):
                    start = time.perf_counter()
                    worker.run(programs[backend], *handles, past, width, cache_layer, config=config)
                    samples[backend].append(time.perf_counter() - start)
            for backend in order:
                compiled = programs[backend]
                worker.run(compiled, *handles, past, width, cache_layer, config=config)
                worker.copy_from(output.data_ptr(), handles[-1].data_ptr, output.numel() * output.element_size())
                for host, original, handle in zip(before, tensors[:-1], handles[:-1], strict=True):
                    worker.copy_from(host.data_ptr(), handle.data_ptr, host.numel() * host.element_size())
                    if not torch.equal(host.view(torch.uint8), original.view(torch.uint8)):
                        raise AssertionError("attention modified a read-only input")
                # BF16 中间舍入与标准 SDPA 的差异单独记录；不要求逐字节相同。
                torch.testing.assert_close(output.float(), native.float(), rtol=.02, atol=.002)
                torch.testing.assert_close(output.float(), expected.float(), rtol=.03, atol=.004)
                row = dict(past=past, width=width, cache_layer=cache_layer, backend=backend,
                           **configurations[backend], passed=True, inputs_byte_exact=True,
                           poisoned_inactive_inputs=True,
                           native_max_abs=(output.float() - native.float()).abs().max().item(),
                           sdpa_max_abs=(output.float() - expected.float()).abs().max().item(),
                           resident_dispatch_median_seconds=None if args.simulate else statistics.median(samples[backend]),
                           resident_dispatch_samples_seconds=[] if args.simulate else samples[backend])
                report["cases"].append(row)
                if args.profile and case_index in {0, len(cases) - 1} and not args.simulate:
                    from dataclasses import replace
                    row["device_profiles"] = []
                    for sample in range(args.profile_repeats):
                        worker.run(compiled, *handles, past, width, cache_layer,
                                   config=replace(config, enable_chip_swimlane=1))
                        profiles = list(Path(compiled.output_dir).rglob("merged_swimlane*.json"))
                        if not profiles:
                            raise FileNotFoundError("missing Attention device profile")
                        path = max(profiles, key=lambda p: p.stat().st_mtime_ns)
                        data = path.read_bytes()
                        saved = args.output.parent / (args.output.stem + "-profiles") / f"c{case_index}-{backend}-{sample}.json"
                        saved.parent.mkdir(exist_ok=True)
                        if saved.exists():
                            raise FileExistsError(saved)
                        shutil.copyfile(path, saved)
                        row["device_profiles"].append(dict(path=str(saved), sha256=hashlib.sha256(data).hexdigest(),
                                                           **summarize(json.loads(data))))
                    row["profile_requested"] = True
                print(json.dumps(row), flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        failures = []
        if worker is not None:
            for handle in reversed(handles):
                try:
                    worker.free_tensor(handle)
                except BaseException as error:
                    failures.append(repr(error))
            try:
                worker.close()
            except BaseException as error:
                failures.append(repr(error))
        if failures:
            report.update(passed=False, cleanup_errors=failures)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if failures:
            raise RuntimeError("attention cleanup failed")


if __name__ == "__main__":
    main()
