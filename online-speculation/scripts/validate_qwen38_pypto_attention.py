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


def exported_runtime_identity(pypto_module, path):
    """导出环境可能不含 .git；仍逐文件校验已登记的源码、绑定与工具链。"""
    import importlib.metadata
    import os
    from pypto.backend._ptoas_locate import find_ptoas_binary
    from pypto.runtime.runtime_pin import check_runtime_pin

    if os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "").lower() not in {"", "0", "false", "no", "off"}:
        raise ValueError("runtime pin checking must remain enabled")
    raw = path.read_bytes()
    expected = json.loads(raw)
    root = Path(pypto_module.__file__).resolve().parents[2]
    if (expected.get("format") != "pypto-export-artifacts-v1"
            or str(root) != expected["pypto_root"]
            or str(Path(pypto_module.__file__).resolve()) not in expected["files"]):
        raise ValueError("exported PyPTO identity does not match the imported package")
    bindings = {str(p.resolve()) for p in (root / "python/pypto").glob("*.so")}
    compiler = find_ptoas_binary()
    if not bindings or not bindings <= expected["files"].keys() or compiler is None or str(
            Path(compiler).resolve()) not in expected["files"]:
        raise ValueError("runtime manifest must bind the native extension and PTOAS executable")
    for name, digest in expected["files"].items():
        actual = hashlib.sha256()
        with Path(name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError("runtime artifact mismatch: " + name)
    versions = {name: importlib.metadata.version(name) for name in expected["packages"]}
    pin = check_runtime_pin()
    # 无 Git 的导出包中 comparable=False 表示未知，不表示版本冲突。
    # 此时由已验证源包的文件哈希和实际加载的 Simpler 扩展共同绑定 ABI。
    if (versions != expected["packages"] or pin.comparable and not pin.matches or pin.off_pin
            or pin.stamp_missing or pin.installed != expected["simpler_commit"]):
        raise ValueError("exported runtime package versions or Simpler pin changed")
    if pin.extension_path is None or str(Path(pin.extension_path).resolve()) not in expected["files"]:
        raise ValueError("runtime manifest must bind the loaded Simpler extension")
    return dict(mode="exported_artifacts", manifest_sha256=hashlib.sha256(raw).hexdigest(),
                checked_files=len(expected["files"]), packages=versions, simpler=pin.installed,
                git_pin_comparable=pin.comparable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="assigned logical NPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--assemble-only", action="store_true",
                        help="build device binaries using the pinned runtime, without creating an NPU worker")
    parser.add_argument("--simulate", action="store_true", help="run functional simulator, not NPU timing")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--query-tile", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--context", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--page-size", type=int, choices=(16, 128), default=16)
    parser.add_argument("--q-heads", type=int, choices=(12, 24), default=24)
    parser.add_argument("--kv-heads", type=int, choices=(2, 4), default=4)
    parser.add_argument("--runtime-artifacts", type=Path,
                        help="verified artifact manifest for exported environments without Git metadata")
    parser.add_argument("--backend", choices=("fused", "staged", "paged", "both", "all"), default="fused")
    parser.add_argument("--profile", action="store_true", help="collect a separate final-case chip swimlane, outside timing")
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--tune-paged", action="store_true", help="compare paged query tiles 16/32 in the same worker")
    parser.add_argument("--compare-full-history", action="store_true",
                        help="compare the paged reference against skipping zero fill for complete history spans")
    args = parser.parse_args()
    if args.device < 0 or args.repeats < 3 or args.output.exists() or args.profile_repeats < 1:
        parser.error("require an assigned device, 3+ repeats and a new output")
    if args.assemble_only and (args.compile_only or args.simulate):
        parser.error("assemble-only uses onboard compilation and excludes compile-only/simulate")
    if args.tune_paged and (args.backend != "all" or args.query_tile != 16):
        parser.error("tune-paged requires backend=all and query-tile=16")
    if args.compare_full_history and args.backend != "paged":
        parser.error("compare-full-history requires backend=paged")
    if args.context != 512 and (args.backend != "paged" or args.query_tile != 16 or args.tune_paged):
        parser.error("1024 context requires standalone paged backend with query tile 16")
    if args.q_heads % args.kv_heads:
        parser.error("query heads must be divisible by KV heads")
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

    shape = AttentionShape(query_tile=args.query_tile, context=args.context, page_size=args.page_size,
                           q_heads=args.q_heads, kv_heads=args.kv_heads)
    report = dict(passed=False, scope="standalone_paged_draft_attention", npu_execution=False,
                  full_model=False, shape=vars(shape), runtime_log_level="WARNING", cases=[])
    worker, handles = None, []
    try:
        torch.set_num_threads(4)
        report["runtime"] = (runtime_identity(pypto) if args.runtime_artifacts is None else
                             exported_runtime_identity(pypto, args.runtime_artifacts))
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
        physical_pages = shape.context // shape.page_size + 3
        logical_pages = shape.context // shape.page_size
        kp, vp = [rand((2, shape.kv_heads, physical_pages, shape.page_size, shape.head_dim)) for _ in range(2)]
        table = torch.randperm(physical_pages, generator=generator)[:logical_pages].int().share_memory_()
        bias = torch.empty((1, shape.context), dtype=torch.float32).share_memory_()
        output = torch.empty_like(q).share_memory_()
        tensors = [q, k, v, gate, kp, vp, table, bias, output]
        backends = (["staged", "fused", "paged"] if args.backend == "all" else
                    ["staged", "fused"] if args.backend == "both" else [args.backend])
        if args.tune_paged:
            backends.append("paged32")
        if args.compare_full_history:
            backends.append("paged_full_history")
        programs = {}
        configurations = {}
        for backend in backends:
            print(f"compiling Attention {backend}", flush=True)
            tile = 32 if backend == "paged32" else args.query_tile
            candidate_shape = AttentionShape(query_tile=tile, context=shape.context, page_size=shape.page_size,
                                             q_heads=shape.q_heads, kv_heads=shape.kv_heads)
            configurations[backend] = dict(query_tile=tile)
            factory = (make_paged_attention(candidate_shape, physical_pages=kp.shape[2], pool_layers=2,
                                            skip_full_history_zero=backend == "paged_full_history") if backend.startswith("paged") else
                       make_attention(shape, backend=backend, physical_pages=kp.shape[2], pool_layers=2))
            programs[backend] = factory[1].compile(*tensors, 0, 32, 0, config=config)
        report["compiled"] = True
        report["artifact_directories"] = {name: str(program.output_dir) for name, program in programs.items()}
        if args.assemble_only:
            # 固定版本的 worker 也是通过该入口编译设备二进制；这里不创建 worker。
            from pypto.runtime.distributed_runner import _assemble_chip_callables
            for program in programs.values():
                _assemble_chip_callables(program)
            report.update(passed=True, scope="attention_device_binary_compile_only", assembled=True)
            return
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
        if shape.context == 1024:
            cases += ((481, 32), (511, 8), (512, 32), (895, 32), (992, 32), (1022, 2))
        if args.compare_full_history:
            cases += ((128, 8), (255, 8), (256, 8), (384, 8), (504, 8))
        for case_index, (past, width) in enumerate(cases):
            cache_layer = case_index % 2
            pages = (past + shape.page_size - 1) // shape.page_size
            table.copy_(torch.randperm(physical_pages, generator=generator)[:logical_pages].int())
            table[pages:] = -1
            bias.fill_(-10000.)
            bias[:, :past + width] = 0
            # 无效页、历史末页尾部和 inactive draft 行用 NaN，检测误读。
            for pool, original in zip((kp, vp), original_pools, strict=True):
                pool.fill_(float("nan"))
                for logical in range(pages):
                    physical = table[logical].item()
                    count = min(shape.page_size, past - logical * shape.page_size)
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
            compared_outputs = {}
            for backend in order:
                compiled = programs[backend]
                worker.run(compiled, *handles, past, width, cache_layer, config=config)
                worker.copy_from(output.data_ptr(), handles[-1].data_ptr, output.numel() * output.element_size())
                if args.compare_full_history:
                    compared_outputs[backend] = output.clone()
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
            if args.compare_full_history and not torch.equal(compared_outputs["paged"], compared_outputs["paged_full_history"]):
                raise AssertionError(f"complete-history path changed output at past={past}, width={width}")
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
