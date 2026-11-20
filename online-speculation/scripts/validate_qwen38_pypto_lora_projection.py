#!/usr/bin/env python3
"""真实投影形状的低秩 VJP/合并检查；不代替完整在线服务验收。"""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from types import SimpleNamespace
import time


def section(graph, targets, stop=(), *, legacy=False):
    """截取产品图里的真实节点，在投影的 X/dY 处停止，不另写一套 VJP。"""
    from blockspec.pypto_qwen38.training_graph import Node, Value
    needed, kept = {v.name for v in targets.values()}, []
    for node in reversed(graph.nodes):
        if any(v.name in needed for v in node.outputs):
            kept.append(node)
            needed.update(v.name for v in node.inputs if v.name not in stop)
    nodes = list(reversed(kept))
    names = {v.name: v for node in nodes for v in (*node.inputs, *node.outputs)}
    rename = {n: Value("legacy_" + n, v.shape, v.dtype) for n, v in names.items()} if legacy else names
    mapped = []
    for node in nodes:
        options = dict(node.options)
        if legacy and "task_group" in options and node.kind == "linear":
            options["task_group"] = 16
        mapped.append(Node(node.kind, tuple(rename[v.name] for v in node.inputs),
                           tuple(rename[v.name] for v in node.outputs), options))
    outputs = {v.name for node in mapped for v in node.outputs}
    inputs = {v.name: v for node in mapped for v in node.inputs if v.name not in outputs}
    inverse = {rename[n].name: n for n in names}
    return SimpleNamespace(shape=graph.shape, nodes=mapped, inputs=inputs,
                           gradients={k: rename[v.name] for k, v in targets.items()},
                           aliases={n: inverse[n] for n in inputs}, renamed=rename)


def load_reference(folder, name):
    spec = importlib.util.spec_from_file_location("blockspec.pypto_qwen38.reference_" + name, folder / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh result path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.training_executor import TrainingPrograms, TrainingWorkspace
    from blockspec.pypto_qwen38.training_graph import SuffixShape
    from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
    from blockspec.pypto_qwen38.training_lora_graph import LoRAMergeGraph, LoRASuffixGraph
    from blockspec.pypto_qwen38.training_optimizer_programs import make_stats
    from blockspec.pypto_qwen38.training_programs import make_element, make_linear

    torch.set_num_threads(4)
    torch.manual_seed(156)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    old = load_reference(args.reference_source, "training_programs")
    old_optimizer = load_reference(args.reference_source, "training_optimizer_programs")
    meta = lambda v: torch.empty(v.shape, dtype=torch.float32 if v.dtype == "fp32" else torch.bfloat16, device="meta")
    examples = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    full, suffix = LoRASuffixGraph(1), LoRASuffixGraph(4)
    tail = LoRASuffixGraph(1, SuffixShape(hidden=128, intermediate=256, vocabulary=128,
        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32),
        GDNShape(heads=8, query_heads=4, dimension=64))
    cases = [(full, "l63_q_weight"), (full, "l63_k_weight"), (full, "l63_o_weight"),
             (suffix, "l60_qkv_weight"), (suffix, "l60_z_weight"), (suffix, "l60_a_weight"),
             (tail, "l63_k_weight")]
    compiled, programs = [], []
    for index, (graph, name) in enumerate(cases):
        adapter = graph.adapters[name]
        producers = {v.name: node for node in graph.nodes for v in node.outputs}
        dw = graph.dense_gradients[name]
        dw_node = producers[dw.name]
        dy, x = dw_node.inputs
        stop = {x.name, dy.name, adapter.a.name, adapter.b.name}
        targets = {p.name: graph.gradients[p.name] for p in (adapter.a, adapter.b)}
        vjp, legacy = section(graph, targets, stop), section(graph, targets, stop, legacy=True)
        merge = LoRAMergeGraph(graph)
        merged = section(merge, {name: merge.targets[name]}, set(merge.inputs))
        parts = []
        for label, current in (("vjp", vjp), ("legacy_vjp", legacy), ("merge", merged)):
            print("compiling", index, name, label, flush=True)
            parts.append(TrainingPrograms(current, config, combined=True))
        vp, lp, mp = parts
        print("compiling old dense gradient", name, flush=True)
        old_dw = old.make_linear(**dw_node.options).compile(meta(dy), meta(x), meta(dw), config=config)
        rows, columns = adapter.projection.shape
        forward = make_linear((32, columns), (rows, columns)).compile(
            examples((32, columns)), examples((rows, columns)), examples((32, rows)), config=config)
        stat, stat_rows = make_stats((rows, columns), input_bf16=True)
        stat = stat.compile(examples((rows, columns)), examples((stat_rows, 8), torch.float32),
                            examples((stat_rows, 8), torch.float32), config=config)
        programs += [*vp.programs.values(), *lp.programs.values(), *mp.programs.values(), old_dw, forward, stat]
        compiled.append(dict(index=index, graph=graph, name=name, adapter=adapter, dw=dw, dy=dy, x=x,
            vjp=vjp, legacy=legacy, merge=merge, merged=merged, vp=vp, lp=lp, mp=mp,
            old_dw=old_dw, forward=forward, stat=stat, stat_rows=stat_rows))

    # 修改过的通用算子默认路径也与固定旧实现逐位比较。
    elements = []
    for kind, dtype in [(k, torch.bfloat16) for k in
                         ("add", "mul", "silu", "sigmoid", "silu_grad", "sigmoid_grad", "scale", "copy")] + [
                         ("add", torch.float32), ("scale", torch.float32)]:
        inp, out = examples((32, 128), dtype), examples((32, 128))
        # 旧入口只接收 BF16；新增 FP32 中间量与独立 Torch 公式对照。
        print("compiling element", kind, dtype, flush=True)
        a = (old.make_element((32, 128), kind, scale=.25).compile(inp, inp, out, config=config)
             if dtype == torch.bfloat16 else None)
        b = make_element((32, 128), kind, scale=.25,
                         input_fp32=(dtype == torch.float32,) * 2).compile(inp, inp, out, config=config)
        elements.append((kind, dtype, a, b)); programs += [b] + ([a] if a is not None else [])
    fp32_stats = []
    for factory in (old_optimizer.make_stats, make_stats):
        entry, count = factory((1, 8192))
        entry = entry.compile(examples((1, 8192), torch.float32), examples((count, 8), torch.float32),
                              examples((count, 8), torch.float32), config=config)
        fp32_stats.append(entry); programs.append(entry)
    copy_cases = []
    for name in ("l60_dt_bias", "l60_norm_weight"):
        merge = LoRAMergeGraph(suffix)
        g = section(merge, {name: merge.targets[name]}, set(merge.inputs))
        p = TrainingPrograms(g, config, combined=True)
        copy_cases.append((g, p)); programs += list(p.programs.values())

    report = dict(completed=False, passed=False, clean_shutdown=False,
        scope="Single-card projection primitives from actual LoRA graphs; not online serving or throughput acceptance",
        compile_only=args.compile_only, projections=[], element_regressions=[], fp32_stat_regressions=[], copies=[])
    if args.compile_only:
        report.update(completed=True, compiled=True, program_count=len(programs),
                      compiled_paths=[str(p.output_dir) for p in programs])
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return

    worker, handles, spaces = None, [], []
    def alloc(tensor):
        tensor = tensor.detach().contiguous()
        handle = worker.alloc_tensor(tuple(tensor.shape), tensor.dtype, init=tensor)
        handles.append(handle)
        return handle
    def read(handle, shape, dtype=torch.float32):
        host = torch.empty(shape, dtype=dtype)
        worker.copy_from(host.data_ptr(), handle.data_ptr, host.nbytes)
        return host
    def run_space(space, bindings):
        return space.run(bindings, past=0, width=32, valid=31)
    def timings(call):
        for _ in range(3): call()
        raw = []
        for _ in range(10):
            start = time.monotonic_ns(); call(); raw.append(time.monotonic_ns() - start)
        return dict(synchronous_call_ns=raw, median_ns=median(raw))
    def clear():
        while spaces: spaces.pop().close()
        while handles: worker.free_tensor(handles.pop())
    try:
        worker = DistributedWorker(programs, config=config)
        for case in compiled:
            p, adapter = case["adapter"].projection, case["adapter"]
            rows, columns = p.shape
            base = torch.randn(p.shape).mul_(.02).bfloat16()
            x = torch.randn((32, columns)).mul_(.2).bfloat16()
            dy = torch.randn((32, rows)).mul_(.01).bfloat16()
            a = torch.empty(adapter.a.shape).uniform_(-columns ** -.5, columns ** -.5).bfloat16()
            b = torch.randn(adapter.b.shape).mul_(.003).bfloat16()
            bindings = {case["x"].name: alloc(x), case["dy"].name: alloc(dy),
                        adapter.a.name: alloc(a), adapter.b.name: alloc(b)}
            base_device = alloc(base)
            vp, lp, mp = [TrainingWorkspace(case[k], worker) for k in ("vp", "lp", "mp")]
            spaces += [vp, lp, mp]
            legacy_bindings = {alias: bindings[name] for alias, name in case["legacy"].aliases.items()}
            merge_bindings = {alias: base_device for alias in case["merge"].base_inputs
                              if alias in case["merged"].inputs}
            merge_bindings.update({alias: bindings[name] for alias, name in case["merge"].optimizer_inputs.items()
                                   if alias in case["merged"].inputs})
            # 合并在反向完成后借用已无读者的 dense dW，不复制或覆盖 base W。
            delta_name = case["merge"].scratch[p.name].name
            retired = mp.values[delta_name]
            mp.handles = [h for h in mp.handles if h is not retired]
            worker.free_tensor(retired)
            mp.values[delta_name] = vp.values[case["dw"].name]
            y_device = alloc(torch.empty((32, rows), dtype=torch.bfloat16))
            old_dw_device = alloc(torch.empty(p.shape, dtype=torch.float32))
            sums = alloc(torch.empty((case["stat_rows"], 8)))
            flags = alloc(torch.empty((case["stat_rows"], 8)))
            worker.run(case["old_dw"], bindings[case["dy"].name], bindings[case["x"].name], old_dw_device, config=config)
            old_gradient = read(old_dw_device, p.shape)
            for zero_b in (True, False):
                bv = torch.zeros_like(b) if zero_b else b
                worker.copy_to(bindings[adapter.b.name].data_ptr, bv.data_ptr(), bv.nbytes)
                am, bm = a.float().requires_grad_(), bv.float().requires_grad_()
                quant = lambda value: value.bfloat16().float()
                merged_expected = quant(base.float() + quant(quant(bm) @ quant(am)))
                y_expected = quant(x.float() @ merged_expected.T)
                expected = torch.autograd.grad((y_expected * dy.float()).sum(), (am, bm))
                run_space(vp, bindings)
                assert torch.equal(read(vp.values[case["dw"].name], p.shape), old_gradient)
                actual = [read(vp.values[case["vjp"].gradients[param.name].name], param.shape)
                          for param in (adapter.a, adapter.b)]
                run_space(lp, legacy_bindings)
                alternative = [read(lp.values[case["legacy"].gradients[param.name].name], param.shape)
                               for param in (adapter.a, adapter.b)]
                assert all(torch.equal(u, v) for u, v in zip(actual, alternative))
                errors = []
                for observed, reference in zip(actual, expected):
                    if torch.count_nonzero(reference) == 0:
                        assert torch.count_nonzero(observed) == 0
                        errors.append(0.)
                    else:
                        relative = float((observed - reference).norm() / reference.norm())
                        assert relative < .02 and float((observed - reference).abs().max()) < .001
                        errors.append(relative)
                run_space(mp, merge_bindings)
                merged_device = mp.values[case["merged"].gradients[p.name].name]
                result = read(merged_device, p.shape, torch.bfloat16)
                reference = merged_expected.detach().bfloat16()
                relative = float((result.float() - reference.float()).norm() / reference.float().norm())
                assert relative < .003 and float((result.float() - reference.float()).abs().max()) < .001
                if zero_b: assert torch.equal(result, base)
                assert torch.equal(read(base_device, p.shape, torch.bfloat16), base)
                worker.run(case["forward"], bindings[case["x"].name], merged_device, y_device, config=config)
                actual_y = read(y_device, y_expected.shape, torch.bfloat16).float()
                torch.testing.assert_close(actual_y, y_expected.detach(), rtol=.02, atol=.005)
                worker.run(case["stat"], merged_device, sums, flags, config=config)
                assert bool((read(flags, (case["stat_rows"], 8)) == 1).all())
                row = dict(name=p.name, shape=p.shape, tail_case=case["index"] == 6, zero_b=zero_b,
                    gradient_relative_errors=errors, merge_relative_error=relative,
                    zero_b_initial_W_exact=zero_b and torch.equal(result, base), base_unchanged=True,
                    adaptive_legacy_gradients_bitwise=True, default_dense_DW_bitwise=True)
                if not zero_b:
                    row["adaptive_vjp"] = timings(lambda: run_space(vp, bindings))
                    row["legacy_vjp"] = timings(lambda: run_space(lp, legacy_bindings))
                    # 确认缺省恢复后正常，再注入私有候选；不改变 base 或任何服务权重。
                    if case["index"] == 6:
                        rejected = []
                        for bad in (float("nan"), float("inf"), -float("inf")):
                            sentinel = torch.full((1, 64), bad, dtype=torch.bfloat16)
                            worker.copy_to(merged_device.data_ptr, sentinel.data_ptr(), sentinel.nbytes)
                            worker.run(case["stat"], merged_device, sums, flags, config=config)
                            assert not bool((read(flags, (case["stat_rows"], 8)) == 1).all())
                            rejected.append(str(bad))
                        row["nonfinite_private_candidates_detected"] = rejected
                report["projections"].append(row)
            clear()

        for kind, dtype, before, after in elements:
            a, b = torch.randn(32, 128, dtype=dtype), torch.randn(32, 128, dtype=dtype)
            da, db = alloc(a), alloc(b)
            outputs = [alloc(torch.empty((32, 128), dtype=torch.bfloat16)) for _ in range(2)]
            worker.run(after, da, db, outputs[1], config=config)
            observed = read(outputs[1], (32, 128), torch.bfloat16)
            if before is not None:
                worker.run(before, da, db, outputs[0], config=config)
                assert torch.equal(read(outputs[0], (32, 128), torch.bfloat16), observed)
            else:
                expected = a + b if kind == "add" else a * .25
                assert torch.equal(observed, expected.bfloat16())
            report["element_regressions"].append(dict(kind=kind, input_dtype=str(dtype),
                default_bitwise=before is not None, independent_formula=before is None))
            clear()
        for bad in (None, float("nan"), float("inf"), -float("inf")):
            value = torch.randn(1, 8192)
            if bad is not None: value[0, 0] = bad
            src = alloc(value); outputs = [(alloc(torch.empty(2, 8)), alloc(torch.empty(2, 8))) for _ in range(2)]
            for program, pair in zip(fp32_stats, outputs): worker.run(program, src, *pair, config=config)
            for column in range(2):
                torch.testing.assert_close(read(outputs[0][column], (2, 8)), read(outputs[1][column], (2, 8)),
                                           rtol=0, atol=0, equal_nan=True)
            report["fp32_stat_regressions"].append(dict(value=str(bad), default_bitwise=True)); clear()
        for graph, program in copy_cases:
            space = TrainingWorkspace(program, worker); spaces.append(space)
            value = next(iter(graph.inputs.values())); host = torch.randn(value.shape).bfloat16()
            run_space(space, {value.name: alloc(host)})
            target = next(iter(graph.gradients.values()))
            assert torch.equal(read(space.values[target.name], target.shape, torch.bfloat16), host)
            report["copies"].append(dict(shape=target.shape, exact=True)); clear()
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            try:
                clear()
                worker.close()
                report["clean_shutdown"] = True
            except BaseException as cleanup:
                report["cleanup_error"] = repr(cleanup)
        report["completed"] = report["passed"] and report["clean_shutdown"]
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
