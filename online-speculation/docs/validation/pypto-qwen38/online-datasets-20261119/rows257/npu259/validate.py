#!/usr/bin/env python3
"""使用真实 27B 权重对照 last-1 的 32/16 行完整前反向。输入激活和教师为合成数据。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")

    import torch
    from safetensors import safe_open
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, configure_log
    from pypto.runtime.task_interface import DataType
    from serving.weight import TextWeights
    from blockspec.pypto_qwen38.oneserving_tp2_online import read_raw_parameter
    from blockspec.pypto_qwen38.training_executor import TrainingPrograms, TrainingWorkspace
    from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
    from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph

    configure_log("warn")
    torch.set_num_threads(4)
    torch.manual_seed(257)
    report = dict(completed=False, stage="compiling", cases=[],
                  scope="Real27B weights/dimensions, synthetic activations/history/teacher. Full last-1 forward/backward; excludes optimizer and publication. Not end-to-end throughput.")

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    worker = None
    workspaces = {}
    try:
        config = RunConfig(platform="a2a3", ring_heap=(512 << 20,) * 4,
                           distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
        shapes = {rows: replace(SuffixShape(), block=rows) for rows in (32, 16)}
        graphs = {(obj, rows): LastLayerGraph(shape, objective=obj)
                  for obj in ("tv", "kl") for rows, shape in shapes.items()}
        programs = {key: TrainingPrograms(graph, config, combined=True)
                    for key, graph in graphs.items()}
        report["compiled_entries"] = sum(len(item.programs) for item in programs.values())
        assert report["compiled_entries"] == 4
        if args.compile_only:
            report.update(completed=True, stage="compiled", npu_execution=False)
            return

        # 所有权重共用同一份设备缓冲。16 行入口只缩短 token 轴的视图，不改变权重或历史。
        metadata = SuffixGraph(1)
        raw = {}
        with safe_open(args.head / "rank0.safetensors", framework="pt", device="cpu") as left, safe_open(
                args.head / "rank1.safetensors", framework="pt", device="cpu") as right:
            for parameter in metadata.parameters:
                value = read_raw_parameter(parameter, (left, right))
                raw[parameter.name.removeprefix("l63_")] = value.float() + 1 if parameter.offset_norm else value
        weights = TextWeights(args.model)
        paths = {"input_gamma": "input_layernorm.weight", "post_gamma": "post_attention_layernorm.weight",
                 "mlp_gate": "mlp.gate_proj.weight", "mlp_up": "mlp.up_proj.weight", "mlp_down": "mlp.down_proj.weight"}
        inputs = {}
        for name, value in graphs["tv", 32].inputs.items():
            if name in raw:
                tensor = raw[name]
            elif name in paths:
                tensor = weights.layer(63, paths[name])
                if name.endswith("gamma"):
                    tensor = (1 + tensor.float()).reshape(value.shape)
            elif name == "lm_head":
                tensor = weights.read("lm_head.weight")
            elif name == "final_gamma":
                tensor = (1 + weights.read("model.language_model.norm.weight").float()).reshape(value.shape)
            elif name in {"cos", "sin"}:
                angle = torch.arange(32)[:, None] * torch.arange(64)[None, :] * .003
                tensor = angle.cos() if name == "cos" else angle.sin()
            else:
                tensor = torch.randn(value.shape) * (.3 if name == "teacher" else .02)
            inputs[name] = tensor.to(torch.float32 if value.dtype == "fp32" else torch.bfloat16).contiguous().share_memory_()
        worker = DistributedWorker([p for item in programs.values() for p in item.programs.values()],
                                   config=config, inherited_host_tensors=list(inputs.values()))
        buffers = {name: worker.alloc_tensor(tuple(t.shape), t.dtype, init=t) for name, t in inputs.items()}
        views = {key: {name: buffers[name].buffer.tensor(value.shape,
                      DataType.FLOAT32 if value.dtype == "fp32" else DataType.BFLOAT16)
                      for name, value in graph.inputs.items()} for key, graph in graphs.items()}
        workspaces = {key: TrainingWorkspace(item, worker) for key, item in programs.items()}
        report["stage"] = "running"
        save()

        def read(handle):
            tensor = torch.empty(handle.shape, dtype=handle.dtype)
            worker.copy_from(tensor.data_ptr(), handle.data_ptr, tensor.nbytes)
            return tensor

        def run(key, past, width, valid):
            result = workspaces[key].run(views[key], past=past, width=width, valid=valid)
            # 读回 loss 作为明确的设备完成点；两种布局的计时使用相同同步方法。
            losses = read(workspaces[key].values[graphs[key].losses.name])
            return result, losses

        for objective in ("tv", "kl"):
            for past, width, valid in ((0, 8, 7), (127, 8, 3), (496, 16, 15)):
                before, old_loss = run((objective, 32), past, width, valid)
                baseline = {name: read(handle) for name, handle in before.items()}
                old_logits = read(workspaces[objective, 32].values[graphs[objective, 32].logits.name])
                after, new_loss = run((objective, 16), past, width, valid)
                new_logits = read(workspaces[objective, 16].values[graphs[objective, 16].logits.name])
                assert torch.equal(new_logits[:width], old_logits[:width]), "active logits differ"
                torch.testing.assert_close(new_loss.sum(0), old_loss.sum(0), atol=1e-6, rtol=1e-5)
                differences = {}
                for name, handle in after.items():
                    old, new = baseline[name], read(handle)
                    assert torch.isfinite(new).all() and torch.count_nonzero(new), name
                    relative = ((new - old).norm() / old.norm().clamp_min(1e-30)).item()
                    differences[name] = dict(relative_l2=relative, max_abs=(new - old).abs().max().item(),
                                             bitwise=torch.equal(new, old))
                    assert relative < .01, (name, relative)
                samples = {32: [], 16: []}
                for _ in range(2):
                    for rows in (32, 16):
                        run((objective, rows), past, width, valid)
                for repetition in range(8):
                    order = [32, 16]
                    random.Random(257 + repetition).shuffle(order)
                    for rows in order:
                        started = time.perf_counter_ns()
                        run((objective, rows), past, width, valid)
                        samples[rows].append((time.perf_counter_ns() - started) / 1e6)
                report["cases"].append(dict(objective=objective, past=past, width=width, valid=valid,
                    active_logits_bitwise=True, gradients=differences,
                    timing_ms={str(rows): dict(median=statistics.median(values), samples=values)
                               for rows, values in samples.items()}))
                save()
        for name in ("hidden", "history_key", "history_value", "teacher", "cos", "sin"):
            assert torch.equal(read(buffers[name]).view(torch.uint8), inputs[name].view(torch.uint8)), name
        report.update(completed=True, stage="completed", npu_execution=True, inputs_byte_exact=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            for workspace in workspaces.values():
                workspace.close()
        finally:
            try:
                if worker is not None:
                    worker.close()
            finally:
                save()


if __name__ == "__main__":
    main()
