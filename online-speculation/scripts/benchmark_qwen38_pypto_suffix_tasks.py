"""Run the combined real-shape last-1 training graph with two dW task schedules."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker, RunConfig
from blockspec.pypto_qwen38.training_executor import TrainingPrograms, TrainingWorkspace
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph


def run(path):
    torch.set_num_threads(4)
    torch.manual_seed(211)
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
    baseline = SuffixGraph(1)
    tuned = SuffixGraph(1)
    gradients = {tuned.gradients[key] for key in ("l63_q_weight", "l63_o_weight")}
    tuned.nodes = [
        replace(node, options={**node.options, "task_group": 32})
        if node.kind == "linear" and gradients.intersection(node.outputs) else node for node in tuned.nodes
    ]
    programs = {g: TrainingPrograms(graph, config, combined=True, external_gradients=True)
                for g, graph in ((16, baseline), (32, tuned))}
    entries = [entry for p in programs.values() for entry in p.programs.values()]
    report = {}
    with DistributedWorker(entries, config=config) as worker:
        workspace = {g: TrainingWorkspace(p, worker) for g, p in programs.items()}
        inputs = {}
        for name, value in baseline.inputs.items():
            dtype = torch.float32 if value.dtype == "fp32" else torch.bfloat16
            source = torch.randn(value.shape, dtype=dtype)
            source.mul_(0.18 if "history" in name or name in ("hidden", "teacher") else .03)
            if "gamma" in name:
                source.add_(1)
            inputs[name] = worker.alloc_tensor(value.shape, dtype, init=source)
        outputs = {g: {n: worker.alloc_tensor(v.shape, torch.float32) for n, v in baseline.gradients.items()}
                   for g in programs}

        def read(handle):
            cpu = torch.empty(handle.shape, dtype=handle.dtype)
            worker.copy_from(cpu.data_ptr(), handle.data_ptr, cpu.nbytes)
            return cpu

        for past, width, valid in [(173, 8, 7), (480, 32, 31)]:

            def execute(group):
                start = time.perf_counter_ns()
                workspace[group].run(inputs, past=past, width=width, valid=valid, gradients=outputs[group])
                return (time.perf_counter_ns() - start) / 1000

            for _ in range(3):
                for g in programs:
                    execute(g)
            for name in baseline.gradients:
                left, right = (read(outputs[g][name]) for g in programs)
                assert torch.isfinite(left).all(), name
                torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)
            for name in ("logits", "losses"):
                left, right = (read(workspace[g].values[getattr(baseline, name).name]) for g in programs)
                assert torch.isfinite(left).all(), name
                torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)
            samples = {g: [] for g in programs}
            for n in range(12):
                for g in ((16, 32) if n % 2 == 0 else (32, 16)):
                    samples[g].append(execute(g))
            report[str(width)] = {
                "past": past, "valid_rows": valid, "bitwise_equal_gradients": sorted(baseline.gradients),
                "logits_and_loss_exact": True, "samples_us": samples,
                "median_us": {g: statistics.median(t) for g, t in samples.items()},
            }
        for w in workspace.values():
            w.close()
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
