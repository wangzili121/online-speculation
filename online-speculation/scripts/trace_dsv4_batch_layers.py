"""Read-only layer tracing for a fixed request/block in Orthrus state tests."""

import json
from pathlib import Path

import torch

from check_dsv4_engine_state import EngineStateWorker
from trace_dsv4_native_layers import LayerTraceWorker, compare_traces


class BatchLayerWorker(EngineStateWorker, LayerTraceWorker):
    def batch_trace_configure(self, directory, block_start, prompt_length, repeat_attention=False, capture_c4=False):
        self._batch_trace = dict(directory=directory, start=block_start,
                                 length=prompt_length, previous=None, paths={},
                                 repeat_attention=repeat_attention, capture_c4=capture_c4)

    def state_probe_phase(self, name, plans=None, skip_head=False):
        config = self._batch_trace
        if config["previous"] is not None:
            phase = config["previous"]
            config["paths"][phase] = self.trace_flush(config["directory"], phase)
            config["previous"] = None
        super().state_probe_phase(name, plans, skip_head)
        if name.startswith("shadow-"):
            self.trace_arm(config["start"], config["length"], config["start"])
            self._layer_trace["repeat_attention"] = config["repeat_attention"]
            self._layer_trace["capture_c4"] = config["capture_c4"]
            config["previous"] = name
        elif hasattr(self, "_layer_trace"):
            self.trace_disable()

    def batch_trace_report(self):
        config = self._batch_trace
        comparisons = {}
        traces = {phase: torch.load(path, weights_only=True) for phase, path in config["paths"].items()}
        pairs = (("reference_repeat", "shadow-token-only-1", "shadow-token-only-2"),
                 ("reference_vs_draft", "shadow-token-only-1", "shadow-real"),
                 ("draft_vs_no_head", "shadow-real", "shadow-no-head"))
        for name, left, right in pairs:
            result = compare_traces(traces[left], traces[right])
            comparisons[name] = result
        report = dict(rank=self.rank, block_start=config["start"], prompt_length=config["length"],
                      paths=config["paths"], comparisons=comparisons)
        if config["repeat_attention"]:
            report["attention_replays"] = {}
            for phase, tensors in traces.items():
                repeats = {key.replace(".repeat_output", ".output"): value
                           for key, value in tensors.items() if ".repeat_output" in key}
                originals = {key: tensors[key] for key in repeats}
                report["attention_replays"][phase] = compare_traces(originals, repeats)
        path = Path(config["directory"]) / f"layer-report-rank{self.rank}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        return report
