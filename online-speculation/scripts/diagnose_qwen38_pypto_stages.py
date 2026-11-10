#!/usr/bin/env python3
"""逐节点采用相同的真实输入，定位第一个局部反向偏差。"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import torch
    from safetensors.torch import load_file
    from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
    from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
    from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
    from blockspec.pypto_qwen38.training_reference import evaluate
    from diagnose_qwen38_pypto_reference import metrics

    torch.set_num_threads(32)
    replay = json.loads(a.replay_report.read_text())
    source = json.loads(Path(replay["capture"]).read_text())
    directory = Path(source["directory"])
    n = source["last_layers"]
    shape = SuffixShape(**source["shape"])
    graph = (
        LastLayerGraph(shape, objective="kl")
        if n == 1
        else SuffixGraph(n, shape, GDNShape(**source["gdn_shape"]), objective="kl")
    )
    inputs = load_file(str(directory / "shared.safetensors"))
    inputs.update(load_file(str(directory / "block-00.safetensors")))
    masters = load_file(str(directory / "initial.master.safetensors"))
    params = {q.name: q for q in graph.parameters} if n > 1 else {}
    for name, master in masters.items():
        offset = params[name].offset_norm if n > 1 else "gamma" in name
        raw = params[name].raw_dtype if n > 1 else "bf16"
        value = master if raw == "fp32" else master.bfloat16().float()
        inputs[name] = value + 1 if offset else value
    inputs = {name: inputs[name].float() for name in graph.inputs}
    meta = source["blocks"][0]
    inputs["teacher"][meta["valid"] :] = (
        0  # 未监督行不参与计算，避免未初始化 padding 干扰诊断。
    )
    actual = load_file(replay["intermediates"])
    observed = {name: tensor.float() for name, tensor in actual.items()}
    expected = evaluate(graph, inputs, **meta, fp32=True, observed=observed)
    entries = []
    for i, node in enumerate(graph.nodes):
        for value in node.outputs:
            x, y = actual[value.name].float(), expected[value.name]
            if node.kind == "kl":
                x, y = x[: meta["valid"]], y[: meta["valid"]]
            item = dict(
                index=i,
                kind=node.kind,
                output=value.name,
                inputs=[v.name for v in node.inputs],
                options=node.options,
                **metrics(x, y),
            )
            entries.append(item)
    report = dict(
        completed=True,
        replay=str(a.replay_report),
        local_same_input=True,
        entries=entries,
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            sorted(entries, key=lambda x: x["relative_l2"], reverse=True)[:10], indent=2
        )
    )


if __name__ == "__main__":
    main()
