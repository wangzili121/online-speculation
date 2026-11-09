#!/usr/bin/env python3
"""在 NPU 上整理训练 head，并保存可直接加载的 PyPTO 推理 bank。"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--head-dir", type=Path)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--runtime-tp", type=int, choices=(1, 2, 4), default=1)
    p.add_argument("--runtime-rank", type=int, default=0)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--mask-token-id", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--online-checkpoint", type=Path, help="export committed last-1 state for fixed-head inference")
    a = p.parse_args()
    if a.output_dir.exists() or a.device < 0:
        p.error("new output directory and assigned device required")
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.prepared_head import save
    runtime_identity(pypto)
    head = HeadSource(a.model, a.head_dir, block_size=a.block_size, mask_token_id=a.mask_token_id,
                      runtime_tp=a.runtime_tp, runtime_rank=a.runtime_rank)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
    head.prepare_on_device(config)
    if a.online_checkpoint is not None:
        step, manifest = export_online(head, a.online_checkpoint, a.output_dir, config)
        print(json.dumps(dict(online_step=step, output=str(a.output_dir), manifest=manifest), indent=2))
        return
    manifest = save(head, a.output_dir)
    print(json.dumps(dict(output=str(a.output_dir), seconds=head.preparation_seconds, manifest=manifest), indent=2))


def export_online(head, checkpoint, output, config):
    import torch
    from pypto.runtime import DistributedWorker
    from blockspec.pypto_qwen38.online_checkpoint import identity, read
    from blockspec.pypto_qwen38.optimizer_pipeline import make_pipeline
    from blockspec.pypto_qwen38.prepared_head import save
    from blockspec.pypto_qwen38.training_graph import LastLayerGraph
    shapes = {name: value.shape for name, value in LastLayerGraph().gradients.items()}
    step, values = read(checkpoint, identity(head), shapes)
    meta, types = [], {}
    for name, shape in shapes.items():
        types[name] = torch.float32 if "gamma" in name else torch.bfloat16
        meta += [torch.empty(shape, dtype=torch.float32, device="meta"),
                 torch.empty(shape, dtype=types[name], device="meta")]
    program = make_pipeline(shapes, update=False).compile(*meta, config=config)
    worker, handles, execution = None, [], {}
    try:
        worker = DistributedWorker([program], config=config)
        arguments = []
        for name, shape in shapes.items():
            master = worker.alloc_tensor(shape, torch.float32, init=values[name+".master"])
            handles.append(master)
            execution[name] = worker.alloc_tensor(shape, types[name])
            handles.append(execution[name])
            arguments += [master, execution[name]]
        worker.run(program, *arguments, config=config)
        banks = tuple(execution[name] for name in ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight"))
        return step, save(head, output, last_layer=banks, worker=worker)
    finally:
        if worker is not None:
            for handle in reversed(handles):
                worker.free_tensor(handle)
            worker.close()


if __name__ == "__main__":
    main()
