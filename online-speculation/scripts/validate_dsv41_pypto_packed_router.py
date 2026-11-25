#!/usr/bin/env python3
"""Check the fixed-Top-K router input VJP with a native W8A8 checkpoint."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None and not args.compile_only:
        parser.error("Pass the real W8A8 checkpoint for numerical validation")

    import torch
    import torch.nn.functional as F
    from blockspec.dsv41_packed_router_vjp import make_packed_router_vjp
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.gate import golden_gate_reference
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    rows, hidden, experts = 8, FLASH.hidden_size, FLASH.n_routed_experts
    topk, temperature, route_scale = FLASH.num_experts_per_tok, FLASH.gate_temperature, FLASH.routed_scaling_factor
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    graph = make_packed_router_vjp(rows, hidden, experts, topk, temperature, route_scale).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, executed=False)))
        return
    model = Checkpoint(args.checkpoint)
    w = model.tensor("layers.39.ffn.gate.weight").float()
    bias = model.tensor("layers.39.ffn.gate.bias")
    generator = torch.Generator().manual_seed(39128)
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, generator=generator)
    _, selected = golden_gate_reference(x, w, bias)
    indices = torch.full((rows, 16), -1, dtype=torch.int32)
    indices[:, :topk] = selected
    route_grad = torch.randn(rows, 8, generator=generator)
    routed_grad = torch.randn(rows, hidden, generator=generator) * .001
    cases = []
    with DistributedWorker([graph], config, persistent=True) as runtime:
        def upload(tensor):
            return runtime.alloc_tensor(tensor.shape, tensor.dtype, init=tensor, worker_id=0)

        bank, normalized, ids, droute = [upload(tensor) for tensor in (w, x, indices, route_grad)]
        result = torch.empty_like(routed_grad)
        device_result = upload(result)
        for active, source_grad in ((1, torch.zeros_like(routed_grad)), (7, routed_grad)):
            grad = upload(source_grad)
            samples = []
            for _ in range(9):
                start = perf_counter()
                runtime.run(graph, normalized, ids, droute, bank, grad, device_result, active, config=config)
                samples.append((perf_counter() - start) * 1000)
            runtime.copy_from(result.data_ptr(), device_result.data_ptr, result.nbytes, worker_id=0)
            replay = x[:active].double().requires_grad_(True)
            logits = ((replay @ w.double().T) / temperature).gather(1, selected[:active].long())
            score = F.softplus(logits).sqrt()
            weight = route_scale * score / (score.sum(1, keepdim=True) + 1e-20)
            expected, = torch.autograd.grad(weight, replay, route_grad[:active, :topk].double())
            observed = result[:active].double() - source_grad[:active]
            delta = observed - expected
            error = float(delta.norm() / expected.norm())
            assert error < 1e-4 and not torch.count_nonzero(result[active:]), (active, error)
            cases.append(dict(active_rows=active, relative_l2=error, max_abs=float(delta.abs().max()),
                              warm_median_ms=median(samples[1:])))
    print(json.dumps(dict(passed=True, compiled=True, executed=True, checkpoint=str(args.checkpoint),
                          checkpoint_layer=39, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
