#!/usr/bin/env python3
"""Check the packed routed-expert VJP with native A3 W8A8 checkpoint matrices."""

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
    if not args.compile_only and args.checkpoint is None:
        parser.error("Pass a W8A8 checkpoint to run numerical validation")

    import torch
    from blockspec.dsv41_packed_expert_vjp import make_packed_expert_vjp, packed_expert_vjp_reference
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.expert_routed import golden_expert_routed
    from models.deepseek_v4_1_flash_a3.gemm import golden_activation_int8
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    torch.set_num_threads(4)
    configure_logging("warn")
    rows, hidden, intermediate, limit = 16, FLASH.hidden_size, FLASH.moe_intermediate_size, FLASH.swiglu_limit
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    graph = make_packed_expert_vjp(rows, hidden, intermediate, limit).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, executed=False)))
        return

    generator = torch.Generator().manual_seed(57131)
    weights = ResidentWeights(Checkpoint(args.checkpoint))
    cases = []
    with DistributedWorker([graph], config, persistent=True) as runtime:
        def upload(value):
            return runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=0)

        dx = torch.zeros(rows, hidden)
        dr = torch.zeros(rows, 16)
        outputs = [upload(item) for item in (dx, dr)]
        for layer, expert, active in ((39, 0, 1), (39, 247, 7)):
            stem = f"layers.{layer}.ffn.experts.{expert}"
            projections = [weights.projection(f"{stem}.{name}", output_major=True) for name in ("w1", "w3", "w2")]
            matrices = [item for pair in projections for item in pair]
            x = torch.randn((rows, hidden), dtype=torch.bfloat16, generator=generator)
            dy = torch.randn((rows, hidden), generator=generator) * .2
            route = torch.zeros(rows, 16)
            route[:active, :1] = torch.linspace(.05, .7, active).reshape(-1, 1)
            if active > 1:
                route[0, 0] = 0
            expected_output, expected_dx, expected_dr = packed_expert_vjp_reference(
                x[:active], dy[:active], route[:active, :1], *matrices, limit=limit)
            quantized = golden_activation_int8(x[:active])
            scale = x[:active].float().abs().amax(-1, keepdim=True).mul(1 / 127).clamp_min(1e-30)
            native = golden_expert_routed(quantized, scale, route[:active, :1],
                                          *(part for weight, scales in projections for part in (weight.T, scales)))
            torch.testing.assert_close(expected_output, native.float(), atol=0, rtol=0)
            device_inputs = [upload(obj) for obj in (x, dy, route, *matrices)]
            elapsed = []
            for _ in range(8):
                start = perf_counter()
                runtime.run(graph, *device_inputs, *outputs, active, config=config)
                elapsed.append((perf_counter() - start) * 1000)
            for dest, source in zip((dx, dr), outputs, strict=True):
                runtime.copy_from(dest.data_ptr(), source.data_ptr, dest.nbytes, worker_id=0)
            assert not torch.count_nonzero(dx[active:]) and not torch.count_nonzero(dr[active:, 0])
            if active > 1:
                assert not torch.count_nonzero(dx[0]) and dr[0, 0] != 0
            metrics = {}
            for name, observed, expected in (("expert_input", dx[:active], expected_dx),
                                              ("route_weight", dr[:active, :1], expected_dr)):
                delta = (observed - expected).double()
                rel = float(delta.norm() / expected.double().norm())
                metrics[name] = dict(relative_l2=rel, max_abs=float(delta.abs().max()))
                assert rel < .006, (layer, expert, name, metrics[name])
            cases.append(dict(layer=layer, expert=expert, active_rows=active, native_golden_byte_exact=True,
                              warm_median_ms=median(elapsed[1:]), metrics=metrics))
    print(json.dumps(dict(passed=True, compiled=True, executed=True, checkpoint=str(args.checkpoint), cases=cases)))


if __name__ == "__main__":
    main()
