#!/usr/bin/env python3
"""Check shared-expert input gradients against the native W8A8 forward reference."""

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
    from blockspec.dsv41_packed_expert_vjp import packed_expert_vjp_reference
    from blockspec.dsv41_packed_shared_expert import make_packed_shared_expert_vjp
    from models.deepseek_v4_1_flash_a3.config import FLASH, SHARED_NZ_PANEL
    from models.deepseek_v4_1_flash_a3.expert_shared import golden_expert_shared
    from models.deepseek_v4_1_flash_a3.gemm import golden_activation_int8
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights, pack_projection_int8_nz
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    rows, hidden, inter = 16, FLASH.hidden_size, FLASH.moe_intermediate_size
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    graph = make_packed_shared_expert_vjp(rows, hidden, inter, FLASH.swiglu_limit).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, executed=False)))
        return
    weights = ResidentWeights(Checkpoint(args.checkpoint))
    generator = torch.Generator().manual_seed(87103)
    cases = []
    with DistributedWorker([graph], config, persistent=True) as runtime:
        def upload(tensor):
            return runtime.alloc_tensor(tensor.shape, tensor.dtype, init=tensor, worker_id=0)

        for layer, active_counts in ((39, (1, 7)), (36, (7,))):
            stem = f"layers.{layer}.ffn.shared_experts"
            w1, s1 = weights.projection(stem + ".w1", output_major=True)
            w3, s3 = weights.projection(stem + ".w3", output_major=True)
            w2, s2 = weights.projection(stem + ".w2", output_major=True)
            paired = torch.stack((w1.reshape(-1, SHARED_NZ_PANEL, hidden),
                                  w3.reshape(-1, SHARED_NZ_PANEL, hidden)), dim=1).reshape(-1, hidden).T
            bank = [upload(value) for value in (pack_projection_int8_nz(paired).reshape(-1, 32), s1, s3,
                                                pack_projection_int8_nz(w2.T).reshape(-1, 32), s2)]
            value = torch.randn(rows, hidden, dtype=torch.bfloat16, generator=generator)
            gradient = torch.randn(rows, hidden, generator=generator) * .1
            dx = torch.zeros_like(gradient)
            x_device, d_device, dx_device = [upload(value) for value in (value, gradient, dx)]
            for active in active_counts:
                times = []
                for _ in range(8):
                    start = perf_counter()
                    runtime.run(graph, x_device, d_device, *bank, dx_device, active, config=config)
                    times.append((perf_counter() - start) * 1000)
                runtime.copy_from(dx.data_ptr(), dx_device.data_ptr, dx.nbytes, worker_id=0)

                x = value[:active]
                exact, reference, _ = packed_expert_vjp_reference(x, gradient[:active], torch.ones(active, 1),
                                                                 w1, s1, w3, s3, w2, s2, FLASH.swiglu_limit)
                qx = golden_activation_int8(x)
                scale = (x.float().abs().amax(-1, keepdim=True) * (1 / 127)).clamp_min(1e-30)
                output = golden_expert_shared(qx, scale, w1.T, s1, w3.T, s3, w2.T, s2)
                torch.testing.assert_close(exact, output.float(), atol=0, rtol=0)
                delta = (reference - dx[:active]).double()
                error = float(delta.norm() / reference.double().norm())
                assert error < .006 and not torch.count_nonzero(dx[active:]), (layer, active, error)
                cases.append(dict(layer=layer, active_rows=active, native_golden_byte_exact=True,
                                  relative_l2=error, max_abs=float(delta.abs().max()), warm_median_ms=median(times[1:])))
    print(json.dumps(dict(passed=True, compiled=True, executed=True, checkpoint=str(args.checkpoint), cases=cases)),
          flush=True)


if __name__ == "__main__":
    main()
