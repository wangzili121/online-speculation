#!/usr/bin/env python3
"""Check native W8A8 FFN preparation and TP ownership at the HCCL layer split."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=39)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--tokens", type=int, choices=(16, 64), default=16)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None and not args.compile_only:
        parser.error("Pass the real W8A8 checkpoint for numerical validation")

    import torch
    from blockspec.dsv41_packed_ep_hccl import make_ep_forward_front, make_ep_forward_tp_rows
    from blockspec.dsv41_packed_ep_runner import compile_hccl_moe
    from models.deepseek_v4_1_flash_a3 import config as native
    from pypto.backend import BackendType
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    native.set_moe_tokens(args.tokens)
    shape = native.FLASH
    rows = args.tokens * native.TP_SIZE
    hidden, topk = shape.hidden_size, shape.num_experts_per_tok
    prefill = args.phase == "prefill"
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       device_id=args.device)
    front = make_ep_forward_front(args.tokens, rows, prefill=prefill)
    scatter = make_ep_forward_tp_rows(args.tokens, hidden, native.TP_SIZE, rows)
    report = dict(passed=True, compiled=True, phase=args.phase, tokens=args.tokens, input_rows=rows)
    if args.compile_only:
        report["resident_stages"] = list(compile_hccl_moe(config, tokens=args.tokens, rows=rows))
    else:
        with ChipWorker(config) as worker:
            stages = (front.compile(config=config), scatter.compile(config=config))
            report["cases"] = validate(worker, config, stages, args, native, rows, topk)
    print(json.dumps(report), flush=True)


def validate(worker, config, programs, args, native, rows, topk):
    import torch
    from models.deepseek_v4_1_flash_a3.expert_shared import golden_expert_shared
    from models.deepseek_v4_1_flash_a3.gate import golden_gate_reference
    from models.deepseek_v4_1_flash_a3.gemm import golden_activation_int8
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights, pack_projection_int8_nz

    checkpoint = Checkpoint(args.checkpoint)
    weights = ResidentWeights(checkpoint)
    width, intermediate = native.FLASH.hidden_size, native.FLASH.moe_intermediate_size
    stem = f"layers.{args.layer}"
    norm = checkpoint.tensor(f"{stem}.ffn_norm.weight").bfloat16()
    gate = checkpoint.tensor(f"{stem}.ffn.gate.weight").float()
    bias = checkpoint.tensor(f"{stem}.ffn.gate.bias").float()
    w1, s1 = weights.projection(f"{stem}.ffn.shared_experts.w1", output_major=True)
    w3, s3 = weights.projection(f"{stem}.ffn.shared_experts.w3", output_major=True)
    w2, s2 = weights.projection(f"{stem}.ffn.shared_experts.w2", output_major=True)
    paired = torch.stack((w1.reshape(-1, native.SHARED_NZ_PANEL, width),
                          w3.reshape(-1, native.SHARED_NZ_PANEL, width)), dim=1).reshape(-1, width).T
    packed_gate = pack_projection_int8_nz(paired).reshape(width, 2 * intermediate)
    packed_down = pack_projection_int8_nz(w2.T).reshape(intermediate, width)
    generator = torch.Generator().manual_seed(60139)
    x = torch.randn(rows, width, generator=generator, dtype=torch.bfloat16)
    compact = torch.empty(args.tokens, width, dtype=torch.bfloat16)
    buffers = [compact, torch.empty_like(compact, dtype=torch.int8), torch.empty(1, args.tokens),
               torch.empty(args.tokens, topk, dtype=torch.int32), torch.empty(args.tokens, topk),
               torch.empty_like(compact)]

    def alloc(tensor, upload=False):
        device = worker.alloc_tensor(tensor.shape, tensor.dtype)
        if upload:
            worker.copy_to(device.data_ptr, tensor.data_ptr(), tensor.nbytes)
        return device

    def read(tensor, device):
        worker.copy_from(tensor.data_ptr(), device.data_ptr, tensor.nbytes)

    params = [alloc(value, upload=True) for value in (x, norm, gate, bias, packed_gate, s1, s3, packed_down, s2)]
    outputs = [alloc(value) for value in buffers]
    restored = torch.empty_like(x)
    restored_buffer = alloc(restored)
    tp = native.TP_SIZE
    results = []
    for active, lane in ((1, 0), (1, tp - 1), (17, 2), (rows - 1, tp - 1)):
        worker.run(programs[0], *params, *outputs, active, lane, config=config)
        worker.run(programs[1], outputs[-1], restored_buffer, active, lane, config=config)
        read(restored, restored_buffer)
        slots = torch.arange(lane, max(lane, active), tp)
        expected = torch.zeros_like(restored)
        receipt = dict(active=active, lane=lane, owner_rows=len(slots))
        if len(slots):
            for host, device in zip(buffers, outputs, strict=True):
                read(host, device)
            live = len(slots)
            normalized, payload, scales, ids, routing, shared = buffers
            x_owned = x[slots].float()
            rms = (x_owned * torch.rsqrt(x_owned.square().mean(dim=-1, keepdim=True) + native.FLASH.rms_norm_eps)
                   * norm.float()).bfloat16()
            torch.testing.assert_close(normalized[:live], rms, atol=0.003, rtol=0.008)
            codes = golden_activation_int8(normalized[:live])
            scale = (normalized[:live].float().abs().amax(-1) / 127).clamp_min(1e-30)
            delta_codes = (codes.short() - payload[:live].short()).abs()
            assert delta_codes.max().item() <= 1
            torch.testing.assert_close(scales[0, :live], scale, atol=1e-8, rtol=1e-5)
            reference_route, reference_ids = golden_gate_reference(normalized[:live], gate, bias)
            torch.testing.assert_close(ids[:live], reference_ids, atol=0, rtol=0)
            torch.testing.assert_close(routing[:live], reference_route, atol=6e-5, rtol=1e-4)
            reference_shared = golden_expert_shared(payload[:live], scales[0, :live], w1.T, s1, w3.T, s3, w2.T, s2)
            difference = (shared[:live].float() - reference_shared.float()).abs()
            outliers = (difference > .003 + .03 * reference_shared.float().abs()).float().mean().item()
            assert outliers <= .005
            receipt.update(route_ids_exact=True, quant_max_abs=int(delta_codes.max()),
                           shared_max_abs=difference.max().item(), shared_outlier_ratio=outliers)
            expected[slots] = shared[:live]
        torch.testing.assert_close(restored, expected, atol=0, rtol=0)
        results.append(dict(receipt, tp_restore_exact=True))
    return results


if __name__ == "__main__":
    main()
