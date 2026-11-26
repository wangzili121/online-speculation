#!/usr/bin/env python3
"""Check the staged draft SWA and TP4 Attention output with a real W8A8 layer."""

import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace
import json

import torch
from blockspec.dsv41_packed_proposer import draft_geometry
from blockspec.pypto_dsv41_draft import pack_draft_input_projection, pack_draft_output_projection, prepare_output
from blockspec.pypto_dsv41_packed_graph import make_packed_draft_graph
from blockspec.pypto_dsv41_packed_history import packed_history_lookup
from blockspec.pypto_dsv41_packed_swa import packed_swa_controls
from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE, set_moe_tokens
from models.deepseek_v4_1_flash_a3.device_views import resident_graph_argument
from models.deepseek_v4_1_flash_a3.hc import golden_hc_collapse, golden_hc_post, golden_mhc_mixes
from models.deepseek_v4_1_flash_a3.weights import Checkpoint, bf16_nz_shape, forward_weight_layouts
from pypto.backend import BackendType
from pypto.ir import DistributedConfig
from pypto.runtime import ExecutionMode, RunConfig
from pypto.runtime.distributed_runner import DistributedWorker
from validate_dsv41_pypto_packed_layer import reference
from validate_dsv41_pypto_packed_qkv import checkpoint_operands

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    args = parser.parse_args()
    devices = list(map(int, args.devices.split(",")))
    if len(devices) != TP_SIZE or len(set(devices)) != TP_SIZE:
        parser.error("Use one complete group of four distinct devices")

    torch.set_num_threads(4)
    set_moe_tokens(16)
    ranks = tuple(range(TP_SIZE))
    block, context = 16, 128
    chk = Checkpoint(args.checkpoint)
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    graph = make_packed_draft_graph(*draft_geometry(context), block, staged_ranks=TP_SIZE).compile(config=config)
    parameters = graph._get_metadata()[0]
    layouts = forward_weight_layouts()
    prefix = "layers.0."
    copies, dim = FLASH.hc_mult, FLASH.head_dim
    swa = packed_swa_controls((0,), 0, draft_geometry(context)[0])
    norm = chk.tensor(prefix + "attn_norm.weight")
    fn, scale, base = [chk.tensor(prefix + "hc_attn_" + name).float() for name in ("fn", "scale", "base")]
    tensors = {key: chk.tensor(prefix + "attn." + key) for key in ("wo_a.weight", "wo_b.weight")}
    sink = chk.tensor(prefix + "attn.attn_sink").float().reshape(TP_SIZE, -1)
    empty = SimpleNamespace(gather=lambda rows: torch.zeros(*rows.shape, dim))
    empty_history = empty, empty
    selected = torch.full((block, FLASH.index_topk), -1, dtype=torch.int32)
    dtype = {k: getattr(torch, v) for k, v in
             (("bfloat16", "bfloat16"), ("int32", "int32"), ("uint8", "uint8"), ("fp32", "float32"))}
    contributions = []

    with DistributedWorker([graph], config, persistent=True, reset_persistent_windows=False) as rt:
        handles = [dict() for _ in ranks]
        for rank in ranks:
            embedding, operands = checkpoint_operands(chk.root, 0, rank)
            wo = prepare_output(tensors, asdict(FLASH), tp_rank=rank, tp=TP_SIZE)
            packed = pack_draft_output_projection(wo)
            w_a, w_b = packed["wo_a"], packed["wo_b"]
            x = embedding[:block, None].expand(-1, copies, -1).float().contiguous()
            incoming = torch.zeros(block, copies)
            incoming[:, 0] = 1
            cos = torch.ones(2, block, FLASH.qk_rope_head_dim // 2)
            sin = torch.zeros_like(cos)
            values = dict(
                x_hc=x, pre_mix=incoming, attn_norm_weight=norm,
                hc_attn_fn=fn, hc_attn_scale=scale, hc_attn_base=base,
                draft_q_a_kv=pack_draft_input_projection(operands),
                draft_q_gamma=operands["q_norm"], draft_q_b=operands["wq_b"], draft_kv_gamma=operands["kv_norm"],
                draft_sink=sink[rank], draft_wo_a=w_a.reshape(bf16_nz_shape(*w_a.shape)),
                draft_wo_b=w_b.reshape(bf16_nz_shape(1, *w_b.shape)),
                draft_cos=cos, draft_sin=sin, count_meta=torch.tensor([block, rank], dtype=torch.int32),
                window_lookup=packed_history_lookup("fp8"), compressed_lookup=packed_history_lookup("fp4"),
                **dict(zip(("window_rows", "window_starts", "window_spans", "window_bias"), swa, strict=True)),
            )
            contributions.append(reference(x, norm, operands, wo, (cos[0], sin[0]), empty_history,
                                           selected, sink[rank], 0).bfloat16().float())
            for param in parameters[:-3]:
                name = param.name.removesuffix("__ssa_v0")
                if name in layouts:
                    shape, dt = layouts[name]
                elif name in ("draft_wo_a", "draft_wo_b") and len(param.shape) == 5:
                    shape, dt = (param.shape[0] // TP_SIZE, *param.shape[1:]), dtype[str(param.dtype)]
                else:
                    shape, dt = param.shape[1:], dtype[str(param.dtype)]
                handle = rt.alloc_tensor(shape, dt, worker_id=rank)
                handles[rank][name] = handle
                if name in values:
                    host = values[name].to(dt).contiguous()
                    rt.copy_to(handle.data_ptr, host.data_ptr(), host.nbytes, worker_id=rank)
                elif name in ("zero_kv", "window_cache_pool", "window_cache_scale_pool"):
                    host = torch.zeros(shape, dtype=dt)
                    rt.copy_to(handle.data_ptr, host.data_ptr(), host.nbytes, worker_id=rank)

        args = [resident_graph_argument([row[p.name.removesuffix("__ssa_v0")] for row in handles], ranks, p)
                for p in parameters[:-3]]
        rt.run(graph, *args, 0, -1, 0)

        def read(name, rank):
            d = handles[rank][name]
            host = torch.empty(d.shape, dtype=d.dtype)
            rt.copy_from(host.data_ptr(), d.data_ptr, host.nbytes, worker_id=rank)
            return host

        times, previous = [], None
        for _ in range(5):
            start = perf_counter()
            rt.run(graph, *args, 0, -1, 1)
            times.append((perf_counter() - start) * 1000)
            result = tuple(read("h_attention", rank) for rank in ranks)
            assert all(torch.equal(result[0], other) for other in result[1:])
            if previous is not None:
                assert torch.equal(previous, result[0])
            previous = result[0]
        pre, post, residual = golden_mhc_mixes(x, fn, scale, base)
        expected = golden_hc_post(sum(contributions).bfloat16().float(), x, post, residual)
        error = (previous - expected).norm() / expected.norm().clamp_min(1e-20)
        assert error < 0.005, error
        actual_pre = read("pre_attention", 0)
        torch.testing.assert_close(actual_pre, pre, atol=2e-5, rtol=2e-4)
        actual_ffn = read("ffn_input", 0)
        reference_ffn = golden_hc_collapse(expected, pre).bfloat16()
        ffn_error = (actual_ffn.float() - reference_ffn.float()).norm() / reference_ffn.float().norm()
        assert ffn_error < 0.005, ffn_error

    print(json.dumps(dict(passed=True, block=block, layer=0, tp=TP_SIZE,
                          devices=devices, repeats=len(times), npu_execution=True, cross_rank_exact=True,
                          attention_relative_l2=error.item(), ffn_relative_l2=ffn_error.item(),
                          warm_median_ms=median(times[1:]))), flush=True)


if __name__ == "__main__":
    main()
