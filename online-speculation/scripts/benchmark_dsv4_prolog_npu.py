"""Measure existing Q/KV operators on loaded DSV4 weights via collective_rpc."""

import statistics
import time

import torch


def benchmark_on_worker(worker):
    from blockspec.vllm_ascend.dsv4 import _qkv_readonly, _query_readonly

    rows = []
    norm_probes = []
    for layer_index, layer in enumerate(worker.model_runner.model.model.layers):
        impl = layer.self_attn.dsa_attn.dsa_attn.impl
        for batch in (1, 2, 4):
            tokens = batch * 32
            generator = torch.Generator(device="npu").manual_seed(184 + layer_index)
            x = torch.randn(tokens, layer.hidden_size, dtype=torch.bfloat16, device="npu", generator=generator)
            angles = torch.randn(tokens, 1, 1, impl.rope_head_dim, device="npu", generator=generator)
            cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

            def separate():
                q, qr, scale = _query_readonly(impl, x, cos, sin)
                kv = impl.kv_norm(impl.wkv(x)).view(-1, 1, impl.head_dim)
                torch.ops._C_ascend.inplace_partial_rotary_mul(
                    kv.unsqueeze(1), cos, sin, rotary_mode="interleave",
                    partial_slice=[impl.nope_head_dim, impl.head_dim],
                )
                return q, qr, scale, kv

            def shared():
                return _qkv_readonly(impl, x, cos, sin)

            with torch.inference_mode():
                q_a = impl.wq_a(x)
                separated, separated_scale = impl.cv_wq_b.quantize(impl.q_norm(q_a))
                fused, fused_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                    q_a, impl.q_norm.weight, epsilon=impl.eps)
                norm_probes.append({"layer": layer_index, "tokens": tokens,
                                    "int8_different": torch.count_nonzero(separated != fused).item(),
                                    "int8_elements": separated.numel(),
                                    "scale_max_abs": (separated_scale - fused_scale).abs().max().item()})
                before, after = separate(), shared()
                equal = [torch.equal(a, b) if a is not None else b is None
                         for a, b in zip(before, after, strict=True)]
                assert all(equal), (layer_index, batch, equal)
                for _ in range(10):
                    separate()
                    shared()
                timings = {"separate": [], "shared": []}
                for repeat in range(8):
                    # Alternate the order to reduce warmup/clock bias.
                    funcs = [("separate", separate), ("shared", shared)]
                    for name, fn in funcs[::1 if repeat % 2 == 0 else -1]:
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        for _ in range(20):
                            fn()
                        torch.npu.synchronize()
                        timings[name].append((time.perf_counter() - start) * 1000 / 20)
            rows.append({"layer": layer_index, "ratio": impl.compress_ratio,
                         "batch": batch, "tokens": tokens, "heads": impl.n_local_heads,
                         "bitwise_equal": equal,
                         "median_ms": {name: statistics.median(values) for name, values in timings.items()},
                         "samples_ms": timings})
    return {"rank": worker.rank, "cases": rows, "norm_quant_probe": norm_probes}
