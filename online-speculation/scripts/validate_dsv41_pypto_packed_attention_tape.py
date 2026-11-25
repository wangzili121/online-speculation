#!/usr/bin/env python3
"""Check persistent Attention tape capture against the native draft TP post path."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]

    import torch
    import pypto.language as pl
    import pypto.language.distributed as pld
    from blockspec.dsv41_packed_attention_tape import (
        make_packed_attention_post_capture,
        packed_attention_tape_shapes,
    )
    from blockspec.pypto_dsv41_packed_draft_attention import make_packed_draft_tp_post
    from models.deepseek_v4_1_flash_a3.config import FLASH as model, TP_SIZE
    from models.deepseek_v4_1_flash_a3.hc import golden_mhc_mixes
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    tp = TP_SIZE
    if len(set(devices)) != tp:
        parser.error("Use a complete native TP group")
    ranks = tuple(range(tp))
    block, last_layers = 32, 4
    hidden, copies, channels, flat = model.hidden_size, model.hc_mult, model.mix_hc, model.hc_dim
    shapes = packed_attention_tape_shapes(block, last_layers, hidden, copies)
    residual_shape, mixes_shape, delta_shape = (value[0] for value in shapes.values())
    original = make_packed_draft_tp_post(block)
    capture = make_packed_attention_post_capture(block, last_layers, model, tp)

    @pl.jit
    def compare(
        x: pl.Tensor[[block, copies, hidden], pl.FP32], premix: pl.Tensor[[block, copies], pl.FP32],
        partial: pl.Tensor[[block, hidden], pl.BF16], f: pl.Tensor[[channels, flat], pl.FP32],
        s: pl.Tensor[[3], pl.FP32], b: pl.Tensor[[channels], pl.FP32],
        output: pl.Out[pl.Tensor[[block, copies, hidden], pl.FP32]],
        next_mix: pl.Out[pl.Tensor[[block, copies], pl.FP32]],
        collapsed: pl.Out[pl.Tensor[[block, hidden], pl.BF16]],
        tape_residual: pl.Out[pl.Tensor[residual_shape, pl.FP32]],
        tape_mixes: pl.Out[pl.Tensor[mixes_shape, pl.FP32]],
        tape_delta: pl.Out[pl.Tensor[delta_shape, pl.BF16]],
        window: pld.DistributedTensor[[block, hidden], pl.BF16],
        signal: pld.DistributedTensor[[tp, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32], capture_owner: pl.Scalar[pl.INT32],
    ):
        if capture_owner >= 0:
            capture(x, premix, partial, f, s, b, window, signal, rank, epoch, output, next_mix, collapsed,
                    tape_residual, tape_mixes, tape_delta, capture_owner, 3)
        else:
            original(x, partial, f, s, b, window, signal, rank, epoch, output, next_mix, collapsed)
        return output, next_mix, collapsed, tape_residual, tape_mixes, tape_delta

    @pl.jit.host
    def entry(
        x: pl.Tensor[[tp, block, copies, hidden], pl.FP32], premix: pl.Tensor[[tp, block, copies], pl.FP32],
        partial: pl.Tensor[[tp, block, hidden], pl.BF16], f: pl.Tensor[[tp, channels, flat], pl.FP32],
        s: pl.Tensor[[tp, 3], pl.FP32], b: pl.Tensor[[tp, channels], pl.FP32],
        output: pl.Out[pl.Tensor[[tp, block, copies, hidden], pl.FP32]],
        next_mix: pl.Out[pl.Tensor[[tp, block, copies], pl.FP32]],
        collapsed: pl.Out[pl.Tensor[[tp, block, hidden], pl.BF16]],
        tape_residual: pl.Out[pl.Tensor[[tp, *residual_shape], pl.FP32]],
        tape_mixes: pl.Out[pl.Tensor[[tp, *mixes_shape], pl.FP32]],
        tape_delta: pl.Out[pl.Tensor[[tp, *delta_shape], pl.BF16]],
        epoch: pl.Scalar[pl.INT32], capture_owner: pl.Scalar[pl.INT32],
    ):
        window_buf = pld.alloc_window_buffer([block, hidden], dtype=pl.BF16)
        signal_buf = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            window = pld.window(window_buf, [block, hidden], dtype=pl.BF16)
            signal = pld.window(signal_buf, [tp, 1], dtype=pl.INT32)
            compare(x[rank], premix[rank], partial[rank], f[rank], s[rank], b[rank],
                    output[rank], next_mix[rank], collapsed[rank], tape_residual[rank], tape_mixes[rank],
                    tape_delta[rank], window, signal, rank, epoch, capture_owner, device=rank)

    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    graph = entry.compile(config=config)
    layer = 39
    checkpoint = Checkpoint(args.checkpoint)
    params = [checkpoint.tensor(f"layers.{layer}.hc_attn_{name}").float() for name in ("fn", "scale", "base")]
    generator = torch.Generator().manual_seed(8442)
    x = torch.randn(block, copies, hidden, generator=generator)
    premix = torch.rand(block, copies, generator=generator)
    partials = torch.randn(tp, block, hidden, generator=generator, dtype=torch.bfloat16)
    timings = {mode: [] for mode in ("off", "on")}

    with DistributedWorker([graph], config, persistent=True, reset_persistent_windows=False) as runtime:
        def bind(values):
            return StackedDeviceTensor([
                runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
                for rank, value in enumerate(values)
            ], (tp, *values[0].shape), ranks)

        def replicated(value):
            return bind([value] * tp)

        def read(bank, rank):
            data = bank.shards[rank]
            value = torch.empty(data.shape, dtype=data.dtype)
            runtime.copy_from(value.data_ptr(), data.data_ptr, value.nbytes, worker_id=rank)
            return value

        inputs = [replicated(x), replicated(premix), bind(list(partials)), *(replicated(x) for x in params)]
        forward = (replicated(torch.zeros_like(x)), replicated(torch.zeros_like(premix)),
                   replicated(torch.zeros((block, hidden), dtype=torch.bfloat16)))
        tapes = tuple(replicated(torch.zeros(shape, dtype=getattr(torch, dtype)))
                      for shape, dtype in shapes.values())
        for step in range(18):
            capture_on = step % 2
            start = perf_counter()
            runtime.run(graph, *inputs, *forward, *tapes, step + 1, 0 if capture_on else -1, config=config)
            timings["on" if capture_on else "off"].append((perf_counter() - start) * 1000)
            if step == 16:
                original_values = tuple(tuple(read(bank, rank) for rank in ranks) for bank in forward)
        for baseline, bank in zip(original_values, forward, strict=True):
            for rank, expected in enumerate(baseline):
                torch.testing.assert_close(read(bank, rank), expected, atol=0, rtol=0)
        residual, mixes, delta = (read(bank, 0) for bank in tapes)
        expected_delta = torch.zeros_like(partials[0], dtype=torch.float32)
        for part in partials:
            expected_delta.add_(part.float())
        torch.testing.assert_close(delta[3 * block:], expected_delta.bfloat16(), atol=0, rtol=0)
        torch.testing.assert_close(residual[3 * block * copies:], x.flatten(0, 1), atol=0, rtol=0)
        torch.testing.assert_close(mixes[3 * block:, :copies], premix, atol=0, rtol=0)
        pre, post, combo = golden_mhc_mixes(x, *params)
        torch.testing.assert_close(mixes[3 * block:, copies:2 * copies], post, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(mixes[3 * block:, 2 * copies:2 * copies + copies * copies],
                                   combo.flatten(1), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(original_values[1][0], pre, atol=2e-6, rtol=2e-5)
        for saved, step_size in ((residual, block * copies), (mixes, block), (delta, block)):
            assert not torch.count_nonzero(saved[:3 * step_size])
        for other_rank in ranks[1:]:
            assert all(not torch.count_nonzero(read(saved, other_rank)) for saved in tapes)
    print(json.dumps(dict(passed=True, checkpoint_layer=layer, checkpoint=str(args.checkpoint), tp=tp,
                          block=block, last_layers=last_layers, forward_byte_exact=True, saved_rows=block,
                          reduction_byte_exact=True, capture_owner=0, tape_slot=3,
                          warm_median_ms={mode: median(samples[1:]) for mode, samples in timings.items()})),
          flush=True)


if __name__ == "__main__":
    main()
