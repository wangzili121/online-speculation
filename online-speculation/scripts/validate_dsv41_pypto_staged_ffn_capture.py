#!/usr/bin/env python3
"""Compare the staged draft FFN outputs and retained TP tape on real devices."""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter
import json

import torch
from blockspec.dsv41_packed_decode import make_decode_ffn_tail
from blockspec.dsv41_packed_ffn_tape import packed_ffn_tape_shapes
from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
from models.deepseek_v4_1_flash_a3.hc import golden_hc_post, golden_mhc_mixes
from models.deepseek_v4_1_flash_a3.weights import Checkpoint
from pypto.backend import BackendType
from pypto.ir import DistributedConfig
from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
from pypto.runtime.distributed_runner import DistributedWorker

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--block", type=int, choices=(16, 32), default=16)
    parser.add_argument("--last-layers", type=int, choices=range(1, 27), default=4, metavar="N")
    args = parser.parse_args()
    devices = list(map(int, args.devices.split(",")))
    if len(devices) != TP_SIZE or len(set(devices)) != TP_SIZE:
        parser.error("Use one complete group of four distinct devices")

    torch.set_num_threads(3)
    ranks = tuple(range(TP_SIZE))
    block, last_layers = args.block, args.last_layers
    hidden, copies, topk = FLASH.hidden_size, FLASH.hc_mult, FLASH.num_experts_per_tok
    local = block // TP_SIZE
    checkpoint = Checkpoint(args.checkpoint)
    prefix = "layers.39.hc_ffn_"
    function, scale, base = (checkpoint.tensor(prefix + name).float() for name in ("fn", "scale", "base"))
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    capture = make_decode_ffn_tail(block, len(ranks), last_layers=last_layers).compile(config=config)
    rng = torch.Generator().manual_seed(6293)
    shape = block, copies, hidden
    attention = torch.randn(*shape, generator=rng)
    combined = torch.randn((block, hidden), dtype=torch.bfloat16, generator=rng)
    owned = torch.zeros((TP_SIZE, block, hidden), dtype=torch.bfloat16)
    for rank in ranks:
        owned[rank, rank::TP_SIZE] = combined[rank::TP_SIZE]
    meta = torch.tensor([(block, lane) for lane in ranks], dtype=torch.int32)
    incoming = torch.rand(block, copies, generator=rng)
    collapsed = torch.randn(block, hidden, dtype=torch.bfloat16, generator=rng)
    normed = torch.randn(TP_SIZE, 16, hidden, dtype=torch.bfloat16, generator=rng)
    ids = torch.randint(0, FLASH.n_routed_experts, (TP_SIZE, 16, topk), dtype=torch.int32, generator=rng)
    weights = torch.randn(TP_SIZE, 16, topk, generator=rng).softmax(-1)
    specs = packed_ffn_tape_shapes(block, last_layers, TP_SIZE, hidden, copies)
    timings = {"uncaptured": [], "capture": []}
    with DistributedWorker([capture], config, persistent=True, reset_persistent_windows=False) as runtime:
        def upload(values):
            shards = [runtime.alloc_tensor(x.shape, x.dtype, init=x, worker_id=rank)
                      for rank, x in enumerate(values)]
            return StackedDeviceTensor(shards, (TP_SIZE, *shards[0].shape), ranks)

        def both(value):
            return upload([value] * TP_SIZE)

        def read(stacked, rank):
            handle = stacked.shards[rank]
            value = torch.empty(handle.shape, dtype=handle.dtype)
            runtime.copy_from(value.data_ptr(), handle.data_ptr, value.nbytes, worker_id=rank)
            return value

        outputs = both(torch.zeros(*shape)), both(torch.zeros(block, copies))
        args = both(attention), upload(owned), both(function), both(scale), both(base), upload(meta), *outputs
        tape = [both(torch.zeros(size, dtype=getattr(torch, dtype))) for size, dtype in specs.values()]
        state = both(incoming), both(collapsed), upload(normed), upload(ids), upload(weights), *tape
        runtime.run(capture, *args, *state, -1, -1, 0)
        expected = None
        for _ in range(6):
            for mode, owner in (("uncaptured", -1), ("capture", 0)):
                start = perf_counter()
                runtime.run(capture, *args, *state, owner, last_layers - 1, 1)
                timings[mode].append((perf_counter() - start) * 1000)
                actual = [[read(value, rank) for value in outputs] for rank in ranks]
                if expected is None:
                    expected = actual
                for candidates, goldens in zip(actual, expected, strict=True):
                    for value, golden in zip(candidates, goldens, strict=True):
                        torch.testing.assert_close(value, golden, rtol=0, atol=0)
        _, post, residual = golden_mhc_mixes(attention, function, scale, base)
        reference = golden_hc_post(combined.float(), attention, post, residual)
        relative = float((expected[0][0] - reference).norm() / reference.norm())
        assert relative < 2e-5, relative
        for rank in ranks:
            expected_tape = (
                attention.flatten(0, 1), incoming,
                torch.cat((collapsed, combined, normed[rank, :local])),
                ids[rank, :local], weights[rank, :local],
            )
            for item, expected_value in zip(tape, expected_tape, strict=True):
                actual = read(item, rank)
                rows, width = expected_value.shape
                torch.testing.assert_close(actual[-rows:, :width], expected_value, rtol=0, atol=0)
                assert not torch.any(actual[:-rows])
    print(json.dumps(dict(passed=True, block=block, last_layers=last_layers, layer=39, tp=TP_SIZE, devices=devices,
                          npu_execution=True, repeats=6, outputs_byte_exact=True, tape_byte_exact=True,
                          ffn_relative_l2=relative, warm_median_ms={k: median(v[1:]) for k, v in timings.items()})))


if __name__ == "__main__":
    main()
