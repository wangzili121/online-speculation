#!/usr/bin/env python3
"""Check the resident 1122 Decode FFN tail and repeated TP window reuse."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--rows", type=int, default=16, choices=(16, 80))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    import torch
    from blockspec.dsv41_packed_decode import make_decode_ffn_tail
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    devices = list(map(int, args.devices.split(",")))
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    program = make_decode_ffn_tail(args.rows, len(devices)).compile(config=config)
    report = dict(passed=True, compiled=True, local_ranks=len(devices), rows=args.rows)
    if not args.compile_only:
        if len(devices) != TP_SIZE or args.checkpoint is None:
            parser.error("Execute with one TP group and --checkpoint")
        report["cases"] = exercise(program, config, args.rows, args.checkpoint, FLASH, TP_SIZE)
    print(json.dumps(report), flush=True)


def exercise(program, config, rows, path, model, tp):
    import torch
    from models.deepseek_v4_1_flash_a3.hc import golden_hc_post, golden_mhc_mixes
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.runtime import StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker

    checkpoint = Checkpoint(path)
    weights = [checkpoint.tensor(f"layers.39.hc_ffn_{key}").float().reshape(shape)
               for key, shape in (("fn", (model.mix_hc, model.hc_dim)), ("scale", (3,)), ("base", (model.mix_hc,)))]
    generator = torch.Generator().manual_seed(80937)
    attention = torch.randn(rows, model.hc_mult, model.hidden_size, generator=generator) * 0.1
    residual = torch.randn(rows, model.hidden_size, generator=generator).bfloat16() * 0.1
    ranks = tuple(range(tp))
    cases = []
    with DistributedWorker([program], config, persistent=True, reset_persistent_windows=False) as runtime:
        def upload(values):
            shards = [runtime.alloc_tensor(value.shape, value.dtype, worker_id=rank, init=value)
                      for rank, value in enumerate(values)]
            return StackedDeviceTensor(shards, (tp, *values[0].shape), ranks)

        def update(tensor, values):
            for rank, (handle, value) in enumerate(zip(tensor.shards, values, strict=True)):
                runtime.copy_to(handle.data_ptr, value.data_ptr(), value.nbytes, worker_id=rank)

        def read(tensor, rank):
            handle = tensor.shards[rank]
            result = torch.empty(handle.shape, dtype=handle.dtype)
            runtime.copy_from(result.data_ptr(), handle.data_ptr, result.nbytes, worker_id=rank)
            return result

        hidden = upload([attention] * tp)
        owned = upload([torch.zeros_like(residual)] * tp)
        mixes = [upload([value] * tp) for value in weights]
        meta = upload([torch.tensor([0, rank], dtype=torch.int32) for rank in ranks])
        output = upload([torch.empty_like(attention)] * tp)
        pre_mix = upload([torch.empty(rows, model.hc_mult)] * tp)
        argv = (hidden, owned, *mixes, meta, output, pre_mix)
        runtime.run(program, *argv, 0)
        for index, active in enumerate((1, 7, rows, 0, 1, 7)):
            token_values = residual if index % 2 == 0 else -residual
            partials = [torch.full_like(residual, 99) for _ in ranks]
            for rank in ranks:
                partials[rank][rank:active:tp] = token_values[rank:active:tp]
            update(owned, partials)
            update(meta, [torch.tensor([active, rank], dtype=torch.int32) for rank in ranks])
            runtime.run(program, *argv, 1)
            if not active:
                cases.append(dict(active=0))
                continue
            expected_mix, post, skip = golden_mhc_mixes(attention[:active], *weights)
            expected = golden_hc_post(token_values[:active].float(), attention[:active], post, skip)
            error = mix_error = 0.0
            for rank in ranks:
                observed = read(output, rank)[:active]
                observed_mix = read(pre_mix, rank)[:active]
                torch.testing.assert_close(observed, expected, atol=2.5e-5, rtol=5e-3)
                torch.testing.assert_close(observed_mix, expected_mix, atol=2.5e-5, rtol=5e-3)
                error = max(error, (observed - expected).abs().max().item())
                mix_error = max(mix_error, (observed_mix - expected_mix).abs().max().item())
            cases.append(dict(active=active, hidden_max_abs=error, pre_mix_max_abs=mix_error))
    return cases


if __name__ == "__main__":
    main()
