#!/usr/bin/env python3
"""真实 TP2 GDN 形状，比较有效行与补齐行递推；不加载主模型。"""

import argparse
import json
from pathlib import Path
from statistics import median
import time


def golden(q, k, v, decay, beta, history, widths):
    import torch
    result = torch.zeros((len(widths) * 32, 3072), dtype=torch.bfloat16)
    for seq, width in enumerate(widths):
        state = history[seq, 47].clone()
        for token in range(width):
            row = seq * 32 + token
            query, key = q[row].reshape(24, 128), k[row].reshape(24, 128)
            faded = state * decay[row, :24].exp()[:, None, None]
            remembered = (faded * key[:, None, :]).sum(-1)
            value = v[row, 2048:].float().reshape(24, 128)
            delta = (value - remembered) * beta[row, :24, None]
            state = faded + delta[:, :, None] * key[:, None, :]
            result[row] = (state * query[:, None, :]).sum(-1).reshape(-1).bfloat16()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compile-only', action='store_true')
    a = p.parse_args()
    if a.output.exists(): p.error('use a fresh result path')
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.tp2_gdn_rows import make_entry
    torch.set_num_threads(2)
    torch.manual_seed(184)
    cfg = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                    distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
    programs = {}
    for batch in (1, 2):
        shapes = [(batch * 32, 3072)] * 2 + [(batch * 32, 5120)] + [(batch * 32, 32)] * 2
        tensors = [torch.empty(s, dtype=torch.bfloat16 if i == 2 else torch.float32, device='meta')
                   for i, s in enumerate(shapes)]
        tensors += [torch.empty((2, 48, 24, 128, 128), dtype=torch.float32, device='meta'),
                    torch.empty((batch * 32, 3072), dtype=torch.bfloat16, device='meta')]
        for valid in (False, True):
            print('compile', batch, valid, flush=True)
            programs[batch, valid] = make_entry(batch, valid_rows=valid, history_slots=2).compile(
                *tensors, 0, 1, 47, 8, 6, config=cfg)
    report = dict(completed=False, passed=False, clean_shutdown=False, cases=[],
                  scope='Isolated TP2-local GDN core with real head dimensions, nonzero history and distinct request widths; synchronous call timing only, not serving throughput')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.compile_only:
        report.update(completed=True, compiled=True, physical_device_count=0,
                      compiled_paths=[str(p.output_dir) for p in programs.values()])
        a.output.write_text(json.dumps(report, indent=2) + '\n')
        return
    worker, handles = None, []
    try:
        worker = DistributedWorker(list(programs.values()), config=cfg)
        history = torch.randn((2, 48, 24, 128, 128)).mul_(.02)
        device_history = worker.alloc_tensor(history.shape, history.dtype, init=history); handles.append(device_history)
        for widths in [(32,), (8,), (2,), (32, 32), (8, 8), (8, 6), (16, 2), (2, 32)]:
            batch, count = len(widths), len(widths) * 32
            q = torch.nn.functional.normalize(torch.randn(count, 24, 128), dim=-1).reshape(count, 3072)
            k = torch.nn.functional.normalize(torch.randn(count, 24, 128), dim=-1).reshape(count, 3072)
            v = torch.randn(count, 5120).mul_(.1).bfloat16()
            decay = -torch.rand(count, 32).mul_(.05)
            beta = torch.rand(count, 32)
            local = [worker.alloc_tensor(x.shape, x.dtype, init=x) for x in (q, k, v, decay, beta)]
            out = [worker.alloc_tensor((count, 3072), torch.bfloat16) for _ in range(2)]
            def run(valid):
                worker.run(programs[batch, valid], *local, device_history, out[int(valid)],
                           0, 1, 47, widths[0], widths[-1], config=cfg)
            actual = []
            try:
                for valid in (False, True):
                    run(valid); x = torch.empty((count, 3072), dtype=torch.bfloat16)
                    worker.copy_from(x.data_ptr(), out[int(valid)].data_ptr, x.nbytes); actual.append(x)
                expected = golden(q, k, v, decay, beta, history, widths)
                active = torch.cat([torch.arange(i * 32, i * 32 + w) for i, w in enumerate(widths)])
                assert torch.equal(actual[0][active].view(torch.int16), actual[1][active].view(torch.int16))
                delta = actual[1][active].float() - expected[active].float()
                relative = float(delta.norm() / expected[active].float().norm())
                assert relative < .01 and float(delta.abs().max()) < .002
                for i, w in enumerate(widths): assert torch.count_nonzero(actual[1][i * 32 + w:(i + 1) * 32]) == 0
                for _ in range(2): run(False); run(True)
                samples = [[], []]
                for i in range(12):
                    for valid in ((False, True) if i % 2 == 0 else (True, False)):
                        start = time.monotonic_ns(); run(valid); samples[int(valid)].append(time.monotonic_ns() - start)
                report['cases'].append(dict(widths=widths,active_prefix_bitwise=True,tail_zero=True,
                    CPU_relative_L2=relative,CPU_max_abs=float(delta.abs().max()),
                    padded_call_ns=samples[0],valid_call_ns=samples[1],
                    padded_median_ns=median(samples[0]),valid_median_ns=median(samples[1])))
            finally:
                for value in [*local, *out]: worker.free_tensor(value)
        after = torch.empty_like(history)
        worker.copy_from(after.data_ptr(), device_history.data_ptr, after.nbytes)
        assert torch.equal(after.view(torch.int32), history.view(torch.int32))
        report.update(passed=True,history_bitwise_unchanged=True)
    except BaseException as err:
        report['error'] = repr(err)
        raise
    finally:
        try:
            if worker is not None:
                for value in handles: worker.free_tensor(value)
                worker.close(); report['clean_shutdown'] = True
        finally:
            report['completed'] = report['passed'] and report['clean_shutdown']
            a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
