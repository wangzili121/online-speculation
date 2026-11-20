#!/usr/bin/env python3
"""只读 GDN 递推的预取和分块对照；不代表模型吞吐。"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time


def golden(q, k, v, decay, beta, history, slots, layer):
    import torch
    outputs = []
    for seq, slot in enumerate(slots):
        state = history.reshape(32, 48, 24, 128, 128)[slot, layer].clone()
        rows = []
        for token in range(32):
            index = seq * 32 + token
            query, key = q[index].reshape(24, 128), k[index].reshape(24, 128)
            faded = state * decay[index, :24].exp()[:, None, None]
            remembered = (faded * key[:, None, :]).sum(-1)
            delta = (v[index, 2048:].float().reshape(24, 128) - remembered) * beta[index, :24, None]
            state = faded + delta[:, :, None] * key[:, None, :]
            rows.append((state * query[:, None, :]).sum(-1).reshape(-1).bfloat16())
        outputs.append(torch.stack(rows))
    return torch.cat(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--profiles', type=int, default=3)
    parser.add_argument('--variants', default='sequential64,prefetch64',
                        help='128-row candidates are optional compilation/memory experiments')
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3 or args.profiles < 0:
        parser.error('fresh output, 3+ repeats and nonnegative profiles required')
    import torch
    import torch.nn.functional as F
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.tp2_gdn_recurrence import make_entry
    from summarize_qwen38_pypto_profile import summarize
    torch.set_num_threads(4)
    torch.manual_seed(115)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    variants = {'sequential64': (False, 64, False), 'prefetch64': (True, 64, False),
                'sequential128': (False, 128, False), 'prefetch128': (True, 128, False),
                'decay64': (False, 64, True)}
    selected = args.variants.split(',')
    if not selected or selected[0] != 'sequential64' or len(set(selected)) != len(selected) or any(name not in variants for name in selected):
        parser.error('unique known variants with sequential64 first required')
    variants = {name: variants[name] for name in selected}
    report = dict(variants=selected, passed=False, clean_shutdown=False, scope='isolated recurrence, not model throughput', compilation_errors=[], cases=[])
    worker, handles = None, []
    try:
        history = torch.zeros(32 * 48 * 24 * 128, 128)
        cache = history.reshape(32, 48, 24, 128, 128)
        settings = [(0, 31, 0), (11, 3, 47), (7, 7, 12)]
        for left, right, layer in settings:
            for slot in set((left, right)):
                cache[slot, layer] = torch.randn(24, 128, 128) * .1
        programs = {}
        for batch in [1, 2]:
            meta = lambda shape, dtype=torch.float32: torch.empty(shape, dtype=dtype, device='meta')
            for name, (prefetch, tile, gates) in variants.items():
                try:
                    programs[batch, name] = make_entry(batch, prefetch=prefetch, state_tile=tile, precompute_decay=gates).compile(
                        meta((batch * 32, 3072)), meta((batch * 32, 3072)),
                        meta((batch * 32, 5120), torch.bfloat16), meta((batch * 32, 32)), meta((batch * 32, 32)),
                        meta(history.shape), meta((batch * 32, 3072), torch.bfloat16), 11, 3, 47, config=config)
                except Exception as error:
                    report['compilation_errors'].append(dict(batch=batch, variant=name, error=repr(error)))
                    print(json.dumps(report['compilation_errors'][-1]), flush=True)
                    if name == 'sequential64': raise
        worker = DistributedWorker(list(programs.values()), config=config)
        def alloc(value):
            handle = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value)
            handles.append(handle)
            return handle
        dh = alloc(history)
        returned = torch.empty_like(history)
        for batch in [1, 2]:
            q = (F.normalize(torch.randn(batch * 32, 24, 128), dim=-1) * 128 ** -.5).reshape(batch * 32, 3072)
            k = F.normalize(torch.randn(batch * 32, 24, 128), dim=-1).reshape(batch * 32, 3072)
            v = torch.randn(batch * 32, 5120).bfloat16()
            decay, beta = -torch.rand(batch * 32, 32) * .4, torch.rand(batch * 32, 32)
            output = torch.empty(batch * 32, 3072, dtype=torch.bfloat16)
            dq, dk, dv, dd, db, dy = [alloc(value)for value in[q, k, v, decay, beta, torch.zeros_like(output)]]
            available = [name for name in variants if (batch, name) in programs]
            for left, right, layer in settings:
                slots = [left] if batch == 1 else [left, right]
                expected = golden(q, k, v, decay, beta, history, slots, layer)
                call = [dq, dk, dv, dd, db, dh, dy, left, right, layer]
                samples = {name: [] for name in available}
                for name in available:
                    worker.run(programs[batch, name], *call, config=config)
                for repeat in range(args.repeats):
                    order = list(available); order = order[repeat % len(available):] + order[:repeat % len(available)]
                    for name in order:
                        started = time.monotonic_ns(); worker.run(programs[batch, name], *call, config=config)
                        samples[name].append(time.monotonic_ns() - started)
                reference = None
                for name in available:
                    program = programs[batch, name]; worker.run(program, *call, config=config)
                    worker.copy_from(output.data_ptr(), dy.data_ptr, output.nbytes)
                    torch.testing.assert_close(output.float(), expected.float(), atol=.002, rtol=.02)
                    if reference is None: reference = output.clone()
                    assert torch.equal(output, reference), f'{batch}/{name} changed rounded recurrence'
                    worker.copy_from(returned.data_ptr(), dh.data_ptr, returned.nbytes)
                    assert torch.equal(returned, history), 'target GDN state changed'
                    profiles = []
                    if layer == 47:
                        for sample in range(args.profiles):
                            worker.run(program, *call, config=replace(config, enable_chip_swimlane=1))
                            source = max(Path(program.output_dir).rglob('merged_swimlane*.json'), key=lambda p:p.stat().st_mtime_ns)
                            target = args.output.parent/'profiles'/f'b{batch}-{name}-{sample}.json'; target.parent.mkdir(exist_ok=True)
                            shutil.copyfile(source, target); data = target.read_bytes()
                            profiles.append(dict(path=str(target), sha256=hashlib.sha256(data).hexdigest(), **summarize(json.loads(data))))
                    row = dict(batch=batch, variant=name, slots=slots, layer=layer, bitwise_equal=True,
                               history_unchanged=True, golden_max_abs=float((output.float()-expected.float()).abs().max()),
                               call_median_ns=statistics.median(samples[name]), call_samples_ns=samples[name], device_profiles=profiles)
                    report['cases'].append(row); print(json.dumps(row), flush=True)
        if report['compilation_errors']:
            raise RuntimeError('Some requested variants did not compile; numeric cases remain diagnostic evidence')
        report['passed'] = True
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        if worker is not None:
            while handles: worker.free_tensor(handles.pop())
            worker.close(); report['clean_shutdown'] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
