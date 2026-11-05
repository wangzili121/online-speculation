"""Interleave private and registered snapshots on identical native transactions."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch

from blockspec.pypto_qwen38.baseline import native_module
from blockspec.pypto_qwen38.greedy import verify_transaction
from blockspec.pypto_qwen38.target import NativeTarget, SnapshotPool


@contextmanager
def profile(target):
    stats, saved = {}, []

    def install(name):
        function = getattr(target, name)
        saved.append((name, function))

        def timed(*args, **kwargs):
            start = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                stats[name] = stats.get(name, 0) + time.monotonic() - start

        setattr(target, name, timed)

    for name in ('snapshot', 'restore', 'forward'):
        install(name)
    try:
        yield stats
    finally:
        for name, function in reversed(saved):
            setattr(target, name, function)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.device < 0:
        parser.error('require an assigned device and a new output')
    from validate_qwen38_pypto_neural import history_checksum

    runtime = native_module(args.source, 'runtime')
    schedule = native_module(args.source, 'schedule')
    engine = native_module(args.source, 'engine').InferenceEngine(
        args.model, args.device, max_active=1, prefix_cache=False)
    native = engine.runner
    pool = SnapshotPool(native.zero_conv.shape, native.zero_recurrent.shape)
    logits = torch.empty((16, 248320), dtype=torch.bfloat16).share_memory_()
    original = runtime.DistributedWorker
    runtime.DistributedWorker = lambda *a, **kw: original(
        *a, inherited_host_tensors=[logits, pool.storage], **kw)
    report = {'passed': False, 'scope': 'native_verification_snapshot_buffers',
              'cases': [], 'shared_state_bytes': pool.storage.numel()}
    request = None
    try:
        engine.start()
        runtime.DistributedWorker = original
        native.verification_logits = logits
        request = schedule.RequestState(
            engine.tokenizer.encode('Hello', add_special_tokens=False), 40, frozenset())
        native.admit(request)
        anchor = native.run_batch([schedule.BatchSlot(
            request, tuple(request.prompt_ids), 0, 'prefill', True)])[0]
        request.generated.append(anchor)
        request.prefill_offset = len(request.prompt_ids)
        target = NativeTarget(native, request, schedule.BatchSlot)
        initial = target.snapshot()  # Independent baseline, outside the measured region.
        expected, token = [], anchor
        for _ in range(35):
            token = target.forward([token])[0]
            expected.append(token)

        for case in ('reject0', 'reject1', 'accept31'):
            candidates = expected[:31].copy()
            if case != 'accept31':
                index = int(case[-1])
                candidates[index] = (candidates[index] + 1) % 248320
            checksums = []
            for iteration in range(4):
                modes = ('private', 'shared') if iteration % 2 == 0 else ('shared', 'private')
                for mode in modes:
                    target.restore(initial)
                    if mode == 'shared':
                        native.verification_states = pool
                    elif hasattr(native, 'verification_states'):
                        del native.verification_states
                    with profile(target) as stats:
                        start = time.monotonic()
                        result = verify_transaction(target, anchor, candidates, budget=32)
                        seconds = time.monotonic() - start
                    assert list(result.tokens) == expected[:len(result.tokens)]
                    assert result.accepted == (31 if case == 'accept31' else int(case[-1]))
                    assert not pool.busy, 'snapshot lease must end with the transaction'
                    checksums.append(history_checksum(native, request))
                    entry = dict(case=case, iteration=iteration, mode=mode,
                                 seconds=seconds, phases=stats, accepted=result.accepted)
                    report['cases'].append(entry)
                    print(json.dumps(entry), flush=True)
            assert len(set(checksums)) == 1, 'accepted target state differs between modes'
        native.release(request)
        request = None
        report['passed'] = True
    finally:
        runtime.DistributedWorker = original
        try:
            if request is not None:
                native.release(request)
            engine.stop()
        finally:
            args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
