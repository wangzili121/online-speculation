#!/usr/bin/env python3
"""Compare speculative round boundaries using the A3 greedy sampler on one NPU."""

import argparse
import json
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter_ns

import pypto.language as pl
import torch
from models.deepseek_v4_1_flash_a3.config import FLASH, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD
from models.deepseek_v4_1_flash_a3.lm_head import greedy_sample
from models.deepseek_v4_1_flash_a3.orthrus_verify import Verification
from pypto.backend import BackendType
from pypto.ir import DistributedConfig
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig, StackedDeviceTensor
from pypto.runtime.distributed_runner import DistributedWorker


ROWS = MAX_LOGIT_ROWS
VOCAB = FLASH.vocab_size
PAD = SAMPLED_IDS_PAD
WIDTH = 8
CANDIDATES = WIDTH - 1
TILE = 1280
REPLY_WORDS = 16


@pl.jit.inline(auto_scope=False)
def finite_rows(logits: pl.Tensor[[ROWS, VOCAB], pl.FP32],
                flags: pl.Out[pl.Tensor[[ROWS, 8], pl.FP32]],
                count: pl.Scalar[pl.INT32]):
    with pl.spmd(ROWS, name_hint="finite_logits") as _ready:
        row = pl.tile.get_block_idx()
        if row < count:
            valid = pl.full([8, TILE // 8], dtype=pl.FP32, value=1.)
            for chunk in pl.range(VOCAB // TILE):
                values = pl.slice(logits, [1, TILE], [row, chunk * TILE])
                finite = pl.cmp(pl.abs(pl.reshape(values, [8, TILE // 8])),
                                3.4028234663852886e38, cmp_type=3)
                valid = pl.minimum(valid, finite)
            flags[row:row + 1, :] = pl.reshape(pl.row_min(valid), [1, 8])


@pl.jit.inline(auto_scope=False)
def sample_draft(logits: pl.Tensor[[ROWS, VOCAB], pl.FP32],
                 ids: pl.Out[pl.Tensor[[ROWS, PAD], pl.INT32]],
                 flags: pl.Out[pl.Tensor[[ROWS, 8], pl.FP32]]):
    greedy_sample(logits, ids)
    finite_rows(logits, flags, CANDIDATES)


@pl.jit.inline(auto_scope=False)
def verify(logits: pl.InOut[pl.Tensor[[ROWS, VOCAB], pl.FP32]],
           candidates: pl.Tensor[[ROWS, PAD], pl.INT32],
           draft_flags: pl.Tensor[[ROWS, 8], pl.FP32],
           ids: pl.Out[pl.Tensor[[ROWS, PAD], pl.INT32]],
           flags: pl.Out[pl.Tensor[[ROWS, 8], pl.FP32]],
           reply: pl.Out[pl.Tensor[[1, REPLY_WORDS], pl.INT32]]):
    # Give the replayed target sampler a real RAW dependency on the draft.
    # Column zero is below the maximum in every frozen replay input.
    for owner in pl.spmd(1, name_hint="wait_for_draft"):
        token = pl.read(candidates, [owner, 0])
        pl.write(logits, [owner, 0], -100000. - pl.cast(token, pl.FP32))
    greedy_sample(logits, ids)
    finite_rows(logits, flags, WIDTH)
    for owner in pl.spmd(1, name_hint="accept_prefix"):
        answer = pl.tile.full([1, REPLY_WORDS], dtype=pl.INT32, value=-1)
        valid = pl.cast(1, pl.INT32)
        matched = pl.cast(0, pl.INT32)
        for row in pl.range(WIDTH):
            for lane in pl.range(8):
                if pl.read(flags, [row, lane]) != 1.:
                    valid = pl.cast(0, pl.INT32)
                if row < CANDIDATES:
                    if pl.read(draft_flags, [row, lane]) != 1.:
                        valid = pl.cast(0, pl.INT32)
            if row < CANDIDATES and row == matched:
                if pl.read(candidates, [row, 0]) == pl.read(ids, [row, 0]):
                    matched = pl.cast(row + 1, pl.INT32)
        pl.write(answer, [0, 0], valid)
        pl.write(answer, [0, 1], matched)
        for row in pl.range(WIDTH):
            if row <= matched:
                pl.write(answer, [0, row + 2], pl.read(ids, [row, 0]))
        pl.store(answer, [owner, 0], reply)


@pl.jit
def raw_sample(logits: pl.Tensor[[ROWS, VOCAB], pl.FP32],
               ids: pl.Out[pl.Tensor[[ROWS, PAD], pl.INT32]]):
    greedy_sample(logits, ids)


@pl.jit
def compact_draft(logits: pl.Tensor[[ROWS, VOCAB], pl.FP32],
                  ids: pl.Out[pl.Tensor[[ROWS, PAD], pl.INT32]],
                  flags: pl.Out[pl.Tensor[[ROWS, 8], pl.FP32]]):
    sample_draft(logits, ids, flags)


@pl.jit
def compact_verify(logits: pl.InOut[pl.Tensor[[ROWS, VOCAB], pl.FP32]],
                   candidates: pl.Tensor[[ROWS, PAD], pl.INT32],
                   draft_flags: pl.Tensor[[ROWS, 8], pl.FP32],
                   ids: pl.Out[pl.Tensor[[ROWS, PAD], pl.INT32]],
                   flags: pl.Out[pl.Tensor[[ROWS, 8], pl.FP32]],
                   reply: pl.Out[pl.Tensor[[1, REPLY_WORDS], pl.INT32]]):
    verify(logits, candidates, draft_flags, ids, flags, reply)


@pl.jit
def resident_round(draft: pl.Tensor[[ROWS, VOCAB], pl.FP32],
                   target: pl.InOut[pl.Tensor[[ROWS, VOCAB], pl.FP32]],
                   candidates: pl.Out[pl.Tensor[[ROWS, PAD], pl.INT32]],
                   draft_flags: pl.Out[pl.Tensor[[ROWS, 8], pl.FP32]],
                   ids: pl.Out[pl.Tensor[[ROWS, PAD], pl.INT32]],
                   flags: pl.Out[pl.Tensor[[ROWS, 8], pl.FP32]],
                   reply: pl.Out[pl.Tensor[[1, REPLY_WORDS], pl.INT32]]):
    sample_draft(draft, candidates, draft_flags)
    verify(target, candidates, draft_flags, ids, flags, reply)


def distributed_entries(ranks):
    @pl.jit.host
    def sample(logits: pl.Tensor[[ranks, ROWS, VOCAB], pl.FP32],
               ids: pl.Out[pl.Tensor[[ranks, ROWS, PAD], pl.INT32]]):
        for rank in pl.range(ranks):
            raw_sample(logits[rank], ids[rank], device=rank)

    @pl.jit.host
    def draft(logits: pl.Tensor[[ranks, ROWS, VOCAB], pl.FP32],
              ids: pl.Out[pl.Tensor[[ranks, ROWS, PAD], pl.INT32]],
              flags: pl.Out[pl.Tensor[[ranks, ROWS, 8], pl.FP32]]):
        for rank in pl.range(ranks):
            compact_draft(logits[rank], ids[rank], flags[rank], device=rank)

    @pl.jit.host
    def verify_round(logits: pl.InOut[pl.Tensor[[ranks, ROWS, VOCAB], pl.FP32]],
                     candidates: pl.Tensor[[ranks, ROWS, PAD], pl.INT32],
                     draft_flags: pl.Tensor[[ranks, ROWS, 8], pl.FP32],
                     ids: pl.Out[pl.Tensor[[ranks, ROWS, PAD], pl.INT32]],
                     flags: pl.Out[pl.Tensor[[ranks, ROWS, 8], pl.FP32]],
                     reply: pl.Out[pl.Tensor[[ranks, 1, REPLY_WORDS], pl.INT32]]):
        for rank in pl.range(ranks):
            compact_verify(logits[rank], candidates[rank], draft_flags[rank], ids[rank],
                           flags[rank], reply[rank], device=rank)

    @pl.jit.host
    def resident(draft_logits: pl.Tensor[[ranks, ROWS, VOCAB], pl.FP32],
                 target: pl.InOut[pl.Tensor[[ranks, ROWS, VOCAB], pl.FP32]],
                 candidates: pl.Out[pl.Tensor[[ranks, ROWS, PAD], pl.INT32]],
                 draft_flags: pl.Out[pl.Tensor[[ranks, ROWS, 8], pl.FP32]],
                 ids: pl.Out[pl.Tensor[[ranks, ROWS, PAD], pl.INT32]],
                 flags: pl.Out[pl.Tensor[[ranks, ROWS, 8], pl.FP32]],
                 reply: pl.Out[pl.Tensor[[ranks, 1, REPLY_WORDS], pl.INT32]]):
        for rank in pl.range(ranks):
            resident_round(draft_logits[rank], target[rank], candidates[rank], draft_flags[rank],
                           ids[rank], flags[rank], reply[rank], device=rank)

    return sample, draft, verify_round, resident


def timed(call, *args, **kwargs):
    start = perf_counter_ns()
    value = call(*args, **kwargs)
    return value, (perf_counter_ns() - start) / 1000


def checked_ids(ids, logits, count):
    values = ids[:count, 0]
    if not torch.isfinite(logits[:count]).all() or not torch.equal(logits[:count].argmax(1), values):
        raise ValueError("Nonfinite logits or an incorrect greedy ID")
    return values.tolist()


def run(args):
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(41127)
    draft = torch.randn(ROWS, VOCAB, generator=rng)
    # Include a tie spanning the first and last vocabulary chunks.
    draft[0, 31] = draft[0, VOCAB - 1] = 100.
    proposed = draft[:CANDIDATES].argmax(1).tolist()
    draft[0, 0] = -100000. - proposed[0]
    targets = {}
    for matched in (0, 3, CANDIDATES):
        logits = torch.randn(ROWS, VOCAB, generator=rng)
        for row, token in enumerate(proposed):
            token = (token + 1024) % VOCAB if row == matched else token
            logits[row, token] = 100.
        logits[0, 0] = -100000. - proposed[0]
        targets[matched] = logits
    expected = {matched: Verification.resolve(proposed, logits[:WIDTH].argmax(1).tolist())
                for matched, logits in targets.items()}
    assert all(value.accepted == matched for matched, value in expected.items())

    candidate_ids = torch.empty((ROWS, PAD), dtype=torch.int32)
    target_ids = torch.empty_like(candidate_ids)
    draft_flags = torch.empty((ROWS, 8), dtype=torch.float32)
    target_flags = torch.empty_like(draft_flags)
    reply = torch.empty((1, REPLY_WORDS), dtype=torch.int32)
    draft_copy = torch.empty_like(draft[:CANDIDATES])
    target_copy = torch.empty_like(draft[:WIDTH])
    options = dict(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD)
    ranks = tuple(range(len(args.devices))) if args.devices else ()
    definitions = {
        "sample": (raw_sample, (draft, candidate_ids)),
        "draft": (compact_draft, (draft, candidate_ids, draft_flags)),
        "verify": (compact_verify, (draft, candidate_ids, draft_flags, target_ids, target_flags, reply)),
        "resident": (resident_round, (draft, draft, candidate_ids, draft_flags, target_ids, target_flags, reply)),
    }
    if ranks:
        config = RunConfig(**options, distributed_config=DistributedConfig(device_ids=args.devices, num_sub_workers=0))
        compiled = {name: entry.compile(config=config)
                    for name, entry in zip(definitions, distributed_entries(len(ranks)), strict=True)}
        runtime = DistributedWorker(list(compiled.values()), config, persistent=True)
    else:
        config = RunConfig(**options, device_id=args.device)
        compiled = {name: entry.compile(*inputs, config=config) for name, (entry, inputs) in definitions.items()}
        runtime = ChipWorker(config)
    modes = ("split_full_logits", "split_compact", "resident_round")
    measurements = {name: [] for name in modes}
    with runtime as worker:
        def allocate(value):
            if ranks:
                shards = [worker.alloc_tensor(tuple(value.shape), value.dtype, init=value, worker_id=rank)
                          for rank in ranks]
                return StackedDeviceTensor(shards, (len(ranks), *value.shape), ranks)
            return worker.alloc_tensor(tuple(value.shape), value.dtype, init=value)

        d_draft = allocate(draft)
        d_targets = {accepted: allocate(value) for accepted, value in targets.items()}
        d_bad = allocate(targets[3].clone())
        d_c, d_t, d_df, d_tf, d_reply = map(
            allocate, (candidate_ids, target_ids, draft_flags, target_flags, reply),
        )

        def launch(name, *values):
            return timed(worker.run, compiled[name], *values, config=config)[1]

        def read(host, device, rank=0):
            if ranks:
                return timed(worker.copy_from, host.data_ptr(), device.shards[rank].data_ptr,
                             host.nbytes, worker_id=rank)[1]
            return timed(worker.copy_from, host.data_ptr(), device.data_ptr, host.nbytes)[1]

        def compact_reply():
            flag, matched, *tokens = reply[0].tolist()
            if flag != 1:
                raise ValueError("Device detected nonfinite logits")
            return Verification(tuple(tokens[:matched + 1]), matched)

        def measure(mode, target):
            start = perf_counter_ns()
            copy_us = check_us = 0.
            if mode == "split_full_logits":
                dispatch_us = launch("sample", d_draft, d_c)
                copy_us += read(candidate_ids[:CANDIDATES], d_c) + read(draft_copy, d_draft)
                draft_tokens, check_us = timed(checked_ids, candidate_ids, draft_copy, CANDIDATES)
                dispatch_us += launch("sample", target, d_t)
                copy_us += read(target_ids[:WIDTH], d_t) + read(target_copy, target)
                target_tokens, check_target_us = timed(checked_ids, target_ids, target_copy, WIDTH)
                result, accept_us = timed(Verification.resolve, draft_tokens, target_tokens)
                check_us += check_target_us + accept_us
            else:
                if mode == "split_compact":
                    dispatch_us = launch("draft", d_draft, d_c, d_df)
                    copy_us += read(candidate_ids[:CANDIDATES], d_c)
                    dispatch_us += launch("verify", target, d_c, d_df, d_t, d_tf, d_reply)
                else:
                    dispatch_us = launch("resident", d_draft, target, d_c, d_df, d_t, d_tf, d_reply)
                copy_us += read(reply, d_reply)
                result, check_us = timed(compact_reply)
            wall_us = (perf_counter_ns() - start) / 1000
            return result, dict(wall_us=wall_us, dispatch_us=dispatch_us, d2h_us=copy_us, host_check_us=check_us)

        for index in range(args.warmup + args.rounds):
            matched = (0, 3, CANDIDATES)[index % 3]
            for offset in range(len(modes)):
                mode = modes[(index + offset) % len(modes)]
                result, times = measure(mode, d_targets[matched])
                assert result == expected[matched], (mode, result, expected[matched])
                if index >= args.warmup:
                    measurements[mode].append(dict(accepted=matched, **times))

        if ranks:
            for matched, target in d_targets.items():
                launch("resident", d_draft, target, d_c, d_df, d_t, d_tf, d_reply)
                for rank in ranks:
                    read(reply, d_reply, rank)
                    assert compact_reply() == expected[matched], (rank, matched)

        # Ensure the compact paths preserve the host safety check on either stage.
        for source in ("draft", "verify"):
            for value in (float("nan"), float("inf"), float("-inf")):
                bad = (draft if source == "draft" else targets[3]).clone()
                bad[1, 121] = value
                if ranks:
                    worker.copy_to(d_bad.shards[0].data_ptr, bad.data_ptr(), bad.nbytes, worker_id=0)
                else:
                    worker.copy_to(d_bad.data_ptr, bad.data_ptr(), bad.nbytes)
                d_source, d_target = (d_bad, d_targets[3]) if source == "draft" else (d_draft, d_bad)
                launch("resident", d_source, d_target, d_c, d_df, d_t, d_tf, d_reply)
                read(reply, d_reply)
                assert reply[0, 0].item() == 0, (source, value)

    summary = {}
    traffic = dict(
        split_full_logits=draft_copy.nbytes + target_copy.nbytes + (CANDIDATES + WIDTH) * PAD * 4,
        split_compact=CANDIDATES * PAD * 4 + reply.nbytes,
        resident_round=reply.nbytes,
    )
    for name, samples in measurements.items():
        stats = {key: round(median(row[key] for row in samples), 2)
                 for key in ("wall_us", "dispatch_us", "d2h_us", "host_check_us")}
        stats["wall_p95_us"] = round(quantiles([row["wall_us"] for row in samples], n=100)[94], 2)
        summary[name] = dict(d2h_bytes=traffic[name], **stats)
    return dict(passed=True, runtime="l3" if ranks else "l2",
                runtime_devices=args.devices or [args.device], rounds=args.rounds, warmup=args.warmup,
                vocab=VOCAB, width=WIDTH, acceptance_cases=list(expected), exact_finite_outputs=True,
                rejected_nonfinite_cases=6, summary=summary, samples=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--devices", type=lambda value: [int(x) for x in value.split(",")],
                        help="Run the resident L3 wrapper on this list of runtime device IDs")
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 3 or args.warmup < 1:
        parser.error("Use at least three timed rounds and one warmup")
    report = run(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: value for k, value in report.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
