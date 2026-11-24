#!/usr/bin/env python3
"""Compare resident TP4 terminal VJPs using matched DSV4.1-sized inputs."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from statistics import median
from threading import RLock
from time import perf_counter
from types import SimpleNamespace

import torch

from blockspec.dsv41_serving_online import DSV41SequenceSpan
from blockspec.pypto_dsv41_resident_primitives import DSV41ResidentPrimitives
from blockspec.pypto_dsv41_runtime import ResidentDraftAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--physical-cards", type=int, nargs=4, metavar="NPU")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")

    from models.deepseek_v4_1_flash_a3 import network
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime

    spec = importlib.util.spec_from_file_location("blockspec.terminal_baseline", args.baseline_file)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    torch.set_num_threads(2)
    torch.manual_seed(4113)
    ranks = tuple(range(4))
    rows, hidden, streams, vocab = 32, 5120, 4, 129280
    width = vocab // len(ranks)
    results = {}

    with ResidentNetworkRuntime(ranks, dict(ep=4, dp=1, tp=4)) as runtime:
        draft = ResidentDraftAttention.__new__(ResidentDraftAttention)
        draft.resident, draft.rank_ids = runtime, {rank: rank for rank in ranks}
        draft.compile_lock, draft.programs = RLock(), {}
        draft.output = args.output.parent / (args.output.stem + "-compiled")
        draft.output.mkdir(parents=True, exist_ok=False)
        draft.native = lambda _: network

        weights = {}
        norm = (1 + torch.randn((1, hidden)) / 16).to(torch.bfloat16)
        runtime.pin(norm, ranks, readonly=True, name="norm.weight")
        for rank in ranks:
            weights[rank] = torch.empty((hidden, width), dtype=torch.bfloat16).normal_(0, .0125)
            runtime.pin(weights[rank], (rank,), readonly=True, name=f"head/{rank}")
        native = SimpleNamespace(weights=SimpleNamespace(head_for_rank=weights.__getitem__))

        def primitive(cls):
            instance = cls.__new__(cls)
            instance.resident, instance.draft, instance.native = runtime, draft, native
            instance.ranks, instance._scalar_reads = ranks, 0
            instance.config = SimpleNamespace(teacher_rows=7)
            return instance

        arms = {"before": primitive(old.DSV41ResidentPrimitives),
                "after": primitive(DSV41ResidentPrimitives)}
        for kind, valid, feedback_capacity, counts in (
            ("tv", 7, 7, (8, 8, 8, 8)),
            ("forward_kl", 7, 7, (8, 8, 8, 8)),
            ("tv", 1, 7, (32, 0, 0, 0)),
            ("tv", 1, 1, (32, 0, 0, 0)),
            ("forward_kl", 31, 31, (8, 8, 8, 8)),
        ):
            arms["after"].config.teacher_rows = feedback_capacity
            data = {}
            start = 0
            for rank, count in enumerate(counts):
                values = {
                    "hidden": torch.randn((rows, streams, hidden), dtype=torch.bfloat16),
                    "pre_mix": torch.rand((rows, streams), dtype=torch.float32),
                }
                for name, value in values.items():
                    runtime.pin(value, (rank,), readonly=True, name=f"{kind}/{valid}/{rank}/{name}")
                data[rank] = dict(values, sp_span=DSV41SequenceSpan(0, start if count else 0, count))
                start += count
            full = torch.empty((valid, vocab)).normal_(0, 2)
            student = full.to(torch.bfloat16)
            teacher = (full + torch.randn_like(full) * .35).to(torch.bfloat16)

            def run(arm, check=False):
                prim = arms[arm]
                collectives, loss_upload = {}, {}
                original = prim._tp_all_reduce
                original_loss = prim._loss_on_rank
                if check:
                    def record(values, *, stage):
                        inputs = [runtime.read_output(values[rank], rank).clone() for rank in ranks]
                        reduced = original(values, stage=stage)
                        collectives[stage] = (inputs, [runtime.read_output(reduced[rank], rank).clone()
                                                      for rank in ranks])
                        return reduced
                    prim._tp_all_reduce = record
                    def record_loss(rank, *args):
                        transfers = runtime.transfer_stats[rank]
                        before = transfers.get("h2d_bytes", 0)
                        outputs = original_loss(rank, *args)
                        loss_upload[rank] = transfers.get("h2d_bytes", 0) - before
                        return outputs
                    prim._loss_on_rank = record_loss
                start = perf_counter()
                try:
                    loss, gradient = prim.loss_vjp(student, teacher, valid, kind)
                    outputs = prim.lm_head_vjp(gradient, data)
                finally:
                    if check:
                        del prim._tp_all_reduce
                        del prim._loss_on_rank
                elapsed = (perf_counter() - start) * 1000
                values = []
                if check:
                    for rank in ranks:
                        for tensor in (outputs.hidden[rank], outputs.pre_mix[rank]):
                            values.append(runtime.read_output(tensor, rank).clone())
                    values.append(torch.tensor(prim.read_scalar(loss)))
                runtime.release_values()
                return elapsed, values, collectives, sum(loss_upload.values())

            print(json.dumps(dict(stage="warm", kind=kind, rows=valid, feedback_capacity=feedback_capacity)), flush=True)
            _, before, before_collectives, before_upload = run("before", check=True)
            _, after, after_collectives, after_upload = run("after", check=True)
            _, repeated, repeated_collectives, _ = run("before", check=True)
            errors = [float((a.float() - b.float()).abs().max()) for a, b in zip(before, after)]
            assert all(torch.isfinite(t).all() for t in before + after)
            assert torch.equal(before[-1], after[-1]) and after[-1] > 0
            updates = {}
            for stage, (old_in, old_out) in before_collectives.items():
                new_in, new_out = after_collectives[stage]
                assert all(torch.equal(a, b) for a, b in zip(old_in, new_in)), stage
                # Up to 3 BF16 additions of identical contributors in the collective.
                bound = (3 * torch.finfo(torch.bfloat16).eps
                         * torch.stack([a.float().abs() for a in old_in]).sum(dim=0))
                for a, b in zip(old_out, new_out):
                    assert ((a.float() - b.float()).abs() <= bound).all(), stage
                updates[stage] = {
                    "before_after_max_abs": max((a.float() - b.float()).abs().max().item()
                                                 for a, b in zip(old_out, new_out)),
                    "before_repeat_max_abs": max((a.float() - b.float()).abs().max().item()
                                                  for a, b in zip(old_out, repeated_collectives[stage][1])),
                }
            for i in range(len(before) - 1):
                if before[i].dtype == torch.float32:
                    torch.testing.assert_close(before[i], after[i], atol=5e-5, rtol=.02)
                else:
                    torch.testing.assert_close(before[i], after[i], atol=5e-6, rtol=.02)
            assert any(out.abs().max() > 0 for out in before[:-1])
            samples = {arm: [] for arm in arms}
            for _ in range(12):
                for arm in arms:
                    elapsed, _, _, _ = run(arm)
                    samples[arm].append(elapsed)
            result = dict(loss=after[-1].item(), exact_collective_inputs=True, collective_outputs=updates,
                          gradient_max_abs=max(errors[:-1]), baseline_repeat_max_abs=max(
                              (a.float() - b.float()).abs().max().item() for a, b in zip(before, repeated)),
                          baseline_median_ms=median(samples["before"]),
                          optimized_median_ms=median(samples["after"]), repetitions=len(samples["before"]),
                          sequence_rows=counts, feedback_capacity=feedback_capacity,
                          loss_h2d_bytes={"before": before_upload, "after": after_upload},
                          timings={arm: median(values) for arm, values in samples.items()})
            key = f"{kind}/{valid}/capacity{feedback_capacity}"
            results[key] = result
            print(json.dumps(result), flush=True)

    report = dict(
        passed=True, device="Ascend 910B3", physical_devices=args.physical_cards,
        tp=4, width=hidden, vocabulary=vocab, block=rows,
        baseline=args.baseline_commit, evidence="synthetic matched terminal VJP with local HCCL",
        cases=results,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
