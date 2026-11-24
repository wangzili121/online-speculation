#!/usr/bin/env python3
"""Compare frozen FFN backward on real W8A8 weights with fixed EP-return inputs."""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from threading import RLock
from time import perf_counter
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 1:
        parser.error("provide a fresh report path and a positive iteration count")

    from blockspec.dsv41_device_update import SuffixGradient
    from blockspec.dsv41_serving_online import DSV41SequenceSpan
    from blockspec.pypto_dsv41_resident_primitives import DSV41ResidentPrimitives
    from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime
    from models.deepseek_v4_1_flash_a3.weights import CheckpointOperands

    spec = importlib.util.spec_from_file_location("blockspec.frozen_tail_baseline", args.baseline_file)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    torch.set_num_threads(2)
    torch.manual_seed(41131)
    block, hidden, copies, layer = 32, 5120, 4, 39
    ranks = tuple(range(4))
    checkpoint = CheckpointOperands(args.model)
    prefix = f"layers.{layer}."
    shared = checkpoint.expert(prefix + "ffn.shared_experts", partitions=4)
    operands = {prefix + name: checkpoint.load(prefix + name) for name in (
        "ffn.gate.weight", "ffn_norm.weight", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    )}
    operands[prefix + "ffn.gate.weight"] = operands[prefix + "ffn.gate.weight"].float()

    def random(shape, dtype=torch.bfloat16):
        return (torch.randn(shape) * .1).to(dtype)

    cases = []
    with ResidentNetworkRuntime(ranks, dict(dp=1, tp=4, ep=4)) as runtime:
        draft = ResidentDraftAttention.__new__(ResidentDraftAttention)
        draft.resident, draft.rank_ids = runtime, {rank: rank for rank in ranks}
        draft.compile_lock, draft.programs, draft.native = RLock(), {}, native
        draft.config = {"routed_scaling_factor": 1.5}
        draft.output = args.output.parent / (args.output.stem + "-compiled")
        draft.output.mkdir(parents=True, exist_ok=False)
        for name, weight in operands.items():
            runtime.pin(weight, ranks, readonly=True, name=name)
        banks = {rank: tuple(runtime.weight(weight, rank) for weight in shared) for rank in ranks}
        weights = SimpleNamespace(for_layer=lambda _: {"expert": lambda rank, _: banks[rank]})
        synchronization = {"update_number": 1, "reverse": {"protocol": "dsv41-synchronized-reverse-v1",
                                                          "layers": [layer], "ep_bucket_rows": [32]}}
        for begin, active in ((0, 32), (32, 96), (96, 128)):
            physical = active // len(ranks)
            captures, gradients, returned = {}, {}, {}
            for rank in ranks:
                span = DSV41SequenceSpan.from_record(
                    {"cp_token_range": (rank * physical, (rank + 1) * physical), "SP_valid_rows": physical},
                    begin, begin + block,
                )
                values = {
                    "normed": random((block, hidden)), "collapsed": random((block, hidden)),
                    "changed": random((block * copies, hidden)),
                    "post": torch.rand((block, copies), dtype=torch.float32),
                    "pre": torch.rand((block, copies), dtype=torch.float32),
                    "comb": torch.softmax(torch.randn((block * copies, copies)), dim=-1),
                    "combined": random((block, hidden)),
                    "expert_outputs": random((block * 6, hidden)),
                    "route_weights": torch.softmax(torch.randn((block, 6)), dim=-1),
                    "route_ids": torch.stack([torch.randperm(384)[:6] for _ in range(block)]).int(),
                    "router_logits": random((block, 384), torch.float32),
                }
                captures[rank] = dict(values, token_begin=begin, token_end=begin + block, sp_span=span)
                dy_hidden, dy_pre = random((block * copies, hidden)), random((block, copies), torch.float32)
                routed = random((block, hidden))
                dy_hidden[span.rows * copies:].zero_()
                dy_pre[span.rows:].zero_()
                routed[span.rows:].zero_()
                gradients[rank] = SuffixGradient(dy_hidden, dy_pre)
                returned[rank] = routed
                values.update(dy_hidden=dy_hidden, dy_pre=dy_pre, routed=routed)
                for name, value in values.items():
                    runtime.pin(value, (rank,), readonly=True, name=f"fixture/{active}/{rank}/{name}")

            arms = {}
            for label, cls in (("before", baseline.DSV41ResidentPrimitives), ("after", DSV41ResidentPrimitives)):
                engine = cls.__new__(cls)
                engine.resident, engine.draft, engine.ranks = runtime, draft, ranks
                engine.plan, engine.native = SimpleNamespace(layers=(layer,)), SimpleNamespace(weights=weights)
                # Give both local backwards the same returned expert dX. The
                # production EP schedule is validated separately by the owner/peer tests.
                engine._ep_dispatch_packets = lambda sources, _: (sources, None)
                engine._ep_exchange = lambda _stage, packet: packet
                engine._ep_owner_expert_packets = lambda _layer, packet, _bucket: packet
                engine._ep_routed_input_gradients = lambda _packet, _old, sources, _bucket: {
                    rank: returned[rank] for rank in sources
                }
                arms[label] = engine

            def run(engine, read=False):
                started = perf_counter()
                output = engine.reverse_frozen_tail_owner_batch(
                    layer=layer, ranks=ranks, captured=captures, states=dict.fromkeys(ranks),
                    output_gradients=gradients, synchronization=synchronization,
                )
                elapsed = (perf_counter() - started) * 1e3
                host = {
                    rank: tuple(runtime.read_output(value, rank).clone() for value in (pair.hidden, pair.pre_mix))
                    for rank, pair in output.items()
                } if read else None
                runtime.release_values()
                return elapsed, host

            before, after = (run(engine, read=True)[1] for engine in arms.values())
            for rank, pair in before.items():
                for a, b in zip(pair, after[rank], strict=True):
                    assert torch.isfinite(a).all()
                    assert torch.equal(a, b), (active, rank, (a.float() - b.float()).abs().max().item())
                if captures[rank]["sp_span"].rows:
                    assert pair[0].count_nonzero()
                else:
                    assert all(not tensor.any() for tensor in pair)
            samples = {label: [] for label in arms}
            for _ in range(args.iterations):
                for label, engine in arms.items():
                    samples[label].append(run(engine)[0])
            cases.append(dict(active_rows=active, first_request_row=begin,
                              local_rows=[captures[rank]["sp_span"].rows for rank in ranks], exact=True,
                              median_ms={label: median(data) for label, data in samples.items()},
                              iterations=args.iterations))

    report = dict(passed=True, phase="resident-frozen-ffn-owner-backward", tp=4, layer=layer,
                  checkpoint=args.model.name, scope="actual W8A8 layer, synthetic captures, fixed EP-return gradients",
                  cases=cases)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
