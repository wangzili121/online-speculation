"""Compare target verifier Engram lookup state and history rollback on the pinned source."""

import argparse
import ast
from contextlib import ExitStack, contextmanager, nullcontext
from inspect import getsource
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from textwrap import dedent
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.deepseek_v4_1_flash_a3 import network, prefill_network_engram as engram


def evaluate_write_history():
    for item in ast.walk(ast.parse(getsource(network.run_network_chunk))):
        if isinstance(item, ast.Call) and getattr(item.func, "id", None) == "prepare_network_engram":
            value = next(key.value for key in item.keywords if key.arg == "write_history")
            return compile(ast.Expression(value), "<verifier Engram write policy>", "eval")
    raise AssertionError("The paired network does not bind HOST Engram lookup")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(41131)
    old_method = args.baseline_file.read_text()
    a = old_method.index("    def verify_candidates(")
    b = old_method.index("    @staticmethod", a)
    globals_ = dict(ExitStack=ExitStack, ORTHRUS_BLOCK=32, perf_counter=perf_counter)
    exec(dedent(old_method[a:b]), globals_)
    baseline = globals_["verify_candidates"]
    write_policy = evaluate_write_history()
    assert all(eval(write_policy, {}, dict(draft=None, phase=p)) for p in ("prefill", "decode", "accept"))
    assert not eval(write_policy, {}, dict(draft=object(), phase="draft"))
    assert not eval(write_policy, {}, dict(draft=None, phase="verify"))

    repeats, ranks, capacity = 24, tuple(range(4)), 32
    topology = dict(dp=1, tp=4, ep=4)
    constants = dict(token_map=torch.arange(256, dtype=torch.int32), layer_ids=(1, 14), pad_id=2,
                     multipliers=torch.arange(8), primes=torch.arange(48), offsets=torch.arange(48))
    contract = dict(capacity=capacity, checkpoint="/test-model", topology=topology, hash_constants=constants)
    buffers = {layer: torch.randn(capacity, 6144, dtype=torch.bfloat16) for layer in (1, 14)}
    rows = tuple(99 + i for i in range(capacity))
    ids = torch.tensor(rows)
    mask = torch.ones(capacity, dtype=torch.int32)
    mask[-1] = 0  # Include a dead row, which the original writer publishes as -1.
    cases = []

    with TemporaryDirectory() as directory, patch.object(
        engram, "engram_owner", side_effect=lambda _config, layer, _role: ("fixture", layer),
    ):
        output = Path(directory)
        counter = 0
        for page_count in (2, 32, 128):
            pages = tuple(range(1, page_count + 1))
            start = (page_count - 1) * 128 - 8
            slots = torch.tensor([(pages[pos // 128], pos % 128) for pos in range(start, start + capacity)],
                                 dtype=torch.int32)
            original = dict(full_input_ids=ids, positions=torch.arange(start, start + capacity),
                            cp_token_range=(0, 8, 8, 32), metadata={"slot_mapping": slots})
            ready = dict(original, full=buffers, local=buffers,
                         hashes=torch.randint(0, 256, (capacity, 2, 24), dtype=torch.int32),
                         mask=mask, local_mask=mask)
            batch = dict(epoch=1, token_map_id="fixture", engram_metadata_owner_layer=1,
                         cases={"fresh": [original] * len(ranks)})
            lookup = dict(contract, epoch=1, token_map_id="fixture", cases={"fresh": [ready] * len(ranks)})
            history = SimpleNamespace(prepared=set(), geometry_only=False,
                                      local_ranks=ranks, caches=[{"fresh": torch.arange(
                                          (page_count + 1) * 128, dtype=torch.int32,
                                      ).neg()} for _ in ranks])
            approved = [entry["fresh"].clone() for entry in history.caches]
            runtime = dict(topology=topology, capacity=capacity, reports=[{} for _ in ranks],
                           entries_for_stage=lambda _: {}, load_rotation=lambda *_: (None, None))

            @contextmanager
            def provisional():
                try:
                    yield
                finally:
                    history.prepared.discard((1, "fixture"))

            def run(method, *, fail=False):
                nonlocal counter
                counter += 1
                runtime["output"] = output / str(counter)
                session = network.GenerationSession.__new__(network.GenerationSession)
                session.closed, session.failed = False, False
                session.epoch, session.contract, session.engram_contracts, session.program = 0, contract, {}, None
                session._candidate_requests = lambda _: {(0, 0, 1, start): rows}
                session.weights = SimpleNamespace(rank_devices=dict(enumerate(ranks)))
                session.provider = SimpleNamespace(
                    max_requests_per_partition=4, request_history=lambda *_: (None, start),
                    capacity=capacity, set_execution_capacity=lambda _: None,
                    verification_ranges=lambda *_: ({}, pages), arenas=[{"fresh": {}}] * len(ranks),
                    history=history, provisional_dispatch=provisional,
                )
                session.runtime = SimpleNamespace(preserve_rank_cache_ranges=lambda _: nullcontext())
                chunk = dict(contract={"capacity": 128}, cache={"cache_ABI": [{}] * len(ranks)},
                             inputs={"input_contract": {"DP_engines": [{"requests": [None], "active": capacity}]}})
                observed = []

                def advance(_chunk, *, phase, **_options):
                    write = method is baseline or eval(write_policy, {}, dict(draft=None, phase=phase))
                    prepared = engram.prepare_network_engram(
                        contract, batch, history, {1: contract, 14: contract}, None,
                        runtime, lookup=lookup, write_history=write,
                    )
                    observed.append(prepared.states)
                    if fail:
                        raise RuntimeError("deliberate verify abort")
                    return dict(timings={})

                session._advance = advance
                before = perf_counter()
                try:
                    result = method(session, chunk, output=output)
                except RuntimeError as error:
                    assert fail and session.failed and str(error) == "deliberate verify abort"
                    return observed[0], None
                elapsed = (perf_counter() - before) * 1000
                assert session._verified_requests == {(0, 0, 1, start): rows}
                return observed[0], dict(result["timings"], wall_ms=elapsed)

            variants = dict(before=baseline, after=network.GenerationSession.verify_candidates)
            for fail in (False, True):
                for method in variants.values():
                    states, _ = run(method, fail=fail)
                    assert all(torch.equal(history.caches[rank]["fresh"], approved[rank]) for rank in ranks)
                    assert not history.prepared
                    for layer, ranks_state in states.items():
                        for record in ranks_state:
                            entry = record["fresh"]
                            assert torch.equal(entry["full"], ready["full"][layer])
                            assert torch.equal(entry["hashes"], ready["hashes"])
            timings = {
                mode: median(run(method)[1]["wall_ms"] for _ in range(repeats))
                for mode, method in variants.items()
            }
            cases.append(dict(pages_per_rank=page_count, verify_rows=capacity, wall_ms=timings))
    print(json.dumps(dict(passed=True, exact=True, abort_restored=True, repeats=repeats, cases=cases)))


if __name__ == "__main__":
    main()
