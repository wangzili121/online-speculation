#!/usr/bin/env python3
"""Run the same real greedy DSV4.1 acceptance workload on two A3 hosts."""

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

from blockspec.pypto_dsv41_cluster import ClusterDispatcher, ControlLink, RankPlan
from validate_dsv41_pypto_cluster import check_visible_devices
from verify_dsv41_pypto_acceptance import TokenDivergence, audit_session, check_generated_prefix, request_parity


PROMPTS = (
    "Continue the list of positive integers from 1.",
    "The capital city of France is",
)
SAMPLING = dict(temperature=0, ignore_eos=True)


def write_report(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            while data := stream.read(8 * 1024 * 1024):
                digest.update(data)
    return digest.hexdigest()


def generate(runner, prompts, steps, assignments, *, reference=None):
    import torch
    from pypto_serving.config.types import DecodeBatch, PrefillBatch, SamplingParams

    pages = {group.name: list(range(group.max_blocks_per_seq)) for group in runner.cache_plan.groups}
    generated = []
    counts = Counter()
    speculative = reference is not None
    mode = "orthrus" if speculative else "target"
    for index, (prompt, partition) in enumerate(zip(prompts, assignments, strict=True)):
        request_id = f"{mode}-request-{index}"
        expected = reference[index]["token_ids"] if speculative else None
        # Exercise the native chunked-prefill path, including the left-over chunk.
        for start in range(0, len(prompt), runner.config.chunk_tokens):
            ids = prompt[start:start + runner.config.chunk_tokens]
            batch = PrefillBatch(request_ids=[request_id], token_ids=torch.tensor(ids),
                input_embeddings=None, seq_lens=[start + len(ids)], chunk_lens=[len(ids)],
                chunk_offsets=[0], chunk_starts=[start], prompt_lens=[len(prompt)],
                block_ids_by_group=[pages], cache_partitions=[partition])
            prefill = runner.run(batch, "prefill")
        output = [int(prefill.sampled_token_ids[0])]
        if speculative:
            check_generated_prefix(index, expected, output, phase="prefill")
        print(json.dumps(dict(stage="prefill_token", mode=mode, request=index, dp=partition,
                              token=output[0], compared=speculative)), flush=True)
        while len(output) < steps:
            committed = len(runner.ledger.requests[request_id].history)
            if committed != len(prompt) + len(output) - 1:
                raise RuntimeError("The next greedy token must be the only uncommitted anchor")
            batch = DecodeBatch(request_ids=[request_id], token_ids=torch.tensor([output[-1]]),
                hidden_states=None, seq_lens=torch.tensor([committed + 1]),
                sampling_params=[SamplingParams(temperature=0, top_p=1)],
                block_ids_by_group=[pages], cache_partitions=[partition])
            result = runner.run_orthrus(batch) if speculative else runner.run(batch, "decode")
            if result.accepted_token_ids is None:
                emitted = result.sampled_token_ids.tolist()
                counts["plain_decode_steps"] += 1
                phase = "decode"
            else:
                emitted = result.accepted_token_ids[0]
                counts["draft_rounds"] += 1
                counts["proposed_candidates"] += result.num_draft_tokens[0]
                counts["accepted_candidates"] += len(emitted) - 1
                phase = "verify/accept"
            if not emitted:
                raise RuntimeError("A greedy decode step must emit at least one token")
            output.extend(emitted[:steps - len(output)])
            if speculative:
                check_generated_prefix(index, expected, output, phase=phase)
        runner.release_finished_requests([request_id])
        generated.append(dict(index=index, dp=partition, prompt_ids=prompt, token_ids=output,
            finish_reason="length", usage=dict(prompt_tokens=len(prompt), completion_tokens=len(output))))
    return generated, dict(counts)


def request_receipts(receipt_path, mode, index, phase):
    request_id = f"{mode}-request-{index}"
    with receipt_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["phase"] != phase:
                continue
            for position, owner in enumerate(row["owners"]):
                if owner["request_id"] == request_id:
                    yield row, position


def prefill_difference(receipt_path, request_index):
    """Summarize the first host-local activation mismatch without sharing tensor payloads."""
    modes = ("target", "orthrus")
    traces = {
        mode: [row for row, _ in request_receipts(receipt_path, mode, request_index, "prefill")]
        for mode in modes
    }
    if not all(traces.values()):
        return {}
    result = {mode: traces[mode][-1]["prefill_topk"] for mode in modes}
    for chunk, (baseline, candidate) in enumerate(zip(*(traces[mode] for mode in modes), strict=True)):
        paths = [row["prefill_activation_trace"] for row in (baseline, candidate)]
        if not all(paths):
            continue
        left, right = (json.loads(Path(path).read_text()) for path in paths)
        for a, b in zip(left, right, strict=True):
            if a != b:
                result["first_activation_difference"] = dict(chunk=chunk, target=a, orthrus=b)
                return result
    return result


def verification_difference(receipt_path, request_index):
    """Return the final candidate block and target decisions for the diverging request."""
    result = {}
    for phase in ("draft", "verify"):
        receipts = list(request_receipts(receipt_path, "orthrus", request_index, phase))
        if receipts:
            row, position = receipts[-1]
            result[phase] = dict(epoch=row["native_epoch"], tokens=row["greedy_rows"][position])
    return result


def run(args):
    import torch
    from pypto_serving_a3.cache import CachePlan
    from pypto_serving_a3.config import A3Config
    from pypto_serving_a3.runner import A3Runner
    from pypto_serving_a3.session import NativeSession
    from pypto_serving_a3.source import KernelSource, integration_identity
    from pypto_serving_a3.weights import A3Checkpoint

    from blockspec.pypto_dsv41_draft import read_draft_artifact
    from blockspec.vllm_ascend.head_contract import find_weight_index

    torch.set_num_threads(args.host_threads)
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "180")
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "180")
    plan = RankPlan(counts=(8, 8), tp=4, node=args.node)
    source = KernelSource.resolve(cluster=True)
    checkpoint = A3Checkpoint(args.model, source)
    checkpoint_identity = checkpoint.identity()
    config = A3Config.from_source(source, max_num_seqs=4, max_context=args.context,
                                 step_tokens=128, chunk_tokens=16)
    tokenize = source.module("prefill_prepare").tokenize_prompt
    prompts = [tokenize(text, args.model, checkpoint.text["vocab_size"]) for text in (args.prompt or PROMPTS)]
    if any(len(prompt) + args.tokens + 31 > args.context for prompt in prompts):
        raise ValueError("Context must cover the prompt, requested output and the full K32 draft block")
    assignments = [0, 2]
    _, head, head_files = read_draft_artifact(args.model, args.head)
    head_identity = fingerprint([args.head / "manifest.json", find_weight_index(args.head)]
                                + [args.head / file for file in sorted(set(head_files.values()))])
    physical = [int(card) for card in os.environ["ORTHRUS_CLUSTER_PHYSICAL_CARDS"].split(",")]
    devices = list(range(len(plan.local_ranks)))
    if len(physical) != len(devices):
        raise ValueError("Each cluster worker must have exactly eight physical cards")
    mapping = check_visible_devices(physical, native=source.module("moe_transport"))
    scripts = Path(__file__).resolve().parent
    blockspec = scripts.parent / "src/blockspec"
    build = dict(
        adapter=integration_identity()["adapter_files_sha256"],
        orthrus=fingerprint([Path(__file__), scripts / "verify_dsv41_pypto_acceptance.py"]
                          + sorted(blockspec.glob("pypto_dsv41_*.py"))),
        native_tree=os.environ["ORTHRUS_CLUSTER_LIB_TREE"],
        torch=torch.__version__,
    )
    identity = hashlib.sha256(json.dumps(build, sort_keys=True).encode()).hexdigest()
    key = args.key_file.read_bytes().strip()
    address = args.master_address, args.port
    print(f"Node {args.node} waiting for peer at {address[0]}:{address[1]}", flush=True)
    with ControlLink.connect(plan.node, address, key, args.connect_timeout) as link:
        link.connection.settimeout(args.timeout)
        request = dict(build=identity, checkpoint=checkpoint_identity, head=head_identity,
                       prompts=prompts, partitions=assignments, tokens=args.tokens, context=args.context,
                       trace_prefill=args.trace_prefill)
        if link.exchange("acceptance/configuration", request) != request:
            raise ValueError("Nodes must use the same checkout, model, head and workload")
        print(f"Node {args.node} preparing HOST Engram contracts", flush=True)
        cache = source.module("prefill_prepare").checkpoint_cache(
            args.model, config.topology, progress=lambda event: print(json.dumps(event), flush=True)
        )
        digests = {str(layer): [item["host_payload_sha256"] for item in value["engram_shards"]]
                   for layer, value in cache[0]["engram_contracts"].items()}
        if link.exchange("acceptance/engram", digests) != digests:
            raise ValueError("Engram checkpoint payloads differ between nodes")
        cache_plan = CachePlan.from_native_templates(source, cache[0]["cache_ABI"], config)
        results, audits = {}, {}
        modes = ("target", "orthrus")
        native_dir = args.output / "native"
        cluster = ClusterDispatcher(plan, link, build_id=identity)
        session = NativeSession(source, args.model, devices, native_dir, config, checkpoint.text["vocab_size"],
                                cache_plan, orthrus_head=str(args.head), cluster=cluster, prepared_cache=cache,
                                defer_draft=True, trace_prefill=args.trace_prefill)
        runner = A3Runner(source, config, cache_plan, session, checkpoint.text["vocab_size"], greedy_tokens=True)
        try:
            for mode in modes:
                print(f"Node {args.node} running {mode} on global ranks {list(plan.local_ranks)}", flush=True)
                expected = results["target"]["requests"] if mode == "orthrus" else None
                rows, stats = generate(runner, prompts, args.tokens, assignments, reference=expected)
                results[mode] = dict(status="executed", devices=devices, sampling=SAMPLING,
                                     requests=rows, speculation=stats)
                write_report(args.output / mode / "result.json", results[mode])
        except TokenDivergence as error:
            diagnostics = {"prefill": prefill_difference, "verify/accept": verification_difference}
            collect = diagnostics.get(error.details["phase"])
            if collect is not None:
                try:
                    error.details["native"] = collect(session.receipt_path, error.details["request"])
                except (OSError, KeyError, ValueError, IndexError) as trace_error:
                    error.details["native_trace_error"] = str(trace_error)
            raise
        finally:
            runner.close()
        for mode in modes:
            _, phases, restored = audit_session(native_dir, speculative=mode == "orthrus", node_local=True,
                                                request_prefix=f"{mode}-request-")
            audits[mode] = dict(phases=phases, verifier_restored_bytes=restored)
        tokens = request_parity(results["target"], results["orthrus"])
        summary = dict(passed=True, node=plan.node, global_ranks=list(plan.local_ranks), build_id=identity,
                       topology=dict(dp=4, tp=4, ep=16), physical_to_runtime=mapping, compared_tokens=tokens,
                       checkpoint=checkpoint_identity, head_sha256=head_identity, head_role=head["artifact_role"],
                       target=audits["target"], orthrus=audits["orthrus"],
                       speculation=results["orthrus"]["speculation"])
        peer = link.exchange("acceptance/results", summary)
        if (peer["node"] != 1 - plan.node or peer["global_ranks"] != list(RankPlan((8, 8), 4, 1 - plan.node).local_ranks)
                or not peer["passed"] or peer["compared_tokens"] != tokens or peer["build_id"] != identity):
            raise ValueError("The peer did not complete the matching eight-rank acceptance")
        summary["peer"] = peer
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--port", type=int, default=29641)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append", help="Supply exactly two raw prompts, one per host")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--host-threads", type=int, default=4)
    parser.add_argument("--trace-prefill", action=argparse.BooleanOptionalAction, default=True,
                        help="Record hashes of actual prefill layer outputs for first-divergence diagnosis")
    parser.add_argument("--connect-timeout", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum seconds of peer inactivity")
    args = parser.parse_args()
    if (args.tokens < 2 or args.host_threads < 1 or args.timeout < 1
            or (args.prompt is not None and len(args.prompt) != 2)):
        parser.error("Require at least two output tokens, positive timeouts/threads and exactly two prompts")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = run(args)
    except Exception as error:
        failure = dict(passed=False, node=args.node, error=f"{type(error).__name__}: {error}")
        if isinstance(error, TokenDivergence):
            failure["divergence"] = error.details
        write_report(args.output / "acceptance.json", failure)
        print(json.dumps(failure), file=sys.stderr, flush=True)
        raise
    write_report(args.output / "acceptance.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
