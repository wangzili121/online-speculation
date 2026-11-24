#!/usr/bin/env python3
"""Run the same real greedy DSV4.1 acceptance workload on two A3 hosts."""

import argparse
from collections import Counter
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import time

from blockspec.pypto_dsv41_cluster import ClusterDispatcher, ControlLink, RankPlan
from blockspec.dsv41_online import read_workload
from blockspec.dsv41_online_profile import OnlineUpdateProfile, active_profile, profile_online
from dsv41_gsm8k_report import study_workload
from summarize_dsv41_pypto_cluster import profile_comparisons, profile_head_versions
from validate_dsv41_pypto_cluster import check_visible_devices
from verify_dsv41_pypto_acceptance import (
    TokenDivergence, audit_online_update, audit_session, check_generated_prefix, request_parity,
)


PROMPTS = (
    "Continue the list of positive integers from 1.",
    "The capital city of France is",
)
SAMPLING = dict(temperature=0, ignore_eos=True)


def load_callable(specification):
    """Load an explicit production extension without an implicit fallback."""
    if not specification or ":" not in specification:
        raise ValueError("online updates require MODULE:CALLABLE resident primitives")
    module, name = specification.rsplit(":", 1)
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise TypeError(f"{specification} is not callable")
    return value


def online_bootstrap(args):
    if args.online_study is None:
        if args.online_primitives is not None or args.online_overrides is not None:
            raise ValueError("online primitives/overrides require --online-study")
        return None
    from blockspec.dsv41_online import DSV41OnlineConfig
    from blockspec.dsv41_online_bootstrap import DSV41OnlineBootstrap

    overrides = None if args.online_overrides is None else json.loads(args.online_overrides)
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("--online-overrides must be a JSON object")
    config = DSV41OnlineConfig.from_study(args.online_study, overrides=overrides)
    return DSV41OnlineBootstrap(load_callable(args.online_primitives), config)


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


def generate(runner, prompts, steps, assignments, *, identities=None, metadata=None,
             decoder=None, reference=None, speculative=False,
             request_start=0, profile_phase="update", capture_online=True):
    import torch
    from pypto_serving.config.types import DecodeBatch, PrefillBatch, SamplingParams

    pages = {group.name: list(range(group.max_blocks_per_seq)) for group in runner.cache_plan.groups}
    generated = []
    counts = Counter()
    mode = "orthrus" if speculative else "target"
    identities = list(range(len(prompts))) if identities is None else identities
    metadata = [{} for _ in prompts] if metadata is None else metadata
    profile = active_profile()
    for index, (prompt, partition, identity, attributes) in enumerate(
            zip(prompts, assignments, identities, metadata, strict=True)):
        request_id = f"{mode}-request-{request_start + index}"
        if profile is not None:
            profile.begin_request(request_id, phase=profile_phase, capture=capture_online)
            phase_started = time.perf_counter()
        expected = None if reference is None else reference[index]["token_ids"]
        # Exercise the native chunked-prefill path, including the left-over chunk.
        for start in range(0, len(prompt), runner.config.chunk_tokens):
            ids = prompt[start:start + runner.config.chunk_tokens]
            batch = PrefillBatch(request_ids=[request_id], token_ids=torch.tensor(ids),
                input_embeddings=None, seq_lens=[start + len(ids)], chunk_lens=[len(ids)],
                chunk_offsets=[0], chunk_starts=[start], prompt_lens=[len(prompt)],
                block_ids_by_group=[pages], cache_partitions=[partition])
            prefill = runner.run(batch, "prefill")
        if profile is not None:
            profile.requests[-1]["prefill_seconds"] = time.perf_counter() - phase_started
        output = [int(prefill.sampled_token_ids[0])]
        acceptance_histogram = Counter()
        if expected is not None:
            check_generated_prefix(index, expected, output, phase="prefill")
        print(json.dumps(dict(stage="prefill_token", mode=mode, request=index, dp=partition,
                              token=output[0], compared=expected is not None)), flush=True)
        while len(output) < steps:
            committed = len(runner.ledger.requests[request_id].history)
            if committed != len(prompt) + len(output) - 1:
                raise RuntimeError("The next greedy token must be the only uncommitted anchor")
            batch = DecodeBatch(request_ids=[request_id], token_ids=torch.tensor([output[-1]]),
                hidden_states=None, seq_lens=torch.tensor([committed + 1]),
                sampling_params=[SamplingParams(temperature=0, top_p=1)],
                block_ids_by_group=[pages], cache_partitions=[partition])
            if profile is not None:
                decode_started = time.perf_counter()
            result = runner.run_orthrus(batch) if speculative else runner.run(batch, "decode")
            if profile is not None:
                profile.requests[-1]["decode_seconds"] += time.perf_counter() - decode_started
            if result.accepted_token_ids is None:
                emitted = result.sampled_token_ids.tolist()
                counts["plain_decode_steps"] += 1
                phase = "decode"
            else:
                emitted = result.accepted_token_ids[0]
                counts["draft_rounds"] += 1
                counts["proposed_candidates"] += result.num_draft_tokens[0]
                counts["accepted_candidates"] += len(emitted) - 1
                acceptance_histogram[len(emitted) - 1] += 1
                phase = "verify/accept"
            if not emitted:
                raise RuntimeError("A greedy decode step must emit at least one token")
            output.extend(emitted[:steps - len(output)])
            if expected is not None:
                check_generated_prefix(index, expected, output, phase=phase)
        if profile is not None:
            phase_started = time.perf_counter()
        runner.release_finished_requests([request_id])
        if profile is not None:
            profile.requests[-1].update(dp=partition, generated_tokens=len(output),
                                        release_seconds=time.perf_counter() - phase_started)
        generated.append(dict(
            index=index, workload_id=identity, dp=partition,
            prompt_ids=prompt, token_ids=output,
            output_text=decoder(output) if decoder is not None else None,
            acceptance_histogram=dict(acceptance_histogram),
            reference_answer=attributes.get("reference_answer"),
            finish_reason="length",
            usage=dict(prompt_tokens=len(prompt), completion_tokens=len(output)),
        ))
    return generated, dict(counts)


def profile_workload(runner, link, profile, prompts, entries, steps, decoder, output):
    """Use one resident session for matched greedy probes and bounded updates."""
    probes = (3, 4)
    probe_tokens = 16
    optional_seconds_per_update = 60
    results = {}

    def phase(name, indices, partitions, *, speculative=True, capture=False, tokens=steps):
        first = len(profile.requests)
        print(json.dumps(dict(stage="online_profile_phase", phase=name)), flush=True)
        started = time.perf_counter()
        metadata = [entries[i] for i in indices]
        rows, stats = generate(
            runner, [prompts[i] for i in indices], tokens, partitions,
            identities=[row["id"] for row in metadata], metadata=metadata, decoder=decoder,
            speculative=speculative, request_start=first, profile_phase=name, capture_online=capture,
        )
        requests = profile.requests[first:]
        decode = sum(row["decode_seconds"] for row in requests)
        generated = sum(len(row["token_ids"]) for row in rows)
        rounds = stats.get("draft_rounds", 0)
        candidates = stats.get("proposed_candidates", 0)
        accepted = stats.get("accepted_candidates", 0)
        results[name] = {
            "requests": rows, "timing": requests, "speculation": stats,
            "wall_seconds": time.perf_counter() - started,
            "decode_seconds": decode, "decode_tokens_per_second": (generated - len(rows)) / decode if decode else None,
            "mean_accepted_candidates": accepted / rounds if rounds else None,
            "candidate_acceptance_rate": accepted / candidates if candidates else None,
        }
        write_report(output / f"online-profile-{name}.json", results[name])

    with profile_online(profile):
        phase("ar", probes, (0, 2), speculative=False, tokens=probe_tokens)
        phase("fixed_before", probes, (0, 2), tokens=probe_tokens)
        phase("update", (0, 0, 0, 0), (0, 2, 0, 0), capture=True)
        # Both nodes must make the same decision. Use the slower host and only
        # the final two repeat requests, where the DP owner is already warm.
        repeat = profile.records[2:4]
        affordable = len(repeat) == 2 and all(
            row["update_seconds"] <= optional_seconds_per_update for row in repeat
        )
        peer_affordable = link.exchange("acceptance/online-profile-extra", affordable)
        if affordable and peer_affordable:
            phase("update_extra", (1, 2), (0, 0), capture=True)
        phase("fixed_after", probes, (0, 2), tokens=probe_tokens)

    return results, profile_comparisons(results)


def benchmark_workload(runner, session, online, prompts, entries, steps, decoder, devices, output):
    """Run AR, frozen draft, then updating draft against the same greedy references."""
    results, warmup_seconds = {}, {}
    for mode in ("target", "fixed", "orthrus"):
        speculative = mode != "target"
        print(json.dumps(dict(stage="gsm8k_benchmark_phase", phase=mode)), flush=True)
        if mode != "orthrus":
            started = time.perf_counter()
            generate(runner, prompts[:1], min(steps, 8), [0], speculative=speculative, request_start=-1)
            warmup_seconds[mode] = time.perf_counter() - started
        else:
            started = time.perf_counter()
            online.bind(session.draft, session.native, session.cluster)
            session.online = online
            warmup_seconds["learner_bind"] = time.perf_counter() - started
        assignments = [index % session.config.topology["dp"] for index in range(len(prompts) - 1)]
        reference = results.get("target", {}).get("requests")
        started = time.perf_counter()
        rows, stats = generate(
            runner, prompts[1:], steps, assignments,
            identities=[row["id"] for row in entries[1:]], metadata=entries[1:], decoder=decoder,
            reference=reference, speculative=speculative, request_start=len(prompts) if mode == "fixed" else 0,
        )
        seconds = time.perf_counter() - started
        tokens = sum(len(row["token_ids"]) for row in rows)
        results[mode] = dict(
            status="executed", devices=devices, sampling=SAMPLING, requests=rows, speculation=stats,
            elapsed_seconds=seconds, output_tokens=tokens, output_tokens_per_second=tokens / seconds,
            online=online.summary() if mode == "orthrus" else None,
        )
        write_report(output / mode / "result.json", results[mode])
    return results, warmup_seconds


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
    import torch

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
        evidence = [json.loads(Path(path).read_text()) for path in paths]
        stages = [item["stages"] if isinstance(item, dict) else item for item in evidence]

        def identity(row):
            return row["stage"], row["rank"], row["case"]

        candidate = {identity(row): row for row in stages[1]}
        for a in stages[0]:
            b = candidate.get(identity(a))
            if b is None or a != b:
                difference = dict(chunk=chunk, target=a, orthrus=b)
                if (b is not None and isinstance(evidence[0], dict) and isinstance(evidence[1], dict)
                        and a.get("sha256") != b.get("sha256")):
                    key = f"{a['stage']}/rank-{a['rank']}/case-{a['case']}"
                    tensors = [torch.load(Path(path).parent / item["tensors"], map_location="cpu",
                                          weights_only=True) for path, item in zip(paths, evidence, strict=True)]
                    left, right = (item[key] for item in tensors)
                    if left.shape == right.shape and left.dtype == right.dtype:
                        delta = (left.float() - right.float()).abs()
                        values, indices = delta.flatten().topk(min(5, delta.numel()))
                        difference["tensor_difference"] = dict(
                            byte_exact=bool(torch.equal(left, right)),
                            mismatched=int(torch.count_nonzero(left != right)),
                            elements=left.numel(),
                            max_abs=float(delta.max()) if delta.numel() else 0.0,
                            mean_abs=float(delta.mean()) if delta.numel() else 0.0,
                            largest_flat_indices=indices.tolist(), largest_abs=values.tolist(),
                        )
                result["trace_position"] = evidence[0].get("position") if isinstance(evidence[0], dict) else None
                result["first_activation_difference"] = difference
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


def profile_acceptance(arms, comparisons, records, versions, *, teacher_rows, publications):
    expected_updates = 6 if "update_extra" in arms else 4
    passed = (all(row["divergence"] is None for row in comparisons.values())
              and len(records) == expected_updates
              and all(row["teacher_rows"] == teacher_rows for row in records)
              and versions.get("fixed_before") == [0]
              and versions.get("fixed_after") == [publications])
    return dict(
        passed=passed, expected_updates=expected_updates,
        compared_tokens=sum(row["matched_tokens"] for row in comparisons.values()),
        speculation=dict(sum((Counter(arm["speculation"]) for arm in arms.values()), Counter())),
    )


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
    online = online_bootstrap(args)
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "180")
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "180")
    plan = RankPlan(counts=(8, 8), tp=4, node=args.node)
    source = KernelSource.resolve(cluster=True)
    checkpoint = A3Checkpoint(args.model, source)
    checkpoint_identity = checkpoint.identity()
    config = A3Config.from_source(source, max_num_seqs=4, max_context=args.context,
                                 step_tokens=128, chunk_tokens=16)
    tokenize = source.module("prefill_prepare").tokenize_prompt
    entries = (read_workload(args.workload, args.max_requests) if args.workload is not None
               else [dict(id=f"prompt-{i}", prompt=text) for i, text in enumerate(args.prompt or PROMPTS)])
    identities = [row["id"] for row in entries]
    raw_prompts = [row["prompt"] for row in entries]
    prompts = [tokenize(text, args.model, checkpoint.text["vocab_size"]) for text in raw_prompts]
    from tokenizers import Tokenizer
    decoder = Tokenizer.from_file(str(args.model / "tokenizer.json")).decode
    required_tokens = max(args.tokens, 16) if args.online_profile else args.tokens
    if any(len(prompt) + required_tokens + 31 > args.context for prompt in prompts):
        raise ValueError("Context must cover the prompt, requested output and the full K32 draft block")
    if args.online_profile and (len(prompts) != 5 or len(set(map(tuple, prompts))) != 5):
        raise ValueError("Online profiling needs five distinct GSM8K prompts")
    if args.online_benchmark and len(prompts) < 2:
        raise ValueError("The benchmark needs one warmup prompt and at least one measured prompt")
    if args.report_train_requests is not None:
        split = args.report_train_requests
        if not 0 < split < len(prompts):
            raise ValueError("The GSM8K report needs both online training and held-out requests")
        if set(map(tuple, prompts[:split])).intersection(map(tuple, prompts[split:])):
            raise ValueError("GSM8K training and held-out prompts overlap after tokenization")
    assignments = [index % config.topology["dp"] for index in range(len(prompts))]
    if args.orthrus_only and args.require_online_update and not args.online_profile:
        # The short capture gate exercises feedback ownership on both hosts.
        assignments = [(0, 2, 1, 3)[index % 4] for index in range(len(prompts))]
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
        orthrus=fingerprint([Path(__file__), scripts / "verify_dsv41_pypto_acceptance.py",
                             scripts / "dsv41_gsm8k_report.py", scripts / "summarize_dsv41_gsm8k.py",
                             scripts / "summarize_dsv41_pypto_cluster.py"]
                          + sorted(blockspec.glob("pypto_dsv41_*.py"))
                          + sorted(blockspec.glob("dsv41_*.py"))),
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
                       prompts=prompts, workload_ids=identities, partitions=assignments,
                       tokens=args.tokens, context=args.context,
                       trace_prefill=args.trace_prefill, trace_position=args.trace_position,
                       short_speculative_dispatch=args.short_speculative_dispatch,
                       orthrus_only=args.orthrus_only, require_online_update=args.require_online_update,
                       online_profile=args.online_profile,
                       online_benchmark=args.online_benchmark,
                       report_train_requests=args.report_train_requests, report_batch_size=args.report_batch_size,
                       online=None if online is None else online.config.contract())
        peer_request = link.exchange("acceptance/configuration", request)
        if peer_request != request:
            mismatched = [name for name in request if peer_request.get(name) != request[name]]
            raise ValueError(
                "Nodes differ in acceptance configuration: " + ", ".join(mismatched)
            )
        print(f"Node {args.node} preparing HOST Engram contracts", flush=True)
        engram_started = time.perf_counter()
        cache = source.module("prefill_prepare").checkpoint_cache(
            args.model, config.topology, progress=lambda event: print(json.dumps(event), flush=True)
        )
        digests = {str(layer): [item["host_payload_sha256"] for item in value["engram_shards"]]
                   for layer, value in cache[0]["engram_contracts"].items()}
        if link.exchange("acceptance/engram", digests) != digests:
            raise ValueError("Engram checkpoint payloads differ between nodes")
        engram_seconds = time.perf_counter() - engram_started
        cache_plan = CachePlan.from_native_templates(source, cache[0]["cache_ABI"], config)
        results, audits = {}, {}
        modes = ("orthrus",) if args.orthrus_only and not args.online_profile else ("target", "orthrus")
        native_dir = args.output / "native"
        cluster = ClusterDispatcher(plan, link, build_id=identity)
        session = NativeSession(source, args.model, devices, native_dir, config, checkpoint.text["vocab_size"],
                                cache_plan, orthrus_head=str(args.head), cluster=cluster, prepared_cache=cache,
                                defer_draft=True, trace_prefill=args.trace_prefill,
                                trace_position=args.trace_position,
                                compact_verifier=args.report_train_requests is not None,
                                short_speculative_dispatch=args.short_speculative_dispatch,
                                online=None if args.online_benchmark or args.report_train_requests else online)
        runner = A3Runner(source, config, cache_plan, session, checkpoint.text["vocab_size"], greedy_tokens=True)
        profile = OnlineUpdateProfile(args.output / "online-profile.jsonl") if args.online_profile else None
        study = None
        try:
            if profile is not None:
                profile_arms, comparisons = profile_workload(
                    runner, link, profile, prompts, entries, args.tokens, decoder, args.output,
                )
                versions = profile_head_versions(session.receipt_path, profile.requests)
            elif args.online_benchmark:
                results, warmup_seconds = benchmark_workload(
                    runner, session, online, prompts, entries, args.tokens, decoder, devices, args.output,
                )
            elif args.report_train_requests is not None:
                eos = checkpoint.text.get("eos_token_id", checkpoint.raw.get("eos_token_id"))
                eos_ids = frozenset(eos if isinstance(eos, list) else () if eos is None else (eos,))
                study = study_workload(
                    runner, session, online, prompts, entries, train_count=args.report_train_requests,
                    steps=args.tokens, batch_size=args.report_batch_size, eos_ids=eos_ids, decoder=decoder,
                    output=args.output, engram_seconds=engram_seconds,
                )
            else:
                for mode in modes:
                    print(f"Node {args.node} running {mode} on global ranks {list(plan.local_ranks)}", flush=True)
                    expected = results.get("target", {}).get("requests") if mode == "orthrus" else None
                    started = time.perf_counter()
                    rows, stats = generate(
                        runner, prompts, args.tokens, assignments,
                        identities=identities, metadata=entries, decoder=decoder, reference=expected,
                        speculative=mode == "orthrus",
                    )
                    elapsed = time.perf_counter() - started
                    results[mode] = dict(status="executed", devices=devices, sampling=SAMPLING,
                                         requests=rows, speculation=stats, elapsed_seconds=elapsed,
                                         output_tokens=sum(len(row["token_ids"]) for row in rows),
                                         output_tokens_per_second=sum(len(row["token_ids"]) for row in rows) / elapsed,
                                         online=None if online is None else online.summary())
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
            _, dispatch_counts, restored = audit_session(
                native_dir, speculative=mode == "orthrus", node_local=True, request_prefix=f"{mode}-request-",
                require_updated_draft=mode == "orthrus" and args.require_online_update and study is None,
            )
            audits[mode] = dict(phases=dispatch_counts, verifier_restored_bytes=restored)
        tokens = (request_parity(results["target"], results["orthrus"])
                  if "target" in results else 0)
        online_summary = None if online is None else online.summary()
        passed = True
        if profile is not None:
            gate = profile_acceptance(profile_arms, comparisons, profile.records, versions,
                                      teacher_rows=online.config.teacher_rows, publications=online_summary["publications"])
            passed, tokens = gate["passed"], gate["compared_tokens"]
            expected_updates, speculation = gate["expected_updates"], gate["speculation"]
        elif study is not None:
            passed, tokens = study["passed"], sum(study["compared_tokens"].values())
            trained = study["arms"]["online"]
            speculation = dict(draft_rounds=trained["draft_rounds"], proposed_candidates=trained["proposed"],
                               accepted_candidates=trained["accepted"])
        else:
            speculation = results["orthrus"]["speculation"]
        summary = dict(passed=passed, node=plan.node, global_ranks=list(plan.local_ranks), build_id=identity,
                       topology=dict(dp=4, tp=4, ep=16), physical_to_runtime=mapping, compared_tokens=tokens,
                       checkpoint=checkpoint_identity, head_sha256=head_identity, head_role=head["artifact_role"],
                       target=audits.get("target"), orthrus=audits["orthrus"],
                       speculation=speculation,
                       online=online_summary)
        peer = link.exchange("acceptance/results", summary)
        if (peer["node"] != 1 - plan.node or peer["global_ranks"] != list(RankPlan((8, 8), 4, 1 - plan.node).local_ranks)
                or peer["passed"] != passed or peer["compared_tokens"] != tokens or peer["build_id"] != identity):
            raise ValueError("The peer did not complete the matching eight-rank acceptance")
        if online is not None:
            audit_online_update(online_summary, peer["online"],
                                publish_every=online.config.publish_every,
                                require_update=args.require_online_update)
        if study is not None:
            study["complete"] = True
            write_report(args.output / "gsm8k-online-report.json", study)
            summary["gsm8k_online_report"] = study
            print(json.dumps(dict(stage="gsm8k_online_report", **study)), flush=True)
        if args.online_benchmark:
            fixed_tokens = request_parity(results["target"], results["fixed"])
            elapsed = {name: result["elapsed_seconds"] for name, result in results.items()}
            peer_elapsed = link.exchange("acceptance/gsm8k-benchmark-seconds", elapsed)
            report = {
                "requests_per_arm": len(prompts) - 1, "compared_tokens": {
                    "fixed": fixed_tokens, "online": tokens,
                },
                "startup_seconds": {"engram": engram_seconds, **warmup_seconds},
                "online_update_seconds": max(online_summary["update_seconds"], peer["online"]["update_seconds"]),
                "online_publication_seconds": max(
                    online_summary["publication_seconds"], peer["online"]["publication_seconds"],
                ),
                "updates": online_summary["updates"], "publications": online_summary["publications"],
                "arms": {},
            }
            for name, result in results.items():
                stats = result["speculation"]
                wall = max(elapsed[name], peer_elapsed[name])
                rounds = stats.get("draft_rounds", 0)
                report["arms"][name] = dict(
                    output_tokens=result["output_tokens"], elapsed_seconds=wall,
                    tokens_per_second=result["output_tokens"] / wall,
                    mean_accepted_candidates=stats.get("accepted_candidates", 0) / rounds if rounds else None,
                )
            report["online_speedup_vs_fixed"] = report["arms"]["fixed"]["elapsed_seconds"] / report["arms"]["orthrus"]["elapsed_seconds"]
            report["online_speedup_vs_ar"] = report["arms"]["target"]["elapsed_seconds"] / report["arms"]["orthrus"]["elapsed_seconds"]
            write_report(args.output / "gsm8k-benchmark-summary.json", report)
            summary["gsm8k_benchmark"] = report
            print(json.dumps(dict(stage="gsm8k_benchmark_summary", **report)), flush=True)
        if profile is not None:
            peer_records = link.exchange("acceptance/online-profile", profile.records)
            node0, node1 = (profile.records, peer_records) if plan.node == 0 else (peer_records, profile.records)
            report = OnlineUpdateProfile.summary(node0, node1)
            report["local_requests"] = profile.requests
            report["engram_seconds"] = engram_seconds
            report["first_prefill_seconds"] = profile.requests[0]["prefill_seconds"]
            report["expected_updates"] = expected_updates
            report["greedy_comparison"] = comparisons
            report["probe_head_versions"] = versions
            report["probe_metrics"] = {
                name: {key: profile_arms[name][key] for key in (
                    "decode_tokens_per_second", "mean_accepted_candidates", "candidate_acceptance_rate",
                )}
                for name in ("ar", "fixed_before", "fixed_after")
            }
            rows = profile_arms["update"]["requests"]
            report["repeat_greedy_outputs_match"] = all(row["token_ids"] == rows[0]["token_ids"] for row in rows)
            write_report(args.output / "online-profile-summary.json", report)
            summary["online_profile"] = {
                "updates_profiled": len(report["updates"]),
                "hot_updates": report["hot_updates"],
                "hot_update_median_seconds": report["hot_update_median_seconds"],
            }
            stages = report["hot_owner_stage_median_seconds"]
            if not stages and report["updates"]:
                stages = report["updates"][-1]["owner_stage_seconds"]
            print(json.dumps({
                "stage": "online_profile_summary", "passed": passed, **summary["online_profile"],
                "expected_updates": expected_updates,
                "engram_seconds": round(engram_seconds, 3),
                "first_prefill_seconds": round(report["first_prefill_seconds"], 3),
                "greedy_comparison": comparisons,
                "probe_head_versions": versions,
                "probe_metrics": report["probe_metrics"],
                "updates": [{
                    "dp": row["owner_dp"], "teacher_rows": row["teacher_rows"],
                    "seconds": round(row["update_seconds"], 3),
                    "publication_seconds": round(row["publication_seconds"], 3),
                    "jit_seconds": round(max(row["jit_seconds_by_host"]), 3),
                    "new_rank_dispatches": row["new_rank_dispatches"],
                } for row in report["updates"]],
                "phase_sample": "hot_owner_median" if report["hot_updates"] else "last_update_owner",
                "phase_seconds": {
                    name: round(stages[name], 3) for name in (
                        "lm_head_loss_and_vjp", "ep_dispatch_exchange", "ep_expert_owner",
                        "ep_return_exchange", "moe_shared_and_mhc_vjp", "attention_vjp",
                        "dp_gradient_reduce", "gradient_norm_clip", "adamw",
                    ) if name in stages
                },
                "repeat_greedy_outputs_match": report["repeat_greedy_outputs_match"],
            }), flush=True)
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
    parser.add_argument("--workload", type=Path,
                        help="JSONL rows with unique id and prompt fields; replaces --prompt")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--host-threads", type=int, default=4)
    parser.add_argument("--trace-prefill", action=argparse.BooleanOptionalAction, default=True,
                        help="Record hashes of actual prefill layer outputs for first-divergence diagnosis")
    parser.add_argument("--trace-position", type=int,
                        help="Record one absolute token position (zero-based) in Decode and K32 Verify")
    parser.add_argument("--short-speculative-dispatch", action="store_true",
                        help="Use the 16-row native program for verify/accept waves that fit on every DP")
    parser.add_argument("--online-study", type=Path,
                        help="Versioned GSM8K online policy; enables resident last-N updates")
    parser.add_argument("--online-primitives", metavar="MODULE:CALLABLE",
                        help="Production resident primitives factory; required with --online-study")
    parser.add_argument("--online-overrides",
                        help="Strict JSON object overriding fields in candidate_default")
    parser.add_argument("--orthrus-only", action="store_true",
                        help="Skip the target parity pass for a short integration gate")
    parser.add_argument("--require-online-update", action="store_true",
                        help="Fail unless the workload captures and applies an online update")
    parser.add_argument("--online-profile", action="store_true",
                        help="Run matched greedy probes and profile four to six updates in one session")
    parser.add_argument("--online-benchmark", action="store_true",
                        help="Benchmark AR, fixed draft, and online draft within one resident session")
    parser.add_argument("--report-train-requests", type=int,
                        help="Run a batched GSM8K study using this many initial training requests")
    parser.add_argument("--report-batch-size", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--connect-timeout", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum seconds of peer inactivity")
    args = parser.parse_args()
    if (args.tokens < 2 or args.host_threads < 1 or args.timeout < 1
            or args.max_requests is not None and args.max_requests < 1
            or args.workload is not None and args.prompt is not None
            or args.workload is None and args.prompt is not None and len(args.prompt) != 2
            or args.online_study is not None and args.online_primitives is None
            or args.require_online_update and args.online_study is None
            or args.online_profile and not (args.orthrus_only and args.require_online_update)
            or args.report_train_requests is not None and (
                args.report_train_requests < 1 or args.online_profile or args.online_benchmark
                or args.orthrus_only or not args.require_online_update or args.workload is None)
            or args.online_benchmark and (args.orthrus_only or not args.require_online_update or args.workload is None)):
        parser.error("Require valid dimensions and exactly one prompt source")
    if args.trace_position is not None and not 0 <= args.trace_position < args.context:
        parser.error("--trace-position must be in the reserved context")
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
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
