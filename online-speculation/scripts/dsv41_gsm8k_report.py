"""Batched online study with a shared base model and disjoint evaluation prompts."""

from collections import Counter, defaultdict
from itertools import accumulate
import json
from time import perf_counter

from summarize_dsv41_gsm8k import numeric
from verify_dsv41_pypto_acceptance import check_generated_prefix, request_parity


def generate_wave(runner, prompts, entries, steps, *, start_index, phase, decoder, eos_ids,
                  speculative, expected=None):
    """Keep one request on each DP partition and release completed requests in order."""
    import torch
    from pypto_serving.config.types import DecodeBatch, PrefillBatch, SamplingParams

    indices = range(len(prompts))
    prefix = "orthrus" if speculative else "target"
    names = [f"{prefix}-request-{phase}-{start_index + i}" for i in indices]
    pages = {group.name: list(range(group.max_blocks_per_seq)) for group in runner.cache_plan.groups}
    output = [[] for _ in prompts]
    counts, histograms = Counter(), [Counter() for _ in prompts]
    chunk = runner.config.chunk_tokens
    started = perf_counter()
    for start in range(0, max(map(len, prompts)), chunk):
        live = [i for i in indices if start < len(prompts[i])]
        packed = [prompts[i][start:start + chunk] for i in live]
        sizes = list(map(len, packed))
        result = runner.run(PrefillBatch(
            request_ids=[names[i] for i in live],
            token_ids=torch.tensor([token for row in packed for token in row], dtype=torch.long),
            input_embeddings=None, seq_lens=[start + size for size in sizes],
            chunk_lens=sizes, chunk_offsets=list(accumulate([0, *sizes[:-1]])),
            chunk_starts=[start] * len(live), prompt_lens=[len(prompts[i]) for i in live],
            block_ids_by_group=[pages] * len(live), cache_partitions=live,
        ), "prefill")
        for i, size, token in zip(live, sizes, result.sampled_token_ids.tolist(), strict=True):
            if start + size == len(prompts[i]):
                output[i].append(token)
                if expected is not None:
                    check_generated_prefix(start_index + i, expected[i]["token_ids"], output[i], phase="prefill")
    prefill_seconds = perf_counter() - started

    def is_finished(i):
        return len(output[i]) >= steps or output[i][-1] in eos_ids

    finished = [i for i in indices if is_finished(i)]
    runner.release_finished_requests([names[i] for i in finished])
    live = [i for i in indices if i not in finished]
    decode_calls = 0
    started = perf_counter()
    while live:
        committed = [len(runner.ledger.requests[names[i]].history) for i in live]
        if any(n != len(prompts[i]) + len(output[i]) - 1 for i, n in zip(live, committed, strict=True)):
            raise RuntimeError("A report request must have exactly one uncommitted anchor")
        batch = DecodeBatch(
            request_ids=[names[i] for i in live], token_ids=torch.tensor([output[i][-1] for i in live]),
            hidden_states=None, seq_lens=torch.tensor([n + 1 for n in committed]),
            sampling_params=[SamplingParams(temperature=0, top_p=1)] * len(live),
            block_ids_by_group=[pages] * len(live), cache_partitions=live,
        )
        result = runner.run_orthrus(batch) if speculative else runner.run(batch, "decode")
        decode_calls += 1
        if result.accepted_token_ids is None:
            emissions = [[token] for token in result.sampled_token_ids.tolist()]
            candidates = [0] * len(live)
        else:
            emissions, candidates = result.accepted_token_ids, result.num_draft_tokens
        for i, emitted, proposed in zip(live, emissions, candidates, strict=True):
            if not emitted:
                raise RuntimeError("A greedy decode step must emit at least one token")
            if proposed:
                counts["draft_rounds"] += 1
                counts["proposed_candidates"] += proposed
                counts["accepted_candidates"] += len(emitted) - 1
                histograms[i][len(emitted) - 1] += 1
            else:
                counts["plain_decode_steps"] += 1
            for token in emitted[:steps - len(output[i])]:
                output[i].append(token)
                if token in eos_ids:
                    break
            if expected is not None:
                check_generated_prefix(start_index + i, expected[i]["token_ids"], output[i], phase="verify/accept"
                                       if proposed else "decode")
        finished = [i for i in live if is_finished(i)]
        runner.release_finished_requests([names[i] for i in finished])
        live = [i for i in live if i not in finished]
        if decode_calls % 8 == 0:
            print(json.dumps(dict(stage="gsm8k_online_decode", phase=phase, wave_start=start_index,
                                  decode_calls=decode_calls, active=len(live),
                                  generated=[len(row) for row in output])), flush=True)
    decode_seconds = perf_counter() - started
    rows = [
        dict(index=start_index + i, workload_id=entries[i]["id"], dp=i,
             prompt_ids=prompts[i], token_ids=output[i], output_text=decoder(output[i]),
             reference_answer=entries[i].get("reference_answer"), acceptance_histogram=dict(histograms[i]),
             finish_reason="stop" if output[i][-1] in eos_ids else "length",
             usage=dict(prompt_tokens=len(prompts[i]), completion_tokens=len(output[i])))
        for i in indices
    ]
    return rows, counts, dict(prefill_seconds=prefill_seconds,
                              decode_and_release_seconds=decode_seconds, decode_calls=decode_calls)


def _write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _collect_phases(path, offset, previous, totals, draft_versions):
    """Read each completed Serving receipt once and attribute runtime byte deltas."""
    counters = ("output_d2h_bytes", "verify_backup_d2d_bytes", "verify_restore_d2d_bytes")
    with path.open() as stream:
        stream.seek(offset)
        for line in iter(stream.readline, ""):
            receipt = json.loads(line)
            phase = receipt["phase"]
            stage = totals[phase]
            stage["calls"] += 1
            if phase == "draft":
                draft_versions.add(receipt["draft_head_version"] or 0)
            stage["serving_seconds"] += receipt.get("seconds", 0.0)
            timings = receipt.get("timings", {})
            stage["native_seconds"] += timings.get("chunk_seconds", 0.0)
            capacity = receipt.get("execution_capacity")
            if capacity is not None:
                stage[f"capacity{capacity}_calls"] += 1
                stage[f"capacity{capacity}_native_seconds"] += timings.get("chunk_seconds", 0.0)
            commit_tokens = receipt.get("commit_tokens_per_dp")
            if commit_tokens:
                largest = max(commit_tokens)
                stage["max_commit_tokens_per_dp"] = max(stage["max_commit_tokens_per_dp"], largest)
                stage["accept_calls_fitting_capacity16"] += int(largest <= 16)
            for name in ("verify_snapshot_seconds", "verify_restore_seconds"):
                if name in timings:
                    stage[name] += timings[name]
            stage["prepare_seconds"] += receipt.get("prepare_seconds", 0.0)
            vocabulary = receipt.get("vocabulary_layout")
            if vocabulary is not None:
                stage[f"{vocabulary}_calls"] += 1
            for name in ("teacher_logit_rows", "student_logit_rows"):
                if receipt.get(name) is not None:
                    stage[name] += receipt[name]
            runtime = receipt.get("runtime") or {}
            for name in counters:
                if name in runtime:
                    stage[name] += runtime[name] - previous[name]
                    previous[name] = runtime[name]
        return stream.tell()


def study_workload(runner, session, online, prompts, entries, *, train_count, steps,
                   batch_size, eos_ids, decoder, output, engram_seconds, min_published_update=0):
    """Persist each wave; compare fixed versions on the identical held-out requests."""
    link = session.cluster.link
    startup = {"engram_seconds": engram_seconds}
    arms, comparisons = {}, {}
    report = dict(format="orthrus-dsv41-gsm8k-online-report-v1", complete=False,
                  max_concurrent_requests=batch_size, max_new_tokens=steps, updates_policy=online.config.contract(),
                  verify_candidate_limit=runner.verify_candidate_limit,
                  short_speculative_dispatch=session.short_speculative_dispatch,
                  training_requests=train_count, heldout_requests=len(entries) - train_count,
                  min_published_update=min_published_update,
                  startup_seconds=startup, arms=arms, compared_tokens=comparisons)
    summary_path = output / "gsm8k-online-report.json"
    output.mkdir(parents=True, exist_ok=True)
    train = slice(0, train_count)
    heldout = slice(train_count, len(entries))
    sampling = dict(temperature=0, ignore_eos=not bool(eos_ids))
    run_results = {}
    receipt_offset, runtime_counters = 0, Counter()

    def warmup(name, *, speculative, expected=None):
        nonlocal receipt_offset
        print(json.dumps(dict(stage="gsm8k_online_startup", phase=name)), flush=True)
        started = perf_counter()
        warmup_rows, _, _ = generate_wave(
            runner, prompts[:batch_size], entries[:batch_size], 2, start_index=0, phase=name,
            eos_ids=eos_ids, decoder=decoder, speculative=speculative,
            expected=None if expected is None else expected["requests"],
        )
        elapsed = perf_counter() - started
        receipt_offset = _collect_phases(
            session.receipt_path, receipt_offset, runtime_counters, defaultdict(Counter), set(),
        )
        startup[name] = max(elapsed, link.exchange(f"report/seconds/{name}", elapsed))
        result = dict(status="executed", devices=list(session.devices), sampling=sampling, requests=warmup_rows)
        if expected is not None:
            report["startup_compared_tokens"] = request_parity(expected, result)
        _write(summary_path, report)
        return result

    warmup_target = warmup("ar_warmup", speculative=False)
    for name, subset in (("ar", heldout), ("initial", heldout), ("online", train), ("frozen", heldout)):
        if name == "initial":
            warmup("draft_warmup", speculative=True, expected=warmup_target)
        print(json.dumps(dict(stage="gsm8k_online_phase", phase=name)), flush=True)
        if name == "online":
            started = perf_counter()
            online.bind(session.draft, session.native, session.cluster)
            session.online = online
            elapsed = perf_counter() - started
            startup["learner_bind"] = max(elapsed, link.exchange("report/seconds/learner-bind", elapsed))
        if name == "frozen":
            session.online = None  # The last published operands remain active in ResidentDraftAttention.
        rows, speculation, timing = [], Counter(), Counter()
        phase_totals = defaultdict(Counter)
        draft_versions = set()
        started = perf_counter()
        for first in range(subset.start, subset.stop, batch_size):
            last = min(first + batch_size, subset.stop)
            baseline = run_results["ar"]["requests"][first - train_count:last - train_count] if (
                name in ("initial", "frozen")
            ) else None
            emitted, counts, spans = generate_wave(
                runner, prompts[first:last], entries[first:last], steps,
                start_index=first - subset.start, phase=name, decoder=decoder, eos_ids=eos_ids,
                speculative=name != "ar", expected=baseline,
            )
            rows.extend(emitted)
            speculation.update(counts)
            timing.update(spans)
            measured_seconds = perf_counter() - started
            receipt_offset = _collect_phases(
                session.receipt_path, receipt_offset, runtime_counters, phase_totals, draft_versions,
            )
            n_tokens = sum(len(row["token_ids"]) for row in rows)
            version = session.draft.online_bank.active_version if name in ("online", "frozen") else 0
            local = dict(
                wall_seconds=measured_seconds, **timing, output_tokens=n_tokens,
                requests=len(rows), learner=online.summary() if name == "online" else None,
                first_update=None, head_version=version, phase_totals=phase_totals,
                draft_head_versions=sorted(draft_versions),
            )
            if name == "online":
                records = online.bridge.controller.records
                local["published_update"] = next((row["update"] for row in reversed(records) if row["published"]), 0)
                if records:
                    record = records[0]
                    local["first_update"] = record["update_seconds"] + record["publication_seconds"]
            peer = link.exchange(f"report/progress/{name}/{first}", local)
            if any(local[key] != peer[key] for key in ("requests", "output_tokens", "head_version", "draft_head_versions")):
                raise ValueError(f"Hosts disagree on {name} workload progress or executed draft versions")
            if name in ("initial", "frozen") and draft_versions - {version}:
                raise ValueError(f"{name} executed draft versions {sorted(draft_versions)}, expected {version}")
            result = dict(status="executed", devices=list(session.devices), sampling=sampling,
                          requests=rows, speculation=dict(speculation))
            run_results[name] = result
            rounds, proposed, accepted = (speculation[key] for key in (
                "draft_rounds", "proposed_candidates", "accepted_candidates",
            ))
            answers = [(numeric(row["output_text"]), numeric(row["reference_answer"])) for row in rows]
            correct = sum(actual == reference for actual, reference in answers if reference is not None)
            scored = sum(reference is not None for _, reference in answers)
            wall = max(local["wall_seconds"], peer["wall_seconds"])
            decode = max(local["decode_and_release_seconds"], peer["decode_and_release_seconds"])
            stats = arms[name] = dict(
                requests=len(rows), output_tokens=n_tokens, wall_seconds=wall,
                wall_tok_s=n_tokens / wall, draft_rounds=rounds, decode_calls=timing["decode_calls"],
                prefill_seconds=max(local["prefill_seconds"], peer["prefill_seconds"]),
                decode_and_release_seconds=decode, decode_tok_s=(n_tokens - len(rows)) / decode if decode else None,
                proposed=proposed, accepted=accepted, acceptance_rate=accepted / proposed if proposed else None,
                accepted_per_round=accepted / rounds if rounds else None, gsm8k_scored=scored,
                gsm8k_correct=correct, gsm8k_exact_match=correct / scored if scored else None,
                gsm8k_answered=sum(actual is not None for actual, _ in answers),
                stopped=sum(row["finish_reason"] == "stop" for row in rows),
                length_limited=sum(row["finish_reason"] == "length" for row in rows), head_version=version,
                draft_head_versions=local["draft_head_versions"],
                serving_phases_by_host={
                    str(node): record["phase_totals"]
                    for node, record in ((session.cluster.plan.node, local), (1 - session.cluster.plan.node, peer))
                },
            )
            if name == "online":
                learners = local["learner"], peer["learner"]
                if (local["published_update"] != peer["published_update"]
                        or learners[0]["updates"] != learners[1]["updates"]):
                    raise ValueError("Hosts disagree on the completed or published optimizer step")
                costs = [item["update_seconds"] + item["publication_seconds"] for item in learners]
                adjusted = max(a["wall_seconds"] - cost for a, cost in zip((local, peer), costs, strict=True))
                after_first = max(a["wall_seconds"] - (a["first_update"] or 0.0) for a in (local, peer))
                stats.update(
                    updates=learners[0]["updates"], publications=learners[0]["publications"],
                    published_update=local["published_update"],
                    first_update_and_publish_seconds=max(local["first_update"] or 0.0, peer["first_update"] or 0.0),
                    update_seconds=max(item["update_seconds"] for item in learners),
                    publication_seconds=max(item["publication_seconds"] for item in learners),
                    wall_tok_s_excluding_first_update_estimate=n_tokens / after_first,
                    wall_tok_s_excluding_all_updates_estimate=n_tokens / adjusted,
                )
                _write(output / "gsm8k-online-updates.json", records)
            _write(output / f"gsm8k-online-{name}.json", result)
            _write(summary_path, report)
            print(json.dumps(dict(stage="gsm8k_online_progress", phase=name, **stats)), flush=True)
            if name == "online" and min_published_update and stats["published_update"] >= min_published_update:
                break
        if name in ("initial", "frozen"):
            comparisons[name] = request_parity(run_results["ar"], run_results[name])
        _write(summary_path, report)
    trained = arms["online"]
    report["passed"] = bool(
        trained["publications"] and trained["published_update"] >= min_published_update
        and arms["frozen"]["draft_head_versions"] == [trained["head_version"]]
    )
    if trained["published_update"] < min_published_update:
        report["failure_reason"] = "training request budget ended before the requested optimizer step was published"
    report["frozen_speedup_vs_initial"] = arms["frozen"]["wall_tok_s"] / arms["initial"]["wall_tok_s"]
    report["frozen_speedup_vs_ar"] = arms["frozen"]["wall_tok_s"] / arms["ar"]["wall_tok_s"]
    return report
