#!/usr/bin/env python3
"""Validate and benchmark 1122 dual-host AR and Orthrus on the same native graphs."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from traceback import print_exc

from blockspec.dsv41_native_forward import ForwardArtifacts, NativeSampleError, check_native_windows, open_native_forward
from blockspec.dsv41_online import read_workload
from validate_dsv41_pypto_packed_generation import OnlineSchedule, benchmark, run_wave


def run_smoke(target, session, report, requests, *, options, draft, schedule):
    arms = [("ar", None)]
    if draft:
        arms.append(("orthrus_online" if schedule else "orthrus_frozen", 0 if schedule else None))
    if schedule and len(requests) > 2:
        arms.append(("orthrus_online_peer", 2))
    for name, owner_dp in arms:
        print(json.dumps(dict(stage="smoke_arm", mode=name, update_dp=owner_dp)), flush=True)
        target.sample_checks = {}
        if owner_dp is not None:
            schedule.next_dp = owner_dp
            schedule.next_capture = schedule.rounds + 1
        try:
            wave = run_wave(session, requests, speculative=name != "ar",
                            online=schedule if owner_dp is not None else None, **options)
        except NativeSampleError as error:
            report[name] = dict(sample_checks=target.sample_checks, sample_failure=error.evidence)
            report["passed"] = False
            print(json.dumps(dict(stage="sample_validation_failed", arm=name, **error.evidence)), flush=True)
            return
        report[name] = dict(wave, sample_checks=target.sample_checks)
        updated_dps = {update["dp"] for update in wave["updates"] if update["applied"]}
        if wave["requests"] != report["ar"]["requests"] or (
            owner_dp is not None and owner_dp not in updated_dps
        ):
            report["passed"] = False
            return


def run_study(session, report, requests, heldout, *, schedule, timing, output):
    holdout_arms = [("heldout_initial", heldout, None)] if heldout else []
    arms = [*holdout_arms, ("online_study", requests, schedule)]
    if heldout:
        arms.append(("heldout_published", heldout, None))
    with output.with_name("study-waves.jsonl").open("w", buffering=1) as waves:
        for name, workload, online in arms:
            print(json.dumps(dict(stage="study_arm", mode=name, requests=len(workload))), flush=True)

            def save_wave(index, total, ar, orthrus, divergence):
                waves.write(json.dumps(dict(mode=name, wave=index, waves=total,
                                            ar=ar, orthrus=orthrus, divergence=divergence)) + "\n")
                if index == 1 or index == total or index % 4 == 0 or divergence:
                    print(json.dumps(dict(stage="study_progress", mode=name, wave=index, waves=total,
                                          updates=schedule.updates, divergence=divergence)), flush=True)

            result = benchmark(session, workload, online=online, on_wave=save_wave, **timing)
            if name == "online_study":
                report.update(result)
            else:
                report[name] = result
            report["passed"] = result["passed"]
            if not report["passed"]:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--peer-listen-host")
    parser.add_argument("--draft-graph", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--online-artifacts", type=Path)
    parser.add_argument("--loss", choices=("tv", "forward_kl"), default="tv")
    parser.add_argument("--update-every", type=int, default=16)
    parser.add_argument("--teacher-rows", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-updates", type=int, default=8)
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--heldout-workload", type=Path)
    parser.add_argument("--warmup-waves", type=int, default=1)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--width", type=int, choices=(8, 16), default=8)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--tokens", type=int, default=8)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--contract-only", action="store_true")
    checks.add_argument("--windows-only", action="store_true",
                        help="Reserve the model's communication windows without loading device weights")
    args = parser.parse_args()
    if not args.contract_only and not (args.endpoint and args.peer_listen_host):
        parser.error("Native execution requires the daemon endpoint and peer listen address")
    if args.tokens < 2:
        parser.error("Generate at least two tokens to exercise Decode")
    if (args.draft_graph is None) != (args.head is None):
        parser.error("Supply the compiled draft graph and its head together")
    if args.study and args.online_artifacts is None:
        parser.error("The online study requires compiled reverse and optimizer artifacts")
    if args.heldout_workload and not args.study:
        parser.error("A held-out workload requires an online study")
    if args.online_artifacts is not None and (
        not 0 < args.teacher_rows < args.width or args.update_every < 1 or args.warmup_updates < 0
        or not isfinite(args.learning_rate) or args.learning_rate <= 0 or args.warmup_waves < 0
    ):
        parser.error("Choose a positive cadence and learning rate, with teacher rows inside Verify width")
    artifacts = ForwardArtifacts(
        args.graphs, args.checkpoint, 1, draft_directory=args.draft_graph, head=args.head,
        online_directory=args.online_artifacts, loss=args.loss,
    )
    if artifacts.draft is not None and args.width > min(artifacts.draft.block, artifacts.capacities[1]):
        parser.error("Verification width exceeds the compiled draft or Decode capacity")
    def workload(path):
        rows = [(row["id"], artifacts.tokenizer.encode(row["prompt"]).ids) for row in read_workload(path)]
        if (not args.study and len(rows) > args.concurrency) or any(
            not prompt or len(prompt) + args.tokens > artifacts.context for _, prompt in rows
        ):
            parser.error("Prompts and output must fit the compiled history and the smoke's one request wave")
        return rows

    requests = workload(args.workload)
    heldout = workload(args.heldout_workload) if args.heldout_workload else []
    if {tuple(prompt) for _, prompt in requests}.intersection(tuple(prompt) for _, prompt in heldout):
        parser.error("Training and held-out workloads contain an identical prompt")
    mode = ("native-contract" if args.contract_only else "native-windows" if args.windows_only else
            "native-study" if args.study else "native-smoke")
    windows = artifacts.native_windows()
    report = dict(
        mode=mode,
        identity=artifacts.identity.hex(), capacities=artifacts.capacities, context=artifacts.context,
        requests=len(requests), output_tokens_per_request=args.tokens,
        last_layers=artifacts.draft.last_layers if artifacts.online else 0,
        workload_sha256=sha256(args.workload.read_bytes()).hexdigest(),
        loss=args.loss if artifacts.online else None,
        planned_windows=windows,
        planned_window_bytes_per_rank=sum(windows.values()),
        passed=True,
    )
    print(json.dumps(dict(stage="artifacts_ready", **{name: value for name, value in report.items()
                                                   if name not in ("planned_windows", "passed")})), flush=True)
    if not args.contract_only:
        start = perf_counter()

        def startup_stage(stage):
            print(json.dumps(dict(stage=stage, elapsed_seconds=perf_counter() - start)), flush=True)

        connection = dict(endpoint=args.endpoint, peer_listen_host=args.peer_listen_host,
                          devices=(tuple(range(8)), tuple(range(8))), timeout=args.timeout, on_stage=startup_stage)
        if args.windows_only:
            try:
                report["reserved_windows"] = check_native_windows(artifacts, **connection)
            except Exception as error:
                print_exc()
                report.update(passed=False, error=str(error))
            else:
                report["passed"] = bool(windows) and report["reserved_windows"] == windows
            report["seconds"] = perf_counter() - start
        else:
            with open_native_forward(artifacts, **connection) as target:
                report["reserved_windows"] = target.reserved_windows
                session = target.session()
                report["startup_seconds"] = perf_counter() - start
                print(json.dumps(dict(stage="resident_ready", seconds=report["startup_seconds"])), flush=True)
                options = dict(chunk=artifacts.capacities[0], tokens=args.tokens, width=args.width)
                schedule = (OnlineSchedule(args.update_every, args.teacher_rows, args.learning_rate, args.warmup_updates)
                            if artifacts.online else None)
                if args.study:
                    timing = dict(**options, concurrency=args.concurrency, warmup=args.warmup_waves)
                    if heldout:
                        report["heldout_workload_sha256"] = sha256(args.heldout_workload.read_bytes()).hexdigest()
                    run_study(session, report, requests, heldout, schedule=schedule, timing=timing, output=args.output)
                else:
                    run_smoke(target, session, report, requests, options=options, draft=artifacts.draft is not None,
                              schedule=schedule)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(stage="complete", passed=report["passed"], path=str(args.output))), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
