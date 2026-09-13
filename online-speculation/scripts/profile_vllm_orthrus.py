"""Capture a bounded torch-NPU window while a fixed Orthrus burst is active."""

import argparse
import asyncio
from pathlib import Path

import aiohttp

from blockspec import reporting as report
from blockspec.vllm_ascend.benchmark import parse_metrics, run_benchmark
from blockspec.vllm_ascend.profiling import set_service_enabled


async def control(session, endpoint, operation):
    async with session.post(f"{endpoint.rstrip('/')}/{operation}") as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(
                f"profile {operation} failed with HTTP {response.status}: {body}")
        return body


async def metrics(session, endpoint):
    async with session.get(f"{endpoint.rstrip('/')}/metrics") as response:
        response.raise_for_status()
        return parse_metrics(await response.text())


def signal_count(path, message):
    if not path:
        return 0
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").count(message)
    except FileNotFoundError:
        return 0


async def wait_for_service_signal(path, message, previous, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    while signal_count(path, message) <= previous:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"service profiler did not report: {message}")
        await asyncio.sleep(0.5)


async def wait_for_decode_window(session, endpoint, baseline, benchmark, args):
    await asyncio.sleep(args.ramp_up_seconds)
    deadline = asyncio.get_running_loop().time() + args.profile_wait_timeout
    while True:
        current = await metrics(session, endpoint)
        drafts = (current.get("vllm:spec_decode_num_drafts_total", 0.0) -
                  baseline.get("vllm:spec_decode_num_drafts_total", 0.0))
        waiting = current.get("vllm:num_requests_waiting", 0.0)
        if drafts >= args.min_spec_drafts and (
                not args.require_backlog or waiting > 0):
            return current, drafts
        if benchmark.done():
            raise RuntimeError(
                "workload completed before a speculative-decode profile window")
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                "timed out waiting for speculative decode under backlog")
        await asyncio.sleep(0.5)


async def run(args):
    async with aiohttp.ClientSession() as session:
        baseline = await metrics(session, args.endpoint)
    benchmark = asyncio.create_task(run_benchmark(
        endpoint=args.endpoint + "/v1/chat/completions",
        metrics_url=args.endpoint + "/metrics", model=args.served_model,
        manifest=args.manifest, requests=args.requests,
        arrival_rate=args.arrival_rate,
        client_concurrency=args.client_concurrency, timeout=args.timeout,
        seed=args.seed))
    started = asyncio.get_running_loop().time()
    enabled = False
    try:
        async with aiohttp.ClientSession() as session:
            state, drafts = await wait_for_decode_window(
                session, args.endpoint, baseline, benchmark, args)
            profile_started = asyncio.get_running_loop().time()
            if args.profiler == "torch":
                await control(session, args.endpoint, "start_profile")
            elif not args.service_startup_capture:
                starts = signal_count(
                    args.service_log, "Received profiler start signal from C++")
                set_service_enabled(args.service_config, True)
                await wait_for_service_signal(
                    args.service_log, "Received profiler start signal from C++",
                    starts, args.service_signal_timeout)
            enabled = True
            await asyncio.sleep(args.profile_seconds)
            if args.profiler == "torch":
                await control(session, args.endpoint, "stop_profile")
            elif not args.service_startup_capture:
                stops = signal_count(
                    args.service_log, "Received profiler stop signal from C++")
                set_service_enabled(args.service_config, False)
                await wait_for_service_signal(
                    args.service_log, "Received profiler stop signal from C++",
                    stops, args.service_signal_timeout)
            enabled = False
        result = await benchmark
    except BaseException:
        if (enabled and args.profiler == "service" and
                not args.service_startup_capture):
            set_service_enabled(args.service_config, False)
        benchmark.cancel()
        await asyncio.gather(benchmark, return_exceptions=True)
        raise
    result["profile_window"] = {
        "profiler": args.profiler,
        "service_startup_capture": args.service_startup_capture,
        "minimum_ramp_up_seconds": args.ramp_up_seconds,
        "started_after_seconds": profile_started - started,
        "profile_seconds": args.profile_seconds,
        "spec_drafts_at_start": drafts,
        "running_at_start": state.get("vllm:num_requests_running"),
        "waiting_at_start": state.get("vllm:num_requests_waiting"),
        "require_backlog": args.require_backlog,
        "stack": False,
        "memory": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model", default="orthrus")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--arrival-rate", type=float, default=0.0)
    parser.add_argument("--client-concurrency", type=int, default=256)
    parser.add_argument("--ramp-up-seconds", type=float, default=20.0)
    parser.add_argument("--profile-wait-timeout", type=float, default=1200.0)
    parser.add_argument("--min-spec-drafts", type=float, default=32.0)
    parser.add_argument("--require-backlog", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--profile-seconds", type=float, default=10.0)
    parser.add_argument("--profiler", choices=("torch", "service"),
                        default="torch")
    parser.add_argument("--service-config")
    parser.add_argument("--service-log")
    parser.add_argument("--service-signal-timeout", type=float, default=90.0)
    parser.add_argument("--service-startup-capture", action="store_true",
                        help="service profiler was enabled before server startup")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.profiler == "service" and not args.service_config:
        parser.error("--service-config is required for service profiling")
    if (args.profiler == "service" and not args.service_startup_capture
            and not args.service_log):
        parser.error("--service-log is required for windowed service profiling")
    result = asyncio.run(run(args))
    with open(args.output, "x", encoding="utf-8") as handle:
        report.dump(result, handle, indent=2)
        handle.write("\n")
    print(report.dumps({"profile_window": result["profile_window"],
                        "summary": result["summary"],
                        "engine": {key: value for key, value in result["engine"].items()
                                   if key != "telemetry"}}, indent=2))


if __name__ == "__main__":
    main()
