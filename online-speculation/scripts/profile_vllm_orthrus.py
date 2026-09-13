"""Capture a bounded torch-NPU window while a fixed Orthrus burst is active."""

import argparse
import asyncio

import aiohttp

from blockspec import reporting as report
from blockspec.vllm_ascend.benchmark import run_benchmark


async def control(session, endpoint, operation):
    async with session.post(f"{endpoint.rstrip('/')}/{operation}") as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(
                f"profile {operation} failed with HTTP {response.status}: {body}")
        return body


async def run(args):
    benchmark = asyncio.create_task(run_benchmark(
        endpoint=args.endpoint + "/v1/chat/completions",
        metrics_url=args.endpoint + "/metrics", model=args.served_model,
        manifest=args.manifest, requests=args.requests,
        arrival_rate=args.arrival_rate,
        client_concurrency=args.client_concurrency, timeout=args.timeout,
        seed=args.seed))
    await asyncio.sleep(args.ramp_up_seconds)
    async with aiohttp.ClientSession() as session:
        await control(session, args.endpoint, "start_profile")
        await asyncio.sleep(args.profile_seconds)
        await control(session, args.endpoint, "stop_profile")
    return await benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model", default="orthrus")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--arrival-rate", type=float, default=0.0)
    parser.add_argument("--client-concurrency", type=int, default=256)
    parser.add_argument("--ramp-up-seconds", type=float, default=20.0)
    parser.add_argument("--profile-seconds", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    result["profile_window"] = {
        "ramp_up_seconds": args.ramp_up_seconds,
        "profile_seconds": args.profile_seconds,
        "stack": False,
        "memory": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        report.dump(result, handle, indent=2)
        handle.write("\n")
    print(report.dumps({"profile_window": result["profile_window"],
                        "summary": result["summary"],
                        "engine": {key: value for key, value in result["engine"].items()
                                   if key != "telemetry"}}, indent=2))


if __name__ == "__main__":
    main()
