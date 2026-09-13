"""Measure Orthrus B1 burst or open-loop serving performance."""

import argparse
import asyncio

from blockspec import reporting as report
from blockspec.vllm_ascend.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--served-model", default="orthrus")
    parser.add_argument("--metrics-url")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--arrival-rate", type=float, default=0.0,
                        help="requests/s; zero submits a burst")
    parser.add_argument("--client-concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--ttft-slo", type=float)
    parser.add_argument("--tpot-slo", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = asyncio.run(run_benchmark(
        endpoint=args.endpoint, model=args.served_model,
        manifest=args.manifest, requests=args.requests,
        arrival_rate=args.arrival_rate,
        client_concurrency=args.client_concurrency, timeout=args.timeout,
        seed=args.seed, ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo,
        metrics_url=args.metrics_url))
    with open(args.output, "x", encoding="utf-8") as handle:
        report.dump(result, handle, indent=2)
        handle.write("\n")
    print(report.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
