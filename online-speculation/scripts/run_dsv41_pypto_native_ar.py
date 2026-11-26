#!/usr/bin/env python3
"""Run a greedy baseline through the 1122 native dual-host Serving path."""

import argparse
import json
from pathlib import Path

from blockspec.dsv41_native_forward import ForwardArtifacts, open_native_forward
from blockspec.dsv41_online import read_workload
from validate_dsv41_pypto_packed_generation import run_wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--peer-listen-host")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    if not args.contract_only and not (args.endpoint and args.peer_listen_host):
        parser.error("Native execution requires the daemon endpoint and peer listen address")
    if args.tokens < 2:
        parser.error("Generate at least two tokens to exercise Decode")
    artifacts = ForwardArtifacts(args.graphs, args.checkpoint, 1)
    requests = [(row["id"], artifacts.tokenizer.encode(row["prompt"]).ids) for row in read_workload(args.workload)]
    if not 1 <= len(requests) <= 4 or any(
        not prompt or len(prompt) + args.tokens > artifacts.context for _, prompt in requests
    ):
        parser.error("Use one to four requests whose prompt and output fit the compiled history")
    report = dict(
        mode="native-contract" if args.contract_only else "native-ar",
        identity=artifacts.identity.hex(), capacities=artifacts.capacities, context=artifacts.context,
        requests=len(requests), output_tokens_per_request=args.tokens,
    )
    print(json.dumps(dict(stage="artifacts_ready", **report)), flush=True)
    if not args.contract_only:
        with open_native_forward(
            artifacts, endpoint=args.endpoint,
            peer_listen_host=args.peer_listen_host, devices=(tuple(range(8)), tuple(range(8))), timeout=args.timeout,
        ) as target:
            result = run_wave(
                target.session(), requests, chunk=artifacts.capacities[0], tokens=args.tokens,
                width=1, speculative=False,
            )
            report["generation"] = result
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(stage="complete", path=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
