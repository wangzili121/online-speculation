"""Create or analyze pinned MS Service Profiler artifacts."""

import argparse

from blockspec import reporting as report
from blockspec.vllm_ascend.profiling import (analyze_service,
                                             write_service_config)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--config", required=True)
    create.add_argument("--profile-directory", required=True)
    create.add_argument("--seconds", type=float, required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "create":
        write_service_config(args.config, args.profile_directory, args.seconds)
        result = {"config": args.config, "profile_directory": args.profile_directory,
                  "seconds": args.seconds}
    else:
        result = analyze_service(args.input, args.output)
    print(report.dumps(result, indent=2))
    if args.command == "analyze" and not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
