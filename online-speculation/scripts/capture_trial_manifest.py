"""Record the private inputs and environment for one Orthrus trial."""

import argparse
import json

from blockspec.vllm_ascend.artifacts import capture_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-json", required=True)
    parser.add_argument("--trial-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = capture_manifest(
        repo=args.repo, workload=args.workload, model=args.model,
        runtime=json.loads(args.runtime_json), trial=json.loads(args.trial_json))
    write_manifest(args.output, value)


if __name__ == "__main__":
    main()
