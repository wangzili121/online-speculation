"""Check the pinned Orthrus runtime before consuming an NPU trial."""

import argparse

from blockspec.vllm_ascend.preflight import check, format_result, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--runtime-root", default="/vllm-workspace/vllm-ascend")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--profiler", choices=("torch", "service"))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check(model=args.model, workload=args.workload,
                   runtime_root=args.runtime_root,
                   max_model_len=args.max_model_len, profiler=args.profiler)
    if args.output:
        write(args.output, result)
    print(format_result(result))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
