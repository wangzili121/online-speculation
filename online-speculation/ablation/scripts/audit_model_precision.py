"""Compare precision policies on the same loaded base. No output files."""

import argparse

from blockspec import reporting as report

import torch

from blockspec_ablation.checkpoint import load_hf_base
from blockspec_ablation.diagnostics import audit_paired_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(87)
    model = load_hf_base(args.base, rank=8, device=args.device, dtype=getattr(torch, args.dtype))
    clean = torch.randint(1, model.config.vocab_size, (1, args.length), device=args.device)
    clean[:, 0] = 0
    for mode, reduced in (("default", True), ("default", False), ("math", False), ("fp32", False)):
        result = audit_paired_teacher(model, clean, attention=mode, reduced_bf16=reduced)
        if not args.trace:
            result.pop("layers")
        print(report.dumps(result), flush=True)


if __name__ == "__main__":
    main()
