"""Emit the original PyTorch Orthrus model's AR greedy tokens for comparison."""

import argparse
import time

import torch

from blockspec import reporting as report
from blockspec.commands.evaluate import prompt_ids
from blockspec.parallel import MaskedAttentionBranch, generate_ar
from blockspec.parallel.weights import load_public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    started = time.perf_counter()
    model = load_public(args.model, device=args.device,
                        dtype=torch.bfloat16).set_backend("sdpa")
    loaded = time.perf_counter()
    prompt = prompt_ids(args.model, 1, path=args.prompt, thinking=False,
                        device=args.device)[0]
    generated = generate_ar(MaskedAttentionBranch(model), prompt,
                            args.max_tokens, eos_id=model.config.eos_token_id)
    finished = time.perf_counter()
    result = {
        "mode": "pytorch_ar_greedy_oracle",
        "model_load_seconds": loaded - started,
        "generation_seconds": finished - loaded,
        "prompt_token_ids": prompt[0].tolist(),
        "output_token_ids": generated.tokens,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        report.dump(result, handle, indent=2)
        handle.write("\n")
    print(report.dumps(result))


if __name__ == "__main__":
    main()
