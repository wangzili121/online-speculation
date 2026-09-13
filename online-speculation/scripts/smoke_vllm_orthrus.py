"""Run one real shared-backbone Orthrus request through vLLM-Ascend."""

import argparse
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from blockspec import reporting as report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Write a Python function that returns the sum of two integers.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--mode", choices=("speculative", "ar"),
                        default="speculative")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--graph-mode", choices=("eager", "FULL_AND_PIECEWISE"),
                        default="eager")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    return parser.parse_args()


def main():
    args = parse_args()
    temperature = ((1.0 if args.mode == "speculative" else 0.0)
                   if args.temperature is None else args.temperature)
    if args.mode == "speculative" and temperature != 1.0:
        raise ValueError("the fixed Orthrus performance contract requires T=1")
    speculative_config = None if args.mode == "ar" else {
        "method": "custom_class",
        "model": "blockspec.vllm_ascend.proposer.OrthrusProposer",
        "num_speculative_tokens": 31,
        "draft_sample_method": "probabilistic",
    }
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    llm = LLM(
        model=args.model,
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.graph_mode == "eager",
        compilation_config={"cudagraph_mode": (
            "NONE" if args.graph_mode == "eager" else args.graph_mode)},
        enable_prefix_caching=True,
        speculative_config=speculative_config,
    )
    loaded = time.perf_counter()
    sampling = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.generate([rendered_prompt], sampling, use_tqdm=False)
    finished = time.perf_counter()
    result = {
        "contract": {
            "mode": args.mode,
            "block_size": 32,
            "num_speculative_tokens": 31,
            "temperature": temperature,
            "top_k": 0,
            "top_p": 1.0,
            "sampling": "tensorized exact correction",
        },
        "model_load_seconds": loaded - started,
        "generation_seconds": finished - loaded,
        "runtime": {
            "graph_mode": args.graph_mode,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "prompt_token_ids": outputs[0].prompt_token_ids,
        "output_token_ids": outputs[0].outputs[0].token_ids,
        "text": outputs[0].outputs[0].text,
        "finish_reason": outputs[0].outputs[0].finish_reason,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        report.dump(result, handle, indent=2)
        handle.write("\n")
    print(report.dumps(result))


if __name__ == "__main__":
    main()
