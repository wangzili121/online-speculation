"""Optional external-engine control, isolated from the independent implementation.

Each measured arm starts in a fresh process with a pinned local research checkout.
The workload is natural-prefix continuation on local hardware; all results go to stdout.
"""

import argparse

from blockspec import reporting as report
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_source(checkout, commit):
    checkout = Path(checkout).resolve()
    command = ["git", "--no-replace-objects", "-C", str(checkout)]
    actual = subprocess.run(command + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual != commit:
        raise ValueError("reference checkout must match the locked commit")
    tree = subprocess.run(command + ["ls-tree", "-rz", commit], check=True, capture_output=True).stdout
    checked = 0
    for record in tree.split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        if not name.endswith(b".py"):
            continue
        mode, kind, blob = metadata.split(b" ", 2)
        path = (checkout / name.decode()).resolve()
        if kind != b"blob" or mode == b"120000" or not path.is_relative_to(checkout):
            raise ValueError("reference source must use regular files within the checkout")
        # Windows checkout CRLF and Git's canonical LF encode the same source.
        contents = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
        if digest != blob.decode():
            raise ValueError(f"reference Python source differs from the pinned checkout: {name.decode()}")
        checked += 1
    extra = subprocess.run(command + ["ls-files", "--others", "--exclude-standard", "--", "*.py"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if not checked or extra:
        raise ValueError("reference Python source differs from the pinned checkout")
    return checkout


def check_artifacts(base, adapter, lock):
    result = {}
    for key, directory in (("base", base), ("adapter", adapter)):
        entry = lock["models"][key]
        digest = file_sha256(Path(directory) / entry["weight_filename"])
        if digest != entry["weight_sha256"]:
            raise ValueError(f"reference weight SHA mismatch: {key}")
        result[key] = digest
    return result


def summarize(rows):
    result = {}
    for arm in ("ar", "static"):
        selected = [row for row in rows if row["arm"] == arm]
        tokens = sum(row["tokens"] for row in selected)
        seconds = sum(row["seconds"] for row in selected)
        result[arm] = {"tokens": tokens, "seconds": seconds, "tps": tokens / seconds,
                       "initialization_seconds": sum(row["initialization_seconds"] for row in selected),
                       "warmup_seconds": sum(row["warmup_seconds"] for row in selected),
                       "forwards": sum(row["stats"]["forwards"] for row in selected),
                       "counted_accepts": sum(row["stats"]["accepts"] for row in selected)}
        result[arm]["tokens_per_sequence_forward"] = result[arm]["counted_accepts"] / result[arm]["forwards"]
    return {"arms": result, "speedup": result["static"]["tps"] / result["ar"]["tps"]}


def run_arm(args, lock):
    # Source isolation happens before importing the reference package.
    checkout = check_source(args.checkout, lock["source"]["commit"])
    os.environ["HF_MODULES_CACHE"] = str(args.cache.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NANO_VLLM_FORCE_FLOAT32"] = "0"
    sys.path.insert(0, str(checkout))
    import numpy as np
    import torch
    import nano_vllm_uno
    from nano_vllm_uno import LLM, SamplingParams
    from nano_vllm_uno.utils.model_tokens import resolve_model_token_ids
    if Path(nano_vllm_uno.__file__).resolve().parent != checkout / "nano_vllm_uno":
        raise ValueError("reference module was loaded from a different checkout")
    torch.set_num_threads(4)
    seed = args.seed + args.repeat_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sequences = [json.loads(line)["input_ids"] for line in args.data.read_text(encoding="utf-8").splitlines()]
    prompts = [ids[:args.prompt_length] for ids in sequences if len(ids) >= args.prompt_length][:args.prompts]
    if len(prompts) != args.prompts:
        raise ValueError("insufficient eligible reference prompts")
    block = 1 if args.arm == "ar" else args.block_size
    start = time.perf_counter()
    engine = LLM(str(args.base), gated_lora_path=str(args.adapter) if args.arm == "static" else None,
                 hf_local_files_only=True, attention_backend="fa2", max_num_seqs=1,
                 max_num_batched_tokens=args.prompt_length + args.tokens + args.block_size,
                 max_model_len=args.prompt_length + args.tokens + args.block_size,
                 max_diffusion_block_size=block, cuda_graph_block_sizes=sorted({1, block}),
                 cuda_graph_batch_sizes=[1], num_kvcache_blocks=16)
    initialized = time.perf_counter() - start
    try:
        if engine.config.dtype != torch.bfloat16:
            raise ValueError("reference control expects the published engine's BF16 policy")
        mask_id, stop_ids, vocabulary = resolve_model_token_ids(str(args.base), engine.tokenizer,
                                                               local_files_only=True, noise_mode="random_uniform")
        token_options = {"mask_token_id": mask_id, "stop_token_ids": stop_ids, "noise_mode": "random_uniform"}
        warm = SamplingParams(temperature=args.temperature, max_tokens=32, ignore_eos=True,
                              diffusion_block_size=block, **token_options)
        start = time.perf_counter()
        engine.generate([prompts[0][:32]], warm, request_max_tokens=[32], use_tqdm=False, detokenize=False)
        torch.cuda.synchronize()
        warm_seconds = time.perf_counter() - start
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        params = SamplingParams(temperature=args.temperature, max_tokens=args.tokens, ignore_eos=True,
                                diffusion_block_size=block, **token_options)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(prompts, params, request_max_tokens=[args.tokens] * len(prompts),
                                  use_tqdm=False, detokenize=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        stats = dict(engine.last_generate_stats)
        row = {"stage": "reference_arm", "arm": args.arm, "repeat": args.repeat_index,
               "tokens": sum(len(o["token_ids"]) for o in outputs), "seconds": seconds, "stats": stats,
               "initialization_seconds": initialized, "warmup_seconds": warm_seconds,
               "dtype": str(engine.config.dtype), "attention_backend": engine.config.attention_backend,
               "mask_token_id": mask_id, "vocab_size": vocabulary, "kv_pages": engine.config.num_kvcache_blocks,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "output_lengths": [len(o["token_ids"]) for o in outputs],
               "budget_lengths_match": all(len(o["token_ids"]) == args.tokens for o in outputs),
               "preemptions": engine.get_stats()["preemptions"],
               "cuda_graph_hits": engine.model_runner.cuda_graph_hits,
               "cuda_graph_misses": engine.model_runner.cuda_graph_misses}
    finally:
        engine.exit()
    print(report.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "base", "adapter", "data", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True, help="local source and artifact checks")
    parser.add_argument("--prompts", type=int, default=17)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--arm", choices=["ar", "static"], help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.prompts, args.prompt_length, args.tokens) < 1 or args.block_size < 2:
        parser.error("positive request dimensions and block size >=2 required")
    if args.repeats < 2 or args.repeats % 2:
        parser.error("a positive even repeat count is required")
    # Sixteen 256-token pages cover this single-sequence control plus scratch pages.
    if args.prompt_length + args.tokens + args.block_size > 3072:
        parser.error("this bounded reference control supports at most 3072 total positions")
    lock = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    if args.arm:
        run_arm(args, lock)
        return
    check_source(args.checkout, lock["source"]["commit"])
    provenance = check_artifacts(args.base, args.adapter, lock)
    rows = []
    for repeat in range(args.repeats):
        for arm in (("ar", "static") if repeat % 2 == 0 else ("static", "ar")):
            print(report.dumps({"stage": "reference_start", "arm": arm, "repeat": repeat}), flush=True)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                     "--arm", arm, "--repeat-index", str(repeat)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            row = next(json.loads(line) for line in reversed(result.stdout.splitlines())
                       if line.startswith('{"stage": "reference_arm"'))
            rows.append(row)
            print(report.dumps(row), flush=True)
    print(report.dumps({"stage": "reference_complete", "source_commit": lock["source"]["commit"],
                      "weights": provenance, "data_sha256": file_sha256(args.data),
                      "benchmark_sha256": file_sha256(__file__), "rows": rows,
                      "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      **summarize(rows), "scope": "external BF16/FA2 engine; natural continuations; fresh process per arm; "
                      "initialization and warmup separate; prefill included; detokenization excluded"}), flush=True)


if __name__ == "__main__":
    main()
