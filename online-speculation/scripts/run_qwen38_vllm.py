#!/usr/bin/env python3
"""Launch native vLLM OpenAI serving with the Qwen3.8 Orthrus plugin."""

import argparse
import json
import os


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--mask-token-id", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--last-layers", type=int, default=1)
    parser.add_argument("--update-stride", type=int, default=16)
    parser.add_argument("--head-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.8)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gdn-prefill-backend", choices=("native", "cann"), default="cann")
    return parser.parse_args()


def native_options(args):
    return dict(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
                enforce_eager=True, max_num_seqs=1, max_model_len=args.max_model_len,
                max_num_batched_tokens=256, gpu_memory_utilization=args.gpu_memory_utilization,
                enable_prefix_caching=False, async_scheduling=False, language_model_only=True,
                additional_config={"qwen38_gdn_prefill_backend": args.gdn_prefill_backend},
                worker_extension_cls="blockspec.vllm_ascend.qwen38_proposer.OrthrusWorkerExtension")


def speculative_options(args):
    options = dict(mask_token_id=args.mask_token_id, online=args.online, last_layers=args.last_layers,
                   update_stride=args.update_stride)
    for name in ("head_dir", "checkpoint_dir"):
        if getattr(args, name, None):
            options[name] = getattr(args, name)
    return dict(speculative_config={
        "method": "custom_class", "model": "blockspec.vllm_ascend.qwen38_proposer.Qwen38OrthrusProposer",
        "num_speculative_tokens": args.block_size - 1, "disable_padded_drafter_batch": True,
        "draft_sample_method": "probabilistic",
    }, additional_config={"qwen38_orthrus": options, "qwen38_gdn_prefill_backend": args.gdn_prefill_backend})


def main():
    args = arguments()
    settings = native_options(args) | speculative_options(args)
    model = settings.pop("model")
    command = ["vllm", "serve", model, "--host", args.host, "--port", str(args.port)]
    for name, value in settings.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            command.extend([flag, json.dumps(value) if isinstance(value, dict) else str(value)])
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
