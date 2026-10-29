"""Text-only hybrid reference smoke: AR, speculative correction, optional online replay.

This does not launch vLLM and does not claim 27B deployment readiness.
"""

import argparse
import json
from pathlib import Path

import torch

from blockspec.parallel.feedback import OnlineFeedback
from blockspec.parallel.generation import generate, generate_ar
from blockspec.parallel.online import SuffixConfig, SuffixLearner
from blockspec.parallel.qwen38 import Qwen38Branch, Qwen38DualView
from blockspec.sampling import SamplingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", action="store_true")
    source.add_argument("--model", help="Local BF16/FP32 Qwen3.8 checkpoint; no remote code")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer-devices", help="Comma-separated contiguous layer devices; not TP")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--prompt", default="Explain what a compiler does in one sentence.")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--output", type=Path, help="Write the generated measurement JSON")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--last-layers", type=int, default=1)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--hybrid-policy", required=True, choices=("orthrus_qwen35_reference",))
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    devices = args.layer_devices.split(",") if args.layer_devices else [args.device]
    if devices[0] != args.device:
        parser.error("the first --layer-devices entry must equal --device")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if args.tiny:
        config = Qwen3_5TextConfig(
            vocab_size=48, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=256,
            linear_num_key_heads=1, linear_num_value_heads=2,
            linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                             "partial_rotary_factor": .25, "mrope_section": [11, 11, 10]},
        )
        config._attn_implementation = "sdpa"
        target = Qwen3_5ForCausalLM(config).eval()
        mask = 1 if args.mask_token_id is None else args.mask_token_id
        prompt = torch.tensor([[3, 5, 8]], device=args.device)
    else:
        if args.mask_token_id is None:
            parser.error("a real checkpoint requires an explicitly agreed --mask-token-id")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        prompt = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
        target = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, args.dtype), attn_implementation="sdpa")
        mask = args.mask_token_id
    model = Qwen38DualView(target, block_size=args.block_size, mask_token_id=mask,
                           hybrid_policy=args.hybrid_policy)
    del target
    model.to(dtype=getattr(torch, args.dtype)).place_layers(devices)
    branch = Qwen38Branch(model)
    sampling = SamplingConfig(temperature=args.temperature)
    ar = generate_ar(branch, prompt, args.tokens, sampling=sampling)
    fixed = generate(branch, prompt, args.tokens, sampling=sampling, audit_cache=True)
    if args.temperature == 0 and ar.tokens != fixed.tokens:
        raise RuntimeError("fixed draft changed greedy AR output")
    report = {"executor": "transformers_hybrid_reference", "model": args.model or "tiny_random",
              "device": args.device, "dtype": args.dtype, "hybrid_policy": args.hybrid_policy,
              "layer_devices": devices,
              "ar": ar.summary(), "fixed": fixed.summary(),
              "greedy_equal": True if args.temperature == 0 else None,
              "temperature": args.temperature,
              "torch_version": torch.__version__,
              "layer_types": model.layer_types,
              "head_dim": model.config.head_dim,
              "linear_head_dim": model.config.linear_key_head_dim,
              "block_size": args.block_size}
    if args.online:
        learner = SuffixLearner(model, SuffixConfig(last_layers=args.last_layers, stride=args.stride, loss="tv"))
        online = generate(branch, prompt, args.tokens, sampling=sampling,
                          feedback=OnlineFeedback(learner=learner), audit_cache=True)
        if args.temperature == 0 and ar.tokens != online.tokens:
            raise RuntimeError("online draft changed greedy AR output")
        report.update(online=online.summary(), last_loss=learner.last_loss,
                      trainable_parameters=learner.trainable_parameters,
                      last_layers=args.last_layers)
        if not learner.updates:
            raise RuntimeError("no online update occurred; lower --stride or increase --tokens")
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
