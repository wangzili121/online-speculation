"""Text-only hybrid reference smoke: AR, speculative correction, optional online replay.

This does not launch vLLM and does not claim 27B deployment readiness.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from blockspec.parallel.feedback import OnlineFeedback
from blockspec.parallel.generation import generate, generate_ar
from blockspec.parallel.online import SuffixConfig, SuffixLearner
from blockspec.parallel.qwen38 import Qwen38Branch, Qwen38DualView, native_reference_logits
from blockspec.parallel.sampling import ProposalSampler
from blockspec.sampling import SamplingConfig


class TracedLearner(SuffixLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def update(self):
        result = super().update()
        if result is not None:
            self.records.append(result)
        return result


@torch.no_grad()
def ar_numerics(model, prompt, output_tokens):
    """AR-only control: compare identical chunking and full recomputation."""
    suffix = prompt.new_tensor([output_tokens[:2]])
    tokens = torch.cat((prompt, suffix), dim=1)
    prefix = model(prompt).cache
    chunk = model(suffix, cache=prefix)
    repeat = model(suffix, cache=model(prompt).cache)
    full = model(tokens)
    rows, repeat_equal = [], True
    for index, (actual, other, whole) in enumerate(zip(chunk.cache, repeat.cache, full.cache, strict=True)):
        for component, (x, y, z) in enumerate(zip(actual, other, whole, strict=True)):
            repeat_equal &= torch.equal(x, y)
            diff = (x.float() - z.float())
            rows.append({"layer": index, "kind": model.layer_types[index], "state": component,
                         "full_vs_chunk_max_abs": float(diff.abs().max()),
                         "full_vs_chunk_relative_l2": float(diff.norm() / z.float().norm().clamp_min(1e-20))})
    return {"matched_chunks_bitwise_equal": repeat_equal,
            "worst_full_vs_chunk_states": sorted(rows, key=lambda row: row["full_vs_chunk_max_abs"], reverse=True)[:8],
            "full_vs_chunk_logits_max_abs": float((chunk.logits.float() - full.logits[:, prompt.shape[1]:].float()).abs().max())}


@torch.no_grad()
def probe_full_block(branch, prompt, sampling):
    """Exercise a full K block even when the generation budget is shorter."""
    sampler = ProposalSampler(sampling)
    logits, history = branch.ar(prompt)
    saved = [tuple(t.clone() for t in pair) for pair in history]
    anchor = prompt.new_tensor([[sampler.sample_ar(logits[0, -1], None)]])
    proposal = branch.draft(anchor, history, branch.default_block_size, sampling, sampler=sampler)
    for pair, other in zip(history, saved, strict=True):
        for value, expected in zip(pair, other, strict=True):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    teacher, verified = branch.ar(proposal.verifier_inputs, cache=history)
    correction, _ = sampler.verify(proposal, teacher[0], None)
    count = len(correction.tokens)
    committed = branch.commit_history(proposal, verified, history.length + count)
    inputs = torch.cat((prompt, proposal.verifier_inputs[:, :count]), dim=1)
    branch.audit_history(committed, inputs, [prompt.shape[1], count])
    return {"query_tokens": proposal.verifier_inputs.shape[1],
            "proposed_tokens": proposal.candidates.numel(), "accepted": correction.accepted,
            "rejected_at": correction.rejected_at, "draft_history_bitwise_unchanged": True,
            "committed_state_bitwise_matches_ar": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", action="store_true")
    source.add_argument("--model", help="Local BF16/FP32 Qwen3.8 checkpoint; no remote code")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer-devices", help="Comma-separated contiguous layer devices; not TP")
    parser.add_argument("--reserve-memory-gb", type=float, default=0.,
                        help="Temporarily allocate GiB per NPU while loading CPU weights")
    parser.add_argument("--native-reference", action="store_true", help="Check unchanged HF model forward")
    parser.add_argument("--diagnose-numerics", action="store_true", help="Run AR-only chunking controls")
    parser.add_argument("--probe-block", action="store_true", help="Verify a full K block independently of output budget")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--prompt", default="Explain what a compiler does in one sentence.")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--sampling-smoke-tokens", type=int, default=0,
                        help="Additionally run temperature=1 with the already-loaded model")
    parser.add_argument("--output", type=Path, help="Write the generated measurement JSON")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--last-layers", type=int, default=1)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--hybrid-policy", required=True, choices=("orthrus_qwen35_reference",))
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.sampling_smoke_tokens < 0:
        parser.error("--sampling-smoke-tokens must be nonnegative")
    if not math.isfinite(args.reserve_memory_gb) or args.reserve_memory_gb < 0:
        parser.error("--reserve-memory-gb must be finite and nonnegative")
    devices = args.layer_devices.split(",") if args.layer_devices else [args.device]
    if devices[0] != args.device:
        parser.error("the first --layer-devices entry must equal --device")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    reservations = []
    if args.reserve_memory_gb:
        for device in dict.fromkeys(devices):
            if device.startswith("npu"):
                reservations.append(torch.empty(int(args.reserve_memory_gb * 1024 ** 3),
                                                dtype=torch.uint8, device=device))
        print("NPU memory allocated for weight loading", file=sys.stderr, flush=True)
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
    print("AR checkpoint loaded; building independent draft mixers", file=sys.stderr, flush=True)
    model = Qwen38DualView(target, block_size=args.block_size, mask_token_id=mask,
                           hybrid_policy=args.hybrid_policy)
    # Keep the freed blocks in the allocator so weight placement can reuse them.
    reservations.clear()
    model.to(dtype=getattr(torch, args.dtype)).place_layers(devices)
    native_report = None
    if args.native_reference:
        expected = native_reference_logits(target, prompt)
        with torch.no_grad():
            actual = model(prompt).logits
        native_report = {"max_abs_diff": float((actual.float() - expected.float()).abs().max()),
                         "bitwise_equal": torch.equal(actual, expected),
                         "argmax_equal": torch.equal(actual.argmax(-1), expected.argmax(-1))}
        print(f"Native HF forward comparison: {native_report}", file=sys.stderr, flush=True)
        torch.testing.assert_close(actual, expected)
    del target
    print("All model layers placed; starting AR", file=sys.stderr, flush=True)
    branch = Qwen38Branch(model)
    sampling = SamplingConfig(temperature=args.temperature)
    ar = generate_ar(branch, prompt, args.tokens, sampling=sampling)
    numerical_control = ar_numerics(model, prompt, ar.tokens) if args.diagnose_numerics else None
    if numerical_control is not None:
        print(f"AR-only numerical control: {numerical_control}", file=sys.stderr, flush=True)
    block_probe = probe_full_block(branch, prompt, sampling) if args.probe_block else None
    if block_probe is not None:
        print(f"Full block probe: {block_probe}", file=sys.stderr, flush=True)
    print("AR completed; starting fixed draft and state audit", file=sys.stderr, flush=True)
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
              "num_hidden_layers": len(model.layers), "hidden_size": model.config.hidden_size,
              "vocab_size": model.config.vocab_size,
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "draft_parameters": sum(p.numel() for name, p in model.named_parameters() if ".attention.draft." in name),
              "prompt_token_ids": prompt[0].tolist(), "full_block_probe": block_probe,
              "layer_types": model.layer_types,
              "head_dim": model.config.head_dim,
              "linear_head_dim": model.config.linear_key_head_dim,
              "block_size": args.block_size, "mask_token_id": mask,
              "native_reference": native_report, "ar_numerics": numerical_control,
              "ar_token_ids": ar.tokens, "fixed_token_ids": fixed.tokens,
              "cache_audit": "bitwise_ar_replay_with_matched_chunks"}
    learner = None
    if args.online:
        print("Fixed draft completed; starting online suffix updates", file=sys.stderr, flush=True)
        learner = TracedLearner(model, SuffixConfig(last_layers=args.last_layers, stride=args.stride, loss="tv"))
        print("Online learner initialized; running suffix replay", file=sys.stderr, flush=True)
        before = {name: p.detach().clone() for name, p in learner.execution.items()}
        online = generate(branch, prompt, args.tokens, sampling=sampling,
                          feedback=OnlineFeedback(learner=learner), audit_cache=True)
        if args.temperature == 0 and ar.tokens != online.tokens:
            raise RuntimeError("online draft changed greedy AR output")
        learner._check()
        changed = sum(int((p != before[name]).sum()) for name, p in learner.execution.items())
        report.update(online=online.summary(), last_loss=learner.last_loss,
                      trainable_parameters=learner.trainable_parameters,
                      last_layers=args.last_layers, update_records=list(learner.records),
                      changed_draft_elements=changed, frozen_versions_unchanged=True,
                      online_token_ids=online.tokens)
        if not learner.updates:
            raise RuntimeError("no online update occurred; lower --stride or increase --tokens")
        if not changed:
            raise RuntimeError("online updates did not change any execution draft elements")
    if args.sampling_smoke_tokens:
        print("Starting temperature=1 smoke with the same loaded model", file=sys.stderr, flush=True)
        distribution = SamplingConfig(temperature=1.)
        probe = probe_full_block(branch, prompt, distribution)
        record_start = 0 if learner is None else len(learner.records)
        sampled = generate(branch, prompt, args.sampling_smoke_tokens, sampling=distribution,
                           feedback=None if learner is None else OnlineFeedback(learner=learner),
                           audit_cache=True)
        if learner is not None:
            learner._check()
        report["sampling_smoke"] = {"temperature": 1., "full_block_probe": probe,
                                    "generation": sampled.summary(), "token_ids": sampled.tokens,
                                    "update_records": [] if learner is None else learner.records[record_start:]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
