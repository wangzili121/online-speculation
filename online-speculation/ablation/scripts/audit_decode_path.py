"""Inspect target logits on the common history at a chosen generation position."""

import argparse

from blockspec import reporting as report
from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from blockspec_ablation.adapter_io import load_peft_adapter, peft_config
from blockspec_ablation.benchmark import compare_tokens, continuation_prompts
from blockspec_ablation.checkpoint import implementation_fingerprint, load_checkpoint, load_hf_base
from blockspec.data import load_sequences
from blockspec_ablation.decoding import generate_ar
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec_ablation.model import cache_length
from blockspec_ablation.online import OnlineConfig, OnlineLearner
from blockspec_ablation.replay_execution import SuffixReplayExecutor
from blockspec_ablation.tree import generate_tree


class TargetTrace:
    """Wrap an executor and retain base rows at one absolute logit position."""

    def __init__(self, engine, position):
        self.engine, self.position, self.rows = engine, position, []

    def validate(self, model):
        self.engine.validate(model)

    def _forward(self, tokens, **kwargs):
        result = self.engine._forward(tokens, **kwargs)
        prefix = cache_length(kwargs.get("cache"))
        positions = kwargs.get("positions")
        if positions is None:
            positions = torch.arange(prefix, prefix + tokens.shape[1], device=tokens.device)[None]
        mask, allowed = kwargs.get("adapter_mask"), kwargs.get("allowed")
        for row in (positions[0] == self.position).nonzero().flatten().tolist():
            if mask is not None and bool(mask[0, row]):
                continue
            visible = (torch.arange(tokens.shape[1], device=tokens.device) <= row if allowed is None else
                       allowed[0, 0, row, prefix:])
            self.rows.append({"prefix": prefix, "path": tokens[0, visible].tolist(),
                              "kind": "draft_root" if mask is not None else "base",
                              "logits": result[0][0, row].detach().cpu().clone()})
        return result


def compare_target_rows(reference, actual):
    p, q = reference.softmax(-1), actual.softmax(-1)
    top = reference.topk(2)
    chosen = actual.topk(2)
    return {"max_logit_error": float((reference - actual).abs().max()),
            "tv": float((p - q).abs().sum() / 2),
            "ar_argmax": int(reference.argmax()), "spec_argmax": int(actual.argmax()),
            "ar_top_ids": top.indices.tolist(), "ar_top_logits": top.values.tolist(),
            "ar_margin": float(top.values[0] - top.values[1]),
            "spec_top_ids": chosen.indices.tolist(), "spec_top_logits": chosen.values.tolist()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "adapter", "data"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--reference-sha256", help="optional local adapter integrity check")
    parser.add_argument("--request", type=int, required=True)
    parser.add_argument("--token-index", type=int, required=True, help="zero-based generated token index")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--prefix-budget", type=int, default=16)
    parser.add_argument("--seed", type=int, default=271844)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution", choices=["eager", "cuda_graph"], default="cuda_graph")
    parser.add_argument("--online-stream", action="store_true", help="replay preceding requests with live adapter updates")
    parser.add_argument("--stream-prompts", type=int, default=17)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--stream-seed", type=int, default=271828, help="benchmark base seed for the online request stream")
    parser.add_argument("--online-last-layers", type=int, default=4)
    parser.add_argument("--update-stride", type=int, default=16)
    parser.add_argument("--replay-blocks", type=int, default=1)
    parser.add_argument("--loss", choices=["l1", "forward_kl"], default="forward_kl")
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--update-policy", choices=["periodic", "coverage"], default="coverage")
    parser.add_argument("--online-execution", choices=["eager", "cuda_graph"], default="cuda_graph")
    args = parser.parse_args()
    if args.request < 0 or not 0 <= args.token_index < args.tokens:
        parser.error("valid request and generation index required")
    if args.online_stream and (not 0 <= args.request < args.stream_prompts or args.repeat_index < 0):
        parser.error("online request index must fit the stream; repeat index must be nonnegative")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    adapter_path = Path(args.adapter)
    if adapter_path.is_dir():
        config = peft_config(adapter_path)
        model = load_hf_base(args.base, rank=config["r"], alpha=config["lora_alpha"], device=args.device)
        provenance = load_peft_adapter(adapter_path, model, expected_sha256=args.reference_sha256)
    else:
        config = torch.load(adapter_path, map_location="cpu", weights_only=True)["config"]
        model = load_hf_base(args.base, rank=config["adapter_rank"], alpha=config["adapter_alpha"], device=args.device)
        model, _ = load_checkpoint(adapter_path, model=model, device=args.device)
        provenance = {"kind": "local_training", "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest()}
    model.set_attention_backend("grouped")
    count = args.stream_prompts if args.online_stream else args.request + 1
    prompts = [p.to(args.device) for p in continuation_prompts(load_sequences(args.data, model.config.vocab_size),
                                                             count=count, length=args.prompt_length)]
    prompt = prompts[args.request]
    learner = None
    if args.online_stream:
        learner = OnlineLearner(model, OnlineConfig(stride=args.update_stride, replay_blocks=args.replay_blocks,
                                                    train_last_layers=args.online_last_layers, loss=args.loss,
                                                    learning_rate=args.learning_rate, update_policy=args.update_policy))
    maximum = max(args.block_size, args.prefix_budget)
    engine = FixedShapeExecutor(model, capacity=args.prompt_length + args.tokens, max_query=maximum,
                                 use_cuda_graph=args.execution == "cuda_graph")
    engine.prepare([(n, False, None) for n in range(1, maximum + 1)] +
                   [(n, True, c) for n in range(2, args.block_size + 1)
                    for c in ({None, learner.capture_layer} if learner is not None else {None})])
    if learner is not None and args.online_execution == "cuda_graph":
        replay = SuffixReplayExecutor(model, start_layer=learner.capture_layer, loss=args.loss,
                                      capacity=args.prompt_length + args.tokens, max_query=args.block_size)
        replay.prepare([(n, m) for n in range(2, args.block_size + 1) for m in range(1, n)])
        learner.replay_executor = replay
    position = args.prompt_length - 1 + args.token_index
    ar_trace, spec_trace = TargetTrace(engine, position), TargetTrace(engine, position)
    ar = generate_ar(model, prompt, args.tokens, executor=ar_trace)
    options = {"block_size": args.block_size, "top_k": args.top_k, "prefix_budget": args.prefix_budget}
    seed = args.seed
    if learner is not None:
        for request in range(args.request):
            rng = torch.Generator(device=args.device).manual_seed(args.stream_seed + args.repeat_index * count + request)
            generate_tree(model, prompts[request], args.tokens, executor=engine, learner=learner, generator=rng, **options)
        seed = args.stream_seed + args.repeat_index * count + args.request
    version_before = learner.version if learner is not None else 0
    spec = generate_tree(model, prompt, args.tokens, executor=spec_trace, block_size=args.block_size,
                         top_k=args.top_k, prefix_budget=args.prefix_budget, learner=learner,
                         generator=torch.Generator(device=args.device).manual_seed(seed))
    comparison = compare_tokens(ar.tokens, spec.tokens)
    if comparison["common_prefix"] < args.token_index:
        raise ValueError("choose a token index on the common history or its first differing prediction")
    history = prompt[0].tolist() + ar.tokens
    rows = [{"prefix": row["prefix"], "path": row["path"], "kind": row["kind"],
             **compare_target_rows(ar_trace.rows[0]["logits"], row["logits"])} for row in spec_trace.rows
            if row["path"] == history[row["prefix"]:position + 1]]
    if not rows:
        raise RuntimeError("the trace contains no matching ancestral target row")
    print(report.dumps({"implementation_sha256": implementation_fingerprint(), "request": args.request,
                      "token_index": args.token_index, "comparison": comparison, "target_rows": rows,
                      "adapter": provenance, "request_seed": seed,
                      "online_config": asdict(learner.config) if learner is not None else None,
                      "adapter_version_before_request": version_before,
                      "adapter_version_after_request": learner.version if learner is not None else 0,
                      "scope": "instrumented numerical audit; elapsed time is excluded from TPS evidence"}))


if __name__ == "__main__":
    main()
