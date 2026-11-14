#!/usr/bin/env python3
"""Run one offline KL update from captured V4.1 verifier feedback and package the head."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from blockspec.vllm_ascend.dsv41_contract import head_manifest, validate_head
from blockspec.losses import divergence
from blockspec.vllm_ascend.dsv41_last_layer import LastLayerLearner, load_feedback
from blockspec.vllm_ascend.head_contract import find_weight_index, read_json, write_json


def save_updated_head(model, head, output, prefix, updated, mask_id):
    """Package all layers while replacing only the trained suffix tensors."""
    index = find_weight_index(head)
    mapping = read_json(index)["weight_map"]
    weights = {prefix + name: value for name, value in updated.items()}
    names = {name for name in mapping if name.startswith(prefix)}
    if names != weights.keys():
        raise ValueError("the update does not contain the complete final attention layer")

    modified_shards = {mapping[name] for name in names}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary)
        for filename in sorted(set(mapping.values())):
            if filename not in modified_shards:
                copyfile(head / filename, staging / filename)
                continue
            with safe_open(head / filename, framework="pt") as shard:
                tensors = {name: shard.get_tensor(name) for name in shard.keys()}
            for name in tensors.keys() & names:
                tensors[name] = weights[name].detach().to(tensors[name].dtype).contiguous()
            save_file(tensors, staging / filename)
        copyfile(head / "config.json", staging / "config.json")
        copyfile(index, staging / index.name)
        write_json(
            staging / "manifest.json",
            head_manifest(model, mask_token_id=mask_id, artifact_role="trained"),
        )
        validate_head(model, staging, mask_token_id=mask_id, require_trained=True)
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "head", "feedback", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.output.exists() or not 0 < args.learning_rate < float("inf"):
        parser.error("the output must be new and the learning rate must be positive")

    mask_id = read_json(args.head / "config.json")["mask_token_id"]
    validate_head(args.model, args.head, mask_token_id=mask_id)
    replay, feedback = load_feedback(args.feedback, args.model, args.head)
    learner = LastLayerLearner(args.model, args.head, learning_rate=args.learning_rate)
    teacher, scheduled, sampled = (feedback[name] for name in ("teacher", "scheduled", "sampled"))
    draft = feedback["draft"].float()
    reached = min(scheduled, sampled)
    with torch.no_grad():
        logits = learner.logits(replay, reached=reached)
    before = {
        name: value.detach().to(torch.float32 if name == "attn_sink" else torch.bfloat16, copy=True)
        for name, value in learner.master.items()
    }
    loss, norm, step = learner.update(replay, teacher, scheduled, sampled, kind="forward_kl")
    with torch.no_grad():
        after = float(learner.loss(replay, teacher, scheduled, sampled, kind="forward_kl"))
    changed = [
        name for name, value in learner.master.items()
        if not torch.equal(before[name], value.to(before[name]))
    ]
    if not changed or not after < loss or not norm > 0:
        raise RuntimeError(f"offline step did not improve loss and change the export: {loss=}, {after=}, {norm=}")
    save_updated_head(args.model, args.head, args.output, learner.base.prefix + "attn.", learner.master, mask_id)
    print(json.dumps({
        "anchor": feedback["anchor"], "scheduled": scheduled, "reached": reached, "step": step,
        "kl_before": loss, "kl_after": after, "gradient_norm": norm, "changed_tensors": changed,
        "draft_vs_replay_logit_max_abs": float((logits - draft).abs().max()),
        "draft_vs_replay_logit_p99_abs": float((logits - draft).abs().flatten().quantile(0.99)),
        "draft_vs_replay_tv_mean": float(divergence(logits, draft, "tv").mean()),
        "draft_vs_replay_argmax_matches": int((logits.argmax(-1) == draft.argmax(-1)).sum()),
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
