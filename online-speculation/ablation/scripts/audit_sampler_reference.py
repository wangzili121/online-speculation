"""Optional CPU-only oracle audit against a pinned, local research checkout.

The serving/training package never imports this source. This audit reads a small
set of CPU definitions from the pinned Git object, not from mutable working files,
and does not load the author's model, engine, package initializers or GPU kernels.
"""

import argparse

from blockspec import reporting as report
import ast
from dataclasses import dataclass, field
import hashlib
import heapq
import math
from pathlib import Path
import subprocess
from typing import Sequence

import torch

from blockspec_ablation.distillation import divergence
from blockspec_ablation.tree import build_tree, traverse_greedy


def load_reference(checkout, revision):
    commit = subprocess.run(["git", "--no-replace-objects", "-C", str(checkout),
                             "rev-parse", "--verify", revision + "^{commit}"],
                            check=True, capture_output=True, text=True).stdout.strip()
    source_file = "nano_vllm_uno/engine/draft_tree.py"
    source = subprocess.run(["git", "--no-replace-objects", "-C", str(checkout), "show",
                             f"{commit}:{source_file}"], check=True, capture_output=True).stdout
    wanted = {"_TreeNode", "DraftTree", "build_tree_from_candidates", "walk_tree"}
    selected = [node for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in wanted]
    if {node.name for node in selected} != wanted:
        raise ValueError("pinned reference does not expose the expected CPU contract")
    # walk_tree's CUDA branch is never reached: every audit tensor is on CPU.
    namespace = {"__name__": __name__, "torch": torch, "Tensor": torch.Tensor,
                 "dataclass": dataclass, "field": field, "Sequence": Sequence,
                 "heapq": heapq, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), f"reference:{commit}:{source_file}", "exec"), namespace)
    return namespace, {"commit": commit, "file": source_file, "source_sha256": hashlib.sha256(source).hexdigest()}


def audit(checkout, *, revision, trials=100):
    if trials < 1:
        raise ValueError("positive trial count required")
    oracle, provenance = load_reference(checkout, revision)
    generator = torch.Generator().manual_seed(8721)
    built = walked = 0
    for depth in (1, 2, 3, 4):
        q = torch.randn(depth, 5, generator=generator, dtype=torch.float64).softmax(-1)
        values, ids = q.topk(3, dim=-1)
        for budget in (1, 2, 5, 12):
            ours = build_tree(2, q, top_k=3, prefix_budget=budget)
            theirs = oracle["build_tree_from_candidates"](2, ids.tolist(), values.log().tolist(), budget)
            assert ours.tokens == list(theirs.token_ids)
            assert ours.parents == list(theirs.parent_indices)
            assert ours.depths == list(theirs.depths)
            torch.testing.assert_close(torch.tensor(ours.scores), torch.tensor(theirs.log_masses), atol=1e-6, rtol=1e-6)
            built += 1
            tree_tokens = torch.tensor([ours.tokens])
            parents = torch.tensor([ours.parents])
            for _ in range(trials):
                # Arbitrary pre-sampled target IDs couple deterministic traversal;
                # their probability law is verified independently by unit tests.
                target = torch.randint(5, tree_tokens.shape, generator=generator)
                emitted, cached, lengths = oracle["walk_tree"](tree_tokens, parents, target,
                                                               max_depth=depth, pad_token_id=-1)
                result = traverse_greedy(ours, target[0], budget=depth + 2)
                assert result.tokens == emitted[0, :int(lengths[0])].tolist()
                assert result.path == cached[0][cached[0] >= 0].tolist()
                walked += 1
    losses = audit_losses(checkout, provenance["commit"])
    return {"reference": provenance, "tree_builds_matched": built, "target_walks_matched": walked,
            "losses": losses,
            "scope": "non-tied prefix scores and CPU pre-sampled target paths; not model/kernel/TPS equivalence"}


def audit_losses(checkout, commit):
    """Compare full-vocabulary L1 values and student gradients with the pinned source."""
    source = subprocess.run(["git", "--no-replace-objects", "-C", str(checkout), "show",
                             f"{commit}:training/losses.py"], check=True, capture_output=True).stdout
    wanted = {"_ChunkedTotalVariation", "token_normalized_total_variation", "token_normalized_reverse_kl"}
    definitions = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                   and n.name in wanted]
    if {n.name for n in definitions} != wanted:
        raise ValueError("pinned source loss contract changed")
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), f"reference:{commit}:losses", "exec"), namespace)
    rng = torch.Generator().manual_seed(315)
    maximum_value = maximum_gradient = 0.0
    cases = 0
    for rows, vocab in ((1, 7), (5, 103), (3, 2111)):
        student = torch.randn(rows, vocab, generator=rng, requires_grad=True)
        teacher = torch.randn(rows, vocab, generator=rng, requires_grad=True)
        for kind, name in (("l1", "token_normalized_total_variation"),
                           ("reverse_kl", "token_normalized_reverse_kl")):
            actual = divergence(student, teacher, kind).mean()
            expected = namespace[name](student, teacher, rows)
            ours = torch.autograd.grad(actual, (student, teacher), allow_unused=True)
            theirs = torch.autograd.grad(expected, (student, teacher), allow_unused=True)
            assert ours[1] is None and theirs[1] is None
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(ours[0], theirs[0], atol=2e-7, rtol=2e-5)
            maximum_value = max(maximum_value, float((actual - expected).abs().detach()))
            maximum_gradient = max(maximum_gradient, float((ours[0] - theirs[0]).abs().max()))
            cases += 1
    return {"cases": cases, "max_value_error": maximum_value, "max_gradient_error": maximum_gradient,
            "source_sha256": hashlib.sha256(source).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="existing read-only Git checkout; no download")
    parser.add_argument("--reference-revision", required=True, help="caller-selected local Git ref")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(report.dumps(audit(args.source, revision=args.reference_revision, trials=args.trials)), flush=True)
