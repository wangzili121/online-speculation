"""Prefix-budget trees and exact target-path traversal (not multiguess p/q).

The tree is a computation schedule: at every reached node we sample the BASE
distribution, then reuse a verified child if present. A deterministic top-k
tree is never mislabeled as iid samples from the original proposal law.
"""

from dataclasses import dataclass
import heapq
import math
import time

import torch

from .decoding import Generation, _check, _inference_forward, _prefill
from .diffusion import UniformNoise
from .model import PackedCache, cache_length, trim_cache
from .online import Feedback, synchronize
from blockspec.sampling import SamplingConfig, draw, greedy_tokens, probabilities, sample_logits, validate_distribution


def tree_scores(logits, sampling=SamplingConfig()):
    """Untruncated soft scores: target temperature if positive, one if greedy.

    Candidate top-k is applied by build_tree; target top-k/top-p filters affect
    the target samples, not this tree ranking law. Scores are never p/q ratios.
    """
    return probabilities(logits, SamplingConfig(temperature=sampling.temperature or 1.0))


@dataclass
class CandidateTree:
    tokens: list[int]
    parents: list[int]
    depths: list[int]
    scores: list[float]
    children: list[dict[int, int]]

    def path(self, node):
        result = []
        while node >= 0:
            result.append(node)
            node = self.parents[node]
        return list(reversed(result))

    def layout(self, prefix_length, *, device):
        count = len(self.tokens)
        allowed = torch.zeros(count, prefix_length + count, dtype=torch.bool)
        allowed[:, :prefix_length] = True
        for node in range(count):
            allowed[node, [prefix_length + i for i in self.path(node)]] = True
        ids = torch.tensor([self.tokens], device=device)
        positions = torch.tensor([self.depths], device=device) + prefix_length
        return ids, positions, allowed[None, None].to(device)


def build_tree(root, q, *, top_k=4, prefix_budget=16):
    """Best-first enumeration of top prefix PRODUCTS. Budget includes the root.

    Products cannot increase on extension, so a best-first heap gives globally
    highest-scoring retained prefixes without enumerating K**depth full leaves.
    """
    validate_distribution(q)
    if q.ndim != 2 or top_k < 1 or prefix_budget < 1 or not 0 <= root < q.shape[-1]:
        raise ValueError("invalid tree dimensions or root")
    values, ids = q.detach().topk(min(top_k, q.shape[-1]), dim=-1, sorted=True)
    values, ids = values.cpu(), ids.cpu()
    tree = CandidateTree([root], [-1], [0], [0.0], [{}])
    frontier = []
    def expand(parent, path):
        depth = tree.depths[parent]
        if depth >= q.shape[0]:
            return
        for probability, token in zip(values[depth].tolist(), ids[depth].tolist()):
            if probability > 0:
                score = tree.scores[parent] + math.log(probability)
                child_path = path + (token,)
                heapq.heappush(frontier, (-score, child_path, parent, token))
    expand(0, ())
    while frontier and len(tree.tokens) < prefix_budget:
        negative_score, path, parent, token = heapq.heappop(frontier)
        node = len(tree.tokens)
        tree.tokens.append(token)
        tree.parents.append(parent)
        tree.depths.append(tree.depths[parent] + 1)
        tree.scores.append(-negative_score)
        tree.children.append({})
        tree.children[parent][token] = node
        expand(node, path)
    return tree


@dataclass
class TreeTraversal:
    tokens: list[int]
    path: list[int]
    teachers: list[int]
    matched: int


def traverse_target(tree, target, *, budget, eos_id=None, generator=None):
    """Sample p at reached nodes; return path nodes and their teacher row indices."""
    validate_distribution(target)
    if target.ndim != 2 or target.shape[0] != len(tree.tokens) or budget < 1:
        raise ValueError("invalid tree target shape or token budget")
    return _traverse(tree, lambda node: int(draw(target[node], generator)), budget, eos_id)


def traverse_greedy(tree, target_ids, *, budget, eos_id=None):
    if (target_ids.shape != (len(tree.tokens),) or budget < 1
            or target_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("integer greedy target per tree node required")
    targets = target_ids.tolist()
    if any(t < 0 for t in targets):
        raise ValueError("negative target id")
    return _traverse(tree, lambda node: targets[node], budget, eos_id)


def _traverse(tree, next_token, budget, eos_id):
    output, path, teachers = [tree.tokens[0]], [0], []
    node, matched = 0, 0
    while len(output) < budget and output[-1] != eos_id:
        token = next_token(node)
        teachers.append(node)
        output.append(token)
        child = tree.children[node].get(token)
        if child is None:
            break
        matched += 1
        node = child
        path.append(node)
    return TreeTraversal(output, path, teachers, matched)


def compact_tree_cache(cache, prefix_length, path):
    """Gather exactly the committed ancestral path, never sibling KV rows."""
    total = cache_length(cache)
    if cache is None or not 0 <= prefix_length <= total:
        raise ValueError("invalid tree cache prefix")
    if len(set(path)) != len(path) or any(n < 0 or prefix_length + n >= total for n in path):
        raise ValueError("invalid tree path indices")
    device = cache[0][0].device
    indices = torch.cat((torch.arange(prefix_length, device=device),
                         prefix_length + torch.tensor(path, dtype=torch.long, device=device)))
    if not indices.numel():
        return None
    if isinstance(cache, PackedCache):
        return PackedCache(cache.packed.index_select(4, indices).detach())
    return tuple((k.index_select(2, indices).detach(), v.index_select(2, indices).detach()) for k, v in cache)


@torch.no_grad()
def generate_tree(model, prompt, max_new_tokens, *, block_size=8, top_k=4, prefix_budget=16,
                  sampling=SamplingConfig(), eos_id=None, generator=None, learner=None, executor=None,
                  noise=UniformNoise()):
    _check(model, prompt, max_new_tokens, eos_id)
    forward = _inference_forward(model, executor)
    if block_size < 2 or top_k < 1 or prefix_budget < 1:
        raise ValueError("invalid tree configuration")
    if learner is not None and learner.model is not model:
        raise ValueError("learner and decoder must share a model")
    synchronize(model)
    start = time.perf_counter()
    initial_updates = learner.updates if learner is not None else 0
    initial_seconds = learner.update_seconds if learner is not None else 0.0
    initial_feedback = learner.feedback_blocks if learner is not None else 0
    initial_skips = learner.coverage_skips if learner is not None else 0
    if learner is not None:
        learner.clear_replay()
    cache = _prefill(forward, prompt) if max_new_tokens else None
    seed, output, matches = prompt[:, -1:], [], []
    forwards = rounds = accepted = proposed = 0
    fully_covered_rounds = 0
    while len(output) < max_new_tokens:
        rounds += 1
        remaining = max_new_tokens - len(output)
        if remaining == 1:
            logits, cache = forward(seed, cache=cache, return_cache=True)
            output.append(int(sample_logits(logits[0, -1], sampling, generator)))
            forwards += 1
            break
        b = min(block_size, remaining)
        noisy = noise.sample((1, b - 1), model.config.vocab_size, device=prompt.device, generator=generator)
        inputs = torch.cat((seed, noisy), dim=1)
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[:, 0] = False
        old_cache = cache
        collect = learner is not None and learner.needs_decoder_feedback
        capture = learner.capture_layer if collect else None
        result = forward(inputs, cache=cache, adapter_mask=mask, return_cache=True, capture_layer=capture)
        draft, temporary = result[:2]
        boundary = result[2] if capture is not None else None
        forwards += 1
        root = int(sample_logits(draft[0, 0], sampling, generator))
        if root == eos_id:
            output.append(root)
            break
        # Soft scores remain useful even when the TARGET is greedy. The chosen
        # tree law is not used as a denominator in any rejection calculation.
        q = tree_scores(draft[0, 1:], sampling)
        tree = build_tree(root, q, top_k=top_k, prefix_budget=prefix_budget)
        prefix_length = prompt.shape[1] + len(output)
        clean_cache = trim_cache(temporary, prefix_length)
        ids, positions, allowed = tree.layout(prefix_length, device=prompt.device)
        teacher, verified = forward(ids, positions=positions, allowed=allowed, cache=clean_cache, return_cache=True)
        forwards += 1
        if sampling.temperature == 0:
            traversal = traverse_greedy(tree, greedy_tokens(teacher[0]), budget=remaining, eos_id=eos_id)
        else:
            traversal = traverse_target(tree, probabilities(teacher[0], sampling), budget=remaining,
                                        eos_id=eos_id, generator=generator)
        output.extend(traversal.tokens)
        # Keep all new tokens except the last one, which is next round's seed.
        committed_path = traversal.path[:len(traversal.tokens) - 1]
        cache = compact_tree_cache(verified, prefix_length, committed_path)
        seed = prompt.new_tensor([[output[-1]]])
        accepted += traversal.matched
        proposed += len(tree.tokens) - 1
        matches.append(traversal.matched)
        fully_covered = traversal.matched == b - 1
        fully_covered_rounds += fully_covered
        done = len(output) >= max_new_tokens or output[-1] == eos_id
        if learner is not None:
            teacher_nodes = traversal.teachers[:b - 1]
            if collect:
                feedback = Feedback(inputs, old_cache, teacher[0, teacher_nodes], len(teacher_nodes), boundary,
                                    fully_covered)
                learner.observe(feedback, may_update=not done)
            else:
                learner._skip_decoder_feedback(len(teacher_nodes))
        if done:
            break
    if learner is not None:
        learner.clear_replay()
    synchronize(model)
    return Generation(output, time.perf_counter() - start, forwards, rounds, accepted, proposed,
                      learner.updates - initial_updates if learner is not None else 0,
                      learner.update_seconds - initial_seconds if learner is not None else 0.0, matches,
                      learner.feedback_blocks - initial_feedback if learner is not None else 0,
                      fully_covered_rounds, learner.coverage_skips - initial_skips if learner is not None else 0)
