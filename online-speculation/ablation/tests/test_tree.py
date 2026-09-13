import itertools
import math

import pytest
import torch

from blockspec_ablation.decoding import generate_ar
from blockspec_ablation.model import Decoder, ModelConfig, cache_length
from blockspec_ablation.online import OnlineConfig, OnlineLearner
from blockspec.sampling import SamplingConfig
from blockspec_ablation.tree import (build_tree, compact_tree_cache, generate_tree, traverse_greedy,
                            traverse_target, tree_scores)


def tiny():
    torch.manual_seed(8)
    return Decoder(ModelConfig(vocab_size=5, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2,
                                num_key_value_heads=1, head_dim=8, adapter_rank=2)).double()


@pytest.mark.parametrize("budget", [1, 2, 4, 8, 15])
def test_heap_selects_globally_best_prefixes(budget):
    q = torch.tensor([[.6, .3, .1], [.1, .4, .5], [.2, .6, .2]], dtype=torch.float64)
    tree = build_tree(0, q, top_k=2, prefix_budget=budget)
    candidates = []
    choices = q.topk(2, dim=-1).indices.tolist()
    for depth in range(1, 4):
        for path in itertools.product(*choices[:depth]):
            score = sum(math.log(float(q[i, t])) for i, t in enumerate(path))
            candidates.append((-score, path))
    expected = [path for score, path in sorted(candidates)[:budget - 1]]
    got = [tuple(tree.tokens[i] for i in tree.path(node)[1:]) for node in range(1, len(tree.tokens))]
    assert got == expected
    assert all(tree.parents[i] < i for i in range(1, len(tree.tokens)))


def test_tree_attention_and_cache_equal_every_individual_path():
    model = tiny()
    prefix = torch.tensor([[0, 1, 3]])
    _, cache = model(prefix, return_cache=True)
    tree = build_tree(2, torch.tensor([[.5, .3, .1, .05, .05], [.2, .1, .5, .1, .1]]),
                      top_k=2, prefix_budget=7)
    ids, positions, allowed = tree.layout(3, device="cpu")
    logits, verified = model(ids, positions=positions, allowed=allowed, cache=cache, return_cache=True)
    for node in range(len(tree.tokens)):
        path = tree.path(node)
        tokens = torch.tensor([[tree.tokens[i] for i in path]])
        direct, direct_cache = model(tokens, cache=cache, return_cache=True)
        torch.testing.assert_close(logits[:, node], direct[:, -1], atol=2e-15, rtol=2e-14)
        compact = compact_tree_cache(verified, 3, path)
        assert cache_length(compact) == 3 + len(path)
        for layer_a, layer_b in zip(compact, direct_cache):
            for a, b in zip(layer_a, layer_b):
                torch.testing.assert_close(a, b, atol=2e-15, rtol=2e-14)


def test_target_traversal_samples_missing_child_and_stops():
    tree = build_tree(0, torch.tensor([[0., 1., 0.], [0., 1., 0.]]), top_k=1, prefix_budget=3)
    # Root -> token 1 is present, token 2 after it is absent and must still emit.
    target = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    result = traverse_target(tree, target, budget=10)
    assert result.tokens == [0, 1, 2]
    assert result.path == [0, 1] and result.teachers == [0, 1] and result.matched == 1


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 17])
@pytest.mark.parametrize("prefix_budget", [1, 4, 12])
def test_tree_greedy_matches_ar_for_tail_and_branch_shapes(budget, prefix_budget):
    model = tiny()
    prompt = torch.tensor([[0, 1, 3]])
    ar = generate_ar(model, prompt, budget)
    tree = generate_tree(model, prompt, budget, block_size=4, top_k=3, prefix_budget=prefix_budget)
    assert tree.tokens == ar.tokens


def test_tree_eos_and_true_online_continuation():
    model = tiny()
    prompt = torch.tensor([[0, 1, 3]])
    ar = generate_ar(model, prompt, 20)
    eos = ar.tokens[3]
    assert generate_tree(model, prompt, 20, eos_id=eos).tokens == generate_ar(model, prompt, 20, eos_id=eos).tokens
    learner = OnlineLearner(model, OnlineConfig(stride=1, replay_blocks=2, loss="forward_kl"))
    generated = generate_tree(model, prompt, 20, block_size=4, prefix_budget=12, learner=learner)
    assert generated.tokens == ar.tokens
    assert generated.updates > 0 and not learner.replay


def test_tree_target_sampling_probabilities_do_not_depend_on_tree_scores(monkeypatch):
    # Exhaustively integrate the first target draw over the three vocabulary items.
    tree = build_tree(0, torch.tensor([[.9, .1, 0.]]), top_k=1, prefix_budget=2)
    p = torch.tensor([[.2, .3, .5], [.6, .1, .3]], dtype=torch.float64)
    law = torch.zeros(3, dtype=torch.float64)
    for token in range(3):
        def forced_draw(distribution, generator, t=token):
            torch.testing.assert_close(distribution, p[0], atol=0, rtol=0)
            return torch.tensor(t)
        monkeypatch.setattr("blockspec_ablation.tree.draw", forced_draw)
        result = traverse_target(tree, p, budget=2)
        law[result.tokens[1]] += p[0, token]
    torch.testing.assert_close(law, p[0], atol=0, rtol=0)


@pytest.mark.parametrize("budget", [1, 2, 3, 5])
@pytest.mark.parametrize("eos_id", [None, 0, 2])
def test_expected_output_length_equals_reachable_node_mass(budget, eos_id, monkeypatch):
    tree = build_tree(2, torch.tensor([[.6, .3, .1], [.2, .3, .5]], dtype=torch.float64),
                      top_k=2, prefix_budget=6)
    target = torch.tensor([[.2, .5, .3], [.6, .1, .3], [.1, .4, .5],
                           [.4, .4, .2], [.3, .2, .5], [.7, .2, .1]], dtype=torch.float64)
    terminals = []

    def enumerate_draws(node, output, mass):
        if len(output) == budget or output[-1] == eos_id:
            terminals.append((output, mass))
            return
        for token in range(target.shape[-1]):
            path, probability = output + [token], mass * target[node, token]
            child = tree.children[node].get(token)
            if child is None:
                terminals.append((path, probability))
            else:
                enumerate_draws(child, path, probability)

    enumerate_draws(0, [tree.tokens[0]], torch.tensor(1.0, dtype=torch.float64))
    measured = torch.tensor(0.0, dtype=torch.float64)
    total_mass = torch.tensor(0.0, dtype=torch.float64)
    for output, mass in terminals:
        draws = iter(output[1:])
        monkeypatch.setattr("blockspec_ablation.tree.draw",
                            lambda distribution, generator, draws=draws: torch.tensor(next(draws)))
        actual = traverse_target(tree, target, budget=budget, eos_id=eos_id)
        assert actual.tokens == output
        measured += len(actual.tokens) * mass
        total_mass += mass
    reach = torch.ones(len(tree.tokens), dtype=torch.float64)
    for node in range(1, len(tree.tokens)):
        parent = tree.parents[node]
        reach[node] = reach[parent] * target[parent, tree.tokens[node]] * (tree.tokens[parent] != eos_id)
    expected = 1 + sum(reach[node] for node in range(len(tree.tokens))
                       if tree.depths[node] <= budget - 2 and tree.tokens[node] != eos_id)
    torch.testing.assert_close(total_mass, torch.ones_like(total_mass), atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(measured, torch.as_tensor(expected, dtype=torch.float64), atol=1e-15, rtol=1e-15)


def test_tree_greedy_specialization_equals_one_hot_traversal():
    tree = build_tree(0, torch.tensor([[.6, .4], [.3, .7]]), top_k=2, prefix_budget=6)
    for targets in itertools.product(range(2), repeat=len(tree.tokens)):
        ids = torch.tensor(targets)
        p = torch.nn.functional.one_hot(ids, 2).float()
        assert traverse_greedy(tree, ids, budget=4) == traverse_target(tree, p, budget=4)


@pytest.mark.parametrize("temperature", [0., .5, 1., 2.])
def test_tree_scores_respect_positive_temperature_without_target_filters(temperature):
    logits = torch.tensor([[2., 1., -.5], [.2, 2.1, -.3]], dtype=torch.float64)
    scores = tree_scores(logits, SamplingConfig(temperature=temperature, top_k=1, top_p=.1))
    torch.testing.assert_close(scores, (logits / (temperature or 1.)).softmax(-1), atol=0, rtol=0)
    assert (scores > 0).all()  # Target truncation must not collapse the candidate tree.


def test_tree_traversal_full_two_token_law_including_exit_and_resume(monkeypatch):
    # Root=0; only the child labelled 1 is retained. Emitting 0 or 2 must exit
    # the tree and resume ordinary target sampling, without renormalizing p.
    tree = build_tree(0, torch.tensor([[.1, .8, .1]]), top_k=1, prefix_budget=2)
    first = torch.tensor([.2, .5, .3], dtype=torch.float64)
    second = torch.tensor([[.6, .1, .3], [.2, .7, .1], [.1, .4, .5]], dtype=torch.float64)
    target = torch.stack((first, second[1]))
    joint = torch.zeros(3, 3, dtype=torch.float64)
    for a, b in itertools.product(range(3), repeat=2):
        forced, encountered = iter((a, b)), []
        def chosen(p, generator, forced=forced, encountered=encountered):
            token = next(forced)
            encountered.append(p[token])
            return torch.tensor(token)
        monkeypatch.setattr("blockspec_ablation.tree.draw", chosen)
        result = traverse_target(tree, target, budget=3)
        assert result.tokens[:2] == [0, a]
        probability = torch.stack(encountered).prod()
        if len(result.tokens) == 2:
            assert a != 1
            emitted = result.tokens[1:] + [b]
            probability *= second[a, b]
        else:
            assert a == 1
            emitted = result.tokens[1:]
        joint[tuple(emitted)] += probability
    torch.testing.assert_close(joint, first[:, None] * second, atol=1e-15, rtol=1e-15)
