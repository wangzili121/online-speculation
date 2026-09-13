"""Exact finite-state checks for the prefix-overlap value and gradient."""

import copy
import itertools

import pytest
import torch

from blockspec import DualViewConfig, DualViewDecoder, MaskedAttentionBranch, generate
from blockspec.parallel.sampling import ProposalSampler
from blockspec.sampling import SamplingConfig
from blockspec_ablation.prefix_objective import (
    PrefixConfig, PrefixFeedback, PrefixLearner, PrefixOnlineFeedback, prefix_overlap)


def teacher(prefix):
    return torch.tensor([.16, .31, .53], dtype=torch.float64).roll(sum(prefix) % 3)


def exact_extra_positions(q, horizon, eos_id):
    total = q.new_zeros(())
    for depth in range(1, horizon + 1):
        for prefix in itertools.product(range(3), repeat=depth):
            mass = q.new_ones(())
            for i, token in enumerate(prefix):
                mass = mass * torch.minimum(q[i, token], teacher(prefix[:i])[token])
                if token == eos_id:
                    mass = mass * 0
            total = total + mass
    return total


def enumerate_committed_count(q, remaining, eos_id, prefix=()):
    if remaining == 1 or len(prefix) == len(q):
        return q.new_ones(())
    overlap = torch.minimum(q[len(prefix)], teacher(prefix))
    # Each rejected proposal emits one correction; its identity leaves the round length unchanged.
    total = 1 - overlap.sum()
    for token in range(3):
        following = (0 if token == eos_id else
                     enumerate_committed_count(q, remaining - 1, eos_id, prefix + (token,)))
        total = total + overlap[token] * (1 + following)
    return total


@pytest.mark.parametrize("remaining", [2, 3, 4])
@pytest.mark.parametrize("eos_id", [None, 0])
def test_survival_sum_counts_actual_accept_reject_eos_and_budget_branches(remaining, eos_id):
    torch.manual_seed(933)
    q = torch.randn(3, 3, dtype=torch.float64).softmax(-1)
    expected = 1 + exact_extra_positions(q, min(3, remaining - 1), eos_id)
    actual = enumerate_committed_count(q, remaining, eos_id)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize("remaining", [2, 3, 4])
@pytest.mark.parametrize("eos_id", [None, 0])
@pytest.mark.parametrize("displacement", [0., .2])
def test_enumerated_estimator_matches_exact_expected_length_and_gradient(remaining, eos_id, displacement):
    torch.manual_seed(921)
    old_logits = torch.randn(3, 3, dtype=torch.float64)
    current = (old_logits + displacement * torch.randn_like(old_logits)).requires_grad_()
    q0, q = old_logits.softmax(-1), current.softmax(-1)
    horizon = min(3, remaining - 1)
    average = current.new_zeros(())
    for prefix in itertools.product(range(3), repeat=horizon):
        ids = torch.tensor(prefix)
        teacher_rows = torch.stack([teacher(prefix[:i]) for i in range(horizon)])
        taken = q0[:horizon].gather(1, ids[:, None]).squeeze(1)
        value = prefix_overlap(q[:horizon], teacher_rows, ids, taken, eos_id=eos_id)
        average = average + taken.prod() * value
    exact = exact_extra_positions(q, horizon, eos_id)
    assert 0 <= float(exact.detach()) <= horizon
    torch.testing.assert_close(average, exact, rtol=1e-12, atol=1e-14)
    estimated_grad, = torch.autograd.grad(average, current, retain_graph=True)
    exact_grad, = torch.autograd.grad(exact, current)
    torch.testing.assert_close(estimated_grad, exact_grad, rtol=1e-11, atol=1e-14)
    assert torch.equal(estimated_grad[horizon:], torch.zeros_like(estimated_grad[horizon:]))


def test_teacher_and_snapshot_are_constants_and_gradient_matches_finite_difference():
    current = torch.tensor([[.2, .4, -.1], [.3, -.5, .1]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[.2, .5, .3], [.4, .1, .5]], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([1, 2])
    taken = torch.tensor([.4, .3], dtype=torch.float64, requires_grad=True)
    value = prefix_overlap(current.softmax(-1), p, ids, taken, eos_id=0)
    gradient, = torch.autograd.grad(value, current)
    assert p.grad is None and taken.grad is None
    epsilon = 1e-6
    for row, col in itertools.product(range(2), range(3)):
        plus, minus = current.detach().clone(), current.detach().clone()
        plus[row, col] += epsilon
        minus[row, col] -= epsilon
        delta = (prefix_overlap(plus.softmax(-1), p, ids, taken, eos_id=0)
                 - prefix_overlap(minus.softmax(-1), p, ids, taken, eos_id=0)) / (2 * epsilon)
        assert float(delta.detach()) == pytest.approx(float(gradient[row, col]), abs=1e-9)


def tiny():
    torch.manual_seed(922)
    return DualViewDecoder(DualViewConfig(
        vocab_size=13, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval().requires_grad_(False)


@torch.no_grad()
def feedback(net, learner, *, rows=3):
    cache = net(torch.tensor([[3, 5, 8]])).cache
    inputs = torch.tensor([[6, 1, 1, 1]])
    draft = net(inputs, view="draft", cache=cache, capture_layer=learner.capture_layer)
    q = draft.logits[0, :-1].float().softmax(-1)
    ids = torch.tensor([2, 4, 7])
    target = net(torch.cat((inputs[:, :1], ids[None]), 1), cache=cache).logits[0, :rows]
    return PrefixFeedback(inputs, cache, target, rows, draft.boundary, False,
                          ids[:rows], q[:rows].gather(1, ids[:rows, None]).squeeze(1), learner.version)


def test_suffix_gradient_matches_full_block_and_state_preserves_the_objective():
    net = tiny()
    complete = copy.deepcopy(net)
    config = PrefixConfig(stride=1, learning_rate=.0001)
    learner = PrefixLearner(net, config)
    item = feedback(net, learner)
    learner.observe(item, may_update=False)
    actual_loss = learner.backward()
    for name, parameter in complete.named_parameters():
        parameter.requires_grad_(name in learner.master)
    q = complete(item.inputs, view="draft", cache=item.cache).logits[0, :item.valid].float().softmax(-1)
    p = item.teacher_logits.float().softmax(-1)
    expected_loss = -prefix_overlap(q, p, item.candidates, item.proposal_taken)
    expected_loss.backward()
    assert actual_loss == pytest.approx(float(expected_loss.detach()), rel=1e-6)
    for name, parameter in complete.named_parameters():
        if name in learner.master:
            torch.testing.assert_close(learner.master[name].grad, parameter.grad, rtol=0, atol=0)
    learner.update()
    with pytest.raises(ValueError, match="fresh"):
        learner.backward()
    learner.clear_replay()
    state = learner.state_dict()
    restored = PrefixLearner(tiny(), config)
    restored.load_state_dict(state)
    assert all(torch.equal(value, restored.master[name]) for name, value in learner.master.items())
    altered = copy.deepcopy(state)
    altered["config"]["temperature"] = .5
    with pytest.raises(ValueError, match="policy"):
        restored.load_state_dict(altered)


@pytest.mark.parametrize("budget", [2, 3, 7, 19])
def test_full_canvas_feedback_respects_remaining_budget_and_freezes_ar(budget):
    net = tiny()
    learner = PrefixLearner(net, PrefixConfig(stride=1, learning_rate=.0001))
    owner = PrefixOnlineFeedback(learner=learner, output_budget=budget)
    prompt = torch.tensor([[3, 5, 8]])
    with torch.no_grad():
        ar_before = net(prompt)
    rows = []
    observe = learner.observe

    def record(item, **kwargs):
        remaining = budget - (owner.emitted - owner.last_commit)
        assert item.valid == min(item.inputs.shape[1] - 1, remaining - 1)
        rows.append(item.valid)
        return observe(item, **kwargs)

    learner.observe = record
    sampling = SamplingConfig(1.)
    result = generate(MaskedAttentionBranch(net), prompt, budget, sampling=sampling, feedback=owner,
                      sampler=ProposalSampler(sampling), generator=torch.Generator().manual_seed(923), audit_cache=True)
    assert len(result.tokens) == owner.emitted == budget and not learner.replay
    with torch.no_grad():
        ar_after = net(prompt)
    assert torch.equal(ar_before.logits, ar_after.logits)
    assert all(torch.equal(a, b) for old, new in zip(ar_before.cache, ar_after.cache) for a, b in zip(old, new))
    assert all(parameter.grad is None for parameter in net.parameters())
    if budget > 2:
        assert rows


def test_rejects_zero_snapshot_probability_and_stale_feedback():
    q, p = torch.tensor([[.3, .7]]), torch.tensor([[.5, .5]])
    with pytest.raises(ValueError, match="positive"):
        prefix_overlap(q, p, torch.tensor([0]), torch.tensor([0.]))
    learner = PrefixLearner(tiny(), PrefixConfig())
    item = feedback(learner.model, learner)
    item.version += 1
    with pytest.raises(ValueError, match="version"):
        learner.observe(item)


def test_prefix_configuration_names_its_actual_loss_and_checks_shared_window_constraints():
    assert PrefixConfig().loss == "prefix_overlap"
    for options in ({"loss": "tv"}, {"temperature": 0.}, {"stride": 1, "replay_blocks": 2}):
        with pytest.raises(ValueError):
            PrefixConfig(**options)
