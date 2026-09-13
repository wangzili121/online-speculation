"""Mainline packaging, tensor execution, and mathematical regression checks."""

import ast
import importlib
from pathlib import Path
import sys

import pytest
import torch

from blockspec import DualViewConfig, DualViewDecoder, MaskedAttentionBranch, generate, generate_ar
from blockspec.losses import divergence
from blockspec.parallel.feedback import OnlineFeedback
from blockspec.parallel.online import SuffixConfig, SuffixLearner
from blockspec.parallel.sampling import ProposalSampler
from blockspec.sampling import SamplingConfig, probabilities
from blockspec.sampling_execution import SamplingExecutor


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA execution check"))]


def tiny(device="cpu"):
    torch.manual_seed(904)
    return DualViewDecoder(DualViewConfig(
        vocab_size=17, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8)).to(device).eval().requires_grad_(False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", [SamplingConfig(1), SamplingConfig(.7, 5, .8)])
@pytest.mark.parametrize("budget", [0, 1, 2, 13])
@torch.no_grad()
def test_mainline_tensor_and_graph_preserve_ar_and_speculative_trajectories(device, config, budget):
    model = tiny(device)
    branch = MaskedAttentionBranch(model)
    prompt = torch.tensor([[3, 5, 8]], device=device)
    plain = SamplingExecutor(17, 4, config, device=device)
    graph = SamplingExecutor(17, 4, config, device=device, use_cuda_graph=device == "cuda")
    gen = lambda: torch.Generator(device=device).manual_seed(905)
    for method in (generate_ar, generate):
        options = {} if method is generate_ar else {"block_size": 4, "audit_cache": True}
        outputs = [method(branch, prompt, budget, sampling=config, generator=gen(),
                          sampler=ProposalSampler(config, executor=engine), **options)
                   for engine in (plain, graph)]
        assert outputs[0].tokens == outputs[1].tokens
        assert outputs[0].accepted_per_round == outputs[1].accepted_per_round
        assert outputs[0].decode_forwards == outputs[1].decode_forwards


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_tensor_probability_snapshots_survive_later_calls(device):
    config = SamplingConfig(1., 5, .8)
    executor = SamplingExecutor(17, 4, config, device=device)
    logits = torch.randn(3, 17, device=device)
    tokens, q, payload = executor.draft(logits)
    assert payload is None
    torch.testing.assert_close(q, probabilities(logits, config), rtol=0, atol=0)
    saved_tokens, saved_q = tokens.clone(), q.clone()
    executor.draft(logits + torch.randn_like(logits))
    assert torch.equal(tokens, saved_tokens) and torch.equal(q, saved_q)
    teacher = torch.randn(4, 17, device=device)
    _, p = executor.verify(tokens, q, teacher)
    saved_p = p.clone()
    executor.verify(tokens, q, teacher + torch.randn_like(teacher))
    assert torch.equal(p, saved_p)


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_online_stream_keeps_immutable_proposals_and_frozen_ar(device):
    model = tiny(device)
    baseline = {n: p.detach().clone() for n, p in model.named_parameters()}
    learner = SuffixLearner(model, SuffixConfig(stride=2, learning_rate=.001))
    config = SamplingConfig(1., 5, .8)
    executor = SamplingExecutor(17, 4, config, device=device)
    result = generate(MaskedAttentionBranch(model), torch.tensor([[3, 5, 8]], device=device), 25,
                      block_size=4, sampling=config, sampler=ProposalSampler(config, executor=executor),
                      feedback=OnlineFeedback(learner=learner),
                      generator=torch.Generator(device=device).manual_seed(906), audit_cache=True)
    assert result.updates > 0 and result.update_seconds > 0 and not learner.replay
    assert any(not torch.equal(baseline[n], p) for n, p in learner.execution.items())
    assert all(torch.equal(baseline[n], p) for n, p in model.named_parameters() if n not in learner.execution)
    assert all(p.grad is None for p in model.parameters())


def test_full_distribution_gradient_and_tv_overlap_identity():
    torch.manual_seed(907)
    logits = torch.randn(11, 7, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(logits)
    q, p = logits.softmax(-1), teacher.softmax(-1)
    loss = divergence(logits, teacher, "forward_kl")
    gradient, = torch.autograd.grad(loss.sum(), logits)
    torch.testing.assert_close(gradient, q - p, rtol=1e-12, atol=1e-14)
    tv = divergence(logits, teacher, "tv")
    torch.testing.assert_close(1 - tv, torch.minimum(q, p).sum(-1), rtol=0, atol=1e-14)
    assert (tv.square() <= .5 * loss + 1e-14).all()


def test_tv_logit_gradient_matches_analytic_subgradient_and_finite_difference():
    torch.manual_seed(932)
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(logits, requires_grad=True)
    q, p = logits.softmax(-1), teacher.detach().softmax(-1)
    direction = torch.sign(q - p)
    expected = .5 * q * (direction - (q * direction).sum(-1, keepdim=True))
    actual, teacher_gradient = torch.autograd.grad(divergence(logits, teacher, "tv").sum(),
                                                 (logits, teacher), allow_unused=True)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    assert teacher_gradient is None
    assert (actual[q < p] <= 0).all() and (actual[q > p] >= 0).all()
    torch.testing.assert_close(actual.sum(-1), torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    assert torch.autograd.gradcheck(lambda u: divergence(u, teacher, "tv"), (logits,))
    identical = logits.detach().clone().requires_grad_()
    zero, = torch.autograd.grad(divergence(identical, identical.detach(), "tv").sum(), identical)
    assert torch.equal(zero, torch.zeros_like(zero))


def test_fresh_online_state_is_continuation_of_the_loaded_draft():
    model = tiny()
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    learner = SuffixLearner(model)
    assert learner.updates == learner.rounds == 0
    assert all(torch.equal(p, weights[n]) for n, p in model.named_parameters())
    assert all(torch.equal(master, weights[n].float()) for n, master in learner.master.items())


def test_parallel_information_loss_decomposes_into_variation_and_fit():
    torch.manual_seed(908)
    teachers = torch.randn(5, 7, dtype=torch.float64).softmax(-1)
    student = torch.randn(7, dtype=torch.float64).softmax(-1)
    average = teachers.mean(0)
    kl = lambda p, q: (p * (p.log() - q.log())).sum(-1)
    actual = kl(teachers, student).mean()
    expected = kl(teachers, average).mean() + kl(average, student)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)


def test_mainline_source_and_installed_commands_are_independent_of_ablation():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "src/blockspec").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("ablation" not in item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert "ablation" not in (node.module or "")
    for name in ("evaluate", "continue_training", "fit"):
        module = importlib.import_module("blockspec.commands." + name)
        assert callable(module.main)
    assert not any(name.startswith("blockspec_ablation") for name in sys.modules)
