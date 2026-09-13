"""Request-local lookup, conditional scan sampling and online mixture gradients."""

import pytest
import torch

from blockspec_ablation.continuation import ContinuationMix, CopyFeedback, SuffixLookup, copy_mixture, copy_tv_gradient


@pytest.mark.parametrize("lookahead", [2, 4])
def test_suffix_index_matches_exhaustive_search_after_every_commit(lookahead):
    torch.manual_seed(730)
    tokens = torch.randint(0, 3, (100,)).tolist()
    memory = SuffixLookup(lookahead=lookahead)
    observed = []
    for token in tokens:
        memory.extend([token])
        observed.append(token)
        expected = ([], 0)
        for n in range(min(8, len(observed)), 1, -1):
            matches = [i for i in range(n, len(observed) - lookahead + 1) if observed[i-n:i] == observed[-n:]]
            if matches:
                expected = observed[matches[-1]:matches[-1] + 6], n
                break
        assert memory.find(6) == expected


def test_lookup_uses_observed_continuations_and_request_reset():
    memory = SuffixLookup([1, 2, 7, 8, 1, 2])
    assert memory.find(4) == ([7, 8, 1, 2], 2)
    mix = ContinuationMix(4, 3)
    mix.begin_request(memory.tokens)
    assert mix.lookup(4) == [7, 8, 1, 2]
    mix.weights.fill_(.4)
    mix.begin_request([9, 10, 11])
    assert mix.lookup(4) == []
    assert mix.weights.min() == .4
    mix.commit([9, 10])
    assert mix.lookup(4) == [11, 9, 10]


@pytest.mark.parametrize("length", [2, 4, 8])
@pytest.mark.parametrize("amount", [0., .3, 1.])
def test_parallel_copy_scan_matches_sequential_conditional_sampler(length, amount):
    torch.manual_seed(731 + length)
    q0 = torch.randn(length, 7, dtype=torch.float64).softmax(-1)
    weights = torch.full((length - 1,), amount, dtype=torch.float64)
    for _ in range(30):
        exponential = torch.empty_like(q0).exponential_()
        copied = torch.randint(0, 7, (length,))
        copied[torch.randint(1, length + 1, ()):] = -1
        tokens, q, feedback = copy_mixture(q0, weights, copied, exponential)
        expected_q = q0.clone()
        expected = [(q0[0] / exponential[0]).argmax().item()]
        active = expected[0] == copied[0]
        for i in range(1, length):
            active = active and copied[i] >= 0
            if active:
                expected_q[i] *= 1 - amount
                expected_q[i, copied[i]] += amount
            expected.append((expected_q[i] / exponential[i]).argmax().item())
            active = active and expected[-1] == copied[i]
        assert tokens.tolist() == expected
        torch.testing.assert_close(q, expected_q, atol=0, rtol=0)
        torch.testing.assert_close(q.sum(-1), torch.ones(length, dtype=q.dtype))
        torch.testing.assert_close(q[0], q0[0], atol=0, rtol=0)
        if amount == 0:
            torch.testing.assert_close(q, q0, atol=0, rtol=0)
        assert not any(feedback.active[i] and not feedback.active[i - 1] for i in range(1, length - 1))


def test_tv_gradient_with_new_support_matches_autograd_and_finite_difference():
    q0 = torch.tensor([[.6, .4, 0.], [.1, .6, .3]], dtype=torch.float64)
    p = torch.tensor([[.2, .1, .7], [.2, .5, .3]], dtype=torch.float64)
    weights = torch.tensor([.2, .3], dtype=torch.float64, requires_grad=True)
    copied = torch.tensor([2, 0])
    feedback = CopyFeedback(q0, copied, weights, torch.ones(2, dtype=torch.bool))
    result, q = copy_tv_gradient(feedback, p)
    expected = torch.autograd.grad(.5 * (q - p).abs().sum(), weights)[0]
    torch.testing.assert_close(result, expected, atol=1e-14, rtol=1e-14)
    for i in range(2):
        difference = torch.zeros_like(weights)
        difference[i] = 1e-6
        losses = []
        for sign in [-1, 1]:
            f = CopyFeedback(q0, copied, weights.detach() + sign * difference, feedback.active)
            losses.append(.5 * (copy_tv_gradient(f, p)[1] - p).abs().sum())
        torch.testing.assert_close((losses[1] - losses[0]) / 2e-6, result[i], atol=1e-10, rtol=1e-10)


def test_online_copy_learning_improves_useful_new_support_and_keeps_snapshot():
    mix = ContinuationMix(3, 2, interval=1, diagnostics=True, dtype=torch.float64)
    q0 = torch.tensor([[.6, .4, 0.]], dtype=torch.float64)
    p = torch.tensor([[.2, .1, .7]], dtype=torch.float64)
    initial = None
    for _ in range(16):
        feedback = CopyFeedback(q0, torch.tensor([2]), mix.weights[0, :1].clone(), torch.tensor([True]))
        _, q = copy_tv_gradient(feedback, p)
        if initial is None:
            initial = q.clone()
        mix.observe(feedback, p)
    assert .5 * (q - p).abs().sum() < .5 * (initial - p).abs().sum() - .4
    torch.testing.assert_close(initial, q0, atol=0, rtol=0)
    assert mix.weights[0, 1] == 0 and mix.weights[1:].sum() == 0
    assert mix.metrics()["depth_observations"] == [16., 0.]


def test_copy_state_restore_resumes_partial_window_with_owned_storage():
    mix = ContinuationMix(3, 2, interval=3, dtype=torch.float64)
    q0 = torch.tensor([[.6, .4, 0.]], dtype=torch.float64)
    p = torch.tensor([[.2, .1, .7]], dtype=torch.float64)
    feedback = CopyFeedback(q0, torch.tensor([2]), torch.zeros(1, dtype=q0.dtype), torch.tensor([True]))
    mix.observe(feedback, p)
    restored = ContinuationMix(3, 2, interval=3, dtype=torch.float64)
    address = restored.weights.data_ptr()
    restored.load_state_dict(mix.state_dict())
    assert address == restored.weights.data_ptr()
    for _ in range(2):
        mix.observe(feedback, p)
        restored.observe(feedback, p)
    assert mix.updates == restored.updates == 1
    torch.testing.assert_close(mix.weights, restored.weights, atol=0, rtol=0)
    corrupt = mix.state_dict()
    corrupt["tensors"]["weights"].fill_(2.)
    before = mix.weights.clone()
    with pytest.raises(ValueError):
        mix.load_state_dict(corrupt)
    torch.testing.assert_close(before, mix.weights, atol=0, rtol=0)


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_copy_graph_matches_eager_walk_and_owns_feedback(device):
    from blockspec.sampling import SamplingConfig, probabilities
    from blockspec_ablation.sampling_execution import SamplingExecutor

    sampling = SamplingConfig(1., 3, .95)
    engine = SamplingExecutor(7, 4, sampling, device=device, temperatures=(), continuation=True,
                              use_cuda_graph=device == "cuda")
    mix = ContinuationMix(4, 3, device=device)
    mix.begin_request([1, 2, 3, 4, 5, 6, 1, 2])
    mix.weights.fill_(.4)
    logits = torch.tensor([[0., 0., 0., 5., 0., 0., 0.], [0., 0., 0., 0., 5., 0., 0.],
                           [0., 0., 0., 0., 0., 5., 0.], [0., 0., 0., 0., 0., 0., 5.]], device=device)
    gen = lambda: torch.Generator(device=device).manual_seed(732)
    ids, q, feedback = engine.draft(logits, gen(), mix)
    expected_ids, expected_q, _ = mix.draft(probabilities(logits, sampling), gen())
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(q, expected_q, atol=0, rtol=0)
    assert feedback.active.all()
    saved = feedback.baseline.clone()
    mix.begin_request([5, 5, 0, 1, 5, 5])
    engine.draft(logits + 1, gen(), mix)
    torch.testing.assert_close(feedback.baseline, saved, atol=0, rtol=0)
    mix.begin_request([0, 1, 2])
    identity_ids, identity, empty = engine.draft(logits, gen(), mix)
    base_ids, base, _ = engine.draft(logits, gen())
    assert empty is None and torch.equal(identity_ids, base_ids)
    torch.testing.assert_close(identity, base, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget", [0, 1, 2, 19])
@torch.no_grad()
def test_copy_decoder_identity_adaptation_budget_and_clean_model(device, budget):
    from blockspec_ablation.decoding import generate_speculative
    from blockspec_ablation.model import Decoder, ModelConfig
    from blockspec.sampling import SamplingConfig
    from blockspec_ablation.sampling_execution import SamplingExecutor

    torch.manual_seed(733)
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).to(device).eval().requires_grad_(False)
    prompt = torch.tensor([[1, 2, 3, 4, 1, 2]], device=device)
    sampling = SamplingConfig(1., 3, .95)
    engine = SamplingExecutor(7, 4, sampling, device=device, temperatures=(), continuation=True,
                              use_cuda_graph=device == "cuda")
    options = dict(block_size=4, sampling=sampling, sampler_executor=engine)
    gen = lambda: torch.Generator(device=device).manual_seed(734)
    base = generate_speculative(model, prompt, budget, generator=gen(), **options)
    mix = ContinuationMix(4, 3, device=device, adaptive=False)
    fixed = generate_speculative(model, prompt, budget, generator=gen(), calibrator=mix, **options)
    assert fixed.tokens == base.tokens and fixed.decode_forwards == base.decode_forwards
    assert mix.memory.tokens == prompt[0].tolist() + fixed.tokens
    state = {k: v.clone() for k, v in model.state_dict().items()}
    adaptive = ContinuationMix(4, 3, device=device, interval=1)
    result = generate_speculative(model, prompt, budget, generator=gen(), calibrator=adaptive, **options)
    assert len(result.tokens) == budget
    assert adaptive.memory.tokens == prompt[0].tolist() + result.tokens
    torch.testing.assert_close(model.state_dict(), state, atol=0, rtol=0)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("amount", [0., .4, 1.])
def test_conditional_copy_and_residual_output_law_by_joint_enumeration(amount):
    from blockspec.sampling import residual

    torch.manual_seed(735)
    root = torch.randn(3, dtype=torch.float64).softmax(-1)
    first = torch.randn(3, 3, dtype=torch.float64).softmax(-1)
    second = torch.randn(3, 3, 3, dtype=torch.float64).softmax(-1)
    q0 = torch.tensor([[.2, .5, .3], [.6, .3, .1]], dtype=torch.float64)
    copy = [1, 2, 0]
    expected = root[:, None, None] * first[:, :, None] * second
    actual = torch.zeros_like(expected)
    for r in range(3):
        q1 = q0[0].clone()
        if r == copy[0]:
            q1 *= 1 - amount
            q1[copy[1]] += amount
        for a in range(3):
            if q1[a] == 0:
                continue
            q2 = q0[1].clone()
            if r == copy[0] and a == copy[1]:
                q2 *= 1 - amount
                q2[copy[2]] += amount
            accept1 = min(1., float(first[r, a] / q1[a]))
            correction1, _ = residual(first[r], q1)
            for b in range(3):
                if q2[b] == 0:
                    continue
                path_mass = root[r] * q1[a] * q2[b]
                # A first-position rejection leaves one ordinary AR token to fill.
                actual[r] += path_mass * (1 - accept1) * correction1[:, None] * second[r]
                accept2 = min(1., float(second[r, a, b] / q2[b]))
                correction2, _ = residual(second[r, a], q2)
                actual[r, a, b] += path_mass * accept1 * accept2
                actual[r, a] += path_mass * accept1 * (1 - accept2) * correction2
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)


def test_late_copy_learning_preserves_initial_proposal_positions():
    mix = ContinuationMix(5, 3, start_depth=3, interval=1)
    q0 = torch.tensor([[.6, .4, 0.]]).expand(4, -1)
    p = torch.tensor([[.2, .1, .7]]).expand(4, -1)
    for _ in range(10):
        feedback = CopyFeedback(q0, torch.full((4,), 2), mix.weights[0].clone(), torch.ones(4, dtype=torch.bool))
        mix.observe(feedback, p)
    assert (mix.weights[:, :2] == 0).all()
    assert (mix.steps[:, :2] == 0).all()
    assert (mix.weights[0, 2:] > .5).all()
    assert mix.metrics()["learned_coefficients"] == 6
    mix.begin_request([1, 2, 7, 8, 1, 2])
    assert mix.lookup(5) == [7, 8, 1, 2]
    mix.begin_request([1, 2, 7, 1, 2])
    assert mix.lookup(5) == []
    bad = mix.state_dict()
    bad["tensors"]["weights"][0, 0] = .1
    with pytest.raises(ValueError):
        mix.load_state_dict(bad)


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_continuation_eos_commits_exactly_the_returned_prefix(device):
    from blockspec_ablation.decoding import generate_speculative
    from blockspec_ablation.model import Decoder, ModelConfig
    from blockspec.sampling import SamplingConfig
    from blockspec_ablation.sampling_execution import SamplingExecutor

    torch.manual_seed(736)
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).to(device).eval().requires_grad_(False)
    prompt = torch.tensor([[1, 2, 3, 4, 1, 2]], device=device)
    sampling = SamplingConfig(1., 5, .95)
    sampler = SamplingExecutor(7, 4, sampling, device=device, temperatures=(), continuation=True,
                               use_cuda_graph=device == "cuda")
    options = dict(block_size=4, sampling=sampling, sampler_executor=sampler)
    gen = lambda: torch.Generator(device=device).manual_seed(737)
    ordinary = generate_speculative(model, prompt, 19, generator=gen(), **options)
    for eos in set(ordinary.tokens[:4]):
        mix = ContinuationMix(4, 5, device=device, adaptive=False, start_depth=3)
        result = generate_speculative(model, prompt, 19, generator=gen(), eos_id=eos, calibrator=mix, **options)
        assert result.tokens == ordinary.tokens[:ordinary.tokens.index(eos) + 1]
        assert mix.memory.tokens == prompt[0].tolist() + result.tokens
