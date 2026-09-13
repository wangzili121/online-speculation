import itertools

import pytest
import torch

from blockspec.sampling import (SamplingConfig, greedy_tokens, probabilities, residual,
                               sample_logits, verify_greedy, verify_linear)


@pytest.mark.parametrize("p,q", [
    ([.2, .5, .3], [.6, .1, .3]),
    ([1., 0., 0.], [0., .5, .5]),
    ([.2, .5, .3], [.2, .5, .3]),
    ([0., .4, .6], [1., 0., 0.]),
])
def test_exhaustive_one_step_output_law(p, q):
    p, q = torch.tensor(p, dtype=torch.float64), torch.tensor(q, dtype=torch.float64)
    r, mass = residual(p, q)
    # Enumerate every proposed token, both acceptance branches and correction.
    law = torch.zeros_like(p)
    for y in range(len(p)):
        a = min(1., float(p[y] / q[y])) if q[y] else 0.
        law[y] += q[y] * a
        law += q[y] * (1 - a) * r
    torch.testing.assert_close(law, p, atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(mass, .5 * (p - q).abs().sum())


def test_adaptive_proposals_preserve_two_token_joint_law():
    # The second proposal is allowed to depend on the first emitted token.
    p0 = torch.tensor([.3, .7], dtype=torch.float64)
    q0 = torch.tensor([.9, .1], dtype=torch.float64)
    p1 = torch.tensor([[.2, .8], [.6, .4]], dtype=torch.float64)
    q1 = torch.tensor([[.8, .2], [0., 1.]], dtype=torch.float64)
    def law(p, q):
        r, _ = residual(p, q)
        return torch.minimum(p, q) + (q - p).clamp_min(0).sum() * r
    actual = law(p0, q0)[:, None] * torch.stack([law(p1[h], q1[h]) for h in range(2)])
    torch.testing.assert_close(actual, p0[:, None] * p1, atol=1e-15, rtol=1e-15)


def test_rejection_and_all_accept_alignment():
    q = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    p = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    result = verify_linear(torch.tensor([0, 1]), q, p, acceptance_uniforms=torch.zeros(2))
    assert result.tokens == [0, 2]
    assert (result.accepted, result.rejected_at, result.supervised) == (1, 1, 2)
    result = verify_linear(torch.tensor([0, 1]), q, torch.cat((q, p[-1:])),
                           acceptance_uniforms=torch.full((2,), .999))
    assert result.tokens == [0, 1, 1]
    assert (result.accepted, result.rejected_at, result.supervised) == (2, None, 2)


def test_empty_proposal_is_a_bonus_draw():
    result = verify_linear(torch.empty(0, dtype=torch.long), torch.empty(0, 2),
                           torch.tensor([[0., 1.]]))
    assert result.tokens == [1] and result.supervised == 0


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.0, float("inf")])
def test_invalid_acceptance_randomness(bad):
    with pytest.raises(ValueError):
        verify_linear(torch.tensor([0]), torch.tensor([[1., 0.]]),
                      torch.tensor([[1., 0.], [1., 0.]]),
                      acceptance_uniforms=torch.tensor([bad]))


def test_temperature_topk_topp_and_greedy():
    logits = torch.tensor([3., 2., 1., 0.], dtype=torch.float64)
    assert torch.equal(probabilities(logits, SamplingConfig()), torch.tensor([1., 0., 0., 0.]))
    p = probabilities(logits, SamplingConfig(temperature=2, top_k=2))
    torch.testing.assert_close(p[:2], (logits[:2] / 2).softmax(-1))
    assert p[2:].sum() == 0
    p = probabilities(logits, SamplingConfig(temperature=1, top_p=.7))
    assert torch.equal(p > 0, torch.tensor([True, True, False, False]))


@pytest.mark.parametrize("config", [SamplingConfig(temperature=1), SamplingConfig(temperature=.4),
                                   SamplingConfig(temperature=2), SamplingConfig(1, 2, 1),
                                   SamplingConfig(1, 0, .7), SamplingConfig(.7, 2, .8),
                                   SamplingConfig(1, 1, 1)])
def test_filtered_output_law_by_executing_every_accept_and_correction_branch(monkeypatch, config):
    p = probabilities(torch.tensor([1.4, -.3, .5, .9], dtype=torch.float64), config)
    q = probabilities(torch.tensor([-.8, 1.5, .4, .3], dtype=torch.float64), config)
    correction, _ = residual(p, q)
    target = torch.stack((p, p))
    actual, accepted_mass = torch.zeros_like(p), p.new_zeros(())
    for y in range(len(q)):
        if q[y] == 0:
            continue
        acceptance = min(1., float(p[y] / q[y]))
        if acceptance > 0:
            # The extra token after acceptance leaves the first emitted token fixed.
            monkeypatch.setattr("blockspec.sampling.draw", lambda distribution, generator: distribution.argmax())
            result = verify_linear(torch.tensor([y]), q[None], target,
                                   acceptance_uniforms=torch.tensor([acceptance / 2], dtype=torch.float64))
            assert result.accepted == 1
            actual[result.tokens[0]] += q[y] * acceptance
            accepted_mass += q[y] * acceptance
        if acceptance < 1:
            for z in range(len(p)):
                if correction[z] == 0:
                    continue
                def force_correction(distribution, generator, z=z):
                    torch.testing.assert_close(distribution, correction, atol=1e-15, rtol=1e-15)
                    return torch.tensor(z)
                monkeypatch.setattr("blockspec.sampling.draw", force_correction)
                result = verify_linear(torch.tensor([y]), q[None], target,
                                       acceptance_uniforms=torch.tensor([(1 + acceptance) / 2], dtype=torch.float64))
                assert result.rejected_at == 0
                actual[result.tokens[0]] += q[y] * (1 - acceptance) * correction[z]
    torch.testing.assert_close(actual, p, atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(accepted_mass, 1 - .5 * (p - q).abs().sum(), atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_exhaustive_greedy_specialization_equals_probability_kernel(length):
    for candidate in itertools.product(range(2), repeat=length):
        for target in itertools.product(range(2), repeat=length + 1):
            proposed = torch.tensor(candidate, dtype=torch.long)
            ids = torch.tensor(target, dtype=torch.long)
            q = torch.nn.functional.one_hot(proposed, 2).float()
            p = torch.nn.functional.one_hot(ids, 2).float()
            reference = verify_linear(proposed, q, p, acceptance_uniforms=torch.zeros(length))
            assert verify_greedy(proposed, ids) == reference


def test_greedy_selection_is_argmax_without_probability_allocation(monkeypatch):
    logits = torch.tensor([[.3, 1., -.5], [2., 2., 1.]])
    def forbidden(*args, **kwargs):
        raise AssertionError("greedy path must not materialize a probability table")
    monkeypatch.setattr("blockspec.sampling.probabilities", forbidden)
    assert torch.equal(sample_logits(logits), torch.tensor([1, 0]))
    with pytest.raises(ValueError):
        greedy_tokens(torch.tensor([float("nan"), 0.]))


@pytest.mark.parametrize("noise", [0., .25, 2., 8.])
def test_clean_ar_kl_bounds_exact_prefix_survival(noise):
    # Exhaustive AR histories and overlap mass implement ALGORITHM section 6.10.
    rng = torch.Generator().manual_seed(912)
    base = torch.randn(4, 3, generator=rng, dtype=torch.float64)
    q = base.softmax(-1)
    histories = {(): (1., 1.)}
    tv_sum, pinsker_sum, exact_length, bounded_length = 0., 0., 1., 1.
    for i in range(4):
        next_histories, mean_tv, mean_kl = {}, 0., 0.
        for prefix, (ar_mass, overlap_mass) in histories.items():
            p = (base[i] + noise * torch.randn(3, generator=rng, dtype=torch.float64)).softmax(-1)
            mean_tv += ar_mass * float(.5 * (p - q[i]).abs().sum())
            mean_kl += ar_mass * float((p * (p.log() - q[i].log())).sum())
            for token in range(3):
                next_histories[prefix + (token,)] = (ar_mass * float(p[token]),
                                                     overlap_mass * float(torch.minimum(p[token], q[i, token])))
        survival = sum(mass[1] for mass in next_histories.values())
        assert sum(mass[0] for mass in next_histories.values()) == pytest.approx(1.)
        tv_sum += mean_tv
        pinsker_sum += (max(0., mean_kl) / 2) ** .5
        assert survival + 1e-12 >= max(0., 1 - tv_sum)
        assert tv_sum <= pinsker_sum + 1e-12
        exact_length += survival
        bounded_length += max(0., 1 - pinsker_sum)
        histories = next_histories
    assert exact_length + 1e-12 >= bounded_length
    if noise == 0:
        assert bounded_length == pytest.approx(5.)
