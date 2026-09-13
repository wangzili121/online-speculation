from functools import lru_cache
import math

import pytest
import torch

from blockspec.parallel import DualViewConfig, DualViewDecoder, MaskedAttentionBranch, generate, generate_ar
from blockspec.parallel.sampling import ProposalSampler
from blockspec.sampling import SamplingConfig, probabilities
from blockspec.state import trim_cache
from blockspec_ablation.virtual_work import (calibration_starts, estimate_trace, expected_work,
                                             inverse_acceptance, score_trace, trace_layout)


def forward_joint(target, proposal, cap, block, eos):
    """Enumerate proposal tokens, forward accept/reject, residual draw, and bonus.

    Each completed output maps to [probability, weighted draft count, weighted
    tail count]. Unused independent proposal suffixes integrate to one.
    """
    def add(out, source, weight, draft=0, tail=0):
        for sequence, (prob, ds, ts) in source.items():
            row = out.setdefault(sequence, [0., 0., 0.])
            row[0] += weight * prob
            row[1] += weight * (ds + draft * prob)
            row[2] += weight * (ts + tail * prob)

    def stopped(prefix):
        return len(prefix) == cap or (prefix and prefix[-1] == eos)

    @lru_cache(None)
    def rounds(prefix):
        if stopped(prefix):
            return {prefix: [1., 0., 0.]}
        out = {}
        if not prefix or cap - len(prefix) == 1:
            for token, mass in enumerate(target(prefix)):
                if mass:
                    add(out, rounds(prefix + (token,)), mass, tail=int(bool(prefix)))
            return out
        count = min(block, cap - len(prefix) + 1) - 1

        def walk(current, offset):
            if stopped(current):
                return {current: [1., 0., 0.]}
            result = {}
            p = target(current)
            if offset == count:
                for token, mass in enumerate(p):
                    if mass:
                        add(result, rounds(current + (token,)), mass)
                return result
            q = proposal(prefix, offset)
            positive = [max(0., a - b) for a, b in zip(p, q)]
            residual_mass = sum(positive)
            for token, mass in enumerate(q):
                if not mass:
                    continue
                alpha = min(1., p[token] / mass)
                if alpha:
                    add(result, walk(current + (token,), offset + 1), mass * alpha)
                if alpha < 1:
                    for correction, value in enumerate(positive):
                        if value:
                            add(result, rounds(current + (correction,)),
                                mass * (1 - alpha) * value / residual_mass)
            return result

        add(out, walk(prefix, 0), 1., draft=1)
        return out

    return rounds(())


def laws(kind, theta=0.):
    def target(prefix):
        if kind == "equal":
            return [.2, .3, .5]
        if kind == "disjoint":
            return [1., 0., 0.] if len(prefix) % 2 == 0 else [0., 0., 1.]
        weights = [1. + ((sum(prefix) + 2 * len(prefix) + i * 3) % 7) for i in range(3)]
        return [v / sum(weights) for v in weights]

    def proposal(prefix, offset):
        if kind == "equal":
            return [.2, .3, .5]
        if kind == "disjoint":
            return [0., 1., 0.]
        bias = torch.tensor([.17 * ((sum(prefix) + i * 2 + offset) % 5) for i in range(3)],
                            dtype=torch.float64)
        logits = bias + theta * torch.tensor([-.5, .3, .9], dtype=torch.float64)
        return logits.softmax(-1)

    return target, proposal


def trace_acceptance(sequence, target, proposal, block, theta=None):
    rows = []
    for j in range(1, len(sequence)):
        values = []
        for offset in range(block - 1):
            index = j + offset
            if index < len(sequence) - 1:
                x = sequence[index]
                q = torch.as_tensor(proposal(sequence[:j], offset)[x], dtype=torch.float64)
                values.append((q / target(sequence[:index])[x]).clamp_max(1))
            else:
                values.append(torch.tensor(1., dtype=torch.float64))
        rows.append(torch.stack(values))
    return torch.stack(rows) if rows else torch.empty(0, block - 1, dtype=torch.float64)


@pytest.mark.parametrize("kind", ["dense", "equal", "disjoint"])
@pytest.mark.parametrize("cap", range(6))
@pytest.mark.parametrize("block", [2, 3, 4])
@pytest.mark.parametrize("eos", [None, 2])
def test_inverse_dp_matches_every_forward_output_and_work(kind, cap, block, eos):
    target, proposal = laws(kind)
    joint = forward_joint(target, proposal, cap, block, eos)
    assert sum(row[0] for row in joint.values()) == pytest.approx(1., abs=1e-12)
    for sequence, (mass, drafts, tails) in joint.items():
        p = math.prod(target(sequence[:i])[x] for i, x in enumerate(sequence))
        assert mass == pytest.approx(p, abs=1e-12)
        gamma = trace_acceptance(sequence, target, proposal, block)
        work = expected_work(gamma, output_tokens=len(sequence), output_budget=cap, block_size=block)
        assert float(work.draft_forwards) == pytest.approx(drafts / mass, abs=2e-12)
        assert float(work.tail_ar_forwards) == pytest.approx(tails / mass, abs=2e-12)
        assert float(work.cost()) == pytest.approx((2 * drafts + tails) / mass, abs=2e-12)


def test_full_request_gradient_matches_forward_enumeration_difference():
    theta = torch.tensor(.237, dtype=torch.float64, requires_grad=True)
    target, proposal = laws("dense", theta)
    reference = forward_joint(target, laws("dense", float(theta.detach()))[1], 5, 3, 2)
    objective = theta * 0
    for sequence, (mass, _, _) in reference.items():
        gamma = trace_acceptance(sequence, target, proposal, 3)
        objective = objective + mass * expected_work(gamma, output_tokens=len(sequence), output_budget=5,
                                                     block_size=3).decode_forwards
    objective.backward()
    def independent_cost(value):
        p, q = laws("dense", value)
        return sum(2 * ds + ts for _, ds, ts in forward_joint(p, q, 5, 3, 2).values())
    step = 1e-5
    finite_difference = (independent_cost(.237 + step) - independent_cost(.237 - step)) / (2 * step)
    assert theta.grad.item() == pytest.approx(finite_difference, rel=1e-7, abs=1e-8)


def test_early_eos_uses_requested_budget_and_includes_terminal_round():
    gamma = torch.ones(1, 3, dtype=torch.float64)
    early = expected_work(gamma, output_tokens=2, output_budget=8, block_size=4)
    capped = expected_work(gamma, output_tokens=2, output_budget=2, block_size=4)
    assert early.decode_forwards == 2
    assert capped.decode_forwards == 1


@pytest.mark.parametrize("bad", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        expected_work(torch.tensor([[bad]]), output_tokens=2, output_budget=2, block_size=2)


def test_inverse_support_and_direction():
    assert inverse_acceptance(torch.tensor([math.log(.1), -torch.inf]),
                              torch.tensor([math.log(.2), math.log(.2)])).tolist() == pytest.approx([.5, 0.])
    with pytest.raises(ValueError):
        inverse_acceptance(torch.tensor([0.]), torch.tensor([-torch.inf]))


def small_model(backend="eager"):
    torch.manual_seed(613)
    config = DualViewConfig(vocab_size=13, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8, block_size=4)
    return DualViewDecoder(config).double().eval().requires_grad_(False).set_backend(backend)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("sampling", [SamplingConfig(1.), SamplingConfig(.7, top_k=4, top_p=.9)])
@pytest.mark.parametrize("cap", [5, 12])
def test_masked_trace_matches_individual_drafts_and_causal_teacher(backend, sampling, cap):
    model = small_model(backend)
    prompt = torch.tensor([[3, 4, 5]])
    # Sample a supported trajectory, then regard its last token as early EOS.
    outputs = generate_ar(MaskedAttentionBranch(model), prompt, 5, sampling=sampling,
                          generator=torch.Generator().manual_seed(8)).tokens
    eos = outputs[-1] if cap > 5 else None
    if eos is not None and eos in outputs[:-1]:
        outputs = outputs[:outputs.index(eos) + 1]
    tokens = torch.tensor(outputs)
    scores = score_trace(model, prompt, tokens, output_budget=cap, eos_id=eos, sampling=sampling,
                          anchors_per_pass=2, chunk_rows=2)
    clean = torch.cat((prompt, tokens[None]), 1)
    teacher = model(clean)
    for j in range(1, len(outputs) - 1):
        anchor = prompt.shape[1] + j - 1
        length = min(4, cap - j + 1)
        block = torch.full((1, length), model.config.mask_token_id)
        block[0, 0] = clean[0, anchor]
        logits = model(block, view="draft", cache=trim_cache(teacher.cache, anchor)).logits[0, :-1]
        q = probabilities(logits, sampling)
        for offset in range(min(length - 1, len(outputs) - j - 1)):
            x = outputs[j + offset]
            p = probabilities(teacher.logits[0, anchor + offset], sampling)[x]
            assert scores.valid[j - 1, offset]
            # The eager backend normalizes attention in FP32 even for FP64 weights.
            assert scores.target_logp[j - 1, offset] == pytest.approx(float(p.log()), abs=5e-8)
            assert scores.proposal_logp[j - 1, offset] == pytest.approx(float(q[offset, x].log()), abs=5e-8)
    scores2 = score_trace(model, prompt, tokens, output_budget=cap, eos_id=eos, sampling=sampling,
                           anchors_per_pass=20, chunk_rows=40)
    torch.testing.assert_close(scores.acceptance, scores2.acceptance, atol=5e-8, rtol=5e-8)


def test_greedy_virtual_work_equals_actual_generation():
    model = small_model()
    prompt = torch.tensor([[2, 3, 4]])
    branch = MaskedAttentionBranch(model)
    ar = generate_ar(branch, prompt, 17)
    actual = generate(branch, prompt, 17)
    assert ar.tokens == actual.tokens
    scores = score_trace(model, prompt, torch.tensor(ar.tokens), output_budget=17, sampling=SamplingConfig())
    assert scores.work().decode_forwards == actual.decode_forwards


def test_trace_layout_isolates_blocks_and_preserves_strict_history():
    clean = torch.tensor([[2, 3, 4, 5]])
    tokens, positions, mask = trace_layout(clean, torch.tensor([1, 3]), 4, 1)
    assert tokens.tolist() == [[3, 1, 1, 1, 5, 1, 1, 1]]
    assert positions.tolist() == [[1, 2, 3, 4, 3, 4, 5, 6]]
    assert mask[0, 0, :4, :4].tolist() == [[True, False, False, False]] * 4
    assert mask[0, 0, 4:, :4].tolist() == [[True, True, True, False]] * 4
    assert mask[0, 0, :4, 4:8].all() and not mask[0, 0, :4, 8:].any()


def test_completed_trace_validation_and_empty_budget():
    model, prompt = small_model(), torch.tensor([[2, 3]])
    empty = score_trace(model, prompt, torch.empty(0, dtype=torch.long), output_budget=0)
    assert empty.work().decode_forwards == 0
    with pytest.raises(ValueError, match="completed answer"):
        score_trace(model, prompt, torch.tensor([3, 4]), output_budget=8)


def test_visit_weighted_calibration_quantiles():
    assert calibration_starts(torch.tensor([1., 0., 2.]), 3) == [1, 3, 3]
    assert calibration_starts(torch.zeros(4), 3) == []
    with pytest.raises(ValueError):
        calibration_starts(torch.ones(4), 0)


def test_timing_estimator_counts_its_work_and_keeps_parameters_fixed():
    model = small_model()
    versions = [(p, p._version) for p in model.parameters()]
    branch, prompt = MaskedAttentionBranch(model), torch.tensor([[2, 3, 4]])
    sampling = SamplingConfig(1.)
    ar = generate_ar(branch, prompt, 9, sampling=sampling, generator=torch.Generator().manual_seed(8), prefix_tokens=1)
    estimate = estimate_trace(branch, prompt, torch.tensor(ar.tokens), output_budget=9,
                               prefill_seconds=ar.prefix_seconds, sampler=ProposalSampler(sampling),
                               generator=torch.Generator().manual_seed(99))
    assert len(estimate.round_seconds) == 3
    assert estimate.measured_seconds == pytest.approx(estimate.score_seconds + estimate.calibration_seconds)
    assert estimate.predicted_seconds > ar.prefix_seconds > 0
    assert estimate.tail_seconds > 0
    assert all(p._version == version for p, version in versions)


def test_report_accepts_first_token_eos():
    from blockspec_ablation.virtual_experiment import summarize
    generation = {"tokens": 1, "seconds": .01, "decode_forwards": 0}
    row = {"ar": generation, "candidate": generation, "predicted_seconds": .01, "virtual_forwards": 0,
           "screen_seconds": .001, "score_seconds": .001, "calibration_seconds": 0.}
    result = summarize([row])
    assert result["candidate_over_ar"] == result["predicted_over_ar"] == 1
    assert result["virtual_decode_tpf"] is None and result["candidate"]["decode_tpf"] is None
