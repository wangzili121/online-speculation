from itertools import product
from math import exp, log

import pytest

from inference_scaling.arllm.backends.stopping import StoppedSequenceBackend
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import ConditionalISConfig, MHConfig, SamplingConfig
from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.algorithms.mh import run_mh_chain
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.output import ThinkingFormat, ThinkingParser
from inference_scaling.arllm.backends.absorbing import AbsorbingEOSBackend


def _backend(*, chunk=2, protected=0, stops=((1,),)):
    base = TabularAutoregressiveBackend({}, fallback=(0.5, 0.3, 0.2))
    return StoppedSequenceBackend(
        base, stop_token_sequences=stops, eos_token_id=2,
        protected_prefix_length=protected, generation_chunk_size=chunk,
    )


def test_stopped_measure_is_normalized_and_retains_boundary_probability():
    backend = _backend()
    scores = backend.score_batch([ScoreRequest((), tuple(product(range(3), repeat=3)))])
    assert sum(exp(sum(score)) for score in scores) == pytest.approx(1)
    assert backend.score_batch([ScoreRequest((), ((0, 1, 2),))])[0] == pytest.approx(
        (log(0.5), log(0.3), 0)
    )
    assert exp(sum(backend.score_batch([ScoreRequest((), ((1, 0, 2),))])[0])) == 0


def test_generation_stops_across_chunks_and_scores_identically():
    backend = _backend(chunk=1, stops=((0, 1),))
    policy = SamplingConfig()
    request = GenerationRequest((), 5, policy, 4, "test", uniforms=(0.1, 0.6, 0.1, 0.1, 0.1))
    sample = backend.sample_batch([request])[0]
    assert sample.token_ids == (0, 1, 2, 2, 2)
    assert sample.token_logprobs == pytest.approx((log(0.5), log(0.3), 0, 0, 0))
    assert backend.score_batch([ScoreRequest((), (sample.token_ids,), policy)])[0] == sample.token_logprobs


def test_stop_marker_can_cross_request_prefix_and_generation():
    backend = _backend(chunk=1, protected=1, stops=((0, 1),))
    # Token 1 in the protected prompt has no stopping effect.
    request = GenerationRequest((1, 0), 3, SamplingConfig(), 4, "cross", uniforms=(0.6, 0.1, 0.1))
    sample = backend.sample_batch([request])[0]
    assert sample.token_ids == (1, 2, 2)
    assert backend.score_batch([ScoreRequest((1, 0), ((1, 2, 2),))])[0] == pytest.approx(
        (log(0.3), 0, 0)
    )


def test_terminal_prefix_is_forced_padding_without_model_calls():
    backend = _backend()
    samples = backend.sample_batch([GenerationRequest((1, 2), 4, SamplingConfig(), 0, "padded")])
    assert samples[0].token_ids == (2,) * 4
    assert samples[0].token_logprobs == (0,) * 4
    with pytest.raises(ValueError, match="non-padding"):
        backend.sample_batch([GenerationRequest((1, 0), 2, SamplingConfig(), 0, "invalid")])


def test_is_and_mh_use_the_same_stopped_backend_contract():
    backend = _backend()
    result = run_conditional_is(
        base_backend=backend, rollout_backend=backend, prompt=(),
        config=ConditionalISConfig(total_length=4, block_size=2, candidate_count=2, rollout_count=2),
        base_sampling=SamplingConfig(eos_token_id=2), rollout_sampling=SamplingConfig(eos_token_id=2),
        reward=lambda prompt, sequence: float(sequence[0] == 0), seeds=SeedStream(7),
    )
    mh = run_mh_chain(
        backend, (), MHConfig(total_length=4, block_size=2, steps_per_block=2),
        SamplingConfig(temperature=0.5), SeedStream(8),
    )
    for tokens in (result.token_ids, mh.token_ids):
        stop = backend._stop_end(tokens)
        if stop is not None:
            assert all(token == 2 for token in tokens[stop:])
        assert exp(sum(backend.score_batch([ScoreRequest((), (tokens,))])[0])) > 0


def test_thinking_and_full_fallback_preserve_the_projected_reward_target():
    raw = TabularAutoregressiveBackend({}, fallback=(0.5, 0.3, 0.2))
    parser = ThinkingParser((ThinkingFormat((1,), (3,)),))
    full = AbsorbingEOSBackend(raw, eos_token_id=2, absorbing_after=1)
    stopped = StoppedSequenceBackend(
        raw, stop_token_sequences=(), eos_token_id=2, protected_prefix_length=1,
        thinking_parser=parser, thinking_prompt=(3,),
    )
    states = tuple(product(range(3), repeat=4))
    full_logs = full.score_batch([ScoreRequest((3,), states)])
    stopped_logs = stopped.score_batch([ScoreRequest((3,), states)])
    projected = {state: 0.0 for state in states}

    def score(tokens):
        segments = parser.split((3,), tokens, eos_token_id=2)
        if segments.has_complete_thinking:
            return 0.25 * len(segments.thinking_token_ids)
        actual = tokens[:tokens.index(2) + 1] if 2 in tokens else tokens
        return -0.1 * sum(actual)  # one fixed full-sequence fallback reward

    for state, logs in zip(states, full_logs, strict=True):
        segments = parser.split((3,), state, eos_token_id=2)
        mapped = state
        if segments.has_complete_thinking:
            end = segments.boundary_end
            mapped = state[:end] + (2,) * (len(state) - end)
        projected[mapped] += exp(sum(logs) + score(state))
    weighted_stopped = {state: exp(sum(logs) + score(state)) for state, logs in zip(states, stopped_logs, strict=True)}
    assert projected == pytest.approx(weighted_stopped, abs=1e-12)


def test_empty_thinking_block_continues_to_full_sequence():
    raw = TabularAutoregressiveBackend({}, fallback=(0.5, 0.3, 0.2))
    parser = ThinkingParser((ThinkingFormat((1,), (3,)),))
    stopped = StoppedSequenceBackend(
        raw, stop_token_sequences=(), eos_token_id=2, protected_prefix_length=1,
        thinking_parser=parser, thinking_prompt=(3,), generation_chunk_size=1,
    )
    request = GenerationRequest((3,), 3, SamplingConfig(), 0, "empty", uniforms=(0.6, 0.1, 0.9))
    result = stopped.sample_batch([request])[0]
    assert result.token_ids == (1, 0, 2)
    assert result.token_logprobs == pytest.approx((log(0.3), log(0.5), log(0.2)))
