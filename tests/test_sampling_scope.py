from fractions import Fraction
from math import exp
from types import SimpleNamespace

import pytest

from inference_scaling.arllm.backends.reference import ReferencePolicyBackend
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest
from inference_scaling.shared.output import ThinkingFormat
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.evaluation import GSM8KProblem
from experiments.arllm.gsm8k_reproduction import _run_method, _apply_overrides


class _Tokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {"<think>": 3, "</think>": 1, "7": 0, "<eos>": 2}


class _Backend(TabularAutoregressiveBackend):
    tokenizer = _Tokenizer()
    parameter_count = 1

    def __init__(self):
        super().__init__({
            (3,): (1, 0, 0, 0), (3, 0): (0, 1, 0, 0),
            (3, 0, 1): (1, 0, 0, 0), (3, 0, 1, 0): (0, 0, 1, 0),
        }, fallback=(0, 0, 1, 0))

    def decode(self, tokens, *, skip_special_tokens=True):
        return "".join("7" for token in tokens if token == 0)

    def encode(self, text, *, add_special_tokens=False):
        return (self.tokenizer.get_vocab()[text],)

    def score_statistics_batch(self, requests, *, confidence_top_k=None):
        # In this fixture token 0 is the entire thinking span. The end marker
        # and the identically worded final content must stay outside scoring.
        assert all(request.prefix == (3,) for request in requests)
        assert all(tokens == (0,) for request in requests for tokens in request.continuations)
        return [
            SimpleNamespace(token_topk_confidences=(1.0,))
            for request in requests for _ in request.continuations
        ]


def test_reference_policy_matches_direct_temperature_scoring():
    raw = TabularAutoregressiveBackend({}, fallback=(0.75, 0.25))
    backend = ReferencePolicyBackend(raw, temperature=0.6)
    score = backend.score_batch([ScoreRequest((), ((0,), (1,)))])
    expected = raw.score_batch([ScoreRequest((), ((0,), (1,)), SamplingConfig(temperature=0.6))])
    assert score == expected
    assert sum(exp(sum(item)) for item in score) == pytest.approx(1)
    request = GenerationRequest((), 3, SamplingConfig(), 1, "sample")
    sample = backend.sample_batch([request])[0]
    assert sample.reference_token_logprobs == sample.token_logprobs
    assert sample.policy_id == SamplingConfig().policy_id


def test_scope_finishes_content_from_original_backend_with_remaining_budget():
    backend = _Backend()
    scope = SamplingScope("thinking", ThinkingFormat((1,), (3,)), generation_chunk_size=1)
    stopped = scope.wrap(backend, (3,))
    thinking = stopped.sample_batch([GenerationRequest((3,), 6, SamplingConfig(), 0, "thinking")])[0]
    assert thinking.token_ids == (0, 1, 2, 2, 2, 2)
    tokens, info = scope.finish(
        backend, (3,), thinking.token_ids, max_new_tokens=6,
        sampling=SamplingConfig(eos_token_id=2), seed=1,
    )
    assert tokens == (0, 1, 0, 2)
    assert info["thinking_token_ids"] == (0,)
    assert info["content_token_ids"] == (0,)
    assert info["final_content_generated_tokens"] == 2
    assert info["thinking_status"] == "complete"


@pytest.mark.parametrize("method", ("mh", "conditional_is", "iterated_conditional_is", "reward_mh"))
@pytest.mark.parametrize("scope", ("full", "thinking"))
@pytest.mark.parametrize("reward_source", ("sequence_log_probability", "consilience"))
def test_experiment_dispatch_preserves_thinking_content_for_all_core_methods(method, scope, reward_source):
    config = {
        "generation": {"max_new_tokens": 6}, "sampling": {"temperature": 0.6},
        "mh": {"alpha": 2, "block_size": 2, "steps_per_block": 1},
        "conditional_is": {"candidate_count": 2, "rollout_count": 1, "block_size": 2, "reward_temperature": 1},
        "reward": {"source": reward_source},
        "output": {"sampling_scope": scope, "generation_chunk_size": 1},
    }
    backend = _Backend()
    problem = GSM8KProblem(index=0, question="seven", gold_solution="#### 7", gold_answer=Fraction(7))
    tokens, diagnostics = _run_method(method, backend, problem, (3,), config, SeedStream(1), None)
    assert backend.decode(tokens) == "77"
    output = diagnostics["output_segments"]
    assert output["thinking_text"] == "7"
    assert output["content_text"] == "7"
    assert output["sampling_scope"] == scope


def test_cli_overrides_common_and_legacy_reward_settings_consistently():
    args = SimpleNamespace(**dict.fromkeys((
        "backend", "limit", "max_new_tokens", "sampling_temperature", "num_beams",
        "best_of_n_samples", "importance_log_ratio_clip", "mh_alpha", "mh_steps",
        "candidate_count", "rollout_count", "block_size",
    )))
    args.method = "reward_mh"
    args.disable_importance_correction = False
    args.conditional_reward = "consilience"
    args.reward_temperature = 4.0
    args.consilience_reward_scale = 2.0
    args.consilience_top_k = 7
    args.sampling_scope = "thinking"
    args.thinking_end_text = "</think>"
    config = {"conditional_is": {}, "reward": {"consilience": {"top_k": 5, "scale": 1.0}}}
    _apply_overrides(config, args)
    assert config["reward"] == {
        "source": "consilience", "temperature": 4.0,
        "consilience": {"top_k": 7, "scale": 2.0},
    }
    assert config["output"] == {"sampling_scope": "thinking", "thinking_end_text": "</think>"}


def test_passk_adapter_preserves_token_format_and_confidence_scoring():
    from experiments.arllm.gsm8k_passk import _MethodBackend
    from inference_scaling.arllm.reward_factory import model_reward_from_config

    raw = _Backend()
    adapter = _MethodBackend(raw, raw)
    reward = model_reward_from_config(adapter, {}, source="consilience")
    assert reward((3,), (0, 1, 0, 2)) == -2.0


def test_thinking_scope_reports_full_fallback_for_final_content_rewards():
    config = {
        "output": {"sampling_scope": "thinking"}, "generation": {"max_new_tokens": 6},
        "conditional_is": {"reward": "self_consistency", "candidate_count": 2, "rollout_count": 1, "block_size": 2},
    }
    problem = GSM8KProblem(index=0, question="seven", gold_solution="#### 7", gold_answer=Fraction(7))
    tokens, info = _run_method("conditional_is", _Backend(), problem, (3,), config, SeedStream(1), None)
    assert _Backend().decode(tokens) == "77"
    assert info["output_segments"]["sampling_scope"] == "full"
    assert info["output_segments"]["sampling_fallback_reason"] == "reward_uses_full_sequence"
