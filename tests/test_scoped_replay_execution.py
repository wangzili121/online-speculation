from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.arllm.gsm8k_replay_benchmark import _run_fresh, _run_warm
from experiments.arllm.gsm8k_dynamic_is_benchmark import _run_method, METHODS
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.backends.transformers_backend import TransformersBackendSnapshot
from inference_scaling.arllm.reward_factory import model_reward_from_config
from inference_scaling.shared.rng import SeedStream


class ThinkingBackend(TabularAutoregressiveBackend):
    parameter_count = 1
    tokenizer = SimpleNamespace(eos_token_id=2, get_vocab=lambda: {"<think>": 3, "</think>": 1, "7": 0, "eos": 2})

    def __init__(self, model_id):
        super().__init__({(3,): (1, 0, 0, 0), (3, 0): (0, 1, 0, 0),
                          (3, 0, 1): (1, 0, 0, 0), (3, 0, 1, 0): (0, 0, 1, 0)},
                         fallback=(0, 0, 1, 0), model_id=model_id)
        self._snapshot = TransformersBackendSnapshot(*([0] * 10))

    def sample_batch(self, requests):
        samples = super().sample_batch(requests)
        slots = sum(len(request.prefix) + len(sample.token_ids) for request, sample in zip(requests, samples, strict=True))
        self._snapshot = replace(self._snapshot, generation_forward_token_slots=self._snapshot.generation_forward_token_slots + slots,
                                 estimated_dense_forward_flops=self._snapshot.estimated_dense_forward_flops + 2 * slots)
        return samples

    def score_statistics_batch(self, requests, **kwargs):
        return [SimpleNamespace(token_topk_confidences=(1.0,) * len(tokens)) for request in requests for tokens in request.continuations]

    def encode(self, text, **kwargs):
        return (self.tokenizer.get_vocab()[text],)

    def decode(self, tokens, **kwargs):
        return "".join("7" for token in tokens if token == 0)

    def snapshot(self):
        return self._snapshot


def configuration(scope="thinking"):
    return {"generation": {"max_new_tokens": 6}, "sampling": {"temperature": 1.0},
            "output": {"sampling_scope": scope, "generation_chunk_size": 1},
            "reward": {"source": "consilience"},
            "conditional_is": {"candidate_count": 2, "rollout_count": 1, "block_size": 2, "reward_temperature": 2.0},
            "replay": {"history_rollouts": 1, "fresh_rollouts": 1, "truncation": 8.0},
            "dynamic_extension": {"auxiliary_mixture": 0.5, "cache_history_rollouts": 1,
                                  "design_rollouts_per_source": 1, "rollouts_per_candidate": 2}}


@pytest.mark.parametrize("mode", ["fresh", "warm", *METHODS])
@pytest.mark.parametrize("scope", ["full", "thinking"])
def test_every_replay_route_preserves_thinking_output_and_final_compute(mode, scope):
    backend, proposal = ThinkingBackend("base"), ThinkingBackend("proposal")
    config = configuration(scope)
    reward = model_reward_from_config(backend, config, source="consilience")
    if mode == "fresh":
        tokens, info = _run_fresh(backend, (3,), reward, "r", config, SeedStream(3))
    elif mode == "warm":
        tokens, info = _run_warm(backend, proposal, (3,), reward, "r", config, SeedStream(3))
    else:
        tokens, info = _run_method(method=mode, backend=backend, proposal_backend=proposal,
                                  prompt=(3,), reward=reward, reward_version="r", config=config, seeds=SeedStream(3))
    assert tokens == (0, 1, 0, 2)
    assert info["output_segments"]["thinking_text"] == "7"
    assert info["output_segments"]["content_text"] == "7"
    assert info["output_segments"]["sampling_scope"] == scope
    assert (info["final_content_forward_token_slots"] > 0) == (scope == "thinking")
