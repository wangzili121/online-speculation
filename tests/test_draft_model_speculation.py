from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_scaling.arllm.acceleration import DraftModelSpeculationConfig
from inference_scaling.arllm.backends import TransformersBackend
from inference_scaling.experimental.arllm.draft_model_speculation import (
    DraftModelSpeculativeBackend,
    _discard_stale_assistant_cache,
)
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest


class _Tokenizer:
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, vocabulary=None) -> None:
        self._vocabulary = vocabulary or {"zero": 0, "one": 1, "eos": 2}

    def get_vocab(self):
        return dict(self._vocabulary)

    def encode(self, _text, *, add_special_tokens=True):
        return [0] if add_special_tokens else [1]

    def decode(self, tokens, *, skip_special_tokens=True):
        values = [token for token in tokens if not skip_special_tokens or token != 2]
        return " ".join(str(token) for token in values)


class _NativeGenerateModel(torch.nn.Module):
    def __init__(self, probabilities, *, target=False) -> None:
        super().__init__()
        self.logits_parameter = torch.nn.Parameter(
            torch.log(torch.tensor(probabilities, dtype=torch.float32)),
            requires_grad=False,
        )
        self.config = SimpleNamespace(
            _name_or_path="native-generate-model",
            model_type="qwen2",
        )
        self.generation_config = SimpleNamespace(
            num_assistant_tokens=7,
            num_assistant_tokens_schedule="heuristic",
            assistant_confidence_threshold=0.4,
        )
        self.target = target

    @property
    def device(self):
        return self.logits_parameter.device

    def forward(self, input_ids, logits_to_keep=0, **_kwargs):
        batch, length = input_ids.shape
        logits = self.logits_parameter.expand(batch, length, -1).clone()
        if logits_to_keep:
            logits = logits[:, -int(logits_to_keep) :, :]
        return SimpleNamespace(logits=logits, past_key_values=None)

    def generate(self, *, input_ids, assistant_model, **_kwargs):
        assert self.target
        assert assistant_model.generation_config.num_assistant_tokens == 3
        assistant_model(input_ids=input_ids[:, -1:], logits_to_keep=1)
        self(input_ids=input_ids, logits_to_keep=3)
        continuation = torch.tensor([[1, 0]], dtype=torch.long, device=input_ids.device)
        row = self.logits_parameter[None, :]
        return SimpleNamespace(
            sequences=torch.cat([input_ids, continuation], dim=-1),
            logits=(row, row),
        )


class _CroppableCache:
    def __init__(self, length: int) -> None:
        self.length = length
        self.removals = []

    def get_seq_length(self):
        return self.length

    def crop(self, tokens_to_remove):
        self.removals.append(tokens_to_remove)
        self.length -= abs(tokens_to_remove)


def _backend(probabilities, *, target=False, tokenizer=None):
    return TransformersBackend(
        _NativeGenerateModel(probabilities, target=target),
        tokenizer or _Tokenizer(),
        device="cpu",
    )


def test_draft_model_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="draft_tokens"):
        DraftModelSpeculationConfig(draft_tokens=0)
    with pytest.raises(ValueError, match="confidence_threshold"):
        DraftModelSpeculationConfig(confidence_threshold=1.0)


def test_rejected_assistant_tokens_are_removed_before_cache_reuse() -> None:
    cache = _CroppableCache(11)

    assert _discard_stale_assistant_cache(cache, 7) == 4
    assert cache.length == 7
    assert cache.removals == [-4]
    assert _discard_stale_assistant_cache(cache, 7) == 0


def test_native_draft_model_path_keeps_target_probabilities_and_separate_costs() -> None:
    target = _backend([0.2, 0.7, 0.1], target=True)
    draft = _backend([0.6, 0.3, 0.1])
    backend = DraftModelSpeculativeBackend(
        target,
        draft,
        config=DraftModelSpeculationConfig(draft_tokens=3),
    )
    request = GenerationRequest((0, 1), 2, SamplingConfig(), 19, "native")

    sample = backend.sample_batch([request])[0]

    assert sample.token_ids == (1, 0)
    assert sample.token_logprobs == pytest.approx(np.log([0.7, 0.2]))
    assert sample.reference_token_logprobs == pytest.approx(np.log([0.7, 0.2]))
    assert sample.model_id == target.model_id
    snapshot = backend.snapshot()
    assert snapshot.speculative_requests == 1
    assert snapshot.speculative_hits == 1
    assert snapshot.draft_tokens_proposed == 2
    assert snapshot.draft_tokens_accepted == 1
    assert snapshot.verification_rounds == 1
    assert snapshot.generation_forward_token_slots == 2
    assert snapshot.draft_model_forward_token_slots == 1
    assert snapshot.estimated_dense_forward_flops == 12
    assert snapshot.draft_model_estimated_dense_forward_flops == 6
    assert snapshot.total_estimated_dense_forward_flops == 18
    assert draft.model.generation_config.num_assistant_tokens == 7
    assert draft.model.generation_config.num_assistant_tokens_schedule == "heuristic"


def test_native_draft_model_path_restores_callback_order() -> None:
    backend = DraftModelSpeculativeBackend(
        _backend([0.2, 0.7, 0.1], target=True),
        _backend([0.6, 0.3, 0.1]),
        config=DraftModelSpeculationConfig(
            draft_tokens=3,
            single_request_only=False,
        ),
    )
    requests = [
        GenerationRequest((0, 1), 2, SamplingConfig(), seed, f"request-{seed}")
        for seed in (3, 5)
    ]
    completed = []

    outputs = backend.sample_batch_with_callback(
        requests,
        lambda index, sample: completed.append((index, sample.request_id)),
    )

    assert [sample.request_id for sample in outputs] == ["request-3", "request-5"]
    assert completed == [(0, "request-3"), (1, "request-5")]


def test_native_draft_model_path_rejects_explicit_uniforms() -> None:
    backend = DraftModelSpeculativeBackend(
        _backend([0.2, 0.7, 0.1], target=True),
        _backend([0.6, 0.3, 0.1]),
    )
    request = GenerationRequest(
        (0, 1),
        2,
        SamplingConfig(),
        19,
        "explicit",
        uniforms=(0.1, 0.2),
    )

    with pytest.raises(ValueError, match="explicit token uniforms"):
        backend.sample_batch([request])


def test_draft_model_requires_identical_tokenizers() -> None:
    target = _backend([0.2, 0.7, 0.1], target=True)
    draft = _backend(
        [0.6, 0.3, 0.1],
        tokenizer=_Tokenizer({"zero": 0, "different": 1, "eos": 2}),
    )

    with pytest.raises(ValueError, match="identical vocabularies"):
        DraftModelSpeculativeBackend(target, draft)
