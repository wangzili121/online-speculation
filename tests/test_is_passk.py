from __future__ import annotations

import threading
from fractions import Fraction
from types import SimpleNamespace

import pytest

import experiments.arllm.gsm8k_is_passk as is_passk
from experiments.arllm.gsm8k_is_passk import (
    _combine_batching_snapshots,
    _combine_numeric_deltas,
    _execution_method_and_config,
    _paired_pass_at_k_difference,
    _summarize_batching_by_model,
    _summarize_model_compute,
    _uses_small_proposal,
)
from inference_scaling.arllm.backends.transformers_backend import TransformersBackendSnapshot
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, SequenceSample


class _Tokenizer:
    eos_token_id = 99


class _CountingRawBackend:
    def __init__(self, model_id: str, parameter_count: int) -> None:
        self._model_id = model_id
        self.parameter_count = parameter_count
        self.tokenizer = _Tokenizer()
        self._lock = threading.Lock()
        self._sample_calls = 0
        self._sampled_sequences = 0
        self._generated_tokens = 0
        self._prefill_tokens = 0
        self._generation_slots = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def sample_batch(self, requests):
        with self._lock:
            self._sample_calls += 1
            self._sampled_sequences += len(requests)
            self._generated_tokens += len(requests)
            self._prefill_tokens += sum(len(request.prefix) for request in requests)
            slots = sum(len(request.prefix) + 1 for request in requests)
            self._generation_slots += slots
        return [
            SequenceSample(
                prefix=request.prefix,
                token_ids=(1,),
                token_logprobs=(-0.1,),
                policy_id="test-policy",
                model_id=self.model_id,
                request_id=request.request_id,
            )
            for request in requests
        ]

    def score_batch(self, requests):
        raise AssertionError("this test path does not score sequences")

    def decode(self, tokens) -> str:
        return "#### 1"

    def snapshot(self) -> TransformersBackendSnapshot:
        with self._lock:
            return TransformersBackendSnapshot(
                sample_calls=self._sample_calls,
                score_calls=0,
                sampled_sequences=self._sampled_sequences,
                generated_tokens=self._generated_tokens,
                prefill_tokens=self._prefill_tokens,
                shared_prefill_tokens_saved=0,
                scored_tokens=0,
                generation_forward_token_slots=self._generation_slots,
                score_forward_token_slots=0,
                estimated_dense_forward_flops=(
                    2 * self.parameter_count * self._generation_slots
                ),
            )


def test_is_passk_combines_base_and_proposal_compute() -> None:
    base = {
        "generation_forward_token_slots": 10,
        "score_forward_token_slots": 20,
        "estimated_dense_forward_flops": 300,
    }
    proposal = {
        "generation_forward_token_slots": 7,
        "score_forward_token_slots": 0,
        "estimated_dense_forward_flops": 40,
    }
    assert _combine_numeric_deltas(base, proposal) == {
        "generation_forward_token_slots": 17,
        "score_forward_token_slots": 20,
        "estimated_dense_forward_flops": 340,
    }
    assert _combine_numeric_deltas(base, None) == base


def test_is_passk_combines_batching_by_sum_and_max() -> None:
    base = {
        "sample_batches": 2,
        "score_batches": 3,
        "sample_requests": 8,
        "score_sequences": 12,
        "maximum_sample_batch": 6,
        "maximum_score_batch": 7,
    }
    proposal = {
        "sample_batches": 4,
        "score_batches": 0,
        "sample_requests": 10,
        "score_sequences": 0,
        "maximum_sample_batch": 8,
        "maximum_score_batch": 0,
    }
    assert _combine_batching_snapshots(base, proposal) == {
        "sample_batches": 6,
        "score_batches": 3,
        "sample_requests": 18,
        "score_sequences": 12,
        "maximum_sample_batch": 8,
        "maximum_score_batch": 7,
    }


def test_is_passk_keeps_model_specific_compute_and_batching() -> None:
    base_delta = {
        "generation_forward_token_slots": 10,
        "score_forward_token_slots": 5,
        "estimated_dense_forward_flops": 30,
    }
    proposal_delta = {
        "generation_forward_token_slots": 7,
        "score_forward_token_slots": 0,
        "estimated_dense_forward_flops": 8,
    }
    base_batching = {
        "sample_batches": 2,
        "score_batches": 1,
        "sample_requests": 5,
        "score_sequences": 3,
        "maximum_sample_batch": 4,
        "maximum_score_batch": 3,
    }
    proposal_batching = {
        "sample_batches": 3,
        "score_batches": 0,
        "sample_requests": 7,
        "score_sequences": 0,
        "maximum_sample_batch": 6,
        "maximum_score_batch": 0,
    }
    chunks = [
        {
            "base_backend_delta": base_delta,
            "proposal_backend_delta": proposal_delta,
            "continuous_batching_by_model": {
                "base": base_batching,
                "proposal": proposal_batching,
            },
        },
        {
            "base_backend_delta": base_delta,
            "proposal_backend_delta": proposal_delta,
            "continuous_batching_by_model": {
                "base": base_batching,
                "proposal": proposal_batching,
            },
        },
    ]
    assert _summarize_model_compute(chunks, "base_backend_delta") == {
        "estimated_dense_forward_flops": 60,
        "generation_forward_token_slots": 20,
        "score_forward_token_slots": 10,
        "total_forward_token_slots": 30,
        "estimated_dense_forward_petaflops": 60 / 1e15,
    }
    assert _summarize_batching_by_model(chunks, "proposal") == {
        "sample_batches": 6,
        "score_batches": 0,
        "sample_requests": 14,
        "score_sequences": 0,
        "maximum_sample_batch": 6,
        "maximum_score_batch": 0,
    }


def test_is_passk_chunk_accounts_both_models_and_keeps_draws_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_method(
        method, backend, problem, prompt, config, seeds, proposal_backend
    ):
        request = GenerationRequest(
            prompt,
            1,
            SamplingConfig(),
            seeds.derive("fake"),
            f"{method}:{problem.question}",
        )
        backend.sample_batch([request])
        assert proposal_backend is not None
        proposal_backend.sample_batch([request])
        return (1,), {"fake": True, "output_segments": {"content_text": "1"}}

    monkeypatch.setattr(is_passk, "_run_method", fake_run_method)
    monkeypatch.setattr(is_passk, "_timed", lambda call: (call(), 0.25))
    base = _CountingRawBackend("base", 10)
    proposal = _CountingRawBackend("proposal", 3)
    chunk = is_passk._run_chunk(
        method="conditional_is_small_proposal",
        chunk_index=0,
        task_keys=((0, 7), (1, 7)),
        problems_by_index={
            7: SimpleNamespace(question="one", gold_answer=Fraction(1))
        },
        prompts_by_index={7: (4, 5)},
        raw_base=base,
        raw_proposal=proposal,
        config={
            "run": {"seed": 11},
            "runtime": {"max_batch_size": 8, "max_batch_tokens": 100},
        },
        workers=2,
        fingerprint="grid",
    )
    assert len(chunk["records"]) == 2
    assert all(record["correct"] for record in chunk["records"])
    assert chunk["base_backend_delta"]["generation_forward_token_slots"] == 6
    assert chunk["proposal_backend_delta"]["generation_forward_token_slots"] == 6
    assert chunk["backend_delta"]["generation_forward_token_slots"] == 12
    assert chunk["backend_delta"]["estimated_dense_forward_flops"] == 156
    assert chunk["continuous_batching_by_model"]["base"]["sample_requests"] == 2
    assert chunk["continuous_batching_by_model"]["proposal"]["sample_requests"] == 2


def test_is_passk_paired_bootstrap_uses_problem_level_differences() -> None:
    standard = {
        "estimated_pass_at_k": {"1": 0.5, "2": 0.75},
        "per_problem": [
            {"problem_index": 1, "correct_draws": 0},
            {"problem_index": 2, "correct_draws": 2},
        ],
    }
    small = {
        "estimated_pass_at_k": {"1": 0.75, "2": 1.0},
        "per_problem": [
            {"problem_index": 1, "correct_draws": 1},
            {"problem_index": 2, "correct_draws": 2},
        ],
    }
    comparison = _paired_pass_at_k_difference(
        standard, small, draws=2, seed=3, replicates=1_000
    )
    assert comparison["1"]["candidate_minus_reference"] == pytest.approx(0.25)
    assert comparison["2"]["candidate_minus_reference"] == pytest.approx(0.5)
    assert comparison["1"]["paired_problem_bootstrap_95"] == [0.0, 0.5]
    assert comparison["2"]["paired_problem_bootstrap_95"] == [0.0, 1.0]


def test_is_passk_unclipped_method_changes_only_the_clip_override() -> None:
    config = {
        "conditional_is": {
            "importance_log_ratio_clip": 10.0,
            "candidate_count": 8,
        }
    }
    method, effective = _execution_method_and_config(
        "conditional_is_small_proposal_unclipped", config
    )
    assert method == "conditional_is_small_proposal"
    assert effective["conditional_is"]["importance_log_ratio_clip"] is None
    assert effective["conditional_is"]["candidate_count"] == 8
    assert config["conditional_is"]["importance_log_ratio_clip"] == 10.0
    assert _uses_small_proposal("conditional_is_small_proposal")
    assert _uses_small_proposal("conditional_is_small_proposal_unclipped")
    assert not _uses_small_proposal("conditional_is")


def test_is_passk_uncorrected_method_disables_rescoring_and_clipping() -> None:
    config = {
        "conditional_is": {
            "importance_log_ratio_clip": 10.0,
            "candidate_count": 8,
        }
    }
    method, effective = _execution_method_and_config(
        "conditional_is_small_proposal_uncorrected", config
    )
    assert method == "conditional_is_small_proposal"
    assert effective["conditional_is"]["importance_log_ratio_clip"] is None
    assert effective["conditional_is"]["apply_importance_correction"] is False
    assert effective["conditional_is"]["candidate_count"] == 8
    assert config["conditional_is"]["importance_log_ratio_clip"] == 10.0
    assert "apply_importance_correction" not in config["conditional_is"]
    assert _uses_small_proposal("conditional_is_small_proposal_uncorrected")
