from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from inference_scaling.dllm.backends import LLaDATransformersBackend
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import (
    DiffusionGenerationRequest,
    DiffusionTrajectoryScoreRequest,
)


class TinyMaskedModel(torch.nn.Module):
    def __init__(self, bias: tuple[float, ...], name: str) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.config = SimpleNamespace(_name_or_path=name, mask_token_id=3)
        self.batch_sizes: list[int] = []

    def forward(self, token_ids):
        batch, length = token_ids.shape
        self.batch_sizes.append(batch)
        logits = self.bias.view(1, 1, -1).expand(batch, length, -1).clone()
        return SimpleNamespace(logits=logits)


class TinyTokenizer:
    mask_token_id = 3

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _backend(bias=(0.0, 0.5, 1.0, -2.0), name="tiny", max_batch_size=None):
    return LLaDATransformersBackend(
        TinyMaskedModel(bias, name),
        TinyTokenizer(),
        max_batch_size=max_batch_size,
    )


class TinyExpertLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
        )
        self.dense = torch.nn.Parameter(torch.ones(1))


class TinyLayeredModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor((0.0, 0.5, 1.0, -2.0)))
        self.layers = torch.nn.ModuleList([TinyExpertLayer() for _ in range(4)])
        self.config = SimpleNamespace(
            _name_or_path="tiny-moe",
            mask_token_id=3,
            num_experts=2,
            num_experts_per_tok=1,
        )
        self.executed_layer_counts: list[int] = []

    def forward(self, token_ids):
        self.executed_layer_counts.append(len(self.layers))
        batch, length = token_ids.shape
        logits = self.bias.view(1, 1, -1).expand(batch, length, -1).clone()
        return SimpleNamespace(logits=logits)


def test_random_remasking_trace_rescores_to_its_recorded_probability():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=0.8,
        remasking="random",
    )
    requests = [
        DiffusionGenerationRequest((0,), 4, sampling, seed, f"sample-{seed}")
        for seed in (7, 11, 19)
    ]

    samples = backend.sample_batch(requests)
    rescored = backend.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, sampling) for sample in samples]
    )

    assert [score for score in rescored] == pytest.approx(
        [sample.trajectory_logprob for sample in samples], abs=1e-6
    )
    assert all(len(sample.trace) == 4 for sample in samples)
    assert all(sorted(position for step in sample.trace for position in step.positions) == [0, 1, 2, 3] for sample in samples)
    snapshot = backend.snapshot()
    assert snapshot.sample_requests == 3
    assert snapshot.score_requests == 3
    assert snapshot.generated_tokens == 12
    assert snapshot.model_token_slots > 0
    assert snapshot.sample_model_token_slots > 0
    assert snapshot.score_model_token_slots > 0
    assert snapshot.model_token_slots == (
        snapshot.sample_model_token_slots + snapshot.score_model_token_slots
    )
    assert snapshot.forward_calls == (
        snapshot.sample_forward_calls + snapshot.score_forward_calls
    )


def test_low_confidence_generation_is_not_mislabeled_as_exact_density():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=1,
        temperature=0.0,
        remasking="low_confidence",
    )
    sample = backend.sample_batch(
        [DiffusionGenerationRequest((0,), 2, sampling, 3, "greedy")]
    )[0]

    assert sample.token_ids == (2, 2)
    assert sample.trajectory_logprob is None
    with pytest.raises(ValueError, match="random or sequential remasking"):
        backend.score_trajectories([DiffusionTrajectoryScoreRequest(sample, sampling)])


def test_target_temperature_changes_the_same_trajectory_score():
    backend = _backend()
    proposal = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=1.5,
        remasking="random",
    )
    target = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=0.5,
        remasking="random",
    )
    sample = backend.sample_batch(
        [DiffusionGenerationRequest((0,), 2, proposal, 31, "proposal")]
    )[0]

    target_score = backend.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, target)]
    )[0]

    assert target_score != pytest.approx(sample.trajectory_logprob)


def test_shared_prefix_layer_proposal_reuses_weights_and_counts_active_moe_parameters():
    model = TinyLayeredModel()
    base = LLaDATransformersBackend(model, TinyTokenizer())
    proposal = base.with_prefix_layers(2)
    sampling = DiffusionSamplingConfig(
        block_length=1,
        steps_per_block=1,
        temperature=1.0,
        remasking="random",
    )

    proposal.sample_batch(
        [DiffusionGenerationRequest((0,), 1, sampling, 7, "early-exit")]
    )

    assert model.executed_layer_counts == [2]
    assert len(model.layers) == 4
    base_snapshot = base.snapshot()
    proposal_snapshot = proposal.snapshot()
    assert base_snapshot.total_parameters == 16
    assert base_snapshot.active_parameters == 12
    assert proposal_snapshot.total_parameters == 10
    assert proposal_snapshot.active_parameters == 8
    assert proposal_snapshot.resident_parameters == base_snapshot.resident_parameters == 16


def test_batch_limit_chunks_sampling_and_scoring_without_changing_results():
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=1,
        temperature=0.8,
        remasking="random",
    )
    requests = [
        DiffusionGenerationRequest((0,), 2, sampling, seed, f"sample-{seed}")
        for seed in range(5)
    ]
    limited = _backend(max_batch_size=2)
    unlimited = _backend()

    limited_samples = limited.sample_batch(requests)
    unlimited_samples = unlimited.sample_batch(requests)
    assert [sample.token_ids for sample in limited_samples] == [
        sample.token_ids for sample in unlimited_samples
    ]
    assert [sample.trajectory_logprob for sample in limited_samples] == pytest.approx(
        [sample.trajectory_logprob for sample in unlimited_samples]
    )

    limited_scores = limited.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, sampling) for sample in limited_samples]
    )
    unlimited_scores = unlimited.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, sampling) for sample in unlimited_samples]
    )
    assert limited_scores == pytest.approx(unlimited_scores)
    assert limited.model.batch_sizes == [2, 2, 1, 2, 2, 1]
    assert limited.snapshot().forward_calls == 6


def test_batch_limit_must_be_positive():
    with pytest.raises(ValueError, match="max_batch_size"):
        _backend(max_batch_size=0)
