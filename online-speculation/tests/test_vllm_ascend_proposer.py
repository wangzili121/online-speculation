"""The vLLM adapter rejects semantically weaker duplicated-model baselines."""

from types import SimpleNamespace

import pytest
import torch

from blockspec.vllm_ascend import OrthrusDraftBatch, OrthrusProposer


class Model:
    def orthrus_propose(self, **kwargs):
        return OrthrusDraftBatch([[3] * 31, [5] * 31], torch.full((2, 31, 17), 1 / 17))


class AsyncModel:
    def __init__(self):
        self.calls = []

    def orthrus_propose(self, **kwargs):
        self.calls.append(kwargs)
        device = kwargs["sampled_token_ids"].device
        counts = (kwargs["sampled_token_ids"] >= 0).sum(dim=1)
        last = kwargs["sampled_token_ids"].gather(
            1, (counts - 1).clamp_min(0)[:, None]).squeeze(1)
        token_ids = torch.full((2, 31), -1, device=device)
        for index in kwargs.get("draft_request_indices") or range(2):
            token_ids[index] = 3 + 2 * index
        return OrthrusDraftBatch(
            token_ids,
            torch.full((2, 31, 17), 1 / 17, device=device),
            accepted_token_counts=counts,
            next_token_ids=last,
        )


class Runner:
    def __init__(self, model):
        self.model = model

    def get_model(self):
        return self.model


class InitializingRunner:
    """Match Ascend's constructor order: the runner exists before its model."""

    def get_model(self):
        raise AttributeError("model has not been loaded")


def config(tokens=31, method="probabilistic"):
    return SimpleNamespace(speculative_config=SimpleNamespace(
        num_speculative_tokens=tokens, draft_sample_method=method))


def test_shared_backbone_proposer_preserves_probabilities():
    proposer, runner = OrthrusProposer(config()), Runner(Model())
    batch = SimpleNamespace(req_ids=["a", "b"])
    ids = proposer.propose(sampled_token_ids=[[1], [2]], input_batch=batch,
                           scheduler_output=object(), common_attn_metadata=object(), runner=runner)
    assert ids == [[3] * 31, [5] * 31]
    assert proposer.take_last_draft_probs().shape == (2, 31, 17)
    assert proposer.take_last_draft_probs() is None


def test_shared_backbone_proposer_exposes_async_state_once():
    model = AsyncModel()
    proposer, runner = OrthrusProposer(config()), Runner(model)
    batch = SimpleNamespace(
        req_ids=["a", "b"], num_tokens_no_spec=[101, 201])
    ids = proposer.propose(
        sampled_token_ids=torch.tensor([
            [1, 11, -1, -1, -1, -1, -1],
            [2, 3, 4, 5, 6, 7, 13],
        ]),
        input_batch=batch,
        scheduler_output=object(), common_attn_metadata=object(), runner=runner)
    assert ids.shape == (2, 31)
    next_ids, counts = proposer.take_last_async_state()
    assert next_ids.tolist() == [11, 13]
    assert counts.tolist() == [2, 7]
    assert proposer.take_last_async_state() == (None, None)
    assert model.calls[0]["history_lengths"] == [100, 200]

    proposer.propose(
        sampled_token_ids=torch.tensor([[7, 8, -1], [9, -1, -1]]),
        input_batch=batch, scheduler_output=object(),
        common_attn_metadata=object(), runner=runner)
    assert model.calls[1]["history_lengths"] == [102, 201]


def test_staggered_phase_scheduler_bounds_orthrus_draft(monkeypatch):
    monkeypatch.setenv("ORTHRUS_PHASE_SCHEDULER", "staggered")
    monkeypatch.setenv("ORTHRUS_DRAFT_MAX_BATCH", "1")
    model = AsyncModel()
    proposer, runner = OrthrusProposer(config()), Runner(model)
    batch = SimpleNamespace(
        req_ids=["a", "b"], num_tokens_no_spec=[101, 201])

    first = proposer.propose(
        sampled_token_ids=torch.tensor([[1, -1], [2, -1]]),
        input_batch=batch, scheduler_output=object(),
        common_attn_metadata=object(), runner=runner)
    second = proposer.propose(
        sampled_token_ids=torch.tensor([[3, -1], [4, -1]]),
        input_batch=batch, scheduler_output=object(),
        common_attn_metadata=object(), runner=runner)

    assert (first >= 0).any(dim=1).sum().item() == 1
    assert (second >= 0).any(dim=1).sum().item() == 1
    assert model.calls[0]["draft_request_indices"] != model.calls[1][
        "draft_request_indices"]


def test_proposer_requires_exact_contract_and_shared_model():
    with pytest.raises(ValueError, match="exactly 31"):
        OrthrusProposer(config(tokens=16))
    with pytest.raises(ValueError, match="probabilistic"):
        OrthrusProposer(config(method="greedy"))
    proposer = OrthrusProposer(config())
    with pytest.raises(TypeError, match="shared-backbone"):
        proposer.load_model(object())


def test_proposer_loads_no_second_model():
    proposer = OrthrusProposer(config())
    model = Model()
    assert proposer.load_model(model) is None
    assert proposer.dummy_run(32, is_profile=True) is None


def test_proposer_binding_does_not_access_model_during_runner_init():
    proposer = OrthrusProposer(config())
    runner = InitializingRunner()
    proposer.bind_runner(runner)
    assert proposer.runner is runner
