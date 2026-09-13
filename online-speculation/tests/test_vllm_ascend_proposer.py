"""The vLLM adapter rejects semantically weaker duplicated-model baselines."""

from types import SimpleNamespace

import pytest
import torch

from blockspec.vllm_ascend import OrthrusDraftBatch, OrthrusProposer


class Model:
    def orthrus_propose(self, **kwargs):
        return OrthrusDraftBatch([[3] * 31, [5] * 31], torch.full((2, 31, 17), 1 / 17))


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
