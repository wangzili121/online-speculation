from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference_scaling.dllm.config import VRPOSamplingConfig
from inference_scaling.dllm.vrpo import (
    AdapterDisabledReference,
    estimate_masked_elbo,
    estimate_vrpo_preference_loss,
    sample_vrpo_mask_plan,
    vrpo_forward_token_slots,
)


class UniformMaskedModel(torch.nn.Module):
    def __init__(self, vocabulary_size=4):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(vocabulary_size))

    def forward(self, token_ids):
        batch, length = token_ids.shape
        return SimpleNamespace(
            logits=self.bias.view(1, 1, -1).expand(batch, length, -1)
        )


def test_optimal_vrpo_layout_spends_one_mask_per_timestep():
    config = VRPOSamplingConfig(timestep_samples=8, masks_per_timestep=1, antithetic=True)
    plan = sample_vrpo_mask_plan(7, config, np.random.default_rng(4))

    assert len(plan.samples) == 8
    assert [sample.timestep_index for sample in plan.samples] == list(range(8))
    assert all(1 <= sample.mask_count <= 7 for sample in plan.samples)


def test_uniform_model_elbo_is_independent_of_mask_count():
    model = UniformMaskedModel()
    config = VRPOSamplingConfig(timestep_samples=7, masks_per_timestep=1)
    plan = sample_vrpo_mask_plan(3, config, np.random.default_rng(8))

    estimate = estimate_masked_elbo(
        model,
        prompt=(0,),
        answer=(1, 2, 1),
        mask_token_id=3,
        plan=plan,
    )

    assert float(estimate.item()) == pytest.approx(-3 * np.log(4), abs=1e-6)


def test_vrpo_antithetic_means_shared_current_reference_masks():
    current = UniformMaskedModel()
    reference = UniformMaskedModel()
    estimate = estimate_vrpo_preference_loss(
        current,
        reference,
        prompt=(0,),
        chosen=(1, 1, 2),
        rejected=(2, 0),
        mask_token_id=3,
        config=VRPOSamplingConfig(timestep_samples=3, masks_per_timestep=1, antithetic=True),
        seed=9,
    )

    assert estimate.current_chosen_plan is estimate.reference_chosen_plan
    assert estimate.current_rejected_plan is estimate.reference_rejected_plan
    assert float(estimate.preference_score.item()) == pytest.approx(0.0, abs=1e-7)
    assert float(estimate.loss.item()) == pytest.approx(np.log(2), abs=1e-7)


def test_non_antithetic_vrpo_draws_independent_reference_masks():
    current = UniformMaskedModel()
    reference = UniformMaskedModel()
    estimate = estimate_vrpo_preference_loss(
        current,
        reference,
        prompt=(0,),
        chosen=(1, 1, 2, 0),
        rejected=(2, 0, 1, 2),
        mask_token_id=3,
        config=VRPOSamplingConfig(timestep_samples=4, masks_per_timestep=1, antithetic=False),
        seed=3,
    )

    assert estimate.current_chosen_plan is not estimate.reference_chosen_plan
    assert estimate.current_rejected_plan is not estimate.reference_rejected_plan


class TinyAdapterMaskedModel(UniformMaskedModel):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor((0.2, -0.1, 0.05, 0.0)))
        self.adapter_enabled = True

    def forward(self, token_ids):
        batch, length = token_ids.shape
        logits = self.bias + (self.adapter if self.adapter_enabled else 0.0)
        return SimpleNamespace(logits=logits.view(1, 1, -1).expand(batch, length, -1))

    @contextmanager
    def disable_adapter(self):
        previous = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = previous


def test_shared_resident_reference_supports_vrpo_backward_and_update():
    current = TinyAdapterMaskedModel()
    current.train()
    reference = AdapterDisabledReference(current)
    optimizer = torch.optim.AdamW((current.adapter,), lr=0.1)

    estimate = estimate_vrpo_preference_loss(
        current,
        reference,
        prompt=(0,),
        chosen=(1, 1, 2),
        rejected=(2, 0, 0),
        mask_token_id=3,
        config=VRPOSamplingConfig(timestep_samples=3, masks_per_timestep=1),
        seed=19,
    )
    before = current.adapter.detach().clone()
    estimate.loss.backward()
    assert current.adapter.grad is not None
    assert torch.isfinite(current.adapter.grad).all()
    optimizer.step()

    assert not torch.equal(before, current.adapter.detach())
    assert current.adapter_enabled
    assert current.training


def test_vrpo_token_slot_accounting_separates_current_and_reference():
    slots = vrpo_forward_token_slots(
        prompt_length=5,
        chosen_length=3,
        rejected_length=2,
        config=VRPOSamplingConfig(timestep_samples=4, masks_per_timestep=1),
    )

    assert slots == {"current_policy": 60, "reference_policy": 60, "total": 120}
