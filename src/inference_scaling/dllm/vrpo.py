"""Variance-reduced ELBO estimators used by dLLM preference optimization.

VRPO calls the shared-randomness construction "antithetic sampling": the
current and reference models evaluate the same timesteps and the same masked
sequences.  It does not pair a timestep ``t`` with ``1 - t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from inference_scaling.dllm.config import VRPOSamplingConfig
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class VRPOMaskSample:
    timestep_index: int
    mask_count: int
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.timestep_index < 0:
            raise ValueError("timestep_index must be non-negative")
        if self.mask_count <= 0:
            raise ValueError("mask_count must be positive")
        if len(self.positions) != self.mask_count:
            raise ValueError("positions must contain exactly mask_count entries")
        if tuple(sorted(self.positions)) != self.positions:
            raise ValueError("masked positions must be sorted")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("masked positions must be unique")


@dataclass(frozen=True, slots=True)
class VRPOMaskPlan:
    answer_length: int
    timestep_samples: int
    masks_per_timestep: int
    samples: tuple[VRPOMaskSample, ...]

    def __post_init__(self) -> None:
        if self.answer_length <= 0:
            raise ValueError("answer_length must be positive")
        if len(self.samples) != self.timestep_samples * self.masks_per_timestep:
            raise ValueError("mask plan size does not match its Monte Carlo layout")
        for sample in self.samples:
            if any(not 0 <= position < self.answer_length for position in sample.positions):
                raise ValueError("masked position lies outside the answer")


def sample_vrpo_mask_plan(
    answer_length: int,
    config: VRPOSamplingConfig,
    rng: np.random.Generator,
) -> VRPOMaskPlan:
    """Draw the doubly Monte Carlo mask plan used by one ELBO estimate.

    A discrete timestep is represented by a mask count sampled uniformly from
    ``1, ..., answer_length``.  All masks assigned to that timestep share the
    count but use independent uniform subsets.  The variance-optimal fixed
    budget setting is ``masks_per_timestep=1``.
    """

    if answer_length <= 0:
        raise ValueError("answer_length must be positive")
    samples: list[VRPOMaskSample] = []
    for timestep_index in range(config.timestep_samples):
        mask_count = int(rng.integers(1, answer_length + 1))
        for _ in range(config.masks_per_timestep):
            positions = tuple(
                sorted(
                    int(value)
                    for value in rng.choice(answer_length, size=mask_count, replace=False)
                )
            )
            samples.append(
                VRPOMaskSample(
                    timestep_index=timestep_index,
                    mask_count=mask_count,
                    positions=positions,
                )
            )
    return VRPOMaskPlan(
        answer_length=answer_length,
        timestep_samples=config.timestep_samples,
        masks_per_timestep=config.masks_per_timestep,
        samples=tuple(samples),
    )


def estimate_masked_elbo(
    model: Any,
    *,
    prompt: TokenSequence,
    answer: TokenSequence,
    mask_token_id: int,
    plan: VRPOMaskPlan,
    cfg_scale: float = 0.0,
) -> Any:
    """Return a differentiable Monte Carlo estimate of the conditional ELBO."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("VRPO estimation requires PyTorch") from exc
    if len(answer) != plan.answer_length:
        raise ValueError("answer length and mask plan do not match")
    if cfg_scale < 0:
        raise ValueError("cfg_scale must be non-negative")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    complete = torch.tensor(prompt + answer, dtype=torch.long, device=device)
    batch = complete.unsqueeze(0).repeat(len(plan.samples), 1)
    mask_matrix = torch.zeros_like(batch, dtype=torch.bool)
    prompt_length = len(prompt)
    for row, sample in enumerate(plan.samples):
        absolute_positions = torch.tensor(
            [prompt_length + position for position in sample.positions],
            dtype=torch.long,
            device=device,
        )
        mask_matrix[row, absolute_positions] = True
    corrupted = torch.where(mask_matrix, mask_token_id, batch)
    if cfg_scale > 0:
        unconditional = corrupted.clone()
        unconditional[:, :prompt_length] = mask_token_id
        logits = model(torch.cat((corrupted, unconditional), dim=0)).logits
        logits, unconditional_logits = torch.chunk(logits, 2, dim=0)
        logits = unconditional_logits + (cfg_scale + 1.0) * (logits - unconditional_logits)
    else:
        logits = model(corrupted).logits
    token_losses = functional.cross_entropy(
        logits[mask_matrix].float(), batch[mask_matrix], reduction="none"
    )
    estimates = []
    cursor = 0
    for sample in plan.samples:
        losses = token_losses[cursor : cursor + sample.mask_count]
        cursor += sample.mask_count
        mask_ratio = sample.mask_count / plan.answer_length
        estimates.append(-losses.sum() / mask_ratio)
    return torch.stack(estimates).mean()


@dataclass(frozen=True, slots=True)
class VRPOPreferenceEstimate:
    loss: Any
    preference_score: Any
    current_chosen_elbo: Any
    reference_chosen_elbo: Any
    current_rejected_elbo: Any
    reference_rejected_elbo: Any
    current_chosen_plan: VRPOMaskPlan
    reference_chosen_plan: VRPOMaskPlan
    current_rejected_plan: VRPOMaskPlan
    reference_rejected_plan: VRPOMaskPlan


class AdapterDisabledReference:
    """Evaluate the frozen base policy through a trainable PEFT model.

    Current-policy and reference-policy calls share one resident base model.
    This avoids loading a second multi-billion-parameter checkpoint while the
    adapter context provides the exact frozen reference used by DPO-style VRPO.
    """

    def __init__(self, model: Any) -> None:
        if not callable(getattr(model, "disable_adapter", None)):
            raise TypeError("model must provide the PEFT disable_adapter context")
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        was_training = bool(getattr(self.model, "training", False))
        try:
            self.model.eval()
            with self.model.disable_adapter():
                return self.model(*args, **kwargs)
        finally:
            self.model.train(was_training)


def vrpo_forward_token_slots(
    *,
    prompt_length: int,
    chosen_length: int,
    rejected_length: int,
    config: VRPOSamplingConfig,
) -> dict[str, int]:
    """Return current/reference forward token slots for one preference pair."""

    for name, value in (
        ("prompt_length", prompt_length),
        ("chosen_length", chosen_length),
        ("rejected_length", rejected_length),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    samples = config.forward_passes
    one_policy = samples * (
        prompt_length + chosen_length + prompt_length + rejected_length
    )
    return {
        "current_policy": one_policy,
        "reference_policy": one_policy,
        "total": 2 * one_policy,
    }


def estimate_vrpo_preference_loss(
    current_model: Any,
    reference_model: Any,
    *,
    prompt: TokenSequence,
    chosen: TokenSequence,
    rejected: TokenSequence,
    mask_token_id: int,
    config: VRPOSamplingConfig = VRPOSamplingConfig(),
    beta: float = 0.2,
    seed: int = 0,
) -> VRPOPreferenceEstimate:
    """Estimate the ELBO-based DPO loss with VRPO variance reduction."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("VRPO estimation requires PyTorch") from exc
    if beta <= 0:
        raise ValueError("beta must be positive")
    seeds = SeedStream(seed)
    current_chosen_plan = sample_vrpo_mask_plan(
        len(chosen), config, seeds.generator("vrpo", "current", "chosen")
    )
    current_rejected_plan = sample_vrpo_mask_plan(
        len(rejected), config, seeds.generator("vrpo", "current", "rejected")
    )
    if config.antithetic:
        reference_chosen_plan = current_chosen_plan
        reference_rejected_plan = current_rejected_plan
    else:
        reference_chosen_plan = sample_vrpo_mask_plan(
            len(chosen), config, seeds.generator("vrpo", "reference", "chosen")
        )
        reference_rejected_plan = sample_vrpo_mask_plan(
            len(rejected), config, seeds.generator("vrpo", "reference", "rejected")
        )

    current_chosen = estimate_masked_elbo(
        current_model,
        prompt=prompt,
        answer=chosen,
        mask_token_id=mask_token_id,
        plan=current_chosen_plan,
    )
    current_rejected = estimate_masked_elbo(
        current_model,
        prompt=prompt,
        answer=rejected,
        mask_token_id=mask_token_id,
        plan=current_rejected_plan,
    )
    with torch.no_grad():
        reference_chosen = estimate_masked_elbo(
            reference_model,
            prompt=prompt,
            answer=chosen,
            mask_token_id=mask_token_id,
            plan=reference_chosen_plan,
        )
        reference_rejected = estimate_masked_elbo(
            reference_model,
            prompt=prompt,
            answer=rejected,
            mask_token_id=mask_token_id,
            plan=reference_rejected_plan,
        )
    preference_score = beta * (
        (current_chosen - reference_chosen)
        - (current_rejected - reference_rejected)
    )
    loss = -functional.logsigmoid(preference_score)
    return VRPOPreferenceEstimate(
        loss=loss,
        preference_score=preference_score,
        current_chosen_elbo=current_chosen,
        reference_chosen_elbo=reference_chosen,
        current_rejected_elbo=current_rejected,
        reference_rejected_elbo=reference_rejected,
        current_chosen_plan=current_chosen_plan,
        reference_chosen_plan=reference_chosen_plan,
        current_rejected_plan=current_rejected_plan,
        reference_rejected_plan=reference_rejected_plan,
    )


__all__ = [
    "AdapterDisabledReference",
    "VRPOMaskPlan",
    "VRPOMaskSample",
    "VRPOPreferenceEstimate",
    "estimate_masked_elbo",
    "estimate_vrpo_preference_loss",
    "sample_vrpo_mask_plan",
    "vrpo_forward_token_slots",
]
