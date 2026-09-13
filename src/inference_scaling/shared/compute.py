"""Transparent token and dominant-matmul FLOP accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Sequence
from typing import Any


def dense_forward_flops(parameter_count: int, forward_token_slots: int) -> int:
    """Conventional ``2 * parameters * tokens`` forward-pass estimate."""

    if parameter_count < 0 or forward_token_slots < 0:
        raise ValueError("parameter_count and forward_token_slots must be non-negative")
    return 2 * parameter_count * forward_token_slots


@dataclass(frozen=True, slots=True)
class GRPOComputeEstimate:
    trainer_observed_prompt_plus_completion_tokens: int
    generated_completions: int
    rollout_generation_forward_token_slots: int
    reference_scoring_forward_token_slots: int
    policy_forward_backward_equivalent_token_slots: int
    total_forward_equivalent_token_slots: int
    total_parameters: int
    trainable_parameters: int
    optimizer_steps: int
    estimated_dense_model_flops: int
    estimated_optimizer_flops: int
    estimated_total_flops: int
    estimated_total_petaflops: float
    accounting_basis: str
    definition: str
    exclusions: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_grpo_compute(
    *,
    model_sequence_tokens: int,
    generated_completions: int,
    total_parameters: int,
    trainable_parameters: int,
    optimizer_steps: int,
    gradient_checkpointing: bool,
    reference_scoring: bool,
) -> GRPOComputeEstimate:
    """Estimate this LoRA GRPO run from observed model-token counts.

    For a frozen dense base, a policy forward/backward pass is approximately
    two forward-equivalent dense passes. Gradient-checkpoint recomputation adds
    one more. The adapter overhead is included conservatively through the total
    parameter count. AdamW is charged only to trainable parameters.
    """

    values = (
        model_sequence_tokens,
        generated_completions,
        total_parameters,
        trainable_parameters,
        optimizer_steps,
    )
    if any(value < 0 for value in values):
        raise ValueError("GRPO compute inputs must be non-negative")
    if generated_completions > model_sequence_tokens:
        raise ValueError("each generated completion requires at least one model token")

    rollout_slots = model_sequence_tokens - generated_completions
    reference_slots = model_sequence_tokens if reference_scoring else 0
    policy_multiplier = 3 if gradient_checkpointing else 2
    policy_slots = policy_multiplier * model_sequence_tokens
    total_slots = rollout_slots + reference_slots + policy_slots
    dense_flops = dense_forward_flops(total_parameters, total_slots)
    optimizer_flops = 10 * trainable_parameters * optimizer_steps
    total_flops = dense_flops + optimizer_flops
    return GRPOComputeEstimate(
        trainer_observed_prompt_plus_completion_tokens=model_sequence_tokens,
        generated_completions=generated_completions,
        rollout_generation_forward_token_slots=rollout_slots,
        reference_scoring_forward_token_slots=reference_slots,
        policy_forward_backward_equivalent_token_slots=policy_slots,
        total_forward_equivalent_token_slots=total_slots,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        optimizer_steps=optimizer_steps,
        estimated_dense_model_flops=dense_flops,
        estimated_optimizer_flops=optimizer_flops,
        estimated_total_flops=total_flops,
        estimated_total_petaflops=total_flops / 1e15,
        accounting_basis=(
            "non-padding trainer token count; use estimate_grpo_compute_from_logs "
            "when per-step mean and maximum completion lengths are available"
        ),
        definition=(
            "rollout generation uses observed prompt+completion tokens minus one input "
            "slot per completion; reference scoring uses one forward pass when beta is "
            "nonzero; a frozen-base LoRA policy update uses two forward-equivalent "
            "passes plus one more when gradient checkpointing recomputes activations; "
            "dominant dense FLOPs are 2 * total parameters * forward-equivalent token "
            "slots; AdamW is 10 * trainable parameters * optimizer steps"
        ),
        exclusions=(
            "quadratic attention, elementwise kernels, reward parsing, data loading, "
            "tokenization, sampling, and host work"
        ),
    )


def estimate_grpo_compute_from_logs(
    *,
    log_history: Sequence[dict[str, Any]],
    sequences_per_optimizer_step: int,
    generated_completions: int,
    total_parameters: int,
    trainable_parameters: int,
    optimizer_steps: int,
    gradient_checkpointing: bool,
    reference_scoring: bool,
) -> GRPOComputeEstimate:
    """Reconstruct padded GRPO token slots from trainer step metrics.

    TRL reports the cumulative non-padding model tokens, the batch mean
    completion length, and the mean of each microbatch's maximum completion
    length. Since every optimizer step has a fixed number of sequences, these
    values recover the prompt tokens and the padded prompt+completion tensor
    shapes used by generation, reference scoring, and policy training.
    """

    if sequences_per_optimizer_step <= 0:
        raise ValueError("sequences_per_optimizer_step must be positive")
    expected_completions = sequences_per_optimizer_step * optimizer_steps
    if generated_completions != expected_completions:
        raise ValueError(
            "generated completion count does not match batch size times optimizer steps"
        )
    complete_by_step = {
        int(entry["step"]): entry
        for entry in log_history
        if all(
            key in entry
            for key in (
                "num_tokens",
                "completions/mean_length",
                "completions/max_length",
                "step",
            )
        )
    }
    step_logs = list(complete_by_step.values())
    if not step_logs:
        raise ValueError("trainer log history has no complete GRPO step metrics")
    step_logs.sort(key=lambda entry: int(entry["step"]))

    prior_tokens = 0
    prior_step = 0
    rollout_slots = 0.0
    full_sequence_slots = 0.0
    for entry in step_logs:
        current_step = int(entry["step"])
        if current_step <= prior_step:
            raise ValueError("trainer steps are not strictly increasing")
        sequences_in_interval = (
            sequences_per_optimizer_step * (current_step - prior_step)
        )
        prior_step = current_step
        cumulative_tokens = int(entry["num_tokens"])
        if cumulative_tokens < prior_tokens:
            raise ValueError("trainer num_tokens is not cumulative")
        observed_step_tokens = cumulative_tokens - prior_tokens
        prior_tokens = cumulative_tokens
        mean_completion = float(entry["completions/mean_length"])
        maximum_completion = float(entry["completions/max_length"])
        if maximum_completion < mean_completion or mean_completion < 0:
            raise ValueError("invalid trainer completion-length metrics")
        completion_tokens = mean_completion * sequences_in_interval
        prompt_tokens = observed_step_tokens - completion_tokens
        if prompt_tokens < -1e-6:
            raise ValueError("completion metrics exceed trainer-observed model tokens")
        prompt_tokens = max(0.0, prompt_tokens)
        padded_completion_tokens = maximum_completion * sequences_in_interval
        full_sequence_slots += prompt_tokens + padded_completion_tokens
        rollout_slots += prompt_tokens + max(
            0.0,
            padded_completion_tokens - sequences_in_interval,
        )

    if prior_step != optimizer_steps:
        raise ValueError(
            "trainer log history does not cover the requested optimizer-step count"
        )

    rollout_slots_int = round(rollout_slots)
    reference_slots = round(full_sequence_slots) if reference_scoring else 0
    policy_multiplier = 3 if gradient_checkpointing else 2
    policy_slots = policy_multiplier * round(full_sequence_slots)
    total_slots = rollout_slots_int + reference_slots + policy_slots
    dense_flops = dense_forward_flops(total_parameters, total_slots)
    optimizer_flops = 10 * trainable_parameters * optimizer_steps
    total_flops = dense_flops + optimizer_flops
    return GRPOComputeEstimate(
        trainer_observed_prompt_plus_completion_tokens=prior_tokens,
        generated_completions=generated_completions,
        rollout_generation_forward_token_slots=rollout_slots_int,
        reference_scoring_forward_token_slots=reference_slots,
        policy_forward_backward_equivalent_token_slots=policy_slots,
        total_forward_equivalent_token_slots=total_slots,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        optimizer_steps=optimizer_steps,
        estimated_dense_model_flops=dense_flops,
        estimated_optimizer_flops=optimizer_flops,
        estimated_total_flops=total_flops,
        estimated_total_petaflops=total_flops / 1e15,
        accounting_basis=(
            "padded forward token slots reconstructed per optimizer step from "
            "cumulative num_tokens, mean completion length, and microbatch maximum "
            "completion length"
        ),
        definition=(
            "generation counts repeated prompts and every padded decode row except the "
            "final generated token; reference scoring counts each padded full sequence; "
            "a frozen-base LoRA policy update uses two forward-equivalent passes plus "
            "one gradient-checkpoint recomputation; dominant dense FLOPs are 2 * total "
            "parameters * reconstructed token slots; AdamW is 10 * trainable parameters "
            "* optimizer steps"
        ),
        exclusions=(
            "quadratic attention, elementwise kernels, reward parsing, data loading, "
            "tokenization, sampling, and host work"
        ),
    )
