"""Output scope and accounting for fixed-reward replay experiment drivers."""

from __future__ import annotations

from functools import wraps
import inspect
import time

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.reward_factory import model_reward_from_config, reward_temperature_from_config
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.shared.generation import generation_config_for_prompt
from experiments.shared.artifacts import json_fingerprint


def fixed_experiment_reward(backend, problem, config, verifier_factory):
    source = config.get("reward", {}).get("source", "verifier")
    if source == "verifier":
        reward = verifier_factory(backend, problem, config)
        return reward, reward.version
    reward = model_reward_from_config(backend, config, source=source)
    return reward, json_fingerprint(reward.describe())


def with_output_scope(operation):
    """Preserve the driver signature; centralize stopped sampling and completion."""
    signature = inspect.signature(operation)

    @wraps(operation)
    def run(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        backend = arguments["backend"]
        proposal = arguments.get("proposal_backend")
        prompt = arguments["prompt"]
        config, budget = generation_config_for_prompt(arguments["config"], len(prompt), [backend, proposal])
        source = config.get("reward", {}).get("source", "verifier")
        if source != "verifier":
            config.setdefault("conditional_is", {})["reward_temperature"] = reward_temperature_from_config(config, source=source)
        scope = SamplingScope.from_config(backend, config).for_prompt(prompt)
        reward_scope = (
            config.get("reward", {}).get("consilience", {}).get("scope", "thinking")
            if source == "consilience" else config.get("reward", {}).get("input_scope", "full")
        )
        if scope.scope == "thinking" and reward_scope != "thinking":
            scope = scope.full_fallback("reward_uses_full_sequence")
        arguments["config"] = config
        arguments["backend"] = scope.wrap(backend, prompt)
        if proposal is not None:
            arguments["proposal_backend"] = scope.wrap(proposal, prompt)
        tokens, info = operation(**arguments)
        before = backend.snapshot()
        started = time.perf_counter()
        tokens, output = scope.finish(
            backend, prompt, tokens, max_new_tokens=budget["effective_max_new_tokens"],
            sampling=SamplingConfig(temperature=float(config.get("sampling", {}).get("temperature", 1.0)),
                                    eos_token_id=backend.tokenizer.eos_token_id),
            seed=arguments["seeds"].derive("scope", "final-content"),
        )
        seconds = time.perf_counter() - started
        after = backend.snapshot()
        slots = (after.generation_forward_token_slots - before.generation_forward_token_slots)
        flops = after.estimated_dense_forward_flops - before.estimated_dense_forward_flops
        info.update(output_segments=output, generation_budget=budget,
                    final_content_forward_token_slots=slots,
                    final_content_estimated_dense_forward_flops=flops,
                    final_content_seconds=seconds)
        # The final continuation is part of online inference, not replay preparation.
        for key in tuple(info):
            if key.startswith(("online_", "steady_online_", "one_shot_", "end_to_end_")) and "proposal" not in key:
                if key.endswith("seconds"):
                    info[key] += seconds
                elif key.endswith("forward_token_slots"):
                    info[key] += slots
                elif key.endswith("estimated_dense_forward_flops"):
                    info[key] += flops
        callback = getattr(arguments["reward"], "describe_completion", None)
        if callback is not None:
            output.update(callback(prompt, tokens))
        return tokens, info

    return run
