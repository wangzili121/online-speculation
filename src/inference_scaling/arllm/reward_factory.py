"""Dataset-independent construction of fixed model-derived sequence rewards."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import thinking_format_from_backend, output_settings_from_config
from inference_scaling.arllm.rewards import ConsilienceReward, SequenceLogProbabilityReward


MODEL_REWARD_SOURCES = ("consilience", "sequence_log_probability")


def model_reward_from_config(
    backend: Any,
    config: Mapping[str, Any],
    *,
    source: str,
    sampling: SamplingConfig | None = None,
) -> ConsilienceReward | SequenceLogProbabilityReward:
    common = config.get("reward", {})
    legacy = config.get("conditional_is", {})
    if source == "sequence_log_probability":
        return SequenceLogProbabilityReward(
            backend,
            sampling,
            scale=float(common.get("logprob_scale", legacy.get("logprob_reward_scale", 1.0))),
        )
    if source != "consilience":
        raise ValueError(f"unknown model reward source: {source}")
    options = common.get("consilience", {})

    def setting(name: str, default: Any) -> Any:
        legacy_name = "reward_scale" if name == "scale" else name
        return options.get(name, legacy.get(f"consilience_{legacy_name}", default))

    output = output_settings_from_config(config)
    legacy_end = legacy.get("consilience_reasoning_end_text")
    if legacy_end is not None:
        output.setdefault("thinking_end_text", legacy_end)
    scope = str(options.get("scope", "thinking"))
    if scope not in {"thinking", "full"}:
        raise ValueError("Consilience scope must be thinking or full")
    format_ = (
        thinking_format_from_backend(backend, output)
        if scope == "thinking" else None
    )
    window_tokens = setting("window_tokens", None)
    return ConsilienceReward(
        backend,
        # Confidence is a fixed property of the reference model, independent
        # of the proposal policy used to obtain the sequence.
        SamplingConfig(temperature=float(options.get("score_temperature", 1.0))),
        top_k=int(setting("top_k", 5)),
        window_fraction=float(setting("window_fraction", 0.2)),
        window_tokens=None if window_tokens is None else int(window_tokens),
        skip_fraction=float(setting("skip_fraction", 0.05)),
        initial_penalty=float(setting("initial_penalty", 3.0)),
        scale=float(setting("scale", 1.0)),
        thinking_format=format_,
        scope="thinking" if scope == "thinking" else "full",
    )


def reward_temperature_from_config(
    config: Mapping[str, Any], *, source: str, default: float = 1.0
) -> float:
    if "temperature" in config.get("reward", {}):
        return float(config["reward"]["temperature"])
    if source == "consilience":
        return 2.0
    return float(config.get("conditional_is", {}).get("reward_temperature", default))


__all__ = ["MODEL_REWARD_SOURCES", "model_reward_from_config", "reward_temperature_from_config"]
