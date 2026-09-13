"""Shared model, output-scope and generation options for AR entry points."""

from __future__ import annotations

import argparse
from typing import Any, Collection

_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("--model", {"help": "base checkpoint directory or Hugging Face model id"}),
    ("--proposal-model", {"help": "rollout proposal checkpoint directory or model id"}),
    ("--model-revision", {}), ("--tokenizer", {}), ("--tokenizer-revision", {}),
    ("--allow-download", {"action": argparse.BooleanOptionalAction, "default": None}),
    ("--max-new-tokens", {"type": int}),
    ("--score-chunk-size", {"type": int}),
    ("--mh-iterations", {"type": int, "help": "full-length MH updates, independent of generation length"}),
    ("--sampling-scope", {"choices": ("full", "thinking")}),
    ("--thinking-mode", {"choices": ("auto", "enabled", "disabled")}),
    ("--thinking-format", {"choices": ("auto", "tags", "json", "xml")}),
    ("--thinking-path", {}), ("--content-path", {}),
    ("--thinking-start-text", {}), ("--thinking-end-text", {}),
    ("--starts-in-thinking", {"action": argparse.BooleanOptionalAction, "default": None}),
)


def add_model_output_arguments(parser: argparse.ArgumentParser, *, exclude: Collection[str] = ()) -> None:
    """Register shared options that are implemented by the calling entry point."""
    for flag, options in _ARGUMENTS:
        if flag not in exclude and flag not in parser._option_string_actions:
            parser.add_argument(flag, **options)


def model_output_cli_arguments(args: argparse.Namespace) -> list[str]:
    values = []
    for flag, _ in _ARGUMENTS:
        value = getattr(args, flag[2:].replace("-", "_"), None)
        if value is None:
            continue
        if isinstance(value, bool):
            values.append(flag if value else "--no-" + flag[2:])
        else:
            values.extend((flag, str(value)))
    return values


def apply_model_output_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    training = "model" in config and "models" not in config
    models = config.setdefault("model" if training else "models", {})
    for argument, role in (("model", "base"), ("proposal_model", "proposal")):
        value = getattr(args, argument, None)
        if value is None:
            continue
        if training and role != "base":
            continue
        previous = models.get(role)
        models[role] = str(value)
        for suffix in ("source", "revision", "weight_sha256", "modelscope_source"):
            models.pop(suffix if training else f"{role}_{suffix}", None)
        if not training and models.get("rl_base") == previous:
            models["rl_base"] = str(value)
        config.pop("_resolved_models", None)
        config.setdefault("model_loading", {}).setdefault(role, {}).pop("revision", None)
    for argument, name in (("model_revision", "revision"), ("tokenizer", "tokenizer_name_or_path"),
                           ("tokenizer_revision", "tokenizer_revision")):
        value = getattr(args, argument, None)
        if value is not None:
            config.setdefault("model_loading", {}).setdefault("base", {})[name] = value
    allow_download = getattr(args, "allow_download", None)
    if allow_download is not None:
        loading = config.setdefault("model_loading", {})
        loading["local_files_only"] = not allow_download
        for role in ("base", "proposal", "rl"):
            if role in loading:
                loading[role]["local_files_only"] = not allow_download
    for argument, section, name in (
        ("max_new_tokens", "generation", "max_new_tokens"),
        ("score_chunk_size", "runtime", "score_chunk_size"),
        ("mh_iterations", "mh", "iterations"),
        *((name, "output", name) for name in (
            "sampling_scope", "thinking_mode", "thinking_format", "thinking_path", "content_path",
            "thinking_start_text", "thinking_end_text", "starts_in_thinking",
        )),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            if argument in {"max_new_tokens", "score_chunk_size", "mh_iterations"} and value <= 0:
                raise ValueError(f"{argument} must be positive")
            config.setdefault(section, {})[name] = value


def require_full_scope(config: dict[str, Any], component: str) -> None:
    """Legacy full-output infra fixtures declare their scope before allocation."""
    if config.get("output", {}).get("sampling_scope", "full") != "full":
        raise ValueError(f"{component} measures full-output execution; use --sampling-scope full. "
                         "Thinking-only IS/MH is available in the quality and pass@k entry points.")
