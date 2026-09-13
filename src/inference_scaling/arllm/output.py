"""Resolve thinking formats and mode from explicit settings and tokenizer metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inference_scaling.shared.output import ThinkingFormat, ThinkingParser
from inference_scaling.shared.structured_output import UnifiedOutputParser


_THINKING_MARKERS = (
    ("think", "<think>", "</think>"),
    ("thinking", "<thinking>", "</thinking>"),
    ("bracket_think", "[THINK]", "[/THINK]"),
    ("reasoning", "<reasoning>", "</reasoning>"),
)


def output_settings_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    options = dict(config.get("output", {}))
    template = config.get("prompt", {}).get("chat_template_kwargs", {})
    if options.get("thinking_mode", "auto") == "auto" and "enable_thinking" in template:
        enabled = template["enable_thinking"]
        if not isinstance(enabled, bool):
            raise ValueError("prompt.chat_template_kwargs.enable_thinking must be boolean")
        options["thinking_mode"] = "enabled" if enabled else "disabled"
    return options


def thinking_format_from_backend(
    backend: Any,
    settings: Mapping[str, Any] | None = None,
    *,
    required: bool = False,
) -> UnifiedOutputParser:
    """Unknown formats have an explicit full-sequence fallback.

    `required` remains a strict-validation option for callers checking a
    deployment. Reward and sampling defaults use the fallback instead.
    """
    options = settings or {}
    kind = str(options.get("thinking_format", "auto"))
    if kind not in {"auto", "tags", "json", "xml"}:
        raise ValueError("thinking_format must be auto, tags, json or xml")
    mode = options.get("thinking_mode", "auto")
    if mode not in {"auto", "enabled", "disabled"}:
        raise ValueError("thinking_mode must be auto, enabled or disabled")
    start_text = options.get("thinking_start_text")
    end_text = options.get("thinking_end_text")
    tokenizer = getattr(backend, "tokenizer", None)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocabulary = get_vocab() if get_vocab is not None else {}
    template = str(getattr(tokenizer, "chat_template", "") or "")
    configured = options.get("thinking_formats")
    if configured is not None and (start_text is not None or end_text is not None):
        raise ValueError("provide thinking_formats or one thinking_start/end_text pair")
    if configured is not None:
        if not isinstance(configured, (list, tuple)) or not configured:
            raise ValueError("thinking_formats must be a nonempty array of tables")
        definitions = list(configured)
    elif end_text is not None:
        definitions = [{
            "name": "configured", "start_text": start_text, "end_text": end_text,
            "starts_in_thinking": options.get("starts_in_thinking"),
        }]
    elif start_text is not None:
        raise ValueError("thinking_start_text requires thinking_end_text")
    else:
        definitions = [
            {"name": name, "start_text": start, "end_text": end,
             "starts_in_thinking": options.get("starts_in_thinking")}
            for name, start, end in _THINKING_MARKERS
            if (start in vocabulary and end in vocabulary) or (start in template and end in template)
        ]
    encode = getattr(backend, "encode", None) or getattr(tokenizer, "encode", None)
    formats = []
    for definition in definitions:
        if not isinstance(definition, Mapping):
            raise ValueError("each thinking format must be a table")
        start, end = definition.get("start_text"), definition.get("end_text")
        if not isinstance(end, str) or not end:
            raise ValueError("thinking end_text must be a nonempty string")
        if start is not None and (not isinstance(start, str) or not start):
            raise ValueError("thinking start_text must be a nonempty string")
        if encode is None:
            raise ValueError("resolving thinking markers requires a tokenizer encoder")
        starts = definition.get("starts_in_thinking", options.get("starts_in_thinking"))
        formats.append(ThinkingFormat(
            end_token_ids=tuple(encode(end, add_special_tokens=False)),
            start_token_ids=None if start is None else tuple(encode(start, add_special_tokens=False)),
            starts_in_thinking=starts, name=str(definition.get("name", "configured")),
        ))
    if required and not formats:
        raise ValueError("thinking requires a configured format or recognized tokenizer delimiters")

    tokenizer_decode = getattr(tokenizer, "decode", None)
    backend_decode = getattr(backend, "decode", None)

    def decode(tokens):
        if tokenizer_decode is not None:
            return str(tokenizer_decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        if backend_decode is None:
            raise ValueError("structured output parsing requires a token decoder")
        return str(backend_decode(tokens, skip_special_tokens=False))

    has_decoder = tokenizer_decode is not None or backend_decode is not None

    def is_blank(tokens):
        if not tokens:
            return True
        return decode(tokens).strip() == ""

    markers = ThinkingParser(
        tuple(formats), mode="disabled" if mode == "disabled" else "enabled" if mode == "enabled" else "auto",
        is_blank=is_blank if has_decoder else None,
    )

    def path_option(name):
        value = options.get(name)
        if value is None:
            return None
        parts = value.split(".") if isinstance(value, str) else value
        if not isinstance(parts, (list, tuple)) or not parts or any(not isinstance(part, str) or not part for part in parts):
            raise ValueError(f"{name} must be a dotted path or nonempty array of field names")
        return tuple(parts)

    return UnifiedOutputParser(
        markers, decode if has_decoder else None, kind,
        path_option("thinking_path"), path_option("content_path"),
    )


__all__ = ["thinking_format_from_backend", "output_settings_from_config"]
