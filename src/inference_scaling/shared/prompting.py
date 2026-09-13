"""Chat-template and plain-text prompt rendering without model-name rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def render_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]], config: Mapping[str, Any]) -> str:
    prompt = config.get("prompt", {})
    format_ = prompt.get("format", "auto")
    if format_ not in {"auto", "chat", "plain"}:
        raise ValueError("prompt.format must be auto, chat or plain")
    template = getattr(tokenizer, "chat_template", None)
    # Test/custom tokenizers may implement a template without exposing its text.
    available = bool(template) or (not hasattr(tokenizer, "chat_template") and callable(getattr(tokenizer, "apply_chat_template", None)))
    if format_ == "plain" or (format_ == "auto" and not available):
        if len(messages) == 1 and messages[0]["role"] == "user":
            return messages[0]["content"]
        return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\n\nassistant:"
    if not available:
        raise ValueError("prompt.format=chat requires a tokenizer chat template")
    kwargs = dict(prompt.get("chat_template_kwargs", {}))
    if {"tokenize", "add_generation_prompt", "return_tensors"} & kwargs.keys():
        raise ValueError("chat_template_kwargs must preserve text rendering and the generation prompt")
    mode = config.get("output", {}).get("thinking_mode", "auto")
    if mode != "auto" and "enable_thinking" in str(template):
        expected = mode == "enabled"
        if "enable_thinking" in kwargs and kwargs["enable_thinking"] != expected:
            raise ValueError("thinking_mode conflicts with chat_template_kwargs.enable_thinking")
        kwargs["enable_thinking"] = expected
    return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs))
