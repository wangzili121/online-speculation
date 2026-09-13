"""Bounded-memory, exact teacher forcing with the complete causal context."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any
from types import SimpleNamespace


def prefill_causal_model(model: Any, input_ids: Any, attention_mask: Any, position_ids: Any, *,
                         chunk_size: int, logits_to_keep: int, supports_logits_to_keep: bool) -> Any:
    """Build the full prefix KV state with bounded logits and attention queries."""
    import torch

    length = input_ids.shape[1]
    retained = min(logits_to_keep, length)
    cache = None
    tail = []
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        needed = max(0, end - max(start, length - retained))
        kwargs = {} if cache is None else {"past_key_values": cache}
        if supports_logits_to_keep:
            kwargs["logits_to_keep"] = max(1, needed)
        output = model(input_ids=input_ids[:, start:end], attention_mask=attention_mask[:, :end],
                       position_ids=position_ids[:, start:end], use_cache=True, return_dict=True, **kwargs)
        cache = getattr(output, "past_key_values", None)
        if cache is None and end < length:
            raise ValueError("long-prefix generation requires past_key_values")
        if needed:
            tail.append(output.logits[:, -needed:, :])
        del output
    return SimpleNamespace(logits=torch.cat(tail, dim=1), past_key_values=cache)


def iter_causal_logits(
    model: Any, prefix: Sequence[int], continuation: Sequence[int], *,
    device: Any, chunk_size: int, supports_logits_to_keep: bool,
    on_forward: Callable[[int], None],
) -> Iterator[tuple[int, Any]]:
    """Yield (continuation offset, predictor logits) without truncating context.

    The caller owns the model lock and inference-mode context. KV state is local
    to this iterator and released on completion/exception. A missing cache is an
    explicit capability error, not a shorter-context approximation.
    """
    import torch

    if not prefix or not continuation or chunk_size <= 0:
        raise ValueError("chunked scoring requires a prefix, continuation and positive chunk_size")
    inputs = tuple(prefix) + tuple(continuation[:-1])
    first_predictor = len(prefix) - 1
    past = None
    try:
        for start in range(0, len(inputs), chunk_size):
            end = min(start + chunk_size, len(inputs))
            kwargs: dict[str, Any] = {}
            if past is not None:
                kwargs["past_key_values"] = past
            needed = end - max(start, first_predictor)
            if supports_logits_to_keep:
                kwargs["logits_to_keep"] = max(1, needed)
            output = model(
                input_ids=torch.tensor([inputs[start:end]], dtype=torch.long, device=device),
                attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                position_ids=torch.arange(start, end, device=device)[None, :],
                use_cache=True, return_dict=True, **kwargs,
            )
            on_forward(end - start)
            past = getattr(output, "past_key_values", None)
            if past is None and end < len(inputs):
                raise ValueError("long-sequence scoring requires a model returning past_key_values")
            if needed > 0:
                if output.logits.shape[1] < needed:
                    raise RuntimeError("model omitted required teacher-forcing logits")
                yield max(0, start - first_predictor), output.logits[0, -needed:, :]
            del output
    finally:
        del past
