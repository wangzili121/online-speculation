"""Greedy block correction and transactional restoration, independent of kernels."""

from dataclasses import dataclass


def host_greedy_tokens(logits, *, name="PyPTO draft"):
    """CPU BF16/FP32 logits：保留非有限值检查，直接归约得到每行 token。"""
    low, high = logits.aminmax()
    if not bool(low.isfinite() & high.isfinite()):
        raise FloatingPointError(f"nonfinite {name} logits")
    return logits.max(dim=-1).indices.tolist()


@dataclass(frozen=True)
class Correction:
    tokens: tuple[int, ...]
    committed_inputs: int
    accepted: int


def correct(candidates, target_tokens, *, budget, stop_tokens=()):
    candidates, target_tokens = tuple(candidates), tuple(target_tokens)
    if (not candidates or len(target_tokens) != len(candidates) + 1
            or type(budget) is not int or budget < 1):
        raise ValueError("one target prediction per candidate plus a bonus and positive budget required")
    if any(type(token) is not int or token < 0 for token in (*candidates, *target_tokens)):
        raise ValueError("candidate and target token IDs must be nonnegative integers")
    emitted = []
    accepted = 0
    for candidate, target in zip(candidates, target_tokens):
        token = target
        if candidate == target:
            accepted += 1
        emitted.append(token)
        if candidate != target or token in stop_tokens or len(emitted) == budget:
            break
    else:
        if len(emitted) < budget:
            emitted.append(target_tokens[-1])
    # The final emitted token remains the next anchor; it has not entered AR yet.
    return Correction(tuple(emitted), len(emitted), accepted)


def verify_transaction(target, anchor, candidates, *, budget, stop_tokens=()):
    """A target implements snapshot/forward/restore; draft state is never committed."""
    inputs = (anchor, *candidates)
    snapshot = target.snapshot()
    try:
        predictions = target.forward(inputs)
        result = correct(candidates, predictions, budget=budget, stop_tokens=stop_tokens)
        if result.committed_inputs < len(inputs):
            target.restore(snapshot)
            target.forward(inputs[:result.committed_inputs])
        return result
    except BaseException:
        target.restore(snapshot)
        raise
