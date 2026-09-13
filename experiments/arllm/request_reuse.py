"""Reuse deterministic experiment requests while charging their cold-run cost.

This is an experiment-only view: snapshot() includes the forward work a method
would perform independently. The underlying backend retains physical counters.
It is not a deployment speedup measurement or a zero-cost rollout proposal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace


class ColdCostRequestReplay:
    """Memoize batch-one, request-seeded ordinary decoding on one fixed model."""

    def __init__(self, backend, *, prefetch_limits=None):
        if getattr(backend, "_speculation", None) is not None:
            raise ValueError("request replay requires ordinary, non-speculative decoding")
        self.backend = backend
        self.model_id = backend.model_id
        self.tokenizer = backend.tokenizer
        self.parameter_count = backend.parameter_count
        self._cache = {}
        self._offset: Counter[str] = Counter()
        self.cache_hits = 0
        self.prefetch_limits = dict(prefetch_limits or {})
        if any(not isinstance(limit, int) or limit <= 0 for limit in self.prefetch_limits.values()):
            raise ValueError("prefetch limits must be positive integers")

    @staticmethod
    def _key(request):
        return (request.prefix, request.sampling, request.seed, request.uniforms, request.arithmetic_uniform)

    def _view(self, request, sample, cost):
        length = min(request.max_new_tokens, len(sample.token_ids))
        removed = len(sample.token_ids) - length
        cost = dict(cost)
        if removed:
            # Ordinary batch-one decoding executes P + G - 1 positions. A
            # shorter maximum removes exactly one forward per removed token.
            for name in ("generation_forward_token_slots", "generated_tokens"):
                if cost[name] < removed:
                    raise ValueError("prefetch requires complete batch-one generation counters")
                cost[name] -= removed
            cost["estimated_dense_forward_flops"] -= 2 * self.parameter_count * removed
        reference = sample.reference_token_logprobs
        return replace(sample, token_ids=sample.token_ids[:length], token_logprobs=sample.token_logprobs[:length],
            reference_token_logprobs=None if reference is None else reference[:length],
            finish_reason="length" if removed else sample.finish_reason, request_id=request.request_id), cost

    def sample_batch(self, requests):
        if len(requests) != 1:
            raise ValueError("cold-cost replay is defined for batch-one experiment requests")
        request = requests[0]
        key = self._key(request)
        allow_prefix_view = (request.prefix in self.prefetch_limits
                             and request.uniforms is None and request.arithmetic_uniform is None)
        for maximum, sample, cost in self._cache.get(key, ()):
            compatible = maximum == request.max_new_tokens or (allow_prefix_view and maximum > request.max_new_tokens) or (
                sample.finish_reason == "eos" and len(sample.token_ids) <= request.max_new_tokens
            )
            if compatible:
                sample, cost = self._view(request, sample, cost)
                self.cache_hits += 1
                self._offset.update(cost)
                return [sample]
        maximum = request.max_new_tokens
        if allow_prefix_view:
            maximum = max(maximum, self.prefetch_limits.get(request.prefix, maximum))
        generated_request = replace(request, max_new_tokens=maximum)
        before = asdict(self.backend.snapshot())
        sample = self.backend.sample_batch([generated_request])[0]
        after = asdict(self.backend.snapshot())
        cost = {key: after[key] - before[key] for key in before}
        self._cache.setdefault(key, []).append((maximum, sample, cost))
        sample, charged = self._view(request, sample, cost)
        # Prefetched suffixes are experimental work, not part of the shorter
        # method's independent cost. A later cache hit charges its own full view.
        self._offset.update({name: charged[name] - cost[name] for name in cost})
        return [sample]

    def score_batch(self, requests):
        return self.backend.score_batch(requests)

    def score_statistics_batch(self, requests, **kwargs):
        return self.backend.score_statistics_batch(requests, **kwargs)

    def snapshot(self):
        current = self.backend.snapshot()
        return replace(current, **{key: getattr(current, key) + value for key, value in self._offset.items()})

    def encode(self, text, **kwargs):
        return self.backend.encode(text, **kwargs)

    def decode(self, tokens, **kwargs):
        return self.backend.decode(tokens, **kwargs)
