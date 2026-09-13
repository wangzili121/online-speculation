"""vLLM custom proposer that makes the native suffix cache accept dynamic K.

vLLM 0.25--0.26's ``SuffixDecodingProposer`` owns the desired global response
cache and per-prompt suffix trees, but asserts that every runtime draft length
equals the fixed startup value.  Dynamic speculative scheduling deliberately
passes a different K as the active batch changes.  This adapter keeps the
upstream implementation and temporarily exposes the scheduler-selected K for
one serialized ``propose`` call.

The module does not import vLLM at import time, so the core package remains
usable when the optional vLLM dependency is absent.
"""

from __future__ import annotations

import threading
from typing import Any


class DynamicSuffixDecodingProposer:
    """Delegate to vLLM's suffix proposer while honoring its runtime K."""

    def __init__(self, vllm_config: Any) -> None:
        from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

        self._delegate = SuffixDecodingProposer(vllm_config)
        self._lock = threading.Lock()

    def propose(
        self,
        num_speculative_tokens: int,
        input_batch: Any,
        sampled_token_ids: list[list[int]],
        slot_mappings: Any = None,
    ) -> list[list[int]]:
        selected = int(num_speculative_tokens)
        if selected < 0:
            raise ValueError("num_speculative_tokens must be non-negative")
        with self._lock:
            configured = self._delegate.num_speculative_tokens
            self._delegate.num_speculative_tokens = selected
            try:
                return self._delegate.propose(
                    selected,
                    input_batch,
                    sampled_token_ids,
                    slot_mappings,
                )
            finally:
                self._delegate.num_speculative_tokens = configured

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate.load_model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)
