"""EOS-only specialization of the shared stopped-sequence backend."""

from __future__ import annotations

from collections.abc import Sequence

from inference_scaling.arllm.backends.stopping import StoppedSequenceBackend
from inference_scaling.arllm.types import AutoregressiveBackend, GenerationRequest, SequenceSample


class AbsorbingEOSBackend(StoppedSequenceBackend):
    """Retain the fixed-length MH API and native EOS generation requests."""

    def __init__(
        self, backend: AutoregressiveBackend, eos_token_id: int, *, absorbing_after: int = 0
    ) -> None:
        super().__init__(
            backend, stop_token_sequences=(), eos_token_id=eos_token_id,
            protected_prefix_length=absorbing_after, generation_chunk_size=None,
        )
        self.absorbing_after = absorbing_after
        self.model_id = f"{backend.model_id}|absorbing-eos={eos_token_id};after={absorbing_after}"

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        if any(request.sampling.eos_token_id is not None for request in requests):
            raise ValueError(
                "the outer fixed-length policy must leave eos_token_id unset; "
                "the adapter supplies absorbing EOS semantics"
            )
        return super().sample_batch(requests)
