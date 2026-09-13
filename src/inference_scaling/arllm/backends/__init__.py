"""Inference backends."""

from inference_scaling.arllm.backends.absorbing import AbsorbingEOSBackend
from inference_scaling.arllm.backends.batching import BatchingSnapshot, ContinuousBatchingBackend
from inference_scaling.arllm.backends.cache import ScoreCacheSnapshot, ScoreCachingBackend
from inference_scaling.arllm.backends.candidate_cache import CachedCandidateBackend
from inference_scaling.arllm.backends.loader import (
    BACKEND_CHOICES,
    close_backend,
    configured_backend,
    load_backend_from_config,
    set_backend_override,
)
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.backends.transformers_backend import (
    SequenceScoreStatistics,
    TransformersBackend,
    TransformersBackendSnapshot,
)
from inference_scaling.arllm.backends.vllm_backend import (
    AsyncVLLMBackend,
    VLLMBackend,
    VLLMBackendSnapshot,
)

__all__ = [
    "AbsorbingEOSBackend",
    "AsyncVLLMBackend",
    "BACKEND_CHOICES",
    "BatchingSnapshot",
    "CachedCandidateBackend",
    "ContinuousBatchingBackend",
    "ScoreCacheSnapshot",
    "ScoreCachingBackend",
    "SequenceScoreStatistics",
    "TabularAutoregressiveBackend",
    "TransformersBackend",
    "TransformersBackendSnapshot",
    "VLLMBackend",
    "VLLMBackendSnapshot",
    "close_backend",
    "configured_backend",
    "load_backend_from_config",
    "set_backend_override",
]
