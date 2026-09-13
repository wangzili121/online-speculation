"""Expose a fixed full-support sampling policy as the reference model law."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from inference_scaling.arllm.config import SamplingConfig


class ReferencePolicyBackend:
    """Outer temperature 1 denotes the configured reference temperature.

    This makes existing MH kernels target the same base policy as an IS run.
    Actual model log probabilities are still returned, with their policy and
    model identities translated into the outer reference-policy contract.
    """

    def __init__(self, backend, *, temperature: float) -> None:
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("reference temperature must be finite and positive")
        self.backend = backend
        self.temperature = temperature
        self.model_id = f"{backend.model_id}|reference-temperature={temperature}"

    def _policy(self, sampling):
        outer = sampling or SamplingConfig()
        return replace(outer, temperature=outer.temperature * self.temperature)

    def sample_batch(self, requests):
        inner_requests = [replace(request, sampling=self._policy(request.sampling)) for request in requests]
        samples = self.backend.sample_batch(inner_requests)
        if len(samples) != len(requests):
            raise RuntimeError("reference backend returned an invalid sample count")
        result = []
        for request, inner_request, sample in zip(requests, inner_requests, samples, strict=True):
            if sample.model_id != self.backend.model_id or sample.policy_id != inner_request.sampling.policy_id:
                raise RuntimeError("reference backend returned a different sampling policy")
            reference = self._policy(SamplingConfig(eos_token_id=request.sampling.eos_token_id))
            if sample.policy_id == reference.policy_id:
                reference_logs = sample.token_logprobs
            elif sample.reference_policy_id == reference.policy_id:
                reference_logs = sample.reference_token_logprobs
            else:
                reference_logs = None
            result.append(replace(
                sample, model_id=self.model_id, policy_id=request.sampling.policy_id,
                reference_token_logprobs=reference_logs,
                reference_policy_id=None if reference_logs is None else SamplingConfig(
                    eos_token_id=request.sampling.eos_token_id
                ).policy_id,
            ))
        return result

    def score_batch(self, requests):
        return self.backend.score_batch([
            replace(request, sampling=self._policy(request.sampling)) for request in requests
        ])


__all__ = ["ReferencePolicyBackend"]
