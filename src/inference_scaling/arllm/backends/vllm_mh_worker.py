"""vLLM 0.26 worker adapter for one-pass MH probability accounting.

The public vLLM output contains probabilities under the processed sampling
policy.  MH also needs the selected tokens' probabilities under the unmodified
model distribution.  This adapter gathers those scalars from the same logits
that produced each token and exposes them through ``LLM.collective_rpc``.

The module is imported only by an opted-in vLLM worker.  Keeping it separate
prevents the base package from importing the optional vLLM and torch stacks.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Sequence
from typing import Any

import torch
from packaging.version import Version
from vllm.v1.outputs import SamplerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as _GPUWorker


def _require_supported_vllm() -> None:
    installed = Version(importlib.metadata.version("vllm"))
    if not Version("0.26") <= installed < Version("0.27"):
        raise RuntimeError(
            "MHFusedLogprobWorker requires vLLM >=0.26,<0.27; "
            f"found {installed}"
        )


_require_supported_vllm()


class _MHFusedLogprobGPUModelRunner(_GPUModelRunner):
    """Gather only the chosen raw-model log-probability at each decode step."""

    def __init__(self, vllm_config: Any, device: Any) -> None:
        super().__init__(vllm_config, device)
        if self.use_async_scheduling:
            raise RuntimeError(
                "MH fused log-probabilities require vLLM async_scheduling=False"
            )
        if self.speculative_config is not None:
            raise RuntimeError(
                "MH fused log-probabilities do not yet support speculative decoding"
            )
        self._mh_step_reference: torch.Tensor | None = None
        self._mh_reference_by_request: dict[str, list[float]] = {}

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: Any | None,
    ) -> SamplerOutput:
        if spec_decode_metadata is not None:
            raise RuntimeError(
                "MH fused log-probabilities do not support speculative decoding"
            )
        sampling_metadata = self.input_batch.sampling_metadata
        needs_logprobs = (
            sampling_metadata.max_num_logprobs is not None
            or bool(sampling_metadata.logprob_token_ids)
        )
        if logits is None or not needs_logprobs:
            self._mh_step_reference = None
            return super()._sample(logits, spec_decode_metadata)

        # The sampling call may transform its logits in place.  Compute the
        # base distribution first, then retain only one scalar per request.
        raw_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
        output = super()._sample(logits, spec_decode_metadata)
        selected = output.sampled_token_ids.to(dtype=torch.int64)
        if selected.ndim != 2 or selected.shape[1] != 1:
            raise RuntimeError(
                "MH fused log-probabilities expected one sampled token per request"
            )
        self._mh_step_reference = raw_logprobs.gather(-1, selected)
        return output

    def _bookkeeping_sync(self, *args: Any, **kwargs: Any) -> Any:
        reference = self._mh_step_reference
        self._mh_step_reference = None
        result = super()._bookkeeping_sync(*args, **kwargs)
        if reference is None:
            return result

        # vLLM has already synchronized generated token IDs in the base method;
        # the additional device-to-host payload is one float per active request.
        values = reference.detach().to(dtype=torch.float32, device="cpu").tolist()
        valid_token_ids = result[2]
        request_ids = result[4]
        if len(values) > len(request_ids) or len(values) > len(valid_token_ids):
            raise RuntimeError("vLLM returned inconsistent MH bookkeeping shapes")
        for index, row in enumerate(values):
            sampled = valid_token_ids[index]
            if not sampled:
                continue
            if len(sampled) != 1 or len(row) != 1:
                raise RuntimeError(
                    "MH fused log-probabilities require one-token decode steps"
                )
            request_id = str(request_ids[index])
            self._mh_reference_by_request.setdefault(request_id, []).append(
                float(row[0])
            )
        return result

    def pop_mh_reference_logprobs(
        self, request_ids: Sequence[str]
    ) -> dict[str, tuple[float, ...]]:
        """Return and release completed request histories held by this worker."""

        result: dict[str, tuple[float, ...]] = {}
        for request_id in request_ids:
            values = self._mh_reference_by_request.pop(str(request_id), None)
            if values is not None:
                result[str(request_id)] = tuple(values)
        return result


class MHFusedLogprobWorker(_GPUWorker):
    """Install the MH runner without replacing vLLM's scheduler or outputs."""

    def init_device(self) -> None:
        if self.use_v2_model_runner:
            raise RuntimeError(
                "MH fused log-probabilities require VLLM_USE_V2_MODEL_RUNNER=0"
            )
        runner_module = importlib.import_module(
            "vllm.v1.worker.gpu_model_runner"
        )
        original = runner_module.GPUModelRunner
        runner_module.GPUModelRunner = _MHFusedLogprobGPUModelRunner
        try:
            super().init_device()
        finally:
            runner_module.GPUModelRunner = original
        if not isinstance(self.model_runner, _MHFusedLogprobGPUModelRunner):
            raise RuntimeError("failed to install the MH fused-logprob model runner")

    def pop_mh_reference_logprobs(
        self, request_ids: Sequence[str]
    ) -> dict[str, tuple[float, ...]]:
        runner = self.model_runner
        if not isinstance(runner, _MHFusedLogprobGPUModelRunner):
            raise RuntimeError("the MH fused-logprob model runner is unavailable")
        return runner.pop_mh_reference_logprobs(request_ids)
