"""Typed boundary between the Ascend runner and a shared-backbone Orthrus model."""

from dataclasses import dataclass

import torch

from .trace import instant, span


@dataclass(frozen=True)
class OrthrusDraftBatch:
    token_ids: list[list[int]] | torch.Tensor
    probabilities: torch.Tensor


class OrthrusProposer:
    """Adapt a vLLM model's Orthrus draft view to speculative scheduling.

    The compatibility patch binds the live model runner. The model itself must
    expose ``orthrus_propose`` so drafting and verification can share one
    physical backbone; loading a second full model is intentionally rejected.
    """

    def __init__(self, vllm_config):
        config = vllm_config.speculative_config
        if config is None or config.num_speculative_tokens != 31:
            raise ValueError("Orthrus K32 requires exactly 31 speculative candidates")
        if getattr(config, "draft_sample_method", None) != "probabilistic":
            raise ValueError("Orthrus exact correction requires probabilistic draft sampling")
        self.runner = None
        self._probabilities = None

    def bind_runner(self, runner):
        # Ascend creates the proposer before load_model() assigns runner.model.
        # Keep only the runner reference here and validate the shared model on
        # the first proposal, after model loading has completed.
        self.runner = runner

    def load_model(self, target_model):
        """Validate the already-loaded target; Orthrus has no second model."""
        if not callable(getattr(target_model, "orthrus_propose", None)):
            raise TypeError("the target model must implement shared-backbone orthrus_propose")

    def dummy_run(self, *args, **kwargs):
        """No second-model graph is profiled by the eager compatibility path."""
        del args, kwargs

    @staticmethod
    def _resolve_model(runner):
        model = runner.get_model()
        if not callable(getattr(model, "orthrus_propose", None)):
            raise TypeError("the target model must implement shared-backbone orthrus_propose")
        return model

    def propose(self, *, sampled_token_ids, input_batch, scheduler_output,
                common_attn_metadata, runner):
        if self.runner is not runner:
            self.bind_runner(runner)
        active = [index for index, row in enumerate(sampled_token_ids) if row]
        history = getattr(input_batch, "num_tokens_no_spec", None)
        args = {
            "request_ids": [input_batch.req_ids[index] for index in active],
            "history_tokens": ([int(history[index]) for index in active]
                               if history is not None else []),
            "batch_requests": len(active),
            "scheduled_tokens": int(getattr(
                scheduler_output, "total_num_scheduled_tokens", 0)),
        }
        instant("orthrus.verify.complete", args)
        with span("orthrus.draft.batch", args):
            with torch.profiler.record_function("orthrus.draft.shared_backbone"):
                result = self._resolve_model(runner).orthrus_propose(
                    sampled_token_ids=sampled_token_ids,
                    input_batch=input_batch,
                    scheduler_output=scheduler_output,
                    common_attn_metadata=common_attn_metadata,
                )
        instant("orthrus.verify.ready", args)
        if not isinstance(result, OrthrusDraftBatch):
            raise TypeError("orthrus_propose must return OrthrusDraftBatch")
        count = len(input_batch.req_ids)
        probabilities = result.probabilities
        if (probabilities.ndim != 3 or probabilities.shape[0] != count
                or probabilities.shape[1] != 31 or not probabilities.is_floating_point()):
            raise ValueError("draft probabilities must have shape [requests,31,vocabulary]")
        if isinstance(result.token_ids, torch.Tensor):
            if result.token_ids.shape != probabilities.shape[:2]:
                raise ValueError("tensor draft ids must have shape [requests,31]")
        elif (len(result.token_ids) != count
              or any(len(row) > 31 for row in result.token_ids)):
            raise ValueError("list draft ids must contain at most 31 tokens per request")
        self._probabilities = probabilities
        return result.token_ids

    def take_last_draft_probs(self):
        result, self._probabilities = self._probabilities, None
        return result
