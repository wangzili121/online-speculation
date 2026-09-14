"""Typed boundary between the Ascend runner and a shared-backbone Orthrus model."""

import os
from dataclasses import dataclass

import torch

from .phase_scheduler import StaggeredDraftAdmission
from .trace import instant, span, writer


@dataclass(frozen=True)
class OrthrusDraftBatch:
    token_ids: list[list[int]] | torch.Tensor
    probabilities: torch.Tensor
    accepted_token_counts: torch.Tensor | None = None
    next_token_ids: torch.Tensor | None = None
    history_lengths: list[int] | None = None


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
        self._accepted_token_counts = None
        self._next_token_ids = None
        self._async_history_lengths: dict[str, int] = {}
        phase_scheduler = os.environ.get("ORTHRUS_PHASE_SCHEDULER", "off")
        if phase_scheduler not in {"off", "staggered"}:
            raise ValueError(
                "ORTHRUS_PHASE_SCHEDULER must be off or staggered")
        self._draft_admission = None
        if phase_scheduler == "staggered":
            self._draft_admission = StaggeredDraftAdmission(
                max_batch=int(os.environ.get(
                    "ORTHRUS_DRAFT_MAX_BATCH", "8")),
                length_bucket=int(os.environ.get(
                    "ORTHRUS_DRAFT_LENGTH_BUCKET", "8192")),
                max_delay_steps=int(os.environ.get(
                    "ORTHRUS_DRAFT_MAX_DELAY_STEPS", "2")),
            )

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
        if torch.is_tensor(sampled_token_ids):
            active = torch.nonzero(
                (sampled_token_ids >= 0).any(dim=1), as_tuple=False
            ).flatten().tolist()
        else:
            active = [index for index, row in enumerate(sampled_token_ids) if row]
        accepted_token_counts = None
        next_token_ids = None
        history_lengths = None
        if torch.is_tensor(sampled_token_ids):
            accepted_token_counts = (sampled_token_ids >= 0).sum(dim=1)
            gather_indices = (accepted_token_counts - 1).clamp_min(0).long()
            next_token_ids = sampled_token_ids.gather(
                1, gather_indices[:, None]).squeeze(1)
            counts_cpu = accepted_token_counts.detach().cpu().tolist()
            history_lengths = []
            for index in active:
                request_id = input_batch.req_ids[index]
                previous = self._async_history_lengths.get(request_id)
                if previous is None:
                    exact = int(input_batch.num_tokens_no_spec[index]) - 1
                else:
                    exact = previous + int(counts_cpu[index])
                self._async_history_lengths[request_id] = exact
                history_lengths.append(exact)
        elif active and self._draft_admission is not None:
            history_lengths = [
                int(input_batch.num_tokens_no_spec[index]) - 1
                for index in active
            ]
        draft_indices = active
        deferred_indices: list[int] = []
        if self._draft_admission is not None and active:
            all_history_lengths = [0] * len(input_batch.req_ids)
            for index, length in zip(active, history_lengths or []):
                all_history_lengths[index] = length
            decision = self._draft_admission.select(
                list(input_batch.req_ids), all_history_lengths, active)
            draft_indices = decision.selected_indices
            deferred_indices = decision.deferred_indices
            length_by_index = dict(zip(active, history_lengths or []))
            history_lengths = [length_by_index[index] for index in draft_indices]
            instant("orthrus.phase.admission", {
                "draft_request_ids": [input_batch.req_ids[index]
                                      for index in draft_indices],
                "deferred_request_ids": [input_batch.req_ids[index]
                                         for index in deferred_indices],
                "draft_ready": len(active),
                "draft_admitted": len(draft_indices),
                "draft_deferred": len(deferred_indices),
                "dominant_length_bucket": decision.dominant_length_bucket,
                "max_queue_age": max(decision.queue_ages.values(), default=0),
            })
        history = getattr(input_batch, "num_tokens_no_spec", None)
        args = {
            "request_ids": [input_batch.req_ids[index] for index in active],
            "history_tokens": ([int(history[index]) for index in active]
                               if history is not None else []),
            "batch_requests": len(active),
            "draft_batch_requests": len(draft_indices),
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
                    accepted_token_counts=accepted_token_counts,
                    next_token_ids=next_token_ids,
                    history_lengths=history_lengths,
                    draft_request_indices=draft_indices,
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
        self._accepted_token_counts = result.accepted_token_counts
        self._next_token_ids = result.next_token_ids
        if writer() is not None:
            state_args = dict(args)
            state_args["draft_request_ids"] = [
                input_batch.req_ids[index] for index in draft_indices]
            state_args["deferred_request_ids"] = [
                input_batch.req_ids[index] for index in deferred_indices]
            if result.history_lengths is not None:
                state_args["draft_history_tokens"] = result.history_lengths
            if result.accepted_token_counts is not None:
                state_args["accepted_token_counts"] = (
                    result.accepted_token_counts.detach().cpu().tolist())
            if result.next_token_ids is not None:
                state_args["next_token_ids"] = (
                    result.next_token_ids.detach().cpu().tolist())
            instant("orthrus.draft.state", state_args)
        return result.token_ids

    def take_last_draft_probs(self):
        result, self._probabilities = self._probabilities, None
        return result

    def take_last_async_state(self):
        result = (self._next_token_ids, self._accepted_token_counts)
        self._next_token_ids = None
        self._accepted_token_counts = None
        return result
