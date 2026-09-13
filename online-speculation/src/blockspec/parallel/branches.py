"""Draft policies supply verifier inputs and clean history to a shared loop."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..state import Cache
from ..sampling import SamplingConfig
from .sampling import ProposalSampler


@dataclass
class DraftBatch:
    verifier_inputs: Tensor
    candidates: Tensor
    proposal: Tensor | None
    cache: Cache | None
    guaranteed: list[int]
    draft_inputs: Tensor | None = None
    draft_cache: Cache | None = None
    boundary: object = None
    auxiliary_feedback: object = None
    collect_feedback: bool = False


class MaskedAttentionBranch:
    """The committed anchor is reprocessed with a bidirectional masked block."""

    name = "masked_attention"
    initial_ar_token = True
    input_budget_extra = 1

    def __init__(self, model):
        self.model = model
        self.default_block_size = model.config.block_size

    def ar(self, tokens, cache=None, logits_to_keep=0):
        result = self.model(tokens, cache=cache, logits_to_keep=logits_to_keep)
        return result.logits, result.cache

    def draft(self, anchor, cache, block_size, sampling=SamplingConfig(), generator=None, *,
              sampler=None, feedback=None):
        sampler = ProposalSampler(sampling) if sampler is None else sampler
        inputs = anchor.new_full((1, block_size), self.model.config.mask_token_id)
        inputs[:, :1] = anchor
        capture = None if feedback is None else feedback.capture_layer
        output = self.model(inputs, view="draft", cache=cache, capture_layer=capture)
        candidates, q, auxiliary = sampler.propose(output.logits[0, :-1], generator)
        return DraftBatch(torch.cat((anchor, candidates[None]), 1), candidates, q, cache, [], inputs, cache,
                          output.boundary, auxiliary, feedback is not None and feedback.learner is not None
                          and feedback.learner.needs_decoder_feedback)
