"""Candidate sampling and exact correction over the same probability policy."""

from ..sampling import (SamplingConfig, draw, greedy_tokens, probabilities, sample_logits,
                        verify_greedy, verify_linear)


class ProposalSampler:
    def __init__(self, config=SamplingConfig(), *, executor=None):
        self.config, self.executor = config, executor

    def sample_ar(self, logits, generator):
        if self.executor is not None:
            return self.executor.sample_ar(logits, generator)
        return int(sample_logits(logits, self.config, generator))

    def propose(self, logits, generator):
        if self.executor is not None:
            return self.executor.draft(logits, generator)
        if self.config.temperature == 0:
            return greedy_tokens(logits), None, None
        q = probabilities(logits, self.config)
        return draw(q, generator), q, None

    def verify(self, proposal, target_logits, generator):
        if self.executor is not None:
            return self.executor.verify(proposal.candidates, proposal.proposal, target_logits, generator)
        if self.config.temperature == 0:
            return verify_greedy(proposal.candidates, greedy_tokens(target_logits)), None
        target = probabilities(target_logits, self.config)
        return verify_linear(proposal.candidates, proposal.proposal, target, generator=generator), target
