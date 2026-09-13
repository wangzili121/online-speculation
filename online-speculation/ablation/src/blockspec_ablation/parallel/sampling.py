"""Proposal transforms and verification execution, independent of backbone layout."""

from blockspec.sampling import (SamplingConfig, draw, greedy_tokens, probabilities, sample_logits,
                        verify_greedy, verify_linear)


class ProposalSampler:
    def __init__(self, config=SamplingConfig(), *, executor=None, calibrator=None):
        self.config = config
        self.protected_rows = 1
        self.executor = executor
        self.calibrator = calibrator
        self.continuation = getattr(calibrator, "kind", None) == "continuation"

    def sample_ar(self, logits, generator):
        if self.executor is not None:
            return self.executor.sample_ar(logits, generator)
        return int(sample_logits(logits, self.config, generator))

    def propose(self, logits, generator, *, protected_rows=None):
        protected_rows = self.protected_rows if protected_rows is None else protected_rows
        if self.calibrator is not None and getattr(self.calibrator, "protected_rows", 1) != protected_rows:
            raise ValueError("proposal calibration must match the branch's protected-row layout")
        if self.executor is not None and self.calibrator is not None and self.executor.protected_rows != protected_rows:
            raise ValueError("prepared calibration must match the branch's protected-row layout")
        if self.executor is not None:
            return self.executor.draft(logits, generator, self.calibrator)
        if self.config.temperature == 0:
            return greedy_tokens(logits), None, None
        q = probabilities(logits, self.config)
        if self.continuation:
            return self.calibrator.draft(q, generator)
        feedback = None
        if self.calibrator is not None:
            q, feedback = self.calibrator.propose(q)
        return draw(q, generator), q, feedback

    def verify(self, proposal, target_logits, generator):
        if self.executor is not None:
            return self.executor.verify(proposal.candidates, proposal.proposal, target_logits, generator)
        if self.config.temperature == 0:
            return verify_greedy(proposal.candidates, greedy_tokens(target_logits)), None
        target = probabilities(target_logits, self.config)
        return verify_linear(proposal.candidates, proposal.proposal, target, generator=generator), target
