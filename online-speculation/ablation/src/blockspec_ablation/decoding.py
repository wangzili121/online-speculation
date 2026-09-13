"""Compatibility entry points over the shared branch-driven generation engine."""

from dataclasses import dataclass, field

from .model import trim_cache
from .diffusion import UniformNoise
from blockspec.sampling import SamplingConfig
from .parallel.branches import CausalLowRankBranch
from .parallel.feedback import OnlineFeedback
from blockspec.parallel.generation import generate as generate_parallel, generate_ar as generate_parallel_ar
from .parallel.sampling import ProposalSampler


@dataclass
class Generation:
    tokens: list[int]
    seconds: float
    decode_forwards: int
    rounds: int
    accepted: int = 0
    proposed: int = 0
    updates: int = 0
    update_seconds: float = 0.0
    accepted_per_round: list[int] = field(default_factory=list)
    feedback_blocks: int = 0
    fully_covered_rounds: int = 0
    coverage_skips: int = 0

    @property
    def tps(self):
        return len(self.tokens) / self.seconds if self.seconds else 0.0

    def summary(self):
        return {"tokens": len(self.tokens), "seconds": self.seconds, "tps": self.tps,
                "decode_forwards": self.decode_forwards, "rounds": self.rounds,
                "accepted": self.accepted, "proposed": self.proposed,
                "updates": self.updates, "update_seconds": self.update_seconds,
                "feedback_blocks": self.feedback_blocks,
                "fully_covered_rounds": self.fully_covered_rounds, "coverage_skips": self.coverage_skips}


def _check(model, prompt, max_new_tokens, eos_id):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("a nonempty batch-one token prompt is required")
    if max_new_tokens < 0 or (eos_id is not None and not 0 <= eos_id < model.config.vocab_size):
        raise ValueError("invalid output budget or EOS token")


def _inference_forward(model, executor):
    if executor is None:
        return model
    executor.validate(model)
    return executor._forward


def _prefill(forward, prompt):
    if prompt.shape[1] == 1:
        return None
    _, cache = forward(prompt[:, :-1], return_cache=True)
    return trim_cache(cache, prompt.shape[1] - 1)


def _legacy_result(result):
    return Generation(result.tokens, result.seconds, result.decode_forwards, result.rounds,
                      sum(result.accepted_per_round), sum(result.proposed_per_round),
                      result.updates, result.update_seconds, result.accepted_per_round,
                      result.feedback_blocks, result.fully_covered_rounds, result.coverage_skips)


def generate_ar(model, prompt, max_new_tokens, *, sampling=SamplingConfig(), eos_id=None,
                generator=None, executor=None, sampler_executor=None):
    _check(model, prompt, max_new_tokens, eos_id)
    if sampler_executor is not None:
        sampler_executor.validate(model, sampling)
    branch = CausalLowRankBranch(model, executor=executor, initial_ar_token=False)
    sampler = ProposalSampler(sampling, executor=sampler_executor)
    return _legacy_result(generate_parallel_ar(branch, prompt, max_new_tokens, sampling=sampling,
                          eos_id=eos_id, generator=generator, sampler=sampler, prefill_output=False))


def generate_speculative(model, prompt, max_new_tokens, *, block_size=8,
                         sampling=SamplingConfig(), eos_id=None, generator=None,
                         learner=None, executor=None, noise=UniformNoise(), calibrator=None,
                         sampler_executor=None):
    """Preserve the legacy call/initialization contract through the common engine.

    Replay learners skip end-of-request updates. Sparse calibration accumulators
    persist across requests; each update consumes the original proposal version.
    Continuation mixtures condition proposal rows on the sampled copy prefix.
    """
    _check(model, prompt, max_new_tokens, eos_id)
    if sampler_executor is not None:
        sampler_executor.validate(model, sampling, block_size, calibrator)
    if block_size < 2:
        raise ValueError("speculative blocks require B>=2; use generate_ar for B=1")
    if learner is not None and learner.model is not model:
        raise ValueError("learner and decoder must share the same model")
    if calibrator is not None and (learner is not None or sampling.temperature <= 0
                                   or sampling.top_k != calibrator.top_k
                                   or block_size != calibrator.block_size):
        raise ValueError("calibration requires matched positive-temperature top-k sampling and a frozen adapter")
    branch = CausalLowRankBranch(model, executor=executor, noise=noise, initial_ar_token=False)
    sampler = ProposalSampler(sampling, executor=sampler_executor, calibrator=calibrator)
    feedback = OnlineFeedback(learner=learner, calibrator=calibrator)
    return _legacy_result(generate_parallel(branch, prompt, max_new_tokens, block_size=block_size,
                          sampling=sampling, eos_id=eos_id, generator=generator,
                          sampler=sampler, feedback=feedback))
