"""Batch-one decoding: pending committed anchor, branch proposal, AR correction."""

from dataclasses import asdict, dataclass, field
import time

import torch

from ..sampling import SamplingConfig
from ..state import cache_length, trim_cache as crop_cache
from .sampling import ProposalSampler


@dataclass
class Generation:
    tokens: list[int] = field(default_factory=list)
    seconds: float = 0.0
    prefill_forwards: int = 0
    prefill_output_tokens: int = 0
    draft_forwards: int = 0
    verifier_forwards: int = 0
    tail_ar_forwards: int = 0
    accepted_per_round: list[int] = field(default_factory=list)
    proposed_per_round: list[int] = field(default_factory=list)
    rounds: int = 0
    fully_covered_rounds: int = 0
    updates: int = 0
    update_seconds: float = 0.0
    feedback_blocks: int = 0
    coverage_skips: int = 0
    prefix_tokens: int = 0
    prefix_seconds: float | None = None
    prefix_capture_seconds: float = 0.0

    @property
    def tps(self):
        return len(self.tokens) / self.seconds if self.seconds else 0.0

    @property
    def decode_forwards(self):
        return self.draft_forwards + self.verifier_forwards + self.tail_ar_forwards

    def summary(self):
        values = asdict(self)
        values["tokens"] = len(self.tokens)
        values["tps"] = self.tps
        values["decode_forwards"] = self.decode_forwards
        values["decode_tpf"] = ((len(self.tokens) - self.prefill_output_tokens) / self.decode_forwards
                                if self.decode_forwards else None)
        return values


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check(branch, prompt, budget, eos_id):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("nonempty batch-one prompt required")
    if budget < 0 or (eos_id is not None and not 0 <= eos_id < branch.model.config.vocab_size):
        raise ValueError("invalid output budget or EOS token")


def _append(result, tokens, budget, eos_id):
    for token in tokens:
        if len(result.tokens) == budget:
            return True
        result.tokens.append(token)
        if token == eos_id:
            return True
    return len(result.tokens) == budget


@torch.no_grad()
def generate_ar(branch, prompt, max_new_tokens, *, sampling=SamplingConfig(), eos_id=None,
                generator=None, sampler=None, prefill_output=True, prefix_tokens=None):
    _check(branch, prompt, max_new_tokens, eos_id)
    if prefix_tokens is not None and (type(prefix_tokens) is not int or prefix_tokens < 1):
        raise ValueError("positive integer AR prefix length required")
    result = Generation()
    if max_new_tokens == 0:
        return result
    _sync(prompt.device)
    start = time.perf_counter()
    sampler = ProposalSampler(sampling) if sampler is None else sampler
    cache, stopped, token = None, False, int(prompt[0, -1])

    def capture_prefix():
        marker = time.perf_counter()
        _sync(prompt.device)
        result.prefix_tokens = len(result.tokens)
        result.prefix_seconds = time.perf_counter() - start
        result.prefix_capture_seconds = time.perf_counter() - marker

    if prefill_output:
        logits, cache = branch.ar(prompt, logits_to_keep=1)
        token = sampler.sample_ar(logits[0, -1], generator)
        result.prefill_forwards = result.prefill_output_tokens = 1
        stopped = _append(result, [token], max_new_tokens, eos_id)
        if prefix_tokens == 1 and not stopped:
            capture_prefix()
    elif prompt.shape[1] > 1:
        _, cache = branch.ar(prompt[:, :-1], logits_to_keep=1)
        result.prefill_forwards = 1
    while not stopped:
        logits, cache = branch.ar(prompt.new_tensor([[token]]), cache=cache, logits_to_keep=1)
        token = sampler.sample_ar(logits[0, -1], generator)
        result.tail_ar_forwards += 1
        stopped = _append(result, [token], max_new_tokens, eos_id)
        if prefix_tokens is not None and len(result.tokens) == prefix_tokens and not stopped:
            capture_prefix()
    _sync(prompt.device)
    result.seconds = time.perf_counter() - start
    if prefix_tokens is not None and result.prefix_seconds is None:
        result.prefix_tokens, result.prefix_seconds = len(result.tokens), result.seconds
    result.rounds = len(result.tokens)
    return result


@torch.no_grad()
def generate(branch, prompt, max_new_tokens, *, block_size=None, sampling=SamplingConfig(),
             eos_id=None, generator=None, audit_cache=False, sampler=None, feedback=None, prefix_tokens=None):
    _check(branch, prompt, max_new_tokens, eos_id)
    if prefix_tokens is not None and (type(prefix_tokens) is not int or prefix_tokens != 1
                                      or not branch.initial_ar_token):
        raise ValueError("speculative prefix timing requires the first exact AR token")
    block_size = branch.default_block_size if block_size is None else block_size
    if block_size < 2:
        raise ValueError("parallel drafting requires block_size >= 2")
    result = Generation()
    if max_new_tokens == 0:
        if feedback is not None:
            feedback.begin(prompt)
            feedback.finish(result)
        return result
    _sync(prompt.device)
    start = time.perf_counter()
    sampler = ProposalSampler(sampling) if sampler is None else sampler
    if feedback is not None:
        feedback.begin(prompt)
    cache = None
    anchor = prompt[:, -1:]
    stopped = False
    if branch.initial_ar_token:
        logits, cache = branch.ar(prompt, logits_to_keep=1)
        token = sampler.sample_ar(logits[0, -1], generator)
        result.prefill_forwards = result.prefill_output_tokens = 1
        stopped = _append(result, [token], max_new_tokens, eos_id)
        if prefix_tokens is not None:
            marker = time.perf_counter()
            _sync(prompt.device)
            result.prefix_tokens = 1
            result.prefix_seconds = time.perf_counter() - start
            result.prefix_capture_seconds = time.perf_counter() - marker
        if feedback is not None:
            feedback.commit([token])
        anchor = prompt.new_tensor([[token]])
    elif prompt.shape[1] > 1:
        _, cache = branch.ar(prompt[:, :-1], logits_to_keep=1)
        result.prefill_forwards = 1
    while not stopped:
        result.rounds += 1
        expected_history = prompt.shape[1] + len(result.tokens) - 1
        if cache_length(cache) != expected_history:
            raise RuntimeError("history must end immediately before the committed anchor")
        remaining = max_new_tokens - len(result.tokens)
        if remaining == 1:
            logits, cache = branch.ar(anchor, cache=cache, logits_to_keep=1)
            token = sampler.sample_ar(logits[0, -1], generator)
            result.tail_ar_forwards += 1
            stopped = _append(result, [token], max_new_tokens, eos_id)
            if feedback is not None:
                feedback.commit([token])
            continue
        length = min(block_size, remaining + branch.input_budget_extra)
        proposal = branch.draft(anchor, cache, length, sampling, generator, sampler=sampler, feedback=feedback)
        result.draft_forwards += 1
        if eos_id is not None and eos_id in proposal.guaranteed:
            begin = len(result.tokens)
            _append(result, proposal.guaranteed, max_new_tokens, eos_id)
            if feedback is not None:
                feedback.commit(result.tokens[begin:])
            break
        logits, verified_cache = branch.ar(proposal.verifier_inputs, cache=proposal.cache)
        result.verifier_forwards += 1
        verified, target = sampler.verify(proposal, logits[0], generator)
        begin = len(result.tokens)
        stopped = _append(result, proposal.guaranteed + verified.tokens, max_new_tokens, eos_id)
        submitted = result.tokens[begin:]
        noise_outputs = max(0, len(submitted) - len(proposal.guaranteed))
        used, kept = min(verified.supervised, noise_outputs), min(verified.accepted, noise_outputs)
        result.accepted_per_round.append(kept)
        result.proposed_per_round.append(proposal.candidates.numel())
        fully_covered = kept == proposal.candidates.numel()
        result.fully_covered_rounds += fully_covered
        committed = prompt.shape[1] + len(result.tokens) - 1
        cache = crop_cache(verified_cache, committed)
        anchor = prompt.new_tensor([[result.tokens[-1]]])
        if audit_cache:
            full = torch.cat((prompt, prompt.new_tensor([result.tokens[:-1]])), 1)
            _, reference = branch.ar(full)
            for (key, value), (ref_key, ref_value) in zip(cache, reference, strict=True):
                torch.testing.assert_close(key, ref_key)
                torch.testing.assert_close(value, ref_value)
        if feedback is not None:
            feedback.commit(submitted)
            feedback.observe(proposal, logits[0], target, used=used, fully_covered=fully_covered, done=stopped)
    if feedback is not None:
        feedback.finish(result)
    _sync(prompt.device)
    result.seconds = time.perf_counter() - start
    return result
