"""Optional mixed proposal maps over the mainline exact tensor correction."""

from dataclasses import fields
import time

import torch

from blockspec.sampling import SamplingConfig, Verification, _probabilities_unchecked
from blockspec.sampling_execution import (exponential_choice as exponential_choice,
                                         linear_correction as linear_correction, _Slot, _VerifySlot)
from .calibration import mix_rows
from .continuation import CopyFeedback, copy_mixture


class _DraftSlot(_Slot):
    @torch.no_grad()
    def __init__(self, length, vocabulary, sampling, device, dtype, temperatures, use_cuda_graph, protected_rows=1):
        self.sampling, self.temperatures = sampling, temperatures
        self.protected_rows = protected_rows
        self.logits = torch.zeros(length, vocabulary, device=device, dtype=dtype)
        self.exponential = torch.ones_like(self.logits)
        self.weights = self.logits.new_zeros(length - protected_rows, len(temperatures))
        if temperatures:
            self.identity = temperatures.index(1.)
            self.weights[:, self.identity] = 1
            self.powers = self.logits.new_tensor(temperatures).reciprocal()[None, :, None]
        self.capture(use_cuda_graph)

    def run(self):
        self.q = _probabilities_unchecked(self.logits, self.sampling)
        self.feedback = None
        if self.temperatures:
            self.q, self.feedback = mix_rows(self.q, self.weights, self.powers, self.identity,
                                            self.sampling.top_k, self.protected_rows)
        self.tokens = exponential_choice(self.q, self.exponential)
        self.valid = (torch.isfinite(self.logits).all() & torch.isfinite(self.q).all() & (self.q >= 0).all()
                      & torch.isclose(self.q.sum(-1), torch.ones_like(self.q[:, 0]), atol=1e-6, rtol=1e-6).all())


class _CopySlot(_Slot):
    @torch.no_grad()
    def __init__(self, length, vocabulary, sampling, device, dtype, use_cuda_graph):
        self.sampling = sampling
        self.logits = torch.zeros(length, vocabulary, device=device, dtype=dtype)
        self.exponential = torch.ones_like(self.logits)
        self.weights = self.logits.new_zeros(length - 1)
        self.copied = torch.full((length,), -1, device=device, dtype=torch.long)
        self.capture(use_cuda_graph)

    def run(self):
        baseline = _probabilities_unchecked(self.logits, self.sampling)
        self.tokens, self.q, self.feedback = copy_mixture(baseline, self.weights, self.copied, self.exponential)
        self.valid = (torch.isfinite(self.logits).all() & torch.isfinite(self.q).all() & (self.q >= 0).all()
                      & torch.isclose(self.q.sum(-1), torch.ones_like(self.q[:, 0]), atol=1e-6, rtol=1e-6).all())


class SamplingExecutor:
    """Prepared sampling maps for one vocabulary, precision and sampling contract."""

    @torch.no_grad()
    def __init__(self, vocabulary, block_size, sampling=SamplingConfig(), *, device="cuda",
                 dtype=torch.float32, temperatures=(.5, .75, 1., 1.25, 1.5), use_cuda_graph=True,
                 continuation=False, protected_rows=1):
        if (type(vocabulary) is not int or vocabulary < 1 or type(block_size) is not int or block_size < 2
                or sampling.temperature <= 0 or dtype not in (torch.float32, torch.float64)
                or (temperatures and (sampling.top_k < 1 or 1. not in temperatures))
                or type(protected_rows) is not int or protected_rows not in (0, 1)
                or (continuation and protected_rows != 1)):
            raise ValueError("positive-temperature floating probabilities and a compatible mixing support required")
        self.vocabulary, self.block_size, self.sampling = vocabulary, block_size, sampling
        self.device, self.dtype = torch.device(device), dtype
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if use_cuda_graph and self.device.type != "cuda":
            raise ValueError("CUDA device required for graph capture")
        self.temperatures = tuple(temperatures)
        self.protected_rows = protected_rows
        self.continuation = continuation
        self.drafts, self.verifiers, self.signature_seconds = {}, {}, {}
        for n in range(1, block_size + 1):
            for mixed in ([False, True] if n > protected_rows and temperatures else [False]):
                start = time.perf_counter()
                self.drafts[n, mixed] = _DraftSlot(n, vocabulary, sampling, self.device, dtype,
                                                 self.temperatures if mixed else (), use_cuda_graph, protected_rows)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self.signature_seconds[n, "mixed" if mixed else "plain"] = time.perf_counter() - start
            if n > 1:
                if continuation:
                    start = time.perf_counter()
                    self.drafts[n, "continuation"] = _CopySlot(n, vocabulary, sampling, self.device, dtype, use_cuda_graph)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    self.signature_seconds[n, "continuation"] = time.perf_counter() - start
                start = time.perf_counter()
                self.verifiers[n] = _VerifySlot(n, vocabulary, sampling, self.device, dtype, use_cuda_graph)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self.signature_seconds[n, "verify"] = time.perf_counter() - start
        self.setup_seconds = sum(self.signature_seconds.values())

    def validate(self, model, sampling, block_size=None, calibrator=None):
        if model.config.vocab_size != self.vocabulary or sampling != self.sampling:
            raise ValueError("sampling executor vocabulary and probability policy must match")
        if block_size is not None and block_size != self.block_size:
            raise ValueError("sampling executor block size must match")
        if calibrator is not None and (((getattr(calibrator, "kind", None) == "continuation" and not self.continuation)
                                       or (getattr(calibrator, "kind", None) != "continuation"
                                           and calibrator.temperatures != self.temperatures))
                                      or calibrator.weights.device != self.device
                                      or calibrator.weights.dtype != self.dtype
                                      or getattr(calibrator, "protected_rows", 1) != self.protected_rows):
            raise ValueError("calibrator temperatures, device and dtype must match prepared maps")

    def _shape(self, logits):
        if (logits.ndim != 2 or logits.shape[-1] != self.vocabulary or logits.device != self.device
                or not 1 <= len(logits) <= self.block_size or torch.is_grad_enabled()):
            raise ValueError("prepared inference-only probability rows required")

    def _draft(self, logits, generator, calibrator):
        self._shape(logits)
        key = calibrator is not None
        if getattr(calibrator, "kind", None) == "continuation":
            copied = calibrator.lookup(len(logits))
            key = "continuation" if copied else False
        slot = self.drafts[len(logits), key]
        slot.logits.copy_(logits)
        slot.exponential.exponential_(generator=generator)
        if key == "continuation":
            slot.weights.copy_(calibrator.weights[calibrator.group, :len(logits) - 1])
            slot.copied.copy_(torch.tensor(copied + [-1] * (len(logits) - len(copied)), device=self.device))
        elif key:
            slot.weights.copy_(calibrator.weights[:len(slot.weights)])
        slot.replay()
        return slot

    def sample_ar(self, logits, generator=None):
        slot = self._draft(logits.reshape(1, -1), generator, None)
        token, valid = torch.stack((slot.tokens[0], slot.valid.long())).tolist()
        if not valid:
            raise ValueError("finite normalized AR probabilities required")
        return token

    def draft(self, logits, generator=None, calibrator=None):
        slot = self._draft(logits, generator, calibrator)
        if not slot.valid:
            raise ValueError("finite normalized proposal probabilities required")
        feedback = (type(slot.feedback)(**{field.name: (getattr(slot.feedback, field.name).clone()
                                                      if isinstance(getattr(slot.feedback, field.name), torch.Tensor)
                                                      else getattr(slot.feedback, field.name)) for field in fields(slot.feedback)})
                    if slot.feedback is not None else None)
        if isinstance(feedback, CopyFeedback):
            feedback.group = calibrator.group
        return slot.tokens.clone(), slot.q.clone(), feedback

    def verify(self, tokens, q, logits, generator=None):
        self._shape(logits)
        if q.shape != (len(logits) - 1, self.vocabulary) or tokens.shape != (len(q),):
            raise ValueError("proposal and verifier block shapes must agree")
        slot = self.verifiers[len(logits)]
        slot.logits.copy_(logits)
        slot.q.copy_(q)
        slot.tokens.copy_(tokens)
        slot.uniforms.uniform_(generator=generator)
        slot.exponential.exponential_(generator=generator)
        slot.replay()
        values = slot.payload.tolist()
        count, tail, valid = values[-3:]
        if not valid:
            raise ValueError("finite target, valid saved proposal and positive correction mass required")
        n = len(q)
        return (Verification(values[:count] + [tail], count, count if count < n else None, min(count + 1, n)),
                slot.p.clone())
