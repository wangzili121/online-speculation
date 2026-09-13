"""Fixed-shape probability transforms, exponential draws and block correction.

AR, static drafting and adaptive drafting use the same tensor probability map.
Independent random variables enter from the caller's generator before replay.
Graph outputs own their storage; updates are published between decode rounds.
"""

import time

import torch

from .sampling import SamplingConfig, Verification, _probabilities_unchecked


def exponential_choice(q, exponential):
    """Argmin E_v/q_v for independent unit-rate exponential waiting times."""
    return (q / exponential.clamp_min(torch.finfo(q.dtype).tiny)).argmax(-1)


def linear_correction(tokens, q, p, uniforms, exponential):
    """Tensor-only accept-prefix and residual draw; inputs have shapes n,nV,(n+1)V."""
    n, vocabulary = q.shape
    safe = tokens.clamp(0, vocabulary - 1)
    selected_q = q.gather(-1, safe[:, None]).squeeze(-1)
    selected_p = p[:-1].gather(-1, safe[:, None]).squeeze(-1)
    ratio = (selected_p / selected_q.clamp_min(torch.finfo(q.dtype).tiny)).clamp_max(1)
    accepted = uniforms < ratio
    ranks = torch.arange(n, device=q.device)
    count = torch.where(accepted, n, ranks).amin()
    chosen_p = p.gather(0, count.expand(1, vocabulary))[0]
    padded_q = torch.cat((q, torch.zeros_like(q[:1])), 0)
    chosen_q = padded_q.gather(0, count.expand(1, vocabulary))[0]
    positive = (chosen_p - chosen_q).clamp_min(0)
    mass = positive.sum()
    tail = exponential_choice(positive, exponential)
    valid = (torch.isfinite(p).all() & torch.isfinite(q).all() & (q >= 0).all()
             & torch.isclose(q.sum(-1), torch.ones_like(selected_q), atol=1e-6, rtol=1e-6).all()
             & (tokens >= 0).all() & (tokens < vocabulary).all() & (selected_q > 0).all()
             & (uniforms >= 0).all() & (uniforms < 1).all() & (exponential > 0).all()
             & torch.isfinite(exponential).all() & (mass > 0))
    return count, tail, valid


class _Slot:
    def capture(self, use_cuda_graph):
        self.graph = None
        if use_cuda_graph:
            stream = torch.cuda.Stream(device=self.logits.device)
            stream.wait_stream(torch.cuda.current_stream(self.logits.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.run()
            torch.cuda.current_stream(self.logits.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.run()
            self.graph = graph

    def replay(self):
        if self.graph is None:
            self.run()
        else:
            self.graph.replay()


class _DraftSlot(_Slot):
    @torch.no_grad()
    def __init__(self, length, vocabulary, sampling, device, dtype, use_cuda_graph):
        self.sampling = sampling
        self.logits = torch.zeros(length, vocabulary, device=device, dtype=dtype)
        self.exponential = torch.ones_like(self.logits)
        self.capture(use_cuda_graph)

    def run(self):
        self.q = _probabilities_unchecked(self.logits, self.sampling)
        self.tokens = exponential_choice(self.q, self.exponential)
        self.valid = (torch.isfinite(self.logits).all() & torch.isfinite(self.q).all() & (self.q >= 0).all()
                      & torch.isclose(self.q.sum(-1), torch.ones_like(self.q[:, 0]), atol=1e-6, rtol=1e-6).all())


class _VerifySlot(_Slot):
    @torch.no_grad()
    def __init__(self, length, vocabulary, sampling, device, dtype, use_cuda_graph):
        self.sampling = sampling
        self.logits = torch.zeros(length, vocabulary, device=device, dtype=dtype)
        self.q = torch.full((length - 1, vocabulary), 1 / vocabulary, device=device, dtype=dtype)
        self.tokens = torch.zeros(length - 1, device=device, dtype=torch.long)
        self.uniforms = torch.zeros(length - 1, device=device, dtype=dtype)
        self.exponential = torch.ones(vocabulary, device=device, dtype=dtype)
        self.capture(use_cuda_graph)

    def run(self):
        self.p = _probabilities_unchecked(self.logits, self.sampling)
        count, tail, valid = linear_correction(self.tokens, self.q, self.p, self.uniforms, self.exponential)
        valid = valid & torch.isfinite(self.logits).all()
        # One transfer supplies the whole accepted-prefix decision and replacement.
        self.payload = torch.cat((self.tokens, torch.stack((count, tail, valid.long()))))


class SamplingExecutor:
    """Prepared, immutable-proposal sampling maps; tensor execution is the default."""

    @torch.no_grad()
    def __init__(self, vocabulary, block_size, sampling=SamplingConfig(), *, device="cuda",
                 dtype=torch.float32, use_cuda_graph=False):
        if (type(vocabulary) is not int or vocabulary < 1 or type(block_size) is not int or block_size < 2
                or sampling.temperature <= 0 or dtype not in (torch.float32, torch.float64)):
            raise ValueError("positive-temperature floating probabilities and block size >= 2 required")
        self.vocabulary, self.block_size, self.sampling = vocabulary, block_size, sampling
        self.device, self.dtype = torch.device(device), dtype
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if use_cuda_graph and self.device.type != "cuda":
            raise ValueError("CUDA device required for graph capture")
        self.drafts, self.verifiers, self.signature_seconds = {}, {}, {}
        for n in range(1, block_size + 1):
            start = time.perf_counter()
            self.drafts[n] = _DraftSlot(n, vocabulary, sampling, self.device, dtype, use_cuda_graph)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.signature_seconds[n, "plain"] = time.perf_counter() - start
            if n > 1:
                start = time.perf_counter()
                self.verifiers[n] = _VerifySlot(n, vocabulary, sampling, self.device, dtype, use_cuda_graph)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self.signature_seconds[n, "verify"] = time.perf_counter() - start
        self.setup_seconds = sum(self.signature_seconds.values())

    def validate(self, model, sampling, block_size=None):
        if model.config.vocab_size != self.vocabulary or sampling != self.sampling:
            raise ValueError("sampling executor vocabulary and probability policy must match")
        if block_size is not None and block_size != self.block_size:
            raise ValueError("sampling executor block size must match")

    def _shape(self, logits):
        if (logits.ndim != 2 or logits.shape[-1] != self.vocabulary or logits.device != self.device
                or not 1 <= len(logits) <= self.block_size or torch.is_grad_enabled()):
            raise ValueError("prepared inference-only probability rows required")

    def _draft(self, logits, generator):
        self._shape(logits)
        slot = self.drafts[len(logits)]
        slot.logits.copy_(logits)
        slot.exponential.exponential_(generator=generator)
        slot.replay()
        return slot

    def sample_ar(self, logits, generator=None):
        slot = self._draft(logits.reshape(1, -1), generator)
        token, valid = torch.stack((slot.tokens[0], slot.valid.long())).tolist()
        if not valid:
            raise ValueError("finite normalized AR probabilities required")
        return token

    def draft(self, logits, generator=None):
        slot = self._draft(logits, generator)
        if not slot.valid:
            raise ValueError("finite normalized proposal probabilities required")
        return slot.tokens.clone(), slot.q.clone(), None

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
