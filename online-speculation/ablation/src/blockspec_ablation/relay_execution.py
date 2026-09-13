"""One GPU graph for the light sequential head and predictable admission state.

Independent Exp(1) race variables are supplied by the caller's RNG. Each
admission bit is computed before that position's categorical choice, and stays
false after the first stop. Only the admitted prefix reaches the verifier.
"""

import time

import torch

from .relay import RelayDraft
from blockspec.sampling import SamplingConfig


def race_sample(q, exponential):
    """Exponential-race categorical draw: argmin E_v/q_v, E_v iid Exp(1)."""
    return (q / exponential).argmax(-1)


def _transform(logits, sampling):
    # Graph-internal transform; finiteness is checked on graph outputs at replay.
    if sampling.temperature == 0:
        return torch.zeros_like(logits).scatter_(-1, logits.argmax(-1, keepdim=True), 1)
    work = logits / sampling.temperature
    if sampling.top_k:
        indices = work.topk(min(sampling.top_k, len(work))).indices
        keep = torch.zeros_like(work, dtype=torch.bool).scatter_(-1, indices, True)
        work = work.masked_fill(~keep, -torch.inf)
    if sampling.top_p < 1:
        ordered, indices = work.sort(descending=True)
        p = ordered.softmax(-1)
        removed = p.cumsum(-1) - p >= sampling.top_p
        removed[:1].zero_()
        work = work.masked_fill(torch.zeros_like(removed).scatter_(-1, indices, removed), -torch.inf)
    return work.softmax(-1)


class _RelaySlot:
    @torch.no_grad()
    def __init__(self, head, length, sampling, threshold, use_cuda_graph):
        self.head, self.length, self.sampling, self.threshold = head, length, sampling, threshold
        options = {"device": next(head.parameters()).device, "dtype": next(head.parameters()).dtype}
        self.logits = torch.zeros(length, head.config.vocab_size, **options)
        self.hidden = torch.zeros(length, head.config.hidden_size, **options)
        self.exponential = torch.ones_like(self.logits)
        self.graph = None
        if use_cuda_graph:
            stream = torch.cuda.Stream(device=options["device"])
            stream.wait_stream(torch.cuda.current_stream(options["device"]))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.run()
            torch.cuda.current_stream(options["device"]).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.run()
            self.graph = graph

    def run(self):
        q0 = _transform(self.logits[0], self.sampling)
        tokens = [q0.argmax(-1) if self.sampling.temperature == 0 else race_sample(q0, self.exponential[0])]
        rows, admissions = [], []
        survival = self.logits.new_ones(())
        live = torch.ones((), device=self.logits.device, dtype=torch.bool)
        for i in range(1, self.length):
            previous = tokens[-1]
            if self.threshold:
                survival = survival * self.head.confidence_logits(self.hidden[i], previous).sigmoid()
                live = live & (survival >= self.threshold)
            admissions.append(live)
            q = _transform(self.head(self.logits[i], previous), self.sampling)
            chosen = q.argmax(-1) if self.sampling.temperature == 0 else race_sample(q, self.exponential[i])
            tokens.append(torch.where(live, chosen, torch.zeros_like(chosen)))
            rows.append(q)
        self.tokens, self.q = torch.stack(tokens), torch.stack(rows)
        self.count = torch.stack(admissions).sum()

    @torch.no_grad()
    def __call__(self, logits, hidden, generator):
        self.logits.copy_(logits)
        self.hidden.copy_(hidden)
        if self.sampling.temperature:
            self.exponential.exponential_(generator=generator)
        if self.graph:
            self.graph.replay()
        else:
            self.run()
        if not torch.isfinite(self.q).all() or not torch.isfinite(self.logits[0]).all():
            raise ValueError("finite proposal probabilities and root logits required")
        n = int(self.count)
        return RelayDraft(self.tokens[:n + 1].clone(), self.q[:n].clone())


class RelayExecutor:
    """In-place head updates are visible on the next replay; snapshots are owned."""
    def __init__(self, head, *, block_size, sampling=SamplingConfig(), threshold=0., use_cuda_graph=True):
        if type(block_size) is not int or block_size < 2 or not 0 <= threshold <= 1:
            raise ValueError("block >=2 and threshold in [0,1] required")
        if use_cuda_graph and next(head.parameters()).device.type != "cuda":
            raise ValueError("CUDA head required for graph execution")
        self.head, self.sampling, self.threshold = head, sampling, threshold
        self.storage = [(id(p), p.data_ptr(), p.device, p.dtype) for p in head.parameters()]
        start = time.perf_counter()
        self.slots = {n: _RelaySlot(head, n, sampling, threshold, use_cuda_graph) for n in range(2, block_size + 1)}
        if use_cuda_graph:
            torch.cuda.synchronize(next(head.parameters()).device)
        self.setup_seconds = time.perf_counter() - start

    def validate(self, head, sampling, threshold):
        if head is not self.head or sampling != self.sampling or threshold != self.threshold:
            raise ValueError("proposal executor must match head and sampling policy")
        if self.storage != [(id(p), p.data_ptr(), p.device, p.dtype) for p in head.parameters()]:
            raise RuntimeError("head storage changed; recapture proposal executor")

    def __call__(self, logits, hidden, *, generator=None):
        if torch.is_grad_enabled():
            raise RuntimeError("proposal executor is inference-only")
        if (logits.ndim != 2 or logits.shape[-1] != self.head.config.vocab_size
                or hidden.shape != (len(logits), self.head.config.hidden_size) or len(logits) not in self.slots):
            raise ValueError("unprepared proposal shape")
        return self.slots[len(logits)](logits, hidden, generator)
