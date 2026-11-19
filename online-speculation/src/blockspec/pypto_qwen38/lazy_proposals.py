"""Sample independent masked-block rows only when verification reaches them."""

import torch

from ..sampling import draw, probabilities


class LazyProposals:
    def __init__(self, count, read_logits, config, generator, *, retain_logits=False):
        if type(count) is not int or count < 1:
            raise ValueError("positive proposal count required")
        self.count, self.read_logits = count, read_logits
        if retain_logits:
            # Several draft groups may reuse the same device output before
            # verification. Preserve its values, while still sampling lazily.
            saved = read_logits(0, count).clone()
            if saved.ndim != 2 or saved.shape[0] != count:
                raise ValueError("reader must return exactly the requested draft rows")
            self.read_logits = lambda first, size: saved[first:first + size]
        self.config, self.generator = config, generator
        self.tokens, self.distributions = [], []
        self.prefetched = None

    def prefetch(self, stop):
        """只读取下一段 logits；采样前可在设备空闲时执行已排队的学习任务。"""
        stop = min(stop, self.count)
        first = len(self.tokens)
        if stop <= first:
            return
        ready = 0 if self.prefetched is None else self.prefetched.shape[0]
        begin = first + ready
        if stop <= begin:
            return
        logits = self.read_logits(begin, stop - begin)
        if logits.ndim != 2 or logits.shape[0] != stop - begin:
            raise ValueError("reader must return exactly the requested draft rows")
        logits = logits.clone()  # 读端可复用 CPU 暂存区，预取后的行必须独立持有。
        self.prefetched = logits if self.prefetched is None else torch.cat((self.prefetched, logits))

    def ensure(self, stop):
        stop = min(stop, self.count)
        first = len(self.tokens)
        if stop <= first:
            return []
        count = stop - first
        if self.prefetched is None:
            logits = self.read_logits(first, count)
            if logits.ndim != 2 or logits.shape[0] != count:
                raise ValueError("reader must return exactly the requested draft rows")
        else:
            self.prefetch(stop)
            logits = self.prefetched[:count]
        q = probabilities(logits, self.config)
        q.div_(q.sum(-1, keepdim=True))
        tokens = draw(q, self.generator).tolist()
        self.distributions.extend(q.unbind(0))
        self.tokens.extend(tokens)
        if self.prefetched is not None:
            self.prefetched = self.prefetched[count:] if self.prefetched.shape[0] > count else None
        return tokens

    def __getitem__(self, position):
        # Verification must never consult a distribution that was not sampled.
        return self.distributions[position]
