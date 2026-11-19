"""Sample independent masked-block rows only when verification reaches them."""

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

    def ensure(self, stop):
        stop = min(stop, self.count)
        first = len(self.tokens)
        if stop <= first:
            return []
        logits = self.read_logits(first, stop - first)
        if logits.ndim != 2 or logits.shape[0] != stop - first:
            raise ValueError("reader must return exactly the requested draft rows")
        q = probabilities(logits, self.config)
        q.div_(q.sum(-1, keepdim=True))
        tokens = draw(q, self.generator).tolist()
        self.distributions.extend(q.unbind(0))
        self.tokens.extend(tokens)
        return tokens

    def __getitem__(self, position):
        # Verification must never consult a distribution that was not sampled.
        return self.distributions[position]
