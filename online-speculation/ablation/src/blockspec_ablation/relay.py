"""PrefixRelay: conditional logit transitions with predictable prefix admission.

Frozen parallel features feed a trainable low-rank Markov head. The confidence
head acts before drawing the current candidate; verification uses saved, fully
transformed conditional proposal rows. See ALGORITHM.md, section 6.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .decoding import Generation, _check, _inference_forward, _prefill
from .diffusion import UniformNoise
from .model import trim_cache
from .online import synchronize
from blockspec.sampling import (SamplingConfig, draw, greedy_tokens, probabilities,
                       sample_logits, verify_greedy, verify_linear)


@dataclass(frozen=True)
class RelayConfig:
    vocab_size: int
    hidden_size: int
    rank: int = 64

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("positive integer head dimensions required")


class RelayHead(nn.Module):
    def __init__(self, config: RelayConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.rank)
        self.projection = nn.Linear(config.rank, config.vocab_size, bias=False)
        self.confidence = nn.Linear(config.hidden_size + config.rank, 1)
        nn.init.normal_(self.embedding.weight, std=config.rank ** -.5)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.confidence.weight)
        nn.init.zeros_(self.confidence.bias)

    def forward(self, logits, previous):
        return logits.to(self.projection.weight.dtype) + self.projection(self.embedding(previous))

    def confidence_logits(self, hidden, previous):
        hidden = hidden.to(self.confidence.weight.dtype)
        normalized = hidden * (hidden.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        return self.confidence(torch.cat((normalized, self.embedding(previous)), -1)).squeeze(-1)

    @torch.no_grad()
    def initialize_from_embedding(self, embedding, *, seed=271828):
        """Gaussian projection of frozen token features; W=0 keeps initial q.

        A private RNG isolates initialization from rollout randomness. Row-wise
        normalization matches the unit expected norm of the random embedding.
        """
        if (embedding.ndim != 2 or embedding.shape != (self.config.vocab_size, self.config.hidden_size)
                or not torch.isfinite(embedding).all() or self.projection.weight.count_nonzero()):
            raise ValueError("finite matching base embeddings and zero transition projection required")
        device = self.embedding.weight.device
        generator = torch.Generator(device=device).manual_seed(seed)
        projection = torch.randn(self.config.hidden_size, self.config.rank, device=device,
                                  dtype=torch.float32, generator=generator) / math.sqrt(self.config.rank)
        features = embedding.detach().to(device=device, dtype=torch.float32) @ projection
        features = F.normalize(features, dim=-1)
        self.embedding.weight.copy_(features.to(self.embedding.weight.dtype))


@dataclass
class RelayDraft:
    tokens: torch.Tensor  # clean root followed by admitted conditional candidates
    q: torch.Tensor      # transformed conditional laws, excluding the root


@torch.no_grad()
def relay_candidates(head, logits, hidden, *, sampling=SamplingConfig(), threshold=0., generator=None):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("prefix threshold must lie in [0,1]")
    if (logits.ndim != 2 or hidden.shape != (len(logits), head.config.hidden_size)
            or logits.shape[-1] != head.config.vocab_size or not len(logits)):
        raise ValueError("one feature and vocabulary logit row per draft position required")
    tokens = [sample_logits(logits[0], sampling, generator)]
    rows, survival = [], logits.new_ones(())
    for i in range(1, len(logits)):
        previous = tokens[-1]
        if threshold:
            survival = survival * head.confidence_logits(hidden[i], previous).sigmoid()
            if float(survival) < threshold:
                break
        corrected = head(logits[i], previous)
        q = probabilities(corrected, sampling)
        token = q.argmax(-1) if sampling.temperature == 0 else draw(q, generator)
        rows.append(q)
        tokens.append(token)
    q = torch.stack(rows) if rows else logits.new_empty((0, logits.shape[-1]))
    return RelayDraft(torch.stack(tokens), q)


@dataclass
class RelayFeedback:
    logits: torch.Tensor
    hidden: torch.Tensor
    previous: torch.Tensor
    teacher: torch.Tensor


def relay_loss(head, feedback, *, sampling=SamplingConfig(1.), confidence_weight=.1):
    """Detached sampled-prefix semi-gradient; both laws use the serving transform.

    Soft distillation requires positive temperature. Greedy-serving training uses
    an explicitly selected positive-temperature surrogate at learner creation.
    """
    if sampling.temperature <= 0 or confidence_weight < 0 or not math.isfinite(confidence_weight):
        raise ValueError("positive training temperature and finite nonnegative confidence weight required")
    corrected = head(feedback.logits.detach(), feedback.previous.detach())
    q = probabilities(corrected, sampling)
    p = probabilities(feedback.teacher.detach(), sampling)
    tv = .5 * (p - q).abs().sum(-1)
    prediction = head.confidence_logits(feedback.hidden.detach(), feedback.previous.detach())
    confidence = F.binary_cross_entropy_with_logits(prediction, (1 - tv).detach())
    return tv.mean() + confidence_weight * confidence, tv.mean().detach(), confidence.detach()


class RelayLearner:
    """Head-only online updates after verification; feedback is bounded per update.

    Detached logits/features replace backbone replay. Optimizer state persists
    across requests, while pending feedback is released at each request boundary.
    """
    def __init__(self, head, *, lr=1e-3, interval=8, sampling=SamplingConfig(1.), confidence_weight=.1,
                 confidence_lr=None):
        if interval < 1 or type(interval) is not int or not math.isfinite(lr) or lr <= 0:
            raise ValueError("positive update interval and learning rate required")
        if sampling.temperature <= 0 or not math.isfinite(confidence_weight) or confidence_weight < 0:
            raise ValueError("positive training temperature and nonnegative confidence weight required")
        if confidence_lr is not None and (not math.isfinite(confidence_lr) or confidence_lr <= 0):
            raise ValueError("positive finite confidence learning rate required")
        self.head, self.interval, self.sampling = head, interval, sampling
        self.confidence_weight = confidence_weight
        parameters = (head.parameters() if confidence_lr is None else [
            {"params": [head.embedding.weight, head.projection.weight], "lr": lr},
            {"params": list(head.confidence.parameters()), "lr": confidence_lr}])
        self.optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.,
                                           fused=next(head.parameters()).device.type == "cuda")
        self.pending = []
        self.updates = self.feedback_blocks = self.examples = 0
        self.update_seconds = 0.
        self.last_metrics = {}

    def clear_replay(self):
        self.pending.clear()

    def observe(self, feedback, *, may_update=True):
        if not len(feedback.previous):
            return
        self.feedback_blocks += 1
        self.pending.append(RelayFeedback(*(getattr(feedback, k).detach().clone()
                                            for k in ("logits", "hidden", "previous", "teacher"))))
        if len(self.pending) >= self.interval:
            if may_update:
                self.step()
            else:
                self.clear_replay()

    @torch.enable_grad()
    def step(self):
        if not self.pending:
            return
        synchronize(self.head)
        start = time.perf_counter()
        feedback = RelayFeedback(*(torch.cat([getattr(row, k) for row in self.pending])
                                   for k in ("logits", "hidden", "previous", "teacher")))
        self.optimizer.zero_grad(set_to_none=True)
        loss, tv, confidence = relay_loss(self.head, feedback, sampling=self.sampling,
                                         confidence_weight=self.confidence_weight)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.examples += len(feedback.previous)
        self.last_metrics = {"tv": float(tv), "confidence_bce": float(confidence), "grad_norm": float(norm)}
        self.clear_replay()
        synchronize(self.head)
        self.update_seconds += time.perf_counter() - start


@dataclass
class RelayGeneration(Generation):
    verified_tokens: int = 0
    depth_proposed: list[int] = field(default_factory=list)
    depth_accepted: list[int] = field(default_factory=list)

    def summary(self):
        return {**super().summary(), "verified_tokens": self.verified_tokens,
                "depth_proposed": self.depth_proposed, "depth_accepted": self.depth_accepted}


@torch.no_grad()
def generate_relay(model, head, prompt, max_new_tokens, *, block_size=8, sampling=SamplingConfig(),
                   threshold=0., eos_id=None, generator=None, executor=None, learner=None, noise=UniformNoise(),
                   proposal_executor=None):
    _check(model, prompt, max_new_tokens, eos_id)
    if type(block_size) is not int or block_size < 2:
        raise ValueError("block size must be an integer >=2")
    if (head.config.vocab_size, head.config.hidden_size) != (model.config.vocab_size, model.config.hidden_size):
        raise ValueError("head and decoder dimensions differ")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("prefix threshold must lie in [0,1]")
    if learner is not None and learner.head is not head:
        raise ValueError("learner must update the serving head")
    if proposal_executor is not None:
        proposal_executor.validate(head, sampling, threshold)
    forward = _inference_forward(model, executor)
    initial = (learner.updates, learner.update_seconds, learner.feedback_blocks) if learner else (0, 0., 0)
    if learner:
        learner.clear_replay()
    synchronize(model)
    start = time.perf_counter()
    result = RelayGeneration([], 0., 0, 0, depth_proposed=[0] * (block_size - 1),
                             depth_accepted=[0] * (block_size - 1))
    cache = _prefill(forward, prompt) if max_new_tokens else None
    seed = prompt[:, -1:]
    try:
        while len(result.tokens) < max_new_tokens:
            result.rounds += 1
            remaining = max_new_tokens - len(result.tokens)
            if remaining == 1:
                logits, cache = forward(seed, cache=cache, return_cache=True)
                result.tokens.append(int(sample_logits(logits[0, -1], sampling, generator)))
                result.decode_forwards += 1
                break
            b = min(block_size, remaining)
            noisy = noise.sample((1, b - 1), model.config.vocab_size, device=prompt.device, generator=generator)
            inputs = torch.cat((seed, noisy), 1)
            mask = torch.ones_like(inputs, dtype=torch.bool)
            mask[:, 0] = False
            logits, temporary, boundary = forward(inputs, cache=cache, adapter_mask=mask, return_cache=True,
                                                   capture_layer=model.config.num_hidden_layers)
            result.decode_forwards += 1
            draft = (relay_candidates(head, logits[0], boundary.hidden[0], sampling=sampling,
                                      threshold=threshold, generator=generator)
                     if proposal_executor is None else proposal_executor(logits[0], boundary.hidden[0], generator=generator))
            root = int(draft.tokens[0])
            if root == eos_id:
                result.tokens.append(root)
                break
            clean = trim_cache(temporary, prompt.shape[1] + len(result.tokens))
            teacher, verified_cache = forward(draft.tokens[None], cache=clean, return_cache=True)
            result.decode_forwards += 1
            n = len(draft.tokens) - 1
            result.verified_tokens += n + 1
            verification = (verify_greedy(draft.tokens[1:], greedy_tokens(teacher[0]))
                            if sampling.temperature == 0 else verify_linear(
                                draft.tokens[1:], draft.q, probabilities(teacher[0], sampling), generator=generator))
            committed = [root] + verification.tokens
            if eos_id is not None and eos_id in committed:
                committed = committed[:committed.index(eos_id) + 1]
            committed = committed[:remaining]
            kept = min(verification.accepted, len(committed) - 1)
            used = min(verification.supervised, len(committed) - 1)
            result.accepted += kept
            result.proposed += n
            result.accepted_per_round.append(kept)
            result.fully_covered_rounds += kept == n
            for i in range(n):
                result.depth_proposed[i] += 1
                result.depth_accepted[i] += i < kept
            result.tokens.extend(committed)
            cache = trim_cache(verified_cache, prompt.shape[1] + len(result.tokens) - 1)
            seed = prompt.new_tensor([[result.tokens[-1]]])
            done = len(result.tokens) == max_new_tokens or result.tokens[-1] == eos_id
            if learner:
                learner.observe(RelayFeedback(logits[0, 1:used + 1], boundary.hidden[0, 1:used + 1],
                                               draft.tokens[:used], teacher[0, :used]), may_update=not done)
            if done:
                break
    finally:
        if learner:
            learner.clear_replay()
    synchronize(model)
    result.seconds = time.perf_counter() - start
    if learner:
        result.updates = learner.updates - initial[0]
        result.update_seconds = learner.update_seconds - initial[1]
        result.feedback_blocks = learner.feedback_blocks - initial[2]
    return result


def save_relay(path, head, *, binding, metadata=None):
    """Exclusive checkpoint creation bound to the frozen base AND draft artifact."""
    if set(binding) != {"base", "adapter"} or any(len(v) != 64 for v in binding.values()):
        raise ValueError("SHA256 bindings for base and adapter required")
    payload = {"format": "prefixrelay-v1", "config": asdict(head.config), "binding": binding,
               "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
               "metadata": json.loads(json.dumps(metadata or {}, allow_nan=False))}
    with Path(path).open("xb") as handle:
        torch.save(payload, handle)


def load_relay(path, *, binding, device="cpu"):
    # Early local metadata stored torch.__version__ as this string subclass.
    # Keep the weights-only loader and narrowly permit that legacy value type.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "prefixrelay-v1" or payload.get("binding") != binding:
        raise ValueError("head checkpoint binding differs from the frozen backbone")
    head = RelayHead(RelayConfig(**payload["config"]))
    expected = head.state_dict()
    state = payload["state"]
    if set(state) != set(expected) or any(state[k].shape != v.shape or not torch.isfinite(state[k]).all()
                                          for k, v in expected.items()):
        raise ValueError("invalid head checkpoint tensors")
    head.load_state_dict(state)
    return head.to(device), payload.get("metadata", {})
