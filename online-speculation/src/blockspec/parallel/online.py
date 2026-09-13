"""Online suffix distillation with FP32 masters and read-only AR history."""

from collections import deque
from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import time

import torch

from ..losses import divergence
from .backbone import DraftBoundary
from .weights import source_identity


@dataclass(frozen=True)
class SuffixConfig:
    last_layers: int = 1
    stride: int = 16
    replay_blocks: int = 1
    learning_rate: float = 1e-5
    clip_norm: float = 1.
    loss: str = "forward_kl"

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in (self.last_layers, self.stride, self.replay_blocks))
                or self.replay_blocks > self.stride
                or any(not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.clip_norm))
                or self.loss not in ("forward_kl", "tv")):
            raise ValueError("positive bounded suffix/update windows, optimizer scales and KL or TV required")


def portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(portable(item) for item in value)
    return copy.deepcopy(value)


class SuffixLearner:
    """Only draft attention in a suffix is published; AR parameters stay frozen.

    Each feedback block stores a detached boundary, clean suffix KV and the
    actual reached teacher rows. The whole masked block is replayed, followed
    by the output projection of its first `valid` rows.
    """

    def __init__(self, model, config=SuffixConfig()):
        if config.last_layers > len(model.layers) or any(p.requires_grad for p in model.parameters()):
            raise ValueError("a frozen dual-view model and a suffix within its layers required")
        self.model, self.config = model, config
        self.capture_layer = len(model.layers) - config.last_layers
        self.execution = {name: p for name, p in model.named_parameters()
                          if name.startswith("layers.") and int(name.split(".")[1]) >= self.capture_layer
                          and ".attention.draft." in name}
        dtype = next(model.parameters()).dtype
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("FP32 or BF16 inference weights required")
        self.master = {name: torch.nn.Parameter(p.detach().float().clone()) for name, p in self.execution.items()}
        self.parameters = list(self.master.values())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate, weight_decay=0,
                                           fused=True if self.parameters[0].is_cuda else None)
        self._frozen = [(p, p._version) for name, p in model.named_parameters() if name not in self.execution]
        fingerprint = hashlib.sha256()
        for name, p in model.named_parameters():
            if name not in self.execution:
                fingerprint.update(name.encode())
                fingerprint.update(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        self.frozen_fingerprint = fingerprint.hexdigest()
        self._backend = model.backend
        self.replay = deque(maxlen=config.replay_blocks)
        self.rounds = self.updates = self.version = self.feedback_blocks = self.coverage_skips = 0
        self.update_seconds, self.last_loss = 0., None

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters)

    @property
    def needs_decoder_feedback(self):
        return self.config.stride - self.rounds % self.config.stride <= self.config.replay_blocks

    def _check(self):
        if (self.model.backend != self._backend or any(p.requires_grad for p in self.model.parameters())
                or any(p._version != version for p, version in self._frozen)):
            raise RuntimeError("frozen representations or attention execution changed; rebuild the learner")

    def _sync(self):
        if self.parameters[0].is_cuda:
            torch.cuda.synchronize(self.parameters[0].device)

    def clear_replay(self):
        self.replay.clear()

    def _skip_decoder_feedback(self, valid):
        if type(valid) is not int or valid < 1 or self.needs_decoder_feedback:
            raise ValueError("positive decoder feedback outside the replay window required")
        self.rounds += 1

    def observe(self, feedback, *, may_update=True):
        self._check()
        if (feedback.inputs.ndim != 2 or feedback.inputs.shape[0] != 1
                or not 0 < feedback.valid < feedback.inputs.shape[1]
                or feedback.teacher_logits.shape != (feedback.valid, self.model.config.vocab_size)
                or not isinstance(feedback.boundary, DraftBoundary)
                or feedback.boundary.start_layer != self.capture_layer
                or feedback.boundary.hidden.shape[:2] != feedback.inputs.shape):
            raise ValueError("aligned masked-block boundary and reached teacher rows required")
        self.rounds += 1
        self.feedback_blocks += 1
        self.replay.append(feedback.detached(cache_start=self.capture_layer))
        if may_update and self.rounds % self.config.stride == 0:
            return self.update()
        return None

    def backward(self):
        """Full-soft-target gradients through a frozen-prefix suffix replay."""
        total, positions = 0., sum(item.valid for item in self.replay)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            for item in self.replay:
                weights = {name: p.to(self.execution[name].dtype) for name, p in self.master.items()}
                output = self.model.forward_suffix(item.boundary, cache=item.cache,
                                                   logit_range=(0, item.valid), draft_weights=weights)
                loss = divergence(output.logits[0], item.teacher_logits, self.config.loss).sum() / positions
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite online suffix loss")
                loss.backward()
                total += float(loss.detach())
        return total

    @torch.no_grad()
    def publish(self):
        for name, master in self.master.items():
            self.execution[name].copy_(master)

    def update(self):
        if not self.replay:
            return None
        self._check()
        self._sync()
        start = time.perf_counter()
        loss = self.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.clip_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.publish()
        self._sync()
        elapsed = time.perf_counter() - start
        self.updates += 1
        self.version += 1
        self.update_seconds += elapsed
        self.last_loss = loss
        return {"loss": loss, "gradient_norm": float(norm), "seconds": elapsed, "version": self.version}

    def state_dict(self):
        if self.replay:
            raise ValueError("save online state at a request boundary after clearing replay")
        return {"format": "blockspec-dual-suffix-v1", "config": asdict(self.config),
                "model_config": self.model.config.to_dict(), "source": source_identity(getattr(self.model, "source", {})),
                "dtype": str(next(self.model.parameters()).dtype), "backend": self._backend,
                "frozen_fingerprint": self.frozen_fingerprint,
                "master": portable(self.master), "optimizer": portable(self.optimizer.state_dict()),
                "rounds": self.rounds, "updates": self.updates, "version": self.version,
                "feedback_blocks": self.feedback_blocks, "update_seconds": self.update_seconds,
                "last_loss": self.last_loss}

    def load_state_dict(self, state):
        self._check()
        if (state.get("format") != "blockspec-dual-suffix-v1" or state.get("config") != asdict(self.config)
                or state.get("model_config") != self.model.config.to_dict()
                or source_identity(state.get("source", {})) != source_identity(getattr(self.model, "source", {}))
                or state.get("dtype") != str(next(self.model.parameters()).dtype)
                or state.get("frozen_fingerprint") != self.frozen_fingerprint
                or state.get("backend") != self._backend or state.get("master", {}).keys() != self.master.keys()):
            raise ValueError("online state must match its model, source, precision and update policy")
        for name, value in state["master"].items():
            if value.shape != self.master[name].shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("finite FP32 suffix master tensors with matching shapes required")
        for key in ("rounds", "updates", "version", "feedback_blocks"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError("nonnegative integer online counters required")
        if not math.isfinite(state.get("update_seconds", -1)) or state["update_seconds"] < 0:
            raise ValueError("finite nonnegative update time required")
        with torch.no_grad():
            for name, value in state["master"].items():
                self.master[name].copy_(value)
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.publish()
        self.clear_replay()
        for key in ("rounds", "updates", "version", "feedback_blocks", "update_seconds", "last_loss"):
            setattr(self, key, state[key])
