"""Synchronous continuation of the actual draft adapter after committed rounds."""

from collections import deque
from dataclasses import dataclass
import math
import time

import torch

from .distillation import LOSS_KINDS, divergence
from blockspec.feedback import Feedback as Feedback
from .model import is_adapter


def synchronize(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class OnlineConfig:
    stride: int = 16
    replay_blocks: int = 4
    learning_rate: float = 1e-4
    loss: str = "l1"
    clip_norm: float = 1.0
    train_last_layers: int | None = None
    optimizer: str = "auto"
    feedback_execution: str = "windowed"
    update_policy: str = "periodic"

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.stride, self.replay_blocks)):
            raise ValueError("positive integer update stride and replay capacity required")
        if not all(math.isfinite(v) and v > 0 for v in (self.learning_rate, self.clip_norm)):
            raise ValueError("finite positive optimizer scales required")
        if self.loss not in LOSS_KINDS:
            raise ValueError("unknown online loss")
        if self.optimizer not in ("auto", "standard", "fused"):
            raise ValueError("unknown optimizer execution")
        if self.feedback_execution not in ("windowed", "all"):
            raise ValueError("unknown feedback execution")
        if self.update_policy not in ("periodic", "coverage"):
            raise ValueError("unknown online update policy")
        if self.train_last_layers is not None and (type(self.train_last_layers) is not int or self.train_last_layers < 1):
            raise ValueError("a positive integer suffix size or None required")


class OnlineLearner:
    """Continue the draft adapter using committed verification feedback.

    Replay recomputes a differentiable noisy student forward with base-only prefix
    KV and detached teacher targets. Learned weights and Adam persist across requests.
    """

    def __init__(self, model, config=OnlineConfig(), *, replay_executor=None):
        self.model, self.config = model, config
        count = model.config.num_hidden_layers
        if config.train_last_layers is not None and config.train_last_layers > count:
            raise ValueError("online suffix exceeds the number of model layers")
        self.capture_layer = None if config.train_last_layers is None else count - config.train_last_layers
        model.train_adapters_only()
        if self.capture_layer is not None:
            for layer in model.model.layers[:self.capture_layer]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)
        self.parameters = [p for p in model.adapter_parameters() if p.requires_grad]
        self._prefix_parameters = ([p for p in model.model.embed_tokens.parameters()] +
                                   list(model.model.layers[:self.capture_layer].parameters())
                                   if self.capture_layer is not None else [])
        # Cached features belong to this fixed prefix and its parameter storage.
        # Reconstruct the learner when replacing prefix weights or parameters.
        self._prefix_versions = [p._version for p in self._prefix_parameters]
        self._attention = model.attention_signature()
        if not self.parameters:
            raise ValueError("online continuation needs trainable adapter parameters")
        use_fused = config.optimizer == "fused" or (config.optimizer == "auto" and self.parameters[0].is_cuda)
        if use_fused and not self.parameters[0].is_cuda:
            raise ValueError("fused online optimizer requires CUDA")
        self.optimizer_backend = "fused" if use_fused else "standard"
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate, weight_decay=0,
                                           fused=True if use_fused else None)
        self.replay = deque(maxlen=config.replay_blocks)
        self.rounds, self.updates, self.version = 0, 0, 0
        self.feedback_blocks = 0
        self.coverage_skips = 0
        self.update_seconds, self.last_loss = 0.0, None
        self.replay_executor = replay_executor
        if replay_executor is not None:
            replay_executor.validate(model, self.capture_layer, config.loss)

    @property
    def needs_decoder_feedback(self):
        """Collect the next positive-feedback decoder round when an update uses it."""
        until_update = self.config.stride - self.rounds % self.config.stride
        return self.config.feedback_execution == "all" or until_update <= self.config.replay_blocks

    def _skip_decoder_feedback(self, valid):
        """Advance a positive-feedback round outside its update's replay window.

        The synchronous decoders provide a valid target on every observed round
        and clear replay at request boundaries. Generic callers use observe().
        """
        if type(valid) is not int or valid < 1 or self.needs_decoder_feedback:
            raise ValueError("a positive-feedback decoder round outside the replay window is required")
        self.rounds += 1

    def clear_replay(self):
        """Release request-specific KV and targets; keep learned weights/optimizer."""
        self.replay.clear()

    def observe(self, feedback, *, may_update=True):
        if type(feedback.fully_covered) is not bool:
            raise ValueError("boolean coverage metadata required")
        if feedback.fully_covered and (feedback.inputs.ndim != 2 or feedback.inputs.shape[1] < 2
                                      or feedback.valid != feedback.inputs.shape[1] - 1):
            raise ValueError("fully covered feedback requires every noisy target row")
        self.rounds += 1
        if feedback.valid:
            if feedback.inputs.shape[0] != 1 or not 0 < feedback.valid < feedback.inputs.shape[1]:
                raise ValueError("invalid replay dimensions")
            if feedback.teacher_logits.shape != (feedback.valid, self.model.config.vocab_size):
                raise ValueError("teacher rows must match the valid prefix")
            if self.capture_layer is not None:
                self._check_prefix()
                if feedback.boundary is None or feedback.boundary.start_layer != self.capture_layer:
                    raise ValueError("suffix continuation requires the matching draft boundary")
            self.replay.append(feedback.detached(cache_start=self.capture_layer or 0))
            self.feedback_blocks += 1
        if may_update and self.replay and self.rounds % self.config.stride == 0:
            return self.update()
        return None

    def _check_prefix(self):
        if self.model.attention_signature() != self._attention:
            raise RuntimeError("attention execution changed; rebuild learner and feedback")
        if any(p.requires_grad or p._version != v for p, v in zip(self._prefix_parameters, self._prefix_versions)):
            raise RuntimeError("frozen draft prefix changed; discard learner and cached boundaries")

    def update(self):
        if not self.replay:
            return None
        if any(p.requires_grad for n, p in self.model.named_parameters() if not is_adapter(n)):
            raise RuntimeError("the target/base model must stay frozen")
        self._check_prefix()
        if self.config.update_policy == "coverage" and all(item.fully_covered for item in self.replay):
            self.coverage_skips += 1
            return None
        synchronize(self.model)
        start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        positions = sum(item.valid for item in self.replay)
        # Accumulate gradients one replay block at a time to bound activation memory.
        if self.replay_executor is not None:
            total = self.replay_executor.backward(self.replay)
        else:
            total = self._eager_backward(positions)
        torch.nn.utils.clip_grad_norm_(self.parameters, self.config.clip_norm,
                                       error_if_nonfinite=True)
        self.optimizer.step()
        synchronize(self.model)
        elapsed = time.perf_counter() - start
        self.updates += 1
        self.version += 1
        self.update_seconds += elapsed
        self.last_loss = total
        return {"loss": total, "seconds": elapsed, "positions": positions, "version": self.version}

    def _eager_backward(self, positions):
        total = 0.0
        with torch.enable_grad():
            for item in self.replay:
                if self.capture_layer is None:
                    mask = torch.ones_like(item.inputs, dtype=torch.bool)
                    mask[:, 0] = False
                    logits = self.model(item.inputs, cache=item.cache, adapter_mask=mask,
                                        logit_range=(1, 1 + item.valid))
                else:
                    logits = self.model.forward_suffix(item.boundary, cache=item.cache,
                                                       logit_range=(1, 1 + item.valid))
                loss = divergence(logits[0], item.teacher_logits, self.config.loss).sum() / positions
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite online loss")
                loss.backward()
                total += float(loss.detach())
        return total
