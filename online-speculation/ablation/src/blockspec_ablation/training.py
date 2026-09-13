"""Offline curriculum runner over local data; optional KL warm-up, then L1."""

from dataclasses import dataclass
import math
import time

import torch

from blockspec.data import sample_batch
from .distillation import LOSS_KINDS, offline_step
from .online import synchronize


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 1
    sequence_length: int = 128
    blocks: tuple[int, ...] = (2, 4, 6, 8)
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    warmup_loss: str = "reverse_kl_l1"
    loss: str = "l1"
    seed: int = 314159
    validation_every: int = 100

    def __post_init__(self):
        if self.steps < 1 or self.batch_size < 1 or self.sequence_length < 2:
            raise ValueError("invalid training dimensions")
        if self.validation_every < 1:
            raise ValueError("positive validation interval required")
        if not self.blocks or any(b < 1 for b in self.blocks):
            raise ValueError("positive block curriculum required")
        if not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warm-up must fit within total steps")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("positive finite learning rate required")
        if any(k not in LOSS_KINDS for k in (self.loss, self.warmup_loss)):
            raise ValueError("unknown training loss")


def train_adapter(model, sequences, config=TrainingConfig(), *, bos_id=0, progress=None, validation=None):
    if not 0 <= bos_id < model.config.vocab_size:
        raise ValueError("BOS id outside vocabulary")
    model.train_adapters_only()
    if not model.adapter_parameters():
        raise ValueError("no adapter parameters to train")
    device = next(model.parameters()).device
    data_rng = torch.Generator().manual_seed(config.seed)
    noise_rng = torch.Generator(device=device).manual_seed(config.seed + 1)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=config.learning_rate, weight_decay=0)
    synchronize(model)
    start = time.perf_counter()
    first_validation = validation.evaluate(model) if validation is not None else None
    if first_validation is not None and progress is not None:
        progress({"step": 0, "validation": first_validation})
    last_validation = first_validation
    for step in range(config.steps):
        clean = sample_batch(sequences, batch_size=config.batch_size, length=config.sequence_length,
                             bos_id=bos_id, device=device, generator=data_rng)
        block = config.blocks[min(len(config.blocks) - 1, step * len(config.blocks) // config.steps)]
        kind = config.warmup_loss if step < config.warmup_steps else config.loss
        stats = offline_step(model, optimizer, clean, block, kind=kind, generator=noise_rng)
        if progress is not None:
            progress({"step": step + 1, "block": block, "loss_kind": kind, **stats})
        if validation is not None and ((step + 1) % config.validation_every == 0 or step + 1 == config.steps):
            last_validation = validation.evaluate(model)
            if progress is not None:
                progress({"step": step + 1, "validation": last_validation})
    synchronize(model)
    return {"steps": config.steps, "seconds": time.perf_counter() - start,
            "training_tokens": config.steps * config.batch_size * config.sequence_length,
            "last_loss": stats["loss"], "last_loss_kind": kind,
            "validation_before": first_validation, "validation_after": last_validation}
