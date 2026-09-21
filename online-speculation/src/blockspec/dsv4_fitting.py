"""Deterministic offline and asynchronous-online fitting for a DSV4 head."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .accelerator import autocast, rng_state, set_rng_state, synchronize
from .dsv4_training import OrthrusDSV4TrainingModel
from .parallel.fitting import BatchStream, TokenDataset


@dataclass(frozen=True)
class DSV4FitConfig:
    steps: int = 1000
    batch_size: int = 1
    sequence_length: int = 4096
    anchors_per_sequence: int = 16
    accumulate: int = 1
    chunk_rows: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    minimum_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    clip_grad: float = 1.0
    precision: str = "bf16"
    objective: str = "forward_kl"
    indexer_weight: float = 0.01
    seed: int = 42

    def __post_init__(self):
        counts = (
            self.steps,
            self.batch_size,
            self.sequence_length,
            self.anchors_per_sequence,
            self.accumulate,
            self.chunk_rows,
        )
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("positive integer DSV4 training dimensions required")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup must fit within the training schedule")
        scales = (
            self.learning_rate,
            self.minimum_lr_ratio,
            self.weight_decay,
            self.clip_grad,
            self.indexer_weight,
        )
        if (
            any(not math.isfinite(value) for value in scales)
            or self.learning_rate <= 0
            or self.clip_grad <= 0
            or self.weight_decay < 0
            or self.indexer_weight < 0
            or not 0 <= self.minimum_lr_ratio <= 1
        ):
            raise ValueError("invalid optimizer or indexer-loss scale")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("DSV4 training precision must be fp32 or bf16")
        if self.objective not in ("forward_kl", "cross_entropy"):
            raise ValueError("DSV4 objective must be forward_kl or cross_entropy")

    def rate(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        fraction = (step - self.warmup_steps) / max(
            1, self.steps - self.warmup_steps - 1
        )
        cosine = (1 + math.cos(math.pi * fraction)) / 2
        return self.learning_rate * (
            self.minimum_lr_ratio + (1 - self.minimum_lr_ratio) * cosine
        )


def _head_signature(model: OrthrusDSV4TrainingModel) -> str:
    digest = hashlib.sha256()
    for name, value in model.head.state_dict().items():
        digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
    digest.update(json.dumps(model.compress_ratios).encode())
    digest.update(str(model.block_size).encode())
    return digest.hexdigest()


def _sample_candidate_anchors(
    candidates: list[torch.Tensor], count: int, generator: torch.Generator
) -> torch.Tensor:
    rows = []
    for valid in candidates:
        indices = torch.randint(len(valid), (count,), generator=generator)
        rows.append(valid[indices])
    return torch.stack(rows)


class DSV4Trainer:
    """Head-only optimizer with exact interruption boundaries and fresh-data continuation."""

    def __init__(
        self,
        model: OrthrusDSV4TrainingModel,
        data: TokenDataset,
        config: DSV4FitConfig,
        base_identity: dict | None = None,
    ):
        if data.sequence_length != config.sequence_length:
            raise ValueError("dataset and training sequence lengths differ")
        if data.vocab_size != model.config.vocab_size:
            raise ValueError("dataset and DSV4 vocabulary sizes differ")
        if config.sequence_length < model.block_size:
            raise ValueError("training sequence must contain a complete diffusion block")
        self.model = model.train()
        self.data = data
        self.config = config
        self.parameters = list(model.head_parameters())
        if not self.parameters or any(not parameter.requires_grad for parameter in self.parameters):
            raise ValueError("the complete DSV4 diffusion head must be trainable")
        if any(parameter.requires_grad for parameter in model.target.parameters()):
            raise ValueError("the DSV4 target must stay frozen")
        self.devices = list(dict.fromkeys(parameter.device for parameter in self.parameters))
        self.device = next(
            (device for device in self.devices if device.type in ("cuda", "npu")),
            self.devices[0],
        )
        if config.precision == "bf16" and all(
            device.type not in ("cuda", "npu") for device in self.devices
        ):
            raise ValueError("BF16 autocast requires CUDA or NPU; use fp32 for CPU smoke tests")
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
            foreach=False,
        )
        self.stream = BatchStream(data, config.seed)
        self.anchors_rng = torch.Generator().manual_seed(config.seed + 1)
        self.step = 0
        self.signature = _head_signature(model)
        self.base_identity = None if base_identity is None else dict(base_identity)

    def _autocast(self):
        return (
            autocast(self.device, torch.bfloat16)
            if self.config.precision == "bf16"
            else nullcontext()
        )

    def _sync(self) -> None:
        for device in self.devices:
            synchronize(device)

    def _clip_grad_norm(self) -> float:
        squared = 0.0
        for parameter in self.parameters:
            if parameter.grad is not None:
                norm = parameter.grad.detach().float().norm(2)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite DSV4 head gradient")
                squared += float(norm) ** 2
        total = math.sqrt(squared)
        scale = min(1.0, self.config.clip_grad / (total + 1e-6))
        if scale < 1.0:
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        return total

    def run(self, until: int | None = None, progress=None) -> list[dict]:
        end = self.config.steps if until is None else until
        if type(end) is not int or not self.step <= end <= self.config.steps:
            raise ValueError("stop boundary must lie within the training schedule")
        records = []
        while self.step < end:
            self._sync()
            started = time.perf_counter()
            rate = self.config.rate(self.step)
            for group in self.optimizer.param_groups:
                group["lr"] = rate
            self.optimizer.zero_grad(set_to_none=True)
            token_losses, indexer_losses = [], []
            try:
                for _ in range(self.config.accumulate):
                    tokens, candidates = self.stream.batch_with_anchor_candidates(
                        self.config.batch_size, self.model.block_size
                    )
                    anchors = _sample_candidate_anchors(
                        candidates,
                        self.config.anchors_per_sequence,
                        self.anchors_rng,
                    )
                    tokens = tokens.to(self.model.input_device)
                    anchors = anchors.to(self.model.input_device)
                    with self._autocast():
                        loss, metrics = self.model.loss(
                            tokens,
                            anchors,
                            objective=self.config.objective,
                            chunk_rows=self.config.chunk_rows,
                            indexer_weight=self.config.indexer_weight,
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite DSV4 Orthrus training loss")
                    (loss / self.config.accumulate).backward()
                    token_losses.append(float(metrics["token_loss"]))
                    indexer_losses.append(float(metrics["indexer_loss"]))
                norm = self._clip_grad_norm()
                self.optimizer.step()
            finally:
                self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self._sync()
            record = {
                "step": self.step,
                "loss": sum(token_losses) / len(token_losses),
                "indexer_loss": sum(indexer_losses) / len(indexer_losses),
                "gradient_norm": norm,
                "learning_rate": rate,
                "seconds": time.perf_counter() - started,
            }
            records.append(record)
            if progress is not None:
                progress(record)
        return records

    @torch.no_grad()
    def evaluate(self, data: TokenDataset, batches: int = 4) -> dict[str, float]:
        if (
            data.sequence_length != self.config.sequence_length
            or data.vocab_size != self.data.vocab_size
            or type(batches) is not int
            or batches < 1
        ):
            raise ValueError("validation data and positive batch count must match training")
        stream = BatchStream(data, self.config.seed + 2)
        anchors_rng = torch.Generator().manual_seed(self.config.seed + 3)
        token_losses, indexer_losses, training = [], [], self.model.training
        self.model.eval()
        try:
            for _ in range(batches):
                tokens, candidates = stream.batch_with_anchor_candidates(
                    self.config.batch_size, self.model.block_size
                )
                anchors = _sample_candidate_anchors(
                    candidates, self.config.anchors_per_sequence, anchors_rng
                )
                with self._autocast():
                    loss, metrics = self.model.loss(
                        tokens.to(self.model.input_device),
                        anchors.to(self.model.input_device),
                        objective=self.config.objective,
                        chunk_rows=self.config.chunk_rows,
                        indexer_weight=self.config.indexer_weight,
                    )
                token_losses.append(float(metrics["token_loss"]))
                indexer_losses.append(float(metrics["indexer_loss"]))
        finally:
            self.model.train(training)
        token_loss = sum(token_losses) / len(token_losses)
        indexer_loss = sum(indexer_losses) / len(indexer_losses)
        return {
            "loss": token_loss + self.config.indexer_weight * indexer_loss,
            "token_loss": token_loss,
            "indexer_loss": indexer_loss,
        }

    def state_dict(self) -> dict:
        return {
            "format": "orthrus-dsv4-trainer-v1",
            "config": asdict(self.config),
            "head_signature": self.signature,
            "base_identity": self.base_identity,
            "head": {
                name: value.detach().cpu().clone()
                for name, value in self.model.head.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "data_sha256": self.data.fingerprint,
            "data_stream": self.stream.state_dict(),
            "anchors_rng": self.anchors_rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "accelerator_rng": {
                str(device): rng_state(device)
                for device in self.devices
                if device.type != "cpu"
            },
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.exists():
            raise ValueError(f"checkpoint path already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def _load_head(self, state: dict) -> None:
        if state.get("head_signature") != self.signature:
            raise ValueError("checkpoint head schema differs from the loaded DSV4 model")
        if state.get("base_identity") != self.base_identity:
            raise ValueError("checkpoint BF16 base identity differs from the loaded DSV4 model")
        expected = self.model.head.state_dict()
        supplied = state.get("head", {})
        if supplied.keys() != expected.keys():
            raise ValueError("checkpoint does not contain the complete diffusion head")
        for name, value in supplied.items():
            if value.shape != expected[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid checkpoint head tensor: {name}")
        self.model.head.load_state_dict(supplied)

    def load_continual_weights(self, path: Path) -> None:
        """Start a fresh optimizer/data schedule from an earlier trained head."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "orthrus-dsv4-trainer-v1":
            raise ValueError("unsupported DSV4 trainer checkpoint")
        self._load_head(state)

    @classmethod
    def resume(
        cls,
        path: Path,
        model: OrthrusDSV4TrainingModel,
        data: TokenDataset,
        base_identity: dict | None = None,
    ) -> "DSV4Trainer":
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "orthrus-dsv4-trainer-v1":
            raise ValueError("unsupported DSV4 trainer checkpoint")
        config = DSV4FitConfig(**state["config"])
        trainer = cls(model, data, config, base_identity=base_identity)
        if state.get("data_sha256") != data.fingerprint:
            raise ValueError("exact resume requires the same tokenized data")
        trainer._load_head(state)
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.stream.load_state_dict(state["data_stream"])
        trainer.anchors_rng.set_state(state["anchors_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        saved_rng = state.get("accelerator_rng", {})
        for device in trainer.devices:
            if device.type != "cpu":
                if str(device) not in saved_rng:
                    raise ValueError(f"checkpoint lacks RNG state for {device}")
                set_rng_state(saved_rng[str(device)], device)
        step = state.get("step")
        if type(step) is not int or not 0 <= step <= config.steps:
            raise ValueError("invalid DSV4 training step")
        trainer.step = step
        return trainer
