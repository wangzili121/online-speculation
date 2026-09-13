"""Single-device full-distribution fitting with deterministic data and resume state."""

from array import array
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .training import distillation_loss, distillation_update, sample_anchors
from .weights import file_sha256, load_checkpoint, save_checkpoint


class TokenDataset:
    """Indexed tokenized JSONL; contiguous crops stay within individual records."""

    def __init__(self, path, vocab_size, sequence_length):
        if any(type(n) is not int or n < 1 for n in (vocab_size, sequence_length)):
            raise ValueError("positive vocabulary and sequence window required")
        self.path = Path(path).resolve()
        self.vocab_size, self.sequence_length = vocab_size, sequence_length
        self.offsets, self.skipped = array("Q"), 0
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                ids = self._decode(line)
                if len(ids) >= sequence_length:
                    self.offsets.append(offset)
                else:
                    self.skipped += 1
        if not self.offsets:
            raise ValueError("training data needs a complete sequence window")
        self.fingerprint = file_sha256(self.path)

    def _decode(self, line):
        ids = json.loads(line).get("input_ids")
        if not isinstance(ids, list) or not ids or any(type(x) is not int or not 0 <= x < self.vocab_size for x in ids):
            raise ValueError("each JSONL record requires valid input_ids")
        return ids

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return torch.tensor(self._decode(handle.readline()), dtype=torch.long)


class BatchStream:
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.rng = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.rng)
        self.cursor = self.epoch = 0

    def batch(self, count):
        rows, length = [], self.dataset.sequence_length
        for _ in range(count):
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.dataset), generator=self.rng)
                self.cursor = 0
                self.epoch += 1
            sequence = self.dataset[int(self.order[self.cursor])]
            self.cursor += 1
            start = int(torch.randint(len(sequence) - length + 1, (), generator=self.rng))
            rows.append(sequence[start:start + length])
        return torch.stack(rows)

    def state_dict(self):
        return {"rng": self.rng.get_state(), "order": self.order.clone(),
                "cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        order = state["order"].cpu()
        if (order.dtype != torch.long or order.shape != (len(self.dataset),)
                or not torch.equal(order.sort().values, torch.arange(len(self.dataset)))
                or type(state["cursor"]) is not int or not 0 <= state["cursor"] <= len(order)
                or type(state["epoch"]) is not int or state["epoch"] < 0):
            raise ValueError("invalid data permutation or cursor")
        self.rng.set_state(state["rng"].cpu())
        self.order, self.cursor, self.epoch = order, state["cursor"], state["epoch"]


@dataclass(frozen=True)
class FitConfig:
    steps: int = 1000
    batch_size: int = 1
    sequence_length: int = 256
    anchors_per_sequence: int = 4
    accumulate: int = 1
    chunk_rows: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    minimum_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    precision: str = "fp32"
    backend: str = "sdpa"
    seed: int = 731
    optimizer_impl: str = "single"

    def __post_init__(self):
        counts = (self.steps, self.batch_size, self.sequence_length, self.anchors_per_sequence,
                  self.accumulate, self.chunk_rows)
        if any(type(n) is not int or n < 1 for n in counts):
            raise ValueError("positive integer training dimensions required")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup must fit within the full training schedule")
        if (any(not math.isfinite(x) for x in (self.learning_rate, self.minimum_lr_ratio,
                                               self.weight_decay, self.clip_grad))
                or self.learning_rate <= 0 or self.clip_grad <= 0 or self.weight_decay < 0
                or not 0 <= self.minimum_lr_ratio <= 1):
            raise ValueError("finite positive learning rate/clip and valid decay required")
        if self.precision not in ("fp32", "bf16") or self.backend not in ("eager", "sdpa"):
            raise ValueError("training supports FP32/BF16 and eager/SDPA masks")
        if self.optimizer_impl not in ("single", "fused"):
            raise ValueError("AdamW implementation must be single or fused")

    def make_optimizer(self, parameters):
        return torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=self.weight_decay,
                                  foreach=False, fused=self.optimizer_impl == "fused")

    def rate(self, step):
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        fraction = (step - self.warmup_steps) / max(1, self.steps - self.warmup_steps - 1)
        multiplier = self.minimum_lr_ratio + (1 - self.minimum_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
        return self.learning_rate * multiplier


def frozen_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if ".attention.draft." in name:
            continue
        digest.update((name + str(tuple(value.shape)) + str(value.dtype)).encode())
        digest.update(memoryview(value.detach().contiguous().cpu().view(torch.uint8).numpy()))
    return digest.hexdigest()


class Trainer:
    """Checkpoint boundaries coincide with complete AdamW updates."""

    def __init__(self, model, data, config=FitConfig()):
        if (data.sequence_length != config.sequence_length or data.vocab_size != model.config.vocab_size
                or config.sequence_length < model.config.block_size):
            raise ValueError("matching data windows and complete draft blocks required")
        if any(value.dtype != torch.float32 for value in model.parameters()):
            raise ValueError("training uses FP32 master parameters with optional BF16 autocast")
        self.model = model.train_draft_only().set_backend(config.backend)
        self.config, self.data = config, data
        self.device = model.embedding.weight.device
        if config.precision == "bf16" and self.device.type != "cuda":
            raise ValueError("BF16 training autocast requires CUDA")
        self.parameters = [value for value in model.parameters() if value.requires_grad]
        self.optimizer = config.make_optimizer(self.parameters)
        self.stream = BatchStream(data, config.seed)
        self.anchors_rng = torch.Generator().manual_seed(config.seed + 1)
        self.step = 0
        self.base_fingerprint = frozen_fingerprint(model)

    def _autocast(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16) if self.config.precision == "bf16"
                else nullcontext())

    def _runtime(self):
        return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                "device_type": self.device.type,
                "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
                "threads": torch.get_num_threads(), "matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic": torch.are_deterministic_algorithms_enabled()}

    def run(self, until=None, progress=None):
        end = self.config.steps if until is None else until
        if type(end) is not int or not self.step <= end <= self.config.steps:
            raise ValueError("stop boundary must lie within the unchanged full schedule")
        records = []
        self.model.train()
        while self.step < end:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            rate = self.config.rate(self.step)
            batches = []
            for _ in range(self.config.accumulate):
                tokens = self.stream.batch(self.config.batch_size)
                anchors = sample_anchors(tokens, self.model.config.block_size,
                                         self.config.anchors_per_sequence, generator=self.anchors_rng)
                batches.append((tokens.to(self.device), anchors.to(self.device)))
            metrics = distillation_update(self.model, batches, self.optimizer, learning_rate=rate,
                                          clip_grad=self.config.clip_grad, chunk_rows=self.config.chunk_rows,
                                          autocast=self._autocast)
            self.step += 1
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            record = {"step": self.step, **metrics, "seconds": time.perf_counter() - start}
            records.append(record)
            if progress is not None:
                progress(record)
        return records

    @torch.no_grad()
    def evaluate(self, data, batches=4):
        if data.sequence_length != self.config.sequence_length or batches < 1:
            raise ValueError("matching validation window and positive batch count required")
        stream = BatchStream(data, self.config.seed + 2)
        rng = torch.Generator().manual_seed(self.config.seed + 3)
        losses, training = [], self.model.training
        self.model.eval()
        try:
            for _ in range(batches):
                tokens = stream.batch(self.config.batch_size)
                anchors = sample_anchors(tokens, self.model.config.block_size,
                                         self.config.anchors_per_sequence, generator=rng)
                with self._autocast():
                    loss = distillation_loss(self.model, tokens.to(self.device), anchors.to(self.device),
                                             chunk_rows=self.config.chunk_rows)
                losses.append(float(loss))
        finally:
            self.model.train(training)
        return sum(losses) / len(losses)

    def save(self, path):
        state = {"format": "dual-fit-v1", "config": asdict(self.config), "data_sha256": self.data.fingerprint,
                 "data_stream": self.stream.state_dict(), "anchors_rng": self.anchors_rng.get_state(),
                 "torch_rng": torch.get_rng_state(), "runtime": self._runtime(),
                 "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                 "base_fingerprint": self.base_fingerprint}
        save_checkpoint(path, self.model, optimizer=self.optimizer, step=self.step, training_state=state)

    @classmethod
    def resume(cls, path, data, *, device="cpu"):
        model, saved = load_checkpoint(path, device=device)
        state = saved["training_state"]
        if not isinstance(state, dict) or state.get("format") != "dual-fit-v1" or saved["optimizer"] is None:
            raise ValueError("a complete training checkpoint is required")
        if not isinstance(data, TokenDataset):
            data = TokenDataset(data, model.config.vocab_size, state["config"]["sequence_length"])
        if state["data_sha256"] != data.fingerprint:
            raise ValueError("resume training data SHA256 differs")
        trainer = cls(model, data, FitConfig(**state["config"]))
        if trainer._runtime() != state["runtime"] or trainer.base_fingerprint != state["base_fingerprint"]:
            raise ValueError("resume requires the same runtime and frozen base")
        if type(saved["step"]) is not int or not 0 <= saved["step"] <= trainer.config.steps:
            raise ValueError("invalid training step")
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.stream.load_state_dict(state["data_stream"])
        trainer.anchors_rng.set_state(state["anchors_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if trainer.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), trainer.device)
        trainer.step = saved["step"]
        return trainer
