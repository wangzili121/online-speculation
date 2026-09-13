"""Fixed-crop, fixed-noise teacher-forced validation, distinct from serving TPS."""

import hashlib

import torch

from blockspec.data import sample_batch
from .distillation import divergence, paired_batch


class FixedValidation:
    def __init__(self, sequences, *, vocab_size, blocks, batches=4, length=128, bos_id=0,
                 seed=735017):
        if batches < 1 or not blocks:
            raise ValueError("nonempty validation configuration required")
        data_rng = torch.Generator().manual_seed(seed)
        noise_rng = torch.Generator().manual_seed(seed + 1)
        self.examples = []
        digest = hashlib.sha256()
        for _ in range(batches):
            clean = sample_batch(sequences, batch_size=1, length=length, bos_id=bos_id,
                                 device="cpu", generator=data_rng)
            noisy = torch.randint(vocab_size, clean.shape, generator=noise_rng)
            for block in blocks:
                self.examples.append((clean, noisy, block))
                digest.update(clean.numpy().tobytes())
                digest.update(noisy.numpy().tobytes())
                digest.update(str(block).encode())
        self.fingerprint = digest.hexdigest()

    @torch.no_grad()
    def evaluate(self, model):
        device = next(model.parameters()).device
        sums = {}
        for clean, noisy, block in self.examples:
            paired = paired_batch(clean.to(device), block, noisy=noisy.to(device))
            logits = model(paired.tokens, positions=paired.positions, allowed=paired.allowed,
                           adapter_mask=paired.adapter_mask)
            length = clean.shape[1]
            teacher, student = logits[:, :length], logits[:, length:]
            valid = paired.eligible
            tv = divergence(student, teacher, "tv")[valid].sum()
            kl = divergence(student, teacher, "forward_kl")[valid].sum()
            correct = ((student.argmax(-1) == teacher.argmax(-1)) & valid).sum()
            row = sums.setdefault(block, {"positions": 0, "tv": 0., "forward_kl": 0., "argmax_agreement": 0.})
            row["positions"] += int(valid.sum())
            row["tv"] += float(tv)
            row["forward_kl"] += float(kl)
            row["argmax_agreement"] += int(correct)
        for row in sums.values():
            for key in ("tv", "forward_kl", "argmax_agreement"):
                row[key] /= row["positions"]
            row["teacher_forced_overlap"] = 1 - row["tv"]
        return {"fingerprint": self.fingerprint, "blocks": sums,
                "scope": "fixed teacher-forced positions, not on-policy acceptance or TPS"}
