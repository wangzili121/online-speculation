"""Random-anchor block distillation from complete, frozen AR distributions."""

from dataclasses import dataclass
from contextlib import nullcontext
import math

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class AnchorLayout:
    tokens: Tensor
    positions: Tensor
    allowed: Tensor
    teacher_rows: Tensor
    student_rows: Tensor


def sample_anchors(tokens, block_size, count, *, generator=None):
    if tokens.ndim != 2 or block_size < 2 or count < 1 or tokens.shape[1] < block_size:
        raise ValueError("nonempty batch, anchor count, and a complete training block required")
    return torch.randint(tokens.shape[1] - block_size + 1, (tokens.shape[0], count),
                         device=tokens.device, generator=generator)


def anchor_layout(tokens, anchors, block_size, mask_token_id):
    if tokens.ndim != 2 or anchors.ndim != 2 or anchors.shape[0] != tokens.shape[0]:
        raise ValueError("tokens[batch,time] and anchors[batch,blocks] required")
    batch, sequence = tokens.shape
    if block_size < 2 or anchors.shape[1] < 1 or anchors.dtype != torch.long:
        raise ValueError("integer anchors and block_size >= 2 required")
    if ((anchors < 0) | (anchors + block_size > sequence)).any():
        raise ValueError("each anchor needs a complete block within the clean sequence")
    blocks = anchors.shape[1]
    offset = torch.arange(block_size, device=tokens.device)
    positions = anchors[..., None] + offset
    noisy = tokens.new_full((batch, blocks, block_size), mask_token_id)
    noisy[:, :, 0] = tokens.gather(1, anchors)
    total = blocks * block_size
    clean_visible = torch.arange(sequence, device=tokens.device)[None, None, :] < (
        anchors.repeat_interleave(block_size, 1)[..., None])
    block_index = torch.arange(total, device=tokens.device) // block_size
    same_block = (block_index[:, None] == block_index[None, :])[None].expand(batch, -1, -1)
    allowed = torch.cat((clean_visible, same_block), dim=-1)[:, None]
    rows = torch.arange(total, device=tokens.device).view(blocks, block_size)[:, :-1].reshape(-1)
    return AnchorLayout(noisy.reshape(batch, total), positions.reshape(batch, total), allowed,
                        positions[:, :, :-1].reshape(batch, -1), rows)


def forward_kl(student_logits, teacher_logits):
    teacher_log = teacher_logits.detach().float().log_softmax(-1)
    student_log = student_logits.float().log_softmax(-1)
    return (teacher_log.exp() * (teacher_log - student_log)).sum(-1).mean()


def distillation_loss(model, tokens, anchors, *, block_size=None, chunk_rows=32, draft_weights=None):
    """All K-1 used draft rows, with row-chunked full-vocabulary soft targets.

    Checkpointed output-head chunks retain hidden vectors for backward, keeping
    full-vocabulary intermediate storage bounded by the selected row chunk.
    """
    block_size = model.config.block_size if block_size is None else block_size
    if chunk_rows < 1:
        raise ValueError("positive output-head chunk size required")
    if any(value.requires_grad for name, value in model.named_parameters()
           if ".attention.draft." not in name):
        raise ValueError("freeze shared/AR parameters with train_draft_only() before distillation")
    layout = anchor_layout(tokens, anchors, block_size, model.config.mask_token_id)
    with torch.no_grad():
        teacher = model(tokens, compute_logits=False)
        index = layout.teacher_rows[..., None].expand(-1, -1, model.config.hidden_size)
        teacher_hidden = teacher.hidden.gather(1, index).reshape(-1, model.config.hidden_size)
    options = dict(view="draft", cache=teacher.cache, positions=layout.positions,
                   allowed=layout.allowed, compute_logits=False)
    if draft_weights is None:
        student = model(layout.tokens, **options)
    else:
        expected = {name for name, _ in model.named_parameters() if ".attention.draft." in name}
        if draft_weights.keys() != expected:
            raise ValueError("full-block overrides must contain every draft attention parameter")
        student = torch.func.functional_call(model, draft_weights, (layout.tokens,), options)
    student_hidden = student.hidden[:, layout.student_rows].reshape(-1, model.config.hidden_size)
    count = student_hidden.shape[0]

    def chunk_loss(student_rows, teacher_rows):
        with torch.no_grad():
            teacher_logits = F.linear(teacher_rows, model.head.weight)
        return forward_kl(F.linear(student_rows, model.head.weight), teacher_logits)

    terms = []
    for start in range(0, count, chunk_rows):
        stop = min(count, start + chunk_rows)
        term = checkpoint(chunk_loss, student_hidden[start:stop], teacher_hidden[start:stop],
                          use_reentrant=False)
        terms.append(term * ((stop - start) / count))
    return torch.stack(terms).sum()


def distillation_update(model, batches, optimizer, *, learning_rate, clip_grad=1., chunk_rows=32,
                        autocast=nullcontext, master=None):
    """One full-block AdamW update, shared by file-backed and streaming trainers.

    `batches` contains fixed (clean tokens, anchors) pairs. Separate FP32 masters
    leave serving tensors unchanged; publication is a caller-controlled boundary.
    """
    if (not batches or any(not math.isfinite(v) or v <= 0 for v in (learning_rate, clip_grad))):
        raise ValueError("nonempty microbatches and finite positive optimizer scales required")
    resident = dict(model.named_parameters())
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    expected = {name: p for name, p in resident.items() if ".attention.draft." in name}
    selected = expected if master is None else master
    if (selected.keys() != expected.keys() or {id(p) for p in parameters} != {id(p) for p in selected.values()}
            or any(p.dtype != torch.float32 or not p.requires_grad for p in selected.values())):
        raise ValueError("optimizer must own exactly the full FP32 draft parameter set")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    losses = []
    try:
        with torch.enable_grad():
            for tokens, anchors in batches:
                weights = (None if master is None else
                           {name: p.to(resident[name].dtype) for name, p in master.items()})
                with autocast():
                    loss = distillation_loss(model, tokens, anchors, chunk_rows=chunk_rows, draft_weights=weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite distillation loss")
                (loss / len(batches)).backward()
                losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad, error_if_nonfinite=True)
        optimizer.step()
    finally:
        optimizer.zero_grad(set_to_none=True)
    return {"loss": sum(losses) / len(losses), "learning_rate": learning_rate, "gradient_norm": float(norm)}
