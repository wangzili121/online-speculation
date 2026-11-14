"""Differentiable reference for one V4.1 C1A Reuse suffix."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F

from ..losses import divergence
from .dsv41_attention import PagedRows
from .dsv41_contract import BLOCK_SIZE, V41Topology, text_config
from .dsv4_hybrid import tail_logical_indices
from .dsv41_precise_attention import precise_draft_attention
from .head_contract import find_weight_index, read_json


@dataclass(frozen=True)
class LastLayerReplay:
    # The first 39 draft layers and the final layer's attention input prelude
    # are frozen, so the replay can start just before the final Q/KV projections.
    attention_input: torch.Tensor
    residual: torch.Tensor
    attn_pre: torch.Tensor
    attn_post: torch.Tensor
    attn_comb: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    swa: PagedRows
    global_kv: PagedRows
    global_indices: torch.Tensor
    anchors: torch.Tensor
    window: int

    @classmethod
    @torch.inference_mode(False)
    def snapshot(cls, x, residual, pre, post, comb, cos, sin, history, selection, window):
        if history.long_kv is None or selection.shape[:2] != (1, BLOCK_SIZE):
            raise ValueError("one C1A request required for the last-layer snapshot")

        def on_host(value):
            return value.detach().to(device="cpu", copy=True)

        local_ids = tail_logical_indices(history.anchors, window)
        local = on_host(history.swa.gather(local_ids))
        local_length = min(int(history.anchors[0]), window)
        swa = PagedRows(local, torch.zeros((1, 1), dtype=torch.long))
        selected, inverse = selection.flatten().unique(sorted=True, return_inverse=True)
        values = on_host(history.long_kv.gather(selected.unsqueeze(0)))
        selected_rows = inverse.view_as(selection).masked_fill(selection < 0, -1)
        return cls(
            *(on_host(t) for t in (x, residual, pre, post, comb, cos, sin)),
            swa, PagedRows(values, torch.zeros((1, 1), dtype=torch.long)),
            on_host(selected_rows), torch.tensor([local_length]), window,
        )


def rms_norm(value, weight, eps):
    fp32 = value.float()
    return (fp32 * (fp32.square().mean(-1, keepdim=True) + eps).rsqrt() * weight.float()).to(value.dtype)


def apply_rope(value, cos, sin):
    tokens = value.shape[0]
    cos = cos.reshape(tokens, -1).float()
    sin = sin.reshape(tokens, -1).float()
    prefix, rotary = value.split((value.shape[-1] - cos.shape[-1], cos.shape[-1]), dim=-1)
    broadcast = (tokens,) + (1,) * (value.ndim - 2) + (cos.shape[-1],)
    even, odd = rotary[..., ::2].float(), rotary[..., 1::2].float()
    rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
    rotary = rotary.float() * cos.view(broadcast) + rotated * sin.view(broadcast)
    return torch.cat((prefix, rotary.to(value.dtype)), dim=-1)


def hc_post(value, residual, post, comb):
    mixed = torch.einsum("tij,tid->tjd", comb.float(), residual.float())
    return (mixed + post.float().unsqueeze(-1) * value.float().unsqueeze(-2)).to(value.dtype)


def hc_pre(residual, pre, weight, scale, bias, *, eps, hc_eps, iterations):
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, weight.float()) * (flat.square().mean(-1, keepdim=True) + eps).rsqrt()
    copies = residual.shape[1]
    pre_logits, post_logits, comb_logits = mixes.split((copies, copies, copies**2), dim=-1)
    pre_bias, post_bias, comb_bias = bias.float().split((copies, copies, copies**2))
    next_pre = (pre_logits * scale[0] + pre_bias).sigmoid() + hc_eps
    post = 2 * (post_logits * scale[1] + post_bias).sigmoid()
    comb = (comb_logits * scale[2] + comb_bias).unflatten(-1, (copies, copies)).softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    value = (pre.float().unsqueeze(-1) * residual.float()).sum(-2).to(residual.dtype)
    return value, post, comb, next_pre


def replay_logits(
    replay: LastLayerReplay,
    head: Mapping[str, torch.Tensor],
    tail: Mapping[str, torch.Tensor],
    moe: Callable[[torch.Tensor], torch.Tensor],
    *,
    eps: float,
    hc_eps: float,
    iterations: int,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    row_reduce: Callable[[torch.Tensor], torch.Tensor],
    reached: int,
) -> torch.Tensor:
    """Compute next-token logits with fixed per-query Top-K from layer 36."""
    x = replay.attention_input
    batch, width, _ = replay.global_indices.shape
    heads = head["attn_sink"].numel()
    dim = head["wkv.weight"].shape[0]
    if batch != 1 or x.shape[0] != width or not 1 <= reached < width:
        raise ValueError("one complete draft block and its reached MASK rows are required")

    q_low = rms_norm(F.linear(x, head["wq_a.weight"]), head["q_norm.weight"], eps)
    query = F.linear(q_low, head["wq_b.weight"]).view(width, heads, dim)
    latent = rms_norm(F.linear(x, head["wkv.weight"]), head["kv_norm.weight"], eps)
    query = apply_rope(query, replay.cos, replay.sin)
    latent = apply_rope(latent, replay.cos, replay.sin)
    attended, _ = precise_draft_attention(
        query.unsqueeze(0), latent.unsqueeze(0), replay.swa, replay.anchors,
        head["attn_sink"], window=replay.window,
        global_kv=replay.global_kv, global_indices=replay.global_indices,
    )
    attended = apply_rope(attended[0].to(x.dtype), replay.cos, -replay.sin)
    groups, channels, _ = head["wo_a.weight"].shape
    grouped = torch.einsum("tgc,gcr->tgr", attended.reshape(width, groups, channels), head["wo_a.weight"])
    projected = row_reduce(F.linear(grouped.flatten(1), head["wo_b.weight"]))

    # Attention already used all K rows. The remaining operations are per
    # token, so only source rows for the reached predictions need replay.
    rows = slice(0, reached)
    residual = hc_post(projected[rows], replay.residual[rows], replay.attn_post[rows], replay.attn_comb[rows])
    x, post, comb, pre = hc_pre(
        residual, replay.attn_pre[rows],
        tail["hc_ffn_fn"], tail["hc_ffn_scale"], tail["hc_ffn_base"],
        eps=eps, hc_eps=hc_eps, iterations=iterations,
    )
    x = rms_norm(x, tail["ffn_norm"], eps)
    residual = hc_post(moe(x), residual, post, comb)
    output = (pre.unsqueeze(-1) * residual.float()).sum(-2).to(x.dtype)
    return lm_head(rms_norm(output, tail["final_norm"], eps))


def last_layer_loss(student_logits, verifier_logits, scheduled_drafts, num_sampled):
    """The last sampled row is the correction, or the bonus after full acceptance."""
    reached = min(scheduled_drafts, num_sampled)
    if reached < 1 or student_logits.shape[0] != reached or verifier_logits.shape[0] != scheduled_drafts + 1:
        raise ValueError("student and verifier must agree on the reached proposal rows")
    return divergence(student_logits, verifier_logits[:reached], "tv").mean()


class FrozenExpert(torch.autograd.Function):
    """Recompute one fixed expert to obtain its input gradient."""

    @staticmethod
    def forward(ctx, rows, execute):
        ctx.execute = execute
        ctx.save_for_backward(rows)
        return execute(rows)

    @staticmethod
    def backward(ctx, gradient):
        with torch.enable_grad():
            (rows,) = ctx.saved_tensors
            rows = rows.detach().requires_grad_(True)
            output = ctx.execute(rows)
            (input_grad,) = torch.autograd.grad(output, rows, gradient)
        return input_grad, None


class W4A8LastLayerWeights:
    """Frozen FP32 replay using the checkpoint's original INT4/INT8 weights."""

    def __init__(self, model: Path):
        self.model = Path(model)
        self.index = read_json(find_weight_index(self.model))["weight_map"]
        self.description = read_json(self.model / "quant_model_description.json")
        config = read_json(self.model / "config.json")
        self.config = text_config(config)
        role = V41Topology.from_config(config).layers[-1]
        if role.ratio != 1 or role.mode != "reuse" or self.config["scoring_func"] != "sqrtsoftplus":
            raise ValueError("expected the V4.1 1017 final C1A Reuse layer and sqrtsoftplus MoE")
        self.prefix = f"layers.{role.layer}."

    def load(self, names):
        from safetensors import safe_open

        shards = {}
        for name in names:
            shards.setdefault(self.index[name], []).append(name)
        values = {}
        for shard, names in shards.items():
            with safe_open(self.model / shard, framework="pt", device="cpu") as weights:
                for name in names:
                    values[name] = weights.get_tensor(name)
        return values

    def linear_weight(self, name, device):
        scheme = self.description[name]
        if scheme == "FLOAT":
            return self.load([name])[name].to(device).float()
        if scheme not in ("W4A8_DYNAMIC", "W8A8_DYNAMIC"):
            raise ValueError(f"unsupported replay quantization for {name}: {scheme}")
        scale_name, offset_name = name + "_scale", name + "_offset"
        values = self.load([name, scale_name, offset_name])
        packed, scale, offset = (values[key] for key in (name, scale_name, offset_name))
        if packed.dtype != torch.int8 or scale.shape != offset.shape or torch.count_nonzero(offset):
            raise ValueError(f"expected symmetric per-channel quantization for {name}")
        if scheme == "W4A8_DYNAMIC":
            low = packed.bitwise_left_shift(4).bitwise_right_shift(4)
            high = packed.bitwise_right_shift(4)
            packed = torch.stack((low, high), dim=1).flatten(0, 1)
        if scale.shape != (packed.shape[0], 1):
            raise ValueError(f"unexpected per-channel scale shape for {name}")
        return packed.to(device).float() * scale.to(device)

    def tail(self, device):
        names = {
            "hc_ffn_fn": self.prefix + "hc_ffn_fn",
            "hc_ffn_scale": self.prefix + "hc_ffn_scale",
            "hc_ffn_base": self.prefix + "hc_ffn_base",
            "ffn_norm": self.prefix + "ffn_norm.weight",
            "final_norm": "norm.weight",
        }
        tensors = self.load(names.values())
        return {key: tensors[name].to(device) for key, name in names.items()}

    def _expert(self, prefix, rows):
        weights = [self.linear_weight(prefix + f".w{i}.weight", rows.device) for i in (1, 3, 2)]
        gate, up = F.linear(rows, weights[0]), F.linear(rows, weights[1])
        limit = self.config["swiglu_limit"]
        return F.linear(F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit), weights[2])

    def moe(self, value):
        rows = value.float()
        prefix = self.prefix + "ffn."
        gate, bias = self.prefix + "ffn.gate.weight", self.prefix + "ffn.gate.bias"
        weights = self.load([gate, bias])
        scores = F.softplus(F.linear(rows, weights[gate].to(rows).float())).sqrt()
        selected = (scores.detach() + weights[bias].to(rows)).topk(
            self.config["num_experts_per_tok"], dim=-1
        ).indices
        probabilities = scores.gather(-1, selected)
        if self.config["norm_topk_prob"]:
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        probabilities = probabilities * self.config["routed_scaling_factor"]

        # Re-read only the selected experts during backward so the tape does
        # not retain an FP32 copy of every visited W4A8 expert.
        result = FrozenExpert.apply(rows, partial(self._expert, prefix + "shared_experts"))
        for expert in selected.unique().tolist():
            row, slot = torch.where(selected == expert)
            expert_output = FrozenExpert.apply(rows.index_select(0, row), partial(self._expert, prefix + f"experts.{expert}"))
            result = result.index_add(0, row, expert_output * probabilities[row, slot, None])
        return result.to(value.dtype)


class LastLayerLearner:
    """Host reference update, with per-request bounded replay storage."""

    def __init__(self, model, head, learning_rate=1e-5):
        from safetensors import safe_open

        self.base = W4A8LastLayerWeights(model)
        self.tail = self.base.tail("cpu")
        self.lm_head = self.base.load(["head.weight"])["head.weight"].float()
        index = read_json(find_weight_index(Path(head)))["weight_map"]
        prefix = self.base.prefix + "attn."
        names = [name for name in index if name.startswith(prefix)]
        if not names:
            raise ValueError("the head has no tensors for the final layer")
        self.master = {}
        for filename in sorted({index[name] for name in names}):
            with safe_open(Path(head) / filename, framework="pt", device="cpu") as source:
                for name in names:
                    if index[name] == filename:
                        self.master[name.removeprefix(prefix)] = torch.nn.Parameter(source.get_tensor(name).float())
        self.optimizer = torch.optim.AdamW(self.master.values(), lr=learning_rate, weight_decay=0)
        self.version = 0

    @torch.inference_mode(False)
    @torch.enable_grad()
    def update(self, replay, teacher, scheduled, sampled):
        if self.version == 0:
            # Bound host work during the synchronous reference update. This
            # worker runs inference on NPU and only rank 0 runs the learner.
            torch.set_num_threads(min(6, torch.get_num_threads()))
        replay = replace(replay, attention_input=replay.attention_input.float(), residual=replay.residual.float())
        weights = dict(self.master)
        weights["wo_a.weight"] = weights["wo_a.weight"].reshape(
            self.base.config["o_groups"], self.base.config["o_lora_rank"], -1,
        ).transpose(1, 2)
        self.optimizer.zero_grad(set_to_none=True)
        config = self.base.config
        logits = replay_logits(
            replay, weights, self.tail, self.base.moe,
            eps=config["rms_norm_eps"], hc_eps=config["hc_eps"], iterations=config["hc_sinkhorn_iters"],
            lm_head=partial(F.linear, weight=self.lm_head), row_reduce=lambda value: value,
            reached=min(scheduled, sampled),
        )
        loss = last_layer_loss(logits, teacher.float().clone(), scheduled, sampled)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite V4.1 online loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.master.values(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        if any(not value.isfinite().all() or not value.bfloat16().isfinite().all() for value in self.master.values()):
            raise FloatingPointError("nonfinite V4.1 draft parameters after update")
        self.version += 1
        return float(loss.detach()), float(norm), self.version
