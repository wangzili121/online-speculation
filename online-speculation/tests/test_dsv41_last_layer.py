from types import SimpleNamespace

import torch
import torch.nn.functional as F

from blockspec.vllm_ascend.dsv41_attention import PagedRows
from blockspec.vllm_ascend.dsv41_last_layer import (
    LastLayerReplay,
    W4A8LastLayerWeights,
    last_layer_loss,
    replay_logits,
)


def test_last_layer_replays_reached_rows_and_leaves_shared_tensors_frozen():
    torch.manual_seed(17)
    width, hidden, dim = 32, 6, 4
    x = torch.randn(width, hidden)
    residual = torch.randn(width, 2, hidden)
    pre = torch.rand(width, 2)
    post = torch.rand(width, 2)
    comb = torch.rand(width, 2, 2)
    angles = torch.randn(width, 1).repeat_interleave(2, dim=-1)
    swa = PagedRows(torch.randn(2, 4, dim), torch.tensor([[1, 0]]))
    long = PagedRows(torch.randn(2, 4, dim), torch.tensor([[0, 1]]))
    selected = torch.tensor([1, 4, -1]).repeat(1, width, 1)
    history = SimpleNamespace(swa=swa, long_kv=long, anchors=torch.tensor([6]))
    with torch.inference_mode():
        snapshot = LastLayerReplay.snapshot(
            x, residual, pre, post, comb, angles.cos(), angles.sin(), history, selected, 4,
        )
    assert not torch.is_inference(snapshot.attention_input)
    torch.testing.assert_close(
        snapshot.global_kv.gather(snapshot.global_indices), long.gather(selected)
    )
    saved_swa = swa.values.clone()
    saved_long = long.values.clone()
    head = {
        name: torch.nn.Parameter(torch.randn(shape) * 0.2)
        for name, shape in {
            "wq_a.weight": (3, hidden), "wq_b.weight": (2 * dim, 3), "wkv.weight": (dim, hidden),
            "wo_a.weight": (2, dim, 2), "wo_b.weight": (hidden, 4), "attn_sink": (2,),
        }.items()
    }
    head["q_norm.weight"] = torch.nn.Parameter(torch.ones(3))
    head["kv_norm.weight"] = torch.nn.Parameter(torch.ones(dim))
    tail = {
        "hc_ffn_fn": torch.randn(8, 2 * hidden) * 0.2,
        "hc_ffn_scale": torch.ones(3),
        "hc_ffn_base": torch.zeros(8),
        "ffn_norm": torch.ones(hidden),
        "final_norm": torch.ones(hidden),
    }
    experts = torch.randn(hidden, hidden) * 0.2
    vocab = torch.randn(11, hidden)
    options = dict(eps=1e-5, hc_eps=1e-6, iterations=3, lm_head=lambda v: F.linear(v, vocab),
                   row_reduce=lambda v: v)
    all_rows = replay_logits(snapshot, head, tail, lambda v: F.linear(v, experts), reached=31, **options)
    got = replay_logits(snapshot, head, tail, lambda v: F.linear(v, experts), reached=2, **options)
    torch.testing.assert_close(got, all_rows[:2])
    teacher = torch.randn(4, 11, requires_grad=True)
    parameters_before = {name: p.detach().clone() for name, p in head.items()}
    optimizer = torch.optim.SGD(head.values(), lr=0.05)
    loss = last_layer_loss(got, teacher, 3, 2)
    loss.backward()
    assert teacher.grad is None
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0 for p in head.values())
    optimizer.step()
    next_logits = replay_logits(snapshot, head, tail, lambda v: F.linear(v, experts), reached=2, **options)
    assert last_layer_loss(next_logits, teacher, 3, 2) < loss
    assert all(not torch.equal(p, parameters_before[name]) for name, p in head.items())
    torch.testing.assert_close(swa.values, saved_swa)
    torch.testing.assert_close(long.values, saved_long)


def test_modelslim_signed_nibbles_unpack_in_output_channel_order():
    base = object.__new__(W4A8LastLayerWeights)
    name = "expert.weight"
    base.description = {name: "W4A8_DYNAMIC"}
    base.load = lambda names: {
        name: torch.tensor([[120, -97], [30, -57]], dtype=torch.int8),
        name + "_scale": torch.arange(1, 5).float().unsqueeze(-1),
        name + "_offset": torch.zeros(4, 1),
    }
    expected = torch.tensor([[-8., -1.], [14., -14.], [-6., 21.], [4., -16.]])
    torch.testing.assert_close(base.linear_weight(name, "cpu"), expected)
