"""Vocabulary-parallel loss/gradients against gathered, complete-vocabulary losses."""

import sys

import pytest
import torch
import torch.distributed as dist

from blockspec.losses import divergence
from blockspec.vllm_ascend.qwen38_loss import vocab_divergence
from blockspec.vllm_ascend.qwen38_weights import SharedHead, TensorParallel


@pytest.mark.parametrize("kind", ["forward_kl", "tv"])
def test_single_rank_loss_has_exact_softmax_gradient(kind):
    torch.manual_seed(61)
    student = torch.randn(5, 19, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    weights = torch.randn(5, dtype=torch.float64)
    actual = vocab_divergence(student, teacher, start=0, tp=TensorParallel(), kind=kind)
    reference = divergence(student, teacher, kind)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(actual, student, weights)[0]
    b = torch.autograd.grad(reference, student, weights)[0]
    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
    assert teacher.grad is None
    assert torch.autograd.gradcheck(
        lambda value: vocab_divergence(value, teacher, start=0, tp=TensorParallel(), kind=kind), (student,))


def _vocab_worker(rank, rendezvous, size):
    dist.init_process_group("gloo", rank=rank, world_size=size, init_method=f"file://{rendezvous}")
    try:
        tp = TensorParallel(dist.group.WORLD, tuple(range(size)))
        for vocab in (3, 19, 24):
            width = (vocab + size - 1) // size
            start = min(rank * width, vocab)
            end = min((rank + 1) * width, vocab)
            for kind in ("forward_kl", "tv"):
                for scale in (1., 100.):
                    torch.manual_seed(61)
                    full = (torch.randn(5, vocab) * scale).requires_grad_()
                    teacher = torch.randn_like(full) * scale
                    local = full.detach()[:, start:end].clone().requires_grad_()
                    weights = torch.randn(5)
                    actual = vocab_divergence(local, teacher, start=start, tp=tp, kind=kind)
                    reference = divergence(full, teacher, kind)
                    actual.backward(weights)
                    reference.backward(weights)
                    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
                    torch.testing.assert_close(local.grad, full.grad[:, start:end], rtol=3e-4, atol=2e-6)
        # 冻结 LM head 只回传 hidden 梯度，padding 行不参与 loss 或参数梯度。
        torch.manual_seed(73)
        weight = torch.randn(24, 7)
        hidden = torch.randn(5, 7, requires_grad=True)
        teacher = torch.randn(5, 19)
        with torch.inference_mode():
            shard = torch.nn.Parameter(weight.chunk(size)[rank].clone(), requires_grad=False)
        head = SharedHead(shard, tp, 19)
        local = head.local_logits(hidden)
        actual = vocab_divergence(local, teacher, start=head.vocab_start, tp=tp, kind="forward_kl")
        actual.sum().backward()
        reference = hidden.detach().clone().requires_grad_()
        divergence(torch.nn.functional.linear(reference, weight)[:, :19], teacher).sum().backward()
        torch.testing.assert_close(hidden.grad, reference.grad, rtol=2e-5, atol=2e-5)
        assert shard.grad is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform != "linux", reason="Gloo integration runs in the Linux image")
@pytest.mark.parametrize("size", [2, 4])
def test_sharded_loss_and_frozen_head_gradient_match_full_vocabulary(tmp_path, size):
    torch.multiprocessing.start_processes(_vocab_worker, args=(str(tmp_path / "gloo"), size),
                                          nprocs=size, join=True, start_method="spawn")
