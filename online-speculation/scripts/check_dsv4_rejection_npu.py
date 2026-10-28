#!/usr/bin/env python3
"""Exercise the installed Ascend rejection sampler, without loading a model.

This validates the sampler, not scheduler rollback or trained-head quality.
Only engine configuration is supplied by the harness; the sampling kernels are
the same installed functions called by the service.
"""

import argparse
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from vllm_ascend.sample import rejection_sampler as sampler
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from blockspec.vllm_ascend.draft_sampling import sample_seeded_parallel_draft


def seeded_drafts(device):
    logits = torch.randn(3, 31, 127, device=device)
    temperatures = torch.tensor([0.0, 0.8, 1.0], device=device)
    generators = {i: torch.Generator(device=device).manual_seed(i) for i in range(3)}
    ids, probabilities = sample_seeded_parallel_draft(
        logits.flatten(0, 1), temperatures, generators, 31
    )
    order = torch.tensor([2, 0, 1], device=device)
    generators = {i: torch.Generator(device=device).manual_seed(seed)
                  for i, seed in enumerate([2, 0, 1])}
    torch.rand(1000, device=device)
    reordered, _ = sample_seeded_parallel_draft(
        logits[order].flatten(0, 1), temperatures[order], generators, 31
    )
    assert torch.equal(reordered.view(3, 31), ids.view(3, 31)[order])
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(93, device=device))
    print(json.dumps({"case": "seeded_parallel_draft_reordering", "passed": True}), flush=True)


def sample(ids, q, p, bonus, temperatures, generators=None):
    batch, width = ids.shape
    metadata = SimpleNamespace(
        all_greedy=bool((temperatures == 0).all()),
        all_random=bool((temperatures > 0).all()),
        temperature=temperatures,
        generators=generators or {},
    )
    lengths = torch.arange(1, batch + 1, device=ids.device, dtype=torch.int32) * width
    return sampler.rejection_sample(
        ids.flatten().contiguous(), [width] * batch, width, lengths,
        q.reshape(-1, q.shape[-1]).contiguous(),
        p.reshape(-1, p.shape[-1]).log().contiguous(), bonus.contiguous(), metadata,
    )


def rejection_positions(device):
    # Row r accepts r tokens, then rejects. The last row accepts all 31.
    width, batch, vocab = 31, 32, 4
    ids = torch.zeros((batch, width), device=device, dtype=torch.int32)
    q = torch.zeros((batch, width, vocab), device=device)
    q[..., 0] = 1
    p = q.clone()
    expected = torch.full((batch, width + 1), -1, dtype=torch.int32)
    for r in range(batch):
        expected[r, :r] = 0
        expected[r, r] = 1 if r < width else 2
        if r < width:
            p[r, r, 0], p[r, r, 1] = 0, 1
    bonus = torch.full((batch, 1), 2, device=device, dtype=torch.int32)
    for mode in ("greedy", "random", "mixed"):
        temperatures = torch.ones(batch, device=device)
        if mode == "greedy":
            temperatures.zero_()
        elif mode == "mixed":
            temperatures[::2] = 0
        output = sample(ids, q, p, bonus, temperatures).cpu()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        print(json.dumps({"case": mode, "rejection_positions": 31,
                          "all_accept_bonus": True, "passed": True}), flush=True)


def distribution(device, samples):
    # Independent target positions make every returned token's marginal known.
    # Select the first verified position: unlike an API's first output token,
    # it actually passes through accept/reject and residual-distribution sampling.
    q_row = torch.tensor([0.8, 0.1, 0.1], device=device)
    p_row = torch.tensor([0.1, 0.3, 0.6], device=device)
    width = 3
    ids = torch.multinomial(q_row, samples * width, replacement=True).view(samples, width)
    q = q_row.expand(samples, width, -1).contiguous()
    p = p_row.expand_as(q).contiguous()
    bonus = torch.multinomial(p_row, samples, replacement=True).view(samples, 1)
    output = sample(ids, q, p, bonus, torch.ones(samples, device=device))
    observed = torch.bincount(output[:, 0].long(), minlength=3).float() / samples
    tolerance = 6 * (p_row * (1 - p_row) / samples).sqrt() + 1 / samples
    assert bool(((observed - p_row).abs() <= tolerance).all()), observed.cpu().tolist()
    print(json.dumps({"case": "distribution", "samples": samples,
                      "target": p_row.cpu().tolist(), "observed": observed.cpu().tolist(),
                      "six_sigma_tolerance": tolerance.cpu().tolist(), "passed": True}), flush=True)


def seeded_rejection(device):
    ids = torch.zeros((4, 31), device=device, dtype=torch.int32)
    q = torch.tensor([1.0, 0.0, 0.0], device=device).expand(4, 31, 3).contiguous()
    p = torch.tensor([0.1, 0.3, 0.6], device=device).expand_as(q).contiguous()
    bonus = torch.full((4, 1), 2, device=device, dtype=torch.int32)
    generators = {i: torch.Generator(device=device).manual_seed(i) for i in range(4)}
    batched = sample(ids, q, p, bonus, torch.ones(4, device=device), generators)
    for i in range(4):
        generator = torch.Generator(device=device).manual_seed(i)
        single = sample(ids[i:i+1], q[i:i+1], p[i:i+1], bonus[i:i+1],
                        torch.ones(1, device=device), {0: generator})
        torch.testing.assert_close(single[0], batched[i], rtol=0, atol=0)
    print(json.dumps({"case": "seeded_rejection_batch_independence", "passed": True}), flush=True)


def zero_draft_rng(device):
    # A fresh prefill can share a batch with a decoding request. It has no
    # rejected draft to recover, so recovery must not advance its request RNG.
    generators = {i: torch.Generator(device=device).manual_seed(i) for i in range(2)}
    metadata = SimpleNamespace(generators=generators)
    before = {i: g.get_offset() for i, g in generators.items()}
    sampler.sample_recovered_tokens(
        31, [0, 31], torch.tensor([0, 31], device=device, dtype=torch.int32),
        torch.zeros(31, device=device, dtype=torch.int32),
        torch.tensor([1.0, 0.0, 0.0], device=device).expand(31, 3).contiguous(),
        torch.tensor([0.1, 0.3, 0.6], device=device).expand(31, 3).contiguous(),
        metadata, torch.device(device),
    )
    after = {i: g.get_offset() for i, g in generators.items()}
    passed = before[0] == after[0] and after[1] > before[1]
    print(json.dumps({"case": "zero_draft_rng", "before": before, "after": after,
                      "passed": passed}), flush=True)
    assert passed, "recovery consumed the RNG of a request with no draft tokens"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--case", choices=("all", "zero-draft-rng"), default="all")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    init_device_properties_triton()
    torch.manual_seed(20261028)
    if args.case == "all":
        seeded_drafts(args.device)
    config = SimpleNamespace(
        enable_reduce_sample=False,
        rejection_sampler_config=SimpleNamespace(
            enable_block_verify=False, enable_entropy_verify=False,
            posterior_threshold=0.0, posterior_alpha=0.0,
        ),
    )
    with patch.object(sampler, "get_ascend_config", return_value=config):
        if args.case == "all":
            rejection_positions(args.device)
            distribution(args.device, args.samples)
            seeded_rejection(args.device)
        zero_draft_rng(args.device)
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
