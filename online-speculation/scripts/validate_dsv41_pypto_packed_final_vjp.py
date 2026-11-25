#!/usr/bin/env python3
"""Check resident final RMSNorm and mHC gradients after a later draft overwrites live state."""

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--fixture-tp", type=int, default=2)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--chain", action="store_true", help="Feed the final VJP from the resident TV loss and LM head VJP")
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")]
    tp, world = args.fixture_tp, len(devices)
    if tp not in (1, 2, 4) or world % tp or len(set(devices)) != world:
        parser.error("Use distinct device IDs with one or more full TP groups")
    if args.chain and tp == 1:
        parser.error("The chained LM head VJP requires TP2 or TP4")

    import torch
    from blockspec.dsv41_packed_final_vjp import PackedFinalVJP, make_packed_final_vjp
    from blockspec.dsv41_packed_lm_head_vjp import PackedLMHeadGradient, PackedLMHeadVJP, make_packed_lm_head_vjp
    from blockspec.dsv41_packed_loss import PackedLoss, make_packed_loss_graph
    from blockspec.dsv41_packed_proposer import PackedFinalState, PackedStudentRows
    from models.deepseek_v4_1_flash_a3 import replay
    from models.deepseek_v4_1_flash_a3.config import FLASH
    import pypto.language as pl
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    rows, hidden, copies, eps = 16, FLASH.hidden_size, FLASH.hc_mult, FLASH.rms_norm_eps
    vocabulary, groups = FLASH.vocab_size, world // tp
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    vjp = make_packed_final_vjp(rows, hidden, copies, groups, tp, eps).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, devices=devices, tp=tp)))
        return

    @pl.jit
    def mutate_state(
        streams: pl.InOut[pl.Tensor[[32, copies, hidden], pl.FP32]],
        premix: pl.InOut[pl.Tensor[[32, copies], pl.FP32]],
    ):
        flat = pl.reshape(streams, [32 * copies, hidden])
        with pl.spmd(rows, name_hint="draft_fixture_hc"):
            start = pl.tile.get_block_idx() * copies
            for lane in pl.range(copies):
                row = start + lane
                flat[row:row + 1, :] = pl.mul(pl.slice(flat, [1, hidden], [row, 0]), .5)
        flat_mix = pl.reshape(premix, [32 // 4, 4 * copies])
        with pl.spmd(rows // 4, name_hint="draft_fixture_pre_mix"):
            row = pl.tile.get_block_idx()
            flat_mix[row:row + 1, :] = pl.mul(pl.slice(flat_mix, [1, 4 * copies], [row, 0]), .25)

    @pl.jit.host
    def draft_fixture(
        streams: pl.InOut[pl.Tensor[[groups, 32, copies, hidden], pl.FP32]],
        premix: pl.InOut[pl.Tensor[[groups, 32, copies], pl.FP32]],
        owner: pl.Scalar[pl.INT32],
    ):
        replica = owner // tp
        mutate_state(streams[replica], premix[replica], device=owner)

    draft_graph = draft_fixture.compile(config=config)
    generator = torch.Generator().manual_seed(72123)
    if args.checkpoint:
        from safetensors import safe_open

        index = json.loads((args.checkpoint / "quant_model_weights.safetensors.index.json").read_text())

        def weight(name):
            with safe_open(args.checkpoint / index["weight_map"][name], framework="pt", device="cpu") as handle:
                return handle.get_tensor(name).to(torch.bfloat16)

        gamma = weight("norm.weight")
    else:
        gamma = torch.linspace(.8, 1.2, hidden, dtype=torch.bfloat16)
    assert gamma.shape == (hidden,)

    # DecodeInputs uses the module's fixed production TP for slot selection.
    # Every owner below is active and receives its dedicated slot 2 image.
    owners = tuple(range(0, world, tp))
    inputs = replay.DecodeInputs(dict(x_hc=torch.zeros(32, copies, hidden), pre_mix=torch.zeros(32, copies)))
    programs = [vjp, draft_graph]
    if args.chain:
        student_logits = torch.randn(rows, vocabulary, generator=generator)
        head_weight = (weight("head.weight") if args.checkpoint else
                       torch.randn(vocabulary, hidden, dtype=torch.bfloat16, generator=generator))
        shards = head_weight.chunk(tp, dim=0)
        loss_graph = make_packed_loss_graph(rows, vocabulary, groups, tp, "tv").compile(config=config)
        head_vjp = make_packed_lm_head_vjp(rows, vocabulary, hidden, groups, tp).compile(config=config)
        programs.extend((loss_graph, head_vjp))

    cases = []
    with DistributedWorker(programs, config, persistent=True, reset_persistent_windows=False) as runtime:
        ranks = [dict(inputs.bind(runtime, rank), final_norm_weight=runtime.alloc_tensor(
            gamma.shape, gamma.dtype, init=gamma, worker_id=rank)) for rank in range(world)]
        proposer = SimpleNamespace(runtime=runtime, config=config, ranks=ranks, world=world, tp=tp)
        if args.chain:
            for rank, bindings in enumerate(ranks):
                shard = shards[rank % tp]
                bindings["lm_head_weight"] = runtime.alloc_tensor(shard.shape, shard.dtype, init=shard, worker_id=rank)
                bindings["logits"] = runtime.alloc_tensor(student_logits.shape, student_logits.dtype,
                                                          init=student_logits, worker_id=rank)
            loss = PackedLoss(proposer, loss_graph)
            lm_head = PackedLMHeadVJP(proposer, loss, head_vjp)
        else:
            normed_grad = [runtime.alloc_tensor((rows, hidden), torch.float32, worker_id=rank) for rank in range(world)]
            lm_head = SimpleNamespace(loss=SimpleNamespace(rows=rows, owners=owners),
                                      output=SimpleNamespace(shards=normed_grad))
        reverse = PackedFinalVJP(proposer, lm_head, vjp)
        for owner, valid in [(0, 16), (owners[-1], 7), (0, 1)]:
            x = torch.randn((rows, copies, hidden), generator=generator)
            mix = torch.randn((rows, copies), generator=generator).softmax(-1)
            inputs.fields[2]["x_hc"][:rows].copy_(x)
            inputs.fields[2]["pre_mix"][:rows].copy_(mix)
            saved, views = inputs.reserve(runtime, owner)
            inputs.upload(runtime, owner, owner, allocation=saved)
            stack = []
            for name in ("x_hc", "pre_mix"):
                rank_views = [views[name] if rank == owner else ranks[rank][name] for rank in owners]
                stack.append(StackedDeviceTensor(rank_views, (groups, *views[name].shape), owners))
            runtime.run(draft_graph, *stack, owner, config=config)
            student = PackedStudentRows(owner, 48, valid, ranks[owner].get("logits"),
                                        PackedFinalState(saved, views["x_hc"], views["pre_mix"]))
            inputs.fields[2]["x_hc"].fill_(31.)
            inputs.fields[2]["pre_mix"].zero_()
            inputs.upload(runtime, owner, owner)
            dy = torch.zeros(rows, hidden)
            if args.chain:
                teacher = student_logits[:valid] + .22 * torch.randn((valid, vocabulary), generator=generator)
                head = lm_head(loss(student, teacher))
            else:
                dy[:valid].normal_(generator=generator)
                runtime.copy_to(normed_grad[owner].data_ptr, dy.data_ptr(), dy.nbytes, worker_id=owner)
                head = PackedLMHeadGradient(owner, valid, normed_grad[owner], 0.)
            outputs = [reverse(student, head) for _ in range(10)]
            result = outputs[-1]
            if args.chain:
                runtime.copy_from(dy.data_ptr(), head.gradient.data_ptr, dy.nbytes, worker_id=owner)
            observed = [torch.empty((rows, copies, hidden)), torch.empty((rows, 16))]
            for cpu, dev in zip(observed, (result.streams, result.premix), strict=True):
                runtime.copy_from(cpu.data_ptr(), dev.data_ptr, cpu.nbytes, worker_id=owner)
            runtime.free_tensor(saved, worker_id=owner)

            streams, premix = (x.double() * .5).requires_grad_(), (mix.double() * .25).requires_grad_()
            products = [streams[:, i] * premix[:, i, None] for i in range(copies)]
            # Use the native FP32 pairwise accumulation and BF16 boundary.
            rounded = ((products[0].float() + products[1].float()) +
                       (products[2].float() + products[3].float())).bfloat16().double()
            unrounded = sum(products)
            collapse = rounded.detach() + (unrounded - unrounded.detach())
            normed = collapse * torch.rsqrt(collapse.square().mean(-1, keepdim=True) + eps) * gamma.double()
            expected = torch.autograd.grad(normed, (streams, premix), grad_outputs=dy.double())
            errors = []
            for cpu, ref in zip((observed[0], observed[1][:, :copies]), expected, strict=True):
                delta = cpu.double() - ref
                relative = float(delta.norm() / ref.norm())
                assert relative < 2e-5, (owner, valid, relative, float(delta.abs().max()))
                assert not torch.any(cpu[valid:])
                errors.append(dict(relative_l2=relative, max_abs=float(delta.abs().max())))
            assert not torch.any(observed[1][:, copies:])
            cases.append(dict(owner=owner, rows=valid, streams=errors[0], premix=errors[1],
                              median_ms=1000 * median(row.seconds for row in outputs[1:])))
    print(json.dumps(dict(passed=True, executed=True, devices=devices, tp=tp, cases=cases,
                          checkpoint=str(args.checkpoint or "synthetic"), loss_lm_head_chain=args.chain)), flush=True)


if __name__ == "__main__":
    main()
