#!/usr/bin/env python3
"""在 NPU 上核对输入准备和状态快照，覆盖真实状态大小及不同槽位。"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output must be new")
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, ExecutionMode
    from blockspec.pypto_qwen38.serving_programs import make_draft_inputs, make_state_copy, make_target_inputs

    torch.set_num_threads(4)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
    rng = torch.Generator().manual_seed(123)
    rand = lambda shape: torch.randn(shape, generator=rng).bfloat16().share_memory_()
    co, si, columns = rand((512, 64)), rand((512, 64)), torch.arange(512, dtype=torch.int32).reshape(1, 512).share_memory_()
    pages = torch.arange(31, -1, -1, dtype=torch.int32).share_memory_()
    ids, dc, ds, table, mask = torch.empty(32, dtype=torch.int32), rand((32, 64)), rand((32, 64)), torch.empty(32, dtype=torch.int32), torch.empty((16, 512))
    counts, offsets = torch.empty(8, dtype=torch.int32), torch.empty(8, dtype=torch.int32)
    tc, ts, tm = rand((8, 16, 64)), rand((8, 16, 64)), torch.empty((8, 16, 512))
    conv_shape, rec_shape = (48, 4, 10240), (48, 48, 128, 128)
    conv = torch.empty((8, *conv_shape), dtype=torch.bfloat16, device="meta")
    rec = torch.empty((8, *rec_shape), dtype=torch.float32, device="meta")
    saved_conv = torch.empty((2, *conv_shape), dtype=torch.bfloat16, device="meta")
    saved_rec = torch.empty((2, *rec_shape), dtype=torch.float32, device="meta")
    input_args = [co, si, columns, pages, ids, dc, ds, table, mask]
    target_args = [co, si, columns, counts, offsets, tc, ts, tm]
    print("compiling input/state kernels", flush=True)
    programs = dict(inputs=make_draft_inputs(mask_token=17)[1].compile(*input_args, 3, 0, 32, config=config),
                    target=make_target_inputs()[1].compile(*target_args, config=config),
                    save=make_state_copy()[1].compile(conv, rec, saved_conv, saved_rec, 0, 0, config=config),
                    restore=make_state_copy(source_slots=2, destination_slots=8)[1].compile(saved_conv, saved_rec, conv, rec, 0, 0, config=config))
    host = list(dict((id(t), t.share_memory_()) for t in [*input_args, *target_args]).values())
    report = dict(passed=False, npu_execution=True, cases=[], artifacts={n: str(p.output_dir) for n, p in programs.items()})
    worker, handles = None, []
    try:
        worker = DistributedWorker(list(programs.values()), config=config, inherited_host_tensors=host)
        device = {}
        for tensor in host:
            handle = worker.alloc_tensor(tuple(tensor.shape), tensor.dtype, init=tensor)
            handles.append(handle)
            device[id(tensor)] = handle
        for past, width in ((0, 32), (15, 17), (16, 32), (479, 32), (510, 2)):
            worker.run(programs["inputs"], *(device[id(t)] for t in input_args), 33, past, width, config=config)
            for tensor in (ids, dc, ds, table, mask):
                worker.copy_from(tensor.data_ptr(), device[id(tensor)].data_ptr, tensor.numel() * tensor.element_size())
            expected_ids = torch.full((32,), 17, dtype=torch.int32)
            expected_ids[0] = 33
            expected_mask = torch.full((16, 512), -10000.)
            expected_mask[:, :past + width] = 0
            positions = torch.arange(past, past + 32).clamp_max(511)
            for actual, expected in ((ids, expected_ids), (dc, co[positions]), (ds, si[positions]), (table, pages), (mask, expected_mask)):
                assert torch.equal(actual, expected), (past, width, actual, expected)
            report["cases"].append(dict(kind="draft_inputs", past=past, width=width, byte_exact=True))
        for count, past in ((1, 0), (7, 15), (16, 16), (16, 496)):
            counts.zero_(); offsets.zero_()
            counts[0], offsets[0] = count, past
            worker.copy_to(device[id(counts)].data_ptr, counts.data_ptr(), counts.numel() * 4)
            worker.copy_to(device[id(offsets)].data_ptr, offsets.data_ptr(), offsets.numel() * 4)
            worker.run(programs["target"], *(device[id(t)] for t in target_args), config=config)
            for tensor in (tc, ts, tm):
                worker.copy_from(tensor.data_ptr(), device[id(tensor)].data_ptr, tensor.numel() * tensor.element_size())
            ec, es, em = torch.ones_like(tc), torch.zeros_like(ts), torch.full_like(tm, -10000.)
            ec[0, :count], es[0, :count] = co[past:past + count], si[past:past + count]
            for row in range(16):
                em[0, row, :min(past + row + 1, past + count)] = 0
            for actual, expected in ((tc, ec), (ts, es), (tm, em)):
                assert torch.equal(actual, expected), ("target", count, past)
            report["cases"].append(dict(kind="target_inputs", past=past, count=count, byte_exact=True))
        state_handles = [worker.alloc_tensor(tuple(t.shape), t.dtype) for t in (conv, rec, saved_conv, saved_rec)]
        handles.extend(state_handles)
        # 独立伪随机原始字节，连 NaN payload 一并检查，避免只比较全零状态。
        state_hosts = [torch.randint(0, 256, shape, dtype=torch.uint8, generator=rng)
                       for shape in ((conv_shape[0] * conv_shape[1] * conv_shape[2] * 2,), (48 * 48 * 128 * 128 * 4,))]
        for slot, buffer in ((0, 0), (7, 1)):
            for host_state, handle in zip(state_hosts, state_handles[:2], strict=True):
                worker.copy_to(handle.data_ptr, host_state.data_ptr(), host_state.numel(), dst_offset=slot * host_state.numel())
            worker.run(programs["save"], *state_handles, slot, buffer, config=config)
            worker.run(programs["restore"], *state_handles[2:], *state_handles[:2], buffer, 3, config=config)
            for original, handle in zip(state_hosts, state_handles[:2], strict=True):
                returned = torch.empty_like(original)
                for read_slot in (slot, 3):
                    worker.copy_from(returned.data_ptr(), handle.data_ptr, returned.numel(), src_offset=read_slot * returned.numel())
                    assert torch.equal(returned, original), ("state", slot, buffer, read_slot)
            report["cases"].append(dict(kind="state_copy", slot=slot, buffer=buffer, destination_slot=3, byte_exact=True))
        report["passed"] = True
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            for handle in reversed(handles):
                worker.free_tensor(handle)
            worker.close()
        a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
