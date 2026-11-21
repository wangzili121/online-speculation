#!/usr/bin/env python3
"""检查双请求 16 行布局的搬运、反馈和只读 GDN；不加载主模型。"""

import argparse
import json
from pathlib import Path


def layout_entry():
    import pypto.language as pl
    from blockspec.pypto_qwen38.tp2_row_layout import make_pack, make_expand, make_feedback
    pack, expand, capture = make_pack(2, 16, 5120), make_expand(2, 16, 5120), make_feedback(16)

    @pl.jit
    def kernel(source: pl.Tensor, packed: pl.Out[pl.Tensor], restored: pl.Out[pl.Tensor],
               feedback: pl.Out[pl.Tensor], start: pl.Scalar[pl.INT32]):
        value = pack(source)
        with pl.spmd(20, name_hint="orthrus_layout_probe_output"):
            task = pl.tile.get_block_idx()
            row = task // 10 * 16
            col = task % 10 * 512
            packed[row:row + 16, col:col + 512] = pl.slice(value, [16, 512], [row, col])
        restored = expand(value, restored)
        feedback = capture(value, feedback, start)
        return packed, restored, feedback

    @pl.jit.host
    def entry(source: pl.Tensor, packed: pl.Out[pl.Tensor], restored: pl.Out[pl.Tensor],
              feedback: pl.Out[pl.Tensor], start: pl.Scalar[pl.INT32]):
        return kernel(source, packed, restored, feedback, start, device=0)
    return entry


def main(guard=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output path')
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.tp2_gdn_convolution import make_entry as convolution
    from blockspec.pypto_qwen38.tp2_gdn_rows import make_entry as recurrence
    from blockspec.pypto_qwen38.attention import AttentionShape, attention_golden
    from blockspec.pypto_qwen38.paged_attention_programs import make_paged_attention
    from validate_qwen38_pypto_gdn_rows import golden
    torch.set_num_threads(2)
    torch.manual_seed(234)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
    meta = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device='meta')
    programs = [layout_entry().compile(meta((64, 5120)), meta((32, 5120)), meta((64, 5120)),
        meta((32, 5120)), 0, config=config)]
    conv, recur, attention = {}, {}, {}
    for rows in (32, 16):
        conv[rows] = convolution(2, backend='parallel', channel_tile=128, block_rows=rows).compile(
            meta((2 * rows, 5120)), meta((4, 5120)), meta((32, 48, 4, 5120)),
            meta((2 * rows, 5120)), 3, 21, 47, config=config)
        inputs = [meta((2 * rows, 3072), torch.float32)] * 2 + [meta((2 * rows, 5120))]
        inputs += [meta((2 * rows, 32), torch.float32)] * 2
        recur[rows] = recurrence(2, valid_rows=True, history_slots=2, block_rows=rows).compile(
            *inputs, meta((2, 48, 24, 128, 128), torch.float32), meta((2 * rows, 3072)),
            0, 1, 47, 8, 6, config=config)
        shape = AttentionShape(block_size=rows, q_heads=12, kv_heads=2, page_size=128, query_tile=rows)
        attention[rows] = make_paged_attention(shape, physical_pages=8, pool_layers=17)[1].compile(
            meta((rows, 3072)), meta((rows, 512)), meta((rows, 512)), meta((rows, 3072)),
            meta((17, 2, 8, 128, 256)), meta((17, 2, 8, 128, 256)),
            meta((4,), torch.int32), meta((1, 512), torch.float32), meta((rows, 3072)),
            127, 8, 15, config=config)
    programs += [*conv.values(), *recur.values(), *attention.values()]
    report = dict(completed=False, passed=False, clean_shutdown=False, cases=[],
        scope='One-card real TP2-local dimensions, different request histories; layout/feedback and GDN only, not throughput')
    worker, handles = None, []
    def alloc(value):
        handle = worker.alloc_tensor(value.shape, value.dtype, init=value.contiguous())
        handles.append(handle)
        return handle
    def read(handle, shape, dtype=torch.bfloat16):
        value = torch.empty(shape, dtype=dtype)
        worker.copy_from(value.data_ptr(), handle.data_ptr, value.nbytes)
        return value
    def packed(value):
        return torch.cat((value[:16], value[32:48])).contiguous()
    def bitwise(a, b):
        return torch.equal(a.view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32),
                           b.view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32))
    try:
        worker = DistributedWorker(programs, config=config)
        if guard is not None:
            guard.release()
            guard.close()
        source = torch.randn(64, 5120).bfloat16()
        device_source = alloc(source)
        pack_out = alloc(torch.full((32, 5120), float('nan'), dtype=torch.bfloat16))
        restore_out = alloc(torch.full_like(source, float('nan')))
        feedback_out = alloc(torch.full((32, 5120), float('nan'), dtype=torch.bfloat16))
        for start in (0, 32):
            worker.run(programs[0], device_source, pack_out, restore_out, feedback_out, start, config=config)
            assert bitwise(read(pack_out, (32, 5120)), packed(source))
            restored = read(restore_out, (64, 5120))
            assert bitwise(packed(restored), packed(source))
            assert torch.count_nonzero(restored[16:32]) == torch.count_nonzero(restored[48:]) == 0
            feedback = read(feedback_out, (32, 5120))
            assert bitwise(feedback[:16], source[start:start + 16])
            assert torch.count_nonzero(feedback[16:]) == 0
        assert bitwise(read(device_source, source.shape), source)
        report['cases'].append(dict(kind='pack/expand/feedback', starts=[0, 32], active_bitwise=True,
                                    tail_zero=True, input_unchanged=True))
        projected, weight = torch.randn(64, 5120).bfloat16(), torch.randn(4, 5120).mul_(.1).bfloat16()
        history = torch.randn(32, 48, 4, 5120).mul_(.1).bfloat16()
        h, w = alloc(history), alloc(weight)
        outputs = []
        for rows in (32, 16):
            x = projected if rows == 32 else packed(projected)
            inputs, out = alloc(x), alloc(torch.full_like(x, float('nan')))
            worker.run(conv[rows], inputs, w, h, out, 3, 21, 47, config=config)
            outputs.append(read(out, x.shape))
        assert bitwise(packed(outputs[0]), outputs[1])
        assert bitwise(read(h, history.shape), history)
        report['cases'].append(dict(kind='convolution', slots=[3, 21], layer=47, active_bitwise=True,
                                    nonzero_history_unchanged=True))
        while handles:
            worker.free_tensor(handles.pop())
        history = torch.randn(2, 48, 24, 128, 128).mul_(.02)
        h = alloc(history)
        for widths in ((8, 6), (16, 2), (2, 16)):
            q = torch.nn.functional.normalize(torch.randn(64, 24, 128), dim=-1).reshape(64, 3072)
            k = torch.nn.functional.normalize(torch.randn(64, 24, 128), dim=-1).reshape(64, 3072)
            v = torch.randn(64, 5120).mul_(.1).bfloat16()
            decay, beta = -torch.rand(64, 32).mul_(.05), torch.rand(64, 32)
            outputs = []
            for rows in (32, 16):
                inputs = [alloc(x if rows == 32 else packed(x)) for x in (q, k, v, decay, beta)]
                out = alloc(torch.full((2 * rows, 3072), float('nan'), dtype=torch.bfloat16))
                worker.run(recur[rows], *inputs, h, out, 0, 1, 47, *widths, config=config)
                outputs.append(read(out, (2 * rows, 3072)))
            assert bitwise(packed(outputs[0]), outputs[1])
            expected = packed(golden(q, k, v, decay, beta, history, widths))
            active = torch.cat([torch.arange(i * 16, i * 16 + width) for i, width in enumerate(widths)])
            delta = outputs[1][active].float() - expected[active].float()
            relative = float(delta.norm() / expected[active].float().norm())
            assert relative < .01 and float(delta.abs().max()) < .002
            for i, width in enumerate(widths):
                assert torch.count_nonzero(outputs[1][i * 16 + width:(i + 1) * 16]) == 0
            report['cases'].append(dict(kind='recurrence', widths=widths, active_bitwise=True, tail_zero=True,
                CPU_relative_L2=relative, CPU_max_abs=float(delta.abs().max())))
        assert bitwise(read(h, history.shape, torch.float32), history)
        while handles:
            worker.free_tensor(handles.pop())
        # 覆盖真实短诊断未到达的 512 档和页边界；无效页及临时尾行填 NaN。
        for past, width in ((0, 8), (127, 8), (128, 16), (255, 8), (256, 16), (496, 16), (504, 8)):
            pages = (past + 127) // 128
            table = torch.tensor([5, 1, 7, 2], dtype=torch.int32)
            table[pages:] = -1
            pools = [torch.full((17, 2, 8, 128, 256), float('nan'), dtype=torch.bfloat16) for _ in range(2)]
            for pool in pools:
                for page in range(pages):
                    count = min(128, past - page * 128)
                    pool[15, :, table[page], :count] = torch.randn(2, count, 256).mul_(.1).bfloat16()
            q, k, v, gate = [torch.randn(32, columns).mul_(.1).bfloat16()
                             for columns in (3072, 512, 512, 3072)]
            k[width:] = float('nan')
            v[width:] = float('nan')
            bias = torch.full((1, 512), -10000.)
            bias[:, :past + width] = 0
            cache = [alloc(x) for x in pools]
            device_table, device_bias = alloc(table), alloc(bias)
            actual = []
            for rows in (32, 16):
                inputs = [alloc(x[:rows]) for x in (q, k, v, gate)]
                out = alloc(torch.full((rows, 3072), float('nan'), dtype=torch.bfloat16))
                worker.run(attention[rows], *inputs, *cache, device_table, device_bias, out,
                           past, width, 15, config=config)
                actual.append(read(out, (rows, 3072))[:width])
            shape = AttentionShape(block_size=16, q_heads=12, kv_heads=2, page_size=128, query_tile=16)
            expected = attention_golden(shape, q[:16], k[:16], v[:16], gate[:16],
                pools[0][15], pools[1][15], table, past, width)[:width]
            for value in actual:
                torch.testing.assert_close(value.float(), expected.float(), rtol=.03, atol=.004)
            torch.testing.assert_close(actual[1].float(), actual[0].float(), rtol=.02, atol=.002)
            delta = actual[1].float() - actual[0].float()
            relative = float(delta.norm() / actual[0].float().norm().clamp_min(1e-30))
            assert relative < .01
            for handle, original in zip(cache, pools, strict=True):
                assert bitwise(read(handle, original.shape), original)
            report['cases'].append(dict(kind='paged_attention', past=past, width=width,
                active_bitwise=bitwise(*actual), maximum_layout_abs=float(delta.abs().max()),
                layout_relative_L2=relative, independent_SDPA_passed=True,
                poisoned_inactive_inputs=True, history_bitwise_unchanged=True))
            while handles:
                worker.free_tensor(handles.pop())
        report.update(passed=True, history_bitwise_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        try:
            if worker is not None:
                while handles:
                    worker.free_tensor(handles.pop())
                worker.close()
                report['clean_shutdown'] = True
        finally:
            report['completed'] = report['passed'] and report['clean_shutdown']
            args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
