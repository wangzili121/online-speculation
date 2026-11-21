#!/usr/bin/env python3
"""只编译不占卡：Attention 必须拒绝不能覆盖上下文的页表及内部截短 view。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")

    import torch
    import pypto.language as pl
    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig
    from blockspec.pypto_qwen38.attention import AttentionShape
    from blockspec.pypto_qwen38.paged_attention_programs import make_paged_attention

    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    report = dict(completed=False, npu_execution=False, cases=[])

    def tensors(context, pages):
        query = empty((32, 3072))
        block = empty((32, 512))
        pool = empty((2, 2, 12, 128, 256))
        return [query, block, block, query, pool, pool, empty((pages,), torch.int32),
                empty((1, context), torch.float32), query]

    try:
        for context in (512, 1024):
            shape = AttentionShape(q_heads=12, kv_heads=2, context=context, page_size=128)
            paged, entry = make_paged_attention(shape, physical_pages=12, pool_layers=2)
            entry.compile(*tensors(context, context // 128), 127, 8, 1, config=config)
            report["cases"].append(dict(name=f"complete_{context}", passed=True))

        def reject(name, compile_call):
            try:
                compile_call()
            except (ValueError, TypeError, RuntimeError) as error:
                message = str(error)
                if not any(word in message.lower() for word in ("reshape", "shape", "dimension", "numel")):
                    raise
                report["cases"].append(dict(name=name, passed=True, rejected=True, error=message))
            else:
                raise AssertionError("compiler accepted an incomplete page table: " + name)

        reject("short_external_table", lambda: entry.compile(*tensors(1024, 4), 600, 8, 1, config=config))

        @pl.jit
        def cropped(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
                    key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor[[8], pl.INT32],
                    bias: pl.Tensor, output: pl.Out[pl.Tensor]):
            incomplete = pl.slice(table, [4], [0])
            return paged(query, key, value, gate, key_pool, value_pool, incomplete, bias, output, 600, 8, 1)

        @pl.jit.host
        def cropped_entry(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
                          key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor[[8], pl.INT32],
                          bias: pl.Tensor, output: pl.Out[pl.Tensor]):
            cropped(query, key, value, gate, key_pool, value_pool, table, bias, output, device=0)

        reject("cropped_internal_table", lambda: cropped_entry.compile(*tensors(1024, 8), config=config))
        report["completed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
