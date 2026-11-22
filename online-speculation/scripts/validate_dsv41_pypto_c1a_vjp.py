#!/usr/bin/env python3
"""Run the production-shape Orthrus-DSV4.1 C1A VJP on one NPU."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")

    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_dsv41_attention_vjp import (
        C1AAttentionVJPGeometry,
        c1a_vjp_factories,
    )

    torch.manual_seed(4101)
    torch.set_num_threads(4)
    geometry = C1AAttentionVJPGeometry()
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
    )
    empty = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
    factories = c1a_vjp_factories(geometry)
    programs = {
        "joined_history": factories["joined_history"]().compile(
            empty((128, geometry.dimension)), empty((2, 3), torch.int32),
            empty((512, geometry.dimension)),
            empty((geometry.block, geometry.compressed), torch.int32),
            empty((geometry.block,), torch.int32),
            empty((geometry.block, geometry.dimension)),
            empty((128, geometry.dimension)),
            empty((geometry.block * geometry.padded_columns, geometry.dimension)),
            config=config,
        ),
        "probability_replay": factories["probability_replay"]().compile(
            empty((geometry.rows, geometry.dimension)),
            empty((geometry.block * geometry.padded_columns, geometry.dimension)),
            empty((1, geometry.swa), torch.float32),
            empty((geometry.block, geometry.compressed), torch.float32),
            empty((geometry.local_heads,), torch.float32),
            empty((geometry.rows, 1), torch.float32),
            empty((geometry.rows, geometry.padded_columns), torch.float32),
            config=config,
        ),
        "probability_vjp": factories["probability_vjp"]().compile(
            empty((geometry.rows, geometry.padded_columns), torch.float32),
            empty((geometry.rows, geometry.padded_columns)),
            empty((geometry.rows, geometry.padded_columns)),
            empty((geometry.rows, 16), torch.float32), config=config,
        ),
        "sink_reduce": factories["sink_reduce"]().compile(
            empty((geometry.rows, 16), torch.float32),
            empty((geometry.local_heads, 16), torch.float32), config=config,
        ),
        "score_block_extract": factories["score_block_extract"]().compile(
            empty((geometry.rows, geometry.padded_columns)),
            empty((geometry.rows, geometry.block)), config=config,
        ),
        "probability_block_extract": factories["probability_block_extract"]().compile(
            empty((geometry.rows, geometry.padded_columns), torch.float32),
            empty((geometry.rows, geometry.block)), config=config,
        ),
        "d_probability": factories["d_probability"]().compile(
            empty((geometry.rows, geometry.dimension)),
            empty((geometry.block * geometry.padded_columns, geometry.dimension)),
            empty((geometry.rows, geometry.padded_columns)), config=config,
        ),
        "d_query": factories["d_query"]().compile(
            empty((geometry.rows, geometry.padded_columns)),
            empty((geometry.block * geometry.padded_columns, geometry.dimension)),
            empty((geometry.rows, geometry.dimension)), config=config,
        ),
        "d_key_block": factories["d_key_block"]().compile(
            empty((geometry.rows, geometry.block)),
            empty((geometry.rows, geometry.dimension)),
            empty((geometry.block, geometry.dimension), torch.float32), config=config,
        ),
        "d_value_block": factories["d_value_block"]().compile(
            empty((geometry.rows, geometry.block)),
            empty((geometry.rows, geometry.dimension)),
            empty((geometry.block, geometry.dimension), torch.float32), config=config,
        ),
    }
    report = {
        "format": "orthrus-dsv41-pypto-c1a-vjp-v1",
        "passed": False,
        "compiled": sorted(programs),
        "executed": False,
        "shape": {
            "block": geometry.block,
            "heads": geometry.local_heads,
            "dimension": geometry.dimension,
            "swa": geometry.swa,
            "compressed": geometry.compressed,
            "padded_columns": geometry.padded_columns,
        },
        "cases": [],
    }
    if args.compile_only:
        report["passed"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return

    worker, handles = None, []
    try:
        worker = DistributedWorker(list(programs.values()), config=config)

        def allocate(value):
            handle = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value.contiguous())
            handles.append(handle)
            return handle

        def read(handle, target):
            worker.copy_from(target.data_ptr(), handle.data_ptr, target.nbytes)
            return target

        swa = (torch.randn(128, geometry.dimension) * .2).bfloat16()
        compressed = (torch.randn(512, geometry.dimension) * .2).bfloat16()
        temporary = (torch.randn(geometry.block, geometry.dimension) * .2).bfloat16()
        spans = torch.tensor(((0, 0, 64), (64, 64, 64)), dtype=torch.int32)
        base = torch.arange(geometry.compressed)
        selected = torch.stack([
            torch.roll(base, token * 17) for token in range(geometry.block)
        ]).int()
        lengths = torch.tensor(
            (0, 1, 31, 32, 127, 128, 257, 511, 512) * 4,
            dtype=torch.int32,
        )[:geometry.block]
        zeros = torch.zeros(128, geometry.dimension, dtype=torch.bfloat16)
        expected_joined = torch.zeros(
            geometry.block, geometry.padded_columns, geometry.dimension,
            dtype=torch.bfloat16,
        )
        expected_joined[:, :geometry.swa] = swa
        for token, count in enumerate(lengths.tolist()):
            if count:
                expected_joined[token, geometry.swa:geometry.swa + count] = compressed[
                    selected[token, :count].long()
                ]
        expected_joined[
            :, geometry.swa + geometry.compressed:geometry.token_columns
        ] = temporary
        joined_handle = allocate(torch.zeros_like(expected_joined.flatten(0, 1)))
        worker.run(
            programs["joined_history"], allocate(swa), allocate(spans), allocate(compressed),
            allocate(selected), allocate(lengths), allocate(temporary), allocate(zeros),
            joined_handle, config=config,
        )
        actual_joined = read(joined_handle, torch.empty_like(expected_joined.flatten(0, 1)))
        torch.testing.assert_close(actual_joined, expected_joined.flatten(0, 1), atol=0, rtol=0)
        report["cases"].append({"operator": "joined_history", "byte_exact": True})

        query = (torch.randn(
            geometry.block, geometry.local_heads, geometry.dimension,
        ) * .2).bfloat16()
        history_bias = torch.zeros(1, geometry.swa, dtype=torch.float32)
        history_bias[:, -7:] = -torch.inf
        compressed_bias = torch.full(
            (geometry.block, geometry.compressed), -torch.inf, dtype=torch.float32,
        )
        for token, count in enumerate(lengths.tolist()):
            compressed_bias[token, :count] = 0
        sink = torch.randn(geometry.local_heads, dtype=torch.float32) * .2
        keys = expected_joined[:, :geometry.token_columns].float()
        score = torch.einsum("qhd,qsd->qhs", query.float(), keys) * geometry.dimension ** -.5
        score[:, :, :geometry.swa] += history_bias
        score[:, :, geometry.swa:geometry.swa + geometry.compressed] += compressed_bias[:, None]
        score = torch.cat((score, sink[None, :, None].expand(geometry.block, -1, -1)), -1)
        lse = score.logsumexp(-1).reshape(geometry.rows, 1)
        expected_probability = torch.zeros(
            geometry.rows, geometry.padded_columns, dtype=torch.float32,
        )
        expected_probability[:, :geometry.sink_column + 1] = (
            score - lse.reshape(geometry.block, geometry.local_heads, 1)
        ).exp().reshape(geometry.rows, geometry.sink_column + 1)
        probability_handle = allocate(torch.zeros_like(expected_probability))
        worker.run(
            programs["probability_replay"], allocate(query.flatten(0, 1)), joined_handle,
            allocate(history_bias), allocate(compressed_bias), allocate(sink), allocate(lse),
            probability_handle, config=config,
        )
        actual_probability = read(probability_handle, torch.empty_like(expected_probability))
        torch.testing.assert_close(
            actual_probability, expected_probability, atol=2e-4, rtol=8e-3,
        )
        report["cases"].append({
            "operator": "probability_replay",
            "max_abs": (actual_probability - expected_probability).abs().max().item(),
        })

        upstream = (torch.randn(geometry.rows, geometry.dimension) * .2).bfloat16()
        joined_value = expected_joined.clone()
        d_probability_handle = allocate(torch.zeros(
            geometry.rows, geometry.padded_columns, dtype=torch.bfloat16,
        ))
        worker.run(
            programs["d_probability"], allocate(upstream), allocate(joined_value.flatten(0, 1)),
            d_probability_handle, config=config,
        )
        d_probability = read(d_probability_handle, torch.empty(
            geometry.rows, geometry.padded_columns, dtype=torch.bfloat16,
        ))
        expected_d_probability = torch.einsum(
            "qhd,qsd->qhs", upstream.reshape(
                geometry.block, geometry.local_heads, geometry.dimension,
            ).float(), joined_value.float(),
        ).reshape(geometry.rows, geometry.padded_columns).bfloat16()
        torch.testing.assert_close(d_probability, expected_d_probability, atol=.125, rtol=.02)

        score_gradient_handle = allocate(torch.zeros_like(d_probability))
        sink_rows_handle = allocate(torch.zeros(geometry.rows, 16, dtype=torch.float32))
        worker.run(
            programs["probability_vjp"], probability_handle, d_probability_handle,
            score_gradient_handle, sink_rows_handle, config=config,
        )
        raw_score_gradient = expected_probability * (
            expected_d_probability.float()
            - (expected_probability * expected_d_probability.float()).sum(-1, keepdim=True)
        )
        expected_score_gradient = (raw_score_gradient * geometry.dimension ** -.5).bfloat16()
        score_gradient = read(score_gradient_handle, torch.empty_like(expected_score_gradient))
        torch.testing.assert_close(score_gradient, expected_score_gradient, atol=.002, rtol=.02)
        sink_gradient_handle = allocate(torch.zeros(
            geometry.local_heads, 16, dtype=torch.float32,
        ))
        worker.run(
            programs["sink_reduce"], sink_rows_handle, sink_gradient_handle, config=config,
        )
        sink_gradient = read(sink_gradient_handle, torch.empty(
            geometry.local_heads, 16, dtype=torch.float32,
        ))[:, 0]
        expected_sink = raw_score_gradient.reshape(
            geometry.block, geometry.local_heads, geometry.padded_columns,
        )[:, :, geometry.sink_column].sum(0)
        torch.testing.assert_close(sink_gradient, expected_sink, atol=2e-4, rtol=.01)

        block_score_handle = allocate(torch.zeros(
            geometry.rows, geometry.block, dtype=torch.bfloat16,
        ))
        block_probability_handle = allocate(torch.zeros_like(
            torch.empty(geometry.rows, geometry.block, dtype=torch.bfloat16),
        ))
        worker.run(
            programs["score_block_extract"], score_gradient_handle, block_score_handle, config=config,
        )
        worker.run(
            programs["probability_block_extract"], probability_handle,
            block_probability_handle, config=config,
        )
        block_score = read(block_score_handle, torch.empty(
            geometry.rows, geometry.block, dtype=torch.bfloat16,
        ))
        block_probability = read(block_probability_handle, torch.empty_like(block_score))
        torch.testing.assert_close(
            block_score,
            score_gradient[:, geometry.block_begin:geometry.block_begin + geometry.block],
            atol=0, rtol=0,
        )
        torch.testing.assert_close(
            block_probability,
            actual_probability[
                :, geometry.block_begin:geometry.block_begin + geometry.block
            ].bfloat16(),
            atol=0, rtol=0,
        )

        d_query_handle = allocate(torch.zeros_like(upstream))
        worker.run(
            programs["d_query"], score_gradient_handle, joined_handle, d_query_handle, config=config,
        )
        d_query = read(d_query_handle, torch.empty_like(upstream))
        expected_d_query = torch.einsum(
            "qhs,qsd->qhd", score_gradient.reshape(
                geometry.block, geometry.local_heads, geometry.padded_columns,
            ).float(), expected_joined.float(),
        ).reshape(geometry.rows, geometry.dimension).bfloat16()
        torch.testing.assert_close(d_query, expected_d_query, atol=.125, rtol=.02)

        d_key_handle = allocate(torch.zeros(
            geometry.block, geometry.dimension, dtype=torch.float32,
        ))
        d_value_handle = allocate(torch.zeros_like(
            torch.empty(geometry.block, geometry.dimension, dtype=torch.float32),
        ))
        worker.run(
            programs["d_key_block"], block_score_handle, allocate(query.flatten(0, 1)),
            d_key_handle, config=config,
        )
        worker.run(
            programs["d_value_block"], block_probability_handle, allocate(upstream),
            d_value_handle, config=config,
        )
        d_key = read(d_key_handle, torch.empty(
            geometry.block, geometry.dimension, dtype=torch.float32,
        ))
        d_value = read(d_value_handle, torch.empty_like(d_key))
        expected_d_key = (block_score.float().T @ query.flatten(0, 1).float()).bfloat16().float()
        expected_d_value = (block_probability.float().T @ upstream.float()).bfloat16().float()
        torch.testing.assert_close(d_key, expected_d_key, atol=.125, rtol=.02)
        torch.testing.assert_close(d_value, expected_d_value, atol=.125, rtol=.02)
        report["cases"].append({
            "operator": "c1a_vjp",
            "d_query_max_abs": (d_query.float() - expected_d_query.float()).abs().max().item(),
            "d_key_max_abs": (d_key - expected_d_key).abs().max().item(),
            "d_value_max_abs": (d_value - expected_d_value).abs().max().item(),
            "sink_max_abs": (sink_gradient - expected_sink).abs().max().item(),
        })
        report["executed"] = report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            while handles:
                worker.free_tensor(handles.pop())
            worker.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
