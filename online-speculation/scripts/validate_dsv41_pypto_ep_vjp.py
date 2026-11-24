#!/usr/bin/env python3
"""Compile and execute the production Orthrus-DSV4.1 EP16 VJP packets."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--iterations", type=int, default=0, help="Timed repeats after each correctness launch")
    parser.add_argument("--active", type=int, default=3, help="Number of active source rows in the K32 block")
    parser.add_argument("--bucket-rows", type=int, help="Override the selected EP wire width")
    parser.add_argument("--routing",
                        choices=("mixed", "balanced", "skewed", "concentrated", "expert-boundaries"),
                        default="mixed")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")
    if args.iterations < 0:
        parser.error("iterations must be nonnegative")

    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
    from blockspec.pypto_dsv41_ep_vjp import (
        EPVJPGeometry,
        EP_BUCKET_WIDTHS,
        EP_WORLD,
        EXPERT_CAPACITY,
        EXPERT_SMALL_ROWS,
        HIDDEN,
        K32_ROWS,
        LOCAL_EXPERTS,
        MAP_PAD,
        PACKET_FIELDS,
        ROUTES,
        SLAB_ROWS,
        TOPK,
        make_ep_vjp_dispatch_pack,
        make_ep_vjp_dispatch_unpack,
        make_ep_vjp_return_pack,
        make_ep_vjp_source_restore,
        route_bucket_width,
    )
    if not 0 <= args.active <= K32_ROWS:
        parser.error(f"active must be in [0, {K32_ROWS}]")

    active, source_rank = args.active, 9
    owners = {
        "mixed": (1, 7, 8, 15, 1, 8),
        "balanced": (1, 3, 7, 8, 12, 15),
        "skewed": (1, 1, 1, 1, 7, 8),
        "concentrated": (9,) * TOPK,
        "expert-boundaries": (9,) * TOPK,
    }[args.routing]
    route_ids = torch.zeros(K32_ROWS, TOPK, dtype=torch.int32)
    for token in range(active):
        for slot in range(TOPK):
            route = token * TOPK + slot
            owner = owners[route % len(owners)]
            if args.routing == "expert-boundaries":
                expert = slot + 8 if slot < 2 and token >= EXPERT_SMALL_ROWS + slot else slot
            else:
                expert = route % LOCAL_EXPERTS
            route_ids[token, slot] = owner * LOCAL_EXPERTS + expert
    selected = route_bucket_width([route_ids[:active]])
    bucket_rows = args.bucket_rows if args.bucket_rows is not None else ROUTES
    if bucket_rows not in EP_BUCKET_WIDTHS or bucket_rows < selected:
        parser.error(f"{args.routing} requires a supported bucket width of at least {selected}")
    PACKET_ROWS = EPVJPGeometry(bucket_rows=bucket_rows).packet_rows

    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.SIM if args.simulate else ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(
            device_ids=[args.device], num_sub_workers=0,
        ),
    )
    meta = lambda shape, dtype=torch.bfloat16: torch.empty(
        shape, dtype=dtype, device="meta",
    )
    programs = {
        "dispatch_pack": make_ep_vjp_dispatch_pack(bucket_rows).compile(
            meta((K32_ROWS, HIDDEN)), meta((ROUTES, HIDDEN)),
            meta((K32_ROWS, TOPK), torch.int32),
            meta((PACKET_ROWS, PACKET_FIELDS), torch.int64),
            meta((PACKET_ROWS, HIDDEN)), meta((PACKET_ROWS, HIDDEN)),
            meta((EP_WORLD, MAP_PAD), torch.int32), K32_ROWS, 9,
            config=config,
        ),
        "dispatch_unpack": make_ep_vjp_dispatch_unpack(bucket_rows).compile(
            meta((PACKET_ROWS, PACKET_FIELDS), torch.int64),
            meta((PACKET_ROWS, HIDDEN)), meta((PACKET_ROWS, HIDDEN)),
            meta((EP_WORLD, MAP_PAD), torch.int32),
            meta((SLAB_ROWS, HIDDEN)), meta((SLAB_ROWS, HIDDEN)),
            meta((LOCAL_EXPERTS,), torch.int32),
            meta((PACKET_ROWS, MAP_PAD), torch.int32),
            meta((MAP_PAD,), torch.int32), 9, config=config,
        ),
        "return_pack": make_ep_vjp_return_pack(bucket_rows).compile(
            meta((SLAB_ROWS, HIDDEN)),
            meta((EP_WORLD, MAP_PAD), torch.int32),
            meta((PACKET_ROWS, MAP_PAD), torch.int32),
            meta((PACKET_ROWS, HIDDEN)), config=config,
        ),
        "source_restore": make_ep_vjp_source_restore(bucket_rows).compile(
            meta((PACKET_ROWS, HIDDEN)),
            meta((PACKET_ROWS, PACKET_FIELDS), torch.int64),
            meta((EP_WORLD, MAP_PAD), torch.int32),
            meta((ROUTES, HIDDEN)), meta((K32_ROWS, HIDDEN)),
            K32_ROWS, 9, config=config,
        ),
    }
    report = {
        "format": "orthrus-dsv41-pypto-ep-vjp-v1",
        "passed": False,
        "compiled": sorted(programs),
        "executed": False,
        "simulator_execution": False,
        "routing": args.routing,
        "selected_bucket_rows": selected,
        "shape": {
            "rows": K32_ROWS, "topk": TOPK, "hidden": HIDDEN,
            "experts": LOCAL_EXPERTS * EP_WORLD, "world": EP_WORLD,
            "packet_rows": PACKET_ROWS, "expert_capacity": EXPERT_CAPACITY,
            "bucket_rows": bucket_rows,
        },
    }
    if args.compile_only:
        report["passed"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return

    torch.manual_seed(4117)
    torch.set_num_threads(4)
    normalized_x = (torch.randn(K32_ROWS, HIDDEN) * .05).bfloat16()
    route_dy = (torch.randn(ROUTES, HIDDEN) * .05).bfloat16()
    expected_route_dx = torch.zeros_like(route_dy)
    for token in range(active):
        for slot in range(TOPK):
            route = token * TOPK + slot
            owner = int(route_ids[token, slot]) // LOCAL_EXPERTS
            expected_route_dx[route] = (
                normalized_x[token].float() + 2 * route_dy[route].float() + owner / 16
            ).bfloat16()
    expected_token_dx = torch.zeros(K32_ROWS, HIDDEN, dtype=torch.bfloat16)
    expected_token_dx[:active] = expected_route_dx.reshape(
        K32_ROWS, TOPK, HIDDEN,
    )[:active].float().sum(1).bfloat16()

    worker, handles = None, []
    try:
        worker = ChipWorker(config=config)
        timings = {name: [] for name in programs}

        def run(name, *operands):
            worker.run(programs[name], *operands, config=config)
            for _ in range(args.iterations):
                started = perf_counter()
                worker.run(programs[name], *operands, config=config)
                timings[name].append((perf_counter() - started) * 1e3)

        def allocate(shape, dtype, value=None):
            keywords = {}
            if value is not None:
                keywords["init"] = value.contiguous()
            handle = worker.alloc_tensor(tuple(shape), dtype, **keywords)
            handles.append(handle)
            return handle

        def upload(handle, value):
            worker.copy_to(handle.data_ptr, value.data_ptr(), value.nbytes)

        def download(handle, shape, dtype):
            value = torch.empty(shape, dtype=dtype)
            worker.copy_from(value.data_ptr(), handle.data_ptr, value.nbytes)
            return value

        def dirty(shape, dtype):
            return allocate(shape, dtype, torch.full(shape, 7, dtype=dtype))

        normalized_h = allocate(normalized_x.shape, normalized_x.dtype, normalized_x)
        route_dy_h = allocate(route_dy.shape, route_dy.dtype, route_dy)
        route_ids_h = allocate(route_ids.shape, route_ids.dtype, route_ids)
        sent_meta_h = dirty((PACKET_ROWS, PACKET_FIELDS), torch.int64)
        sent_x_h = dirty((PACKET_ROWS, HIDDEN), torch.bfloat16)
        sent_dy_h = dirty((PACKET_ROWS, HIDDEN), torch.bfloat16)
        sent_counts_h = dirty((EP_WORLD, MAP_PAD), torch.int32)
        run(
            "dispatch_pack", normalized_h, route_dy_h, route_ids_h,
            sent_meta_h, sent_x_h, sent_dy_h, sent_counts_h, active,
            source_rank,
        )
        sent_meta = download(
            sent_meta_h, (PACKET_ROWS, PACKET_FIELDS), torch.int64,
        )
        sent_counts = download(sent_counts_h, (EP_WORLD, MAP_PAD), torch.int32)
        sent_x = download(sent_x_h, (PACKET_ROWS, HIDDEN), torch.bfloat16)
        sent_dy = download(sent_dy_h, (PACKET_ROWS, HIDDEN), torch.bfloat16)
        if int(sent_counts[:, 0].sum()) != active * TOPK:
            raise AssertionError("dispatch did not emit exactly one packet per active route")
        packet_rows = {}
        for owner in range(EP_WORLD):
            for row in range(int(sent_counts[owner, 0])):
                packet = owner * bucket_rows + row
                token, slot = sent_meta[packet, 1:3].tolist()
                actual_x = sent_x[packet]
                actual_dy = sent_dy[packet]
                torch.testing.assert_close(actual_x, normalized_x[token], atol=0, rtol=0)
                torch.testing.assert_close(
                    actual_dy, route_dy[token * TOPK + slot], atol=0, rtol=0,
                )
                packet_rows[packet] = (actual_x, actual_dy)
        for owner in range(EP_WORLD):
            invalid = owner * bucket_rows + int(sent_counts[owner, 0])
            end = (owner + 1) * bucket_rows
            if torch.count_nonzero(sent_x[invalid:end]):
                raise AssertionError("inactive dispatch x row was not zero")
            if torch.count_nonzero(sent_dy[invalid:end]):
                raise AssertionError("inactive dispatch dY row was not zero")

        recv_meta_h = allocate((PACKET_ROWS, PACKET_FIELDS), torch.int64)
        recv_x_h = allocate((PACKET_ROWS, HIDDEN), torch.bfloat16)
        recv_dy_h = allocate((PACKET_ROWS, HIDDEN), torch.bfloat16)
        recv_counts_h = allocate((EP_WORLD, MAP_PAD), torch.int32)
        expert_x_h = dirty((SLAB_ROWS, HIDDEN), torch.bfloat16)
        expert_dy_h = dirty((SLAB_ROWS, HIDDEN), torch.bfloat16)
        untouched_slab = torch.full((SLAB_ROWS, HIDDEN), 7, dtype=torch.bfloat16)
        expert_counts_h = dirty((LOCAL_EXPERTS,), torch.int32)
        packet_map_h = dirty((PACKET_ROWS, MAP_PAD), torch.int32)
        received_total_h = dirty((MAP_PAD,), torch.int32)
        expert_dx_h = allocate((SLAB_ROWS, HIDDEN), torch.bfloat16)
        returned_owner_h = dirty((PACKET_ROWS, HIDDEN), torch.bfloat16)
        assembled_return = torch.zeros(PACKET_ROWS, HIDDEN, dtype=torch.bfloat16)
        owner_reports = []

        for owner in sorted(set(owners)):
            count = int(sent_counts[owner, 0])
            received_meta = torch.zeros(PACKET_ROWS, PACKET_FIELDS, dtype=torch.int64)
            received_counts = torch.zeros(EP_WORLD, MAP_PAD, dtype=torch.int32)
            received_x = torch.zeros(PACKET_ROWS, HIDDEN, dtype=torch.bfloat16)
            received_dy = torch.zeros_like(received_x)
            received_counts[source_rank, 0] = count
            source_begin = owner * bucket_rows
            receive_begin = source_rank * bucket_rows
            received_meta[receive_begin:receive_begin + count] = sent_meta[
                source_begin:source_begin + count
            ]
            received_x[receive_begin:receive_begin + count] = sent_x[
                source_begin:source_begin + count
            ]
            received_dy[receive_begin:receive_begin + count] = sent_dy[
                source_begin:source_begin + count
            ]
            upload(recv_meta_h, received_meta)
            upload(recv_counts_h, received_counts)
            upload(recv_x_h, received_x)
            upload(recv_dy_h, received_dy)
            upload(expert_x_h, untouched_slab)
            upload(expert_dy_h, untouched_slab)
            run(
                "dispatch_unpack", recv_meta_h, recv_x_h, recv_dy_h,
                recv_counts_h, expert_x_h, expert_dy_h, expert_counts_h,
                packet_map_h, received_total_h, owner,
            )
            packet_map = download(packet_map_h, (PACKET_ROWS, MAP_PAD), torch.int32)
            expert_counts = download(expert_counts_h, (LOCAL_EXPERTS,), torch.int32)
            total = int(download(received_total_h, (MAP_PAD,), torch.int32)[0])
            expert_x = download(expert_x_h, (SLAB_ROWS, HIDDEN), torch.bfloat16)
            expert_dy = download(expert_dy_h, (SLAB_ROWS, HIDDEN), torch.bfloat16)
            if total != count or int(expert_counts.sum()) != count:
                raise AssertionError("unpack lost an owner route")
            computed = torch.zeros(SLAB_ROWS, dtype=torch.bool)
            for local_expert, rows in enumerate(expert_counts.tolist()):
                if rows:
                    start = local_expert * EXPERT_CAPACITY
                    capacity = EXPERT_SMALL_ROWS if rows <= EXPERT_SMALL_ROWS else EXPERT_CAPACITY
                    computed[start:start + capacity] = True
            unused = torch.ones(SLAB_ROWS, dtype=torch.bool)
            for row in range(count):
                source_packet = source_begin + row
                receive_packet = receive_begin + row
                slab = int(packet_map[receive_packet, 0])
                unused[slab] = False
                if int(packet_map[receive_packet, 1]) != 1:
                    raise AssertionError("valid packet has no slab map")
                if not torch.equal(
                    packet_map[receive_packet, 2:6].long(), sent_meta[source_packet].long(),
                ):
                    raise AssertionError("packet map did not preserve route metadata")
                slab_x = expert_x[slab]
                slab_dy = expert_dy[slab]
                torch.testing.assert_close(slab_x, packet_rows[source_packet][0], atol=0, rtol=0)
                torch.testing.assert_close(slab_dy, packet_rows[source_packet][1], atol=0, rtol=0)
                token, slot = sent_meta[source_packet, 1:3].tolist()
                route = token * TOPK + slot
            padding = unused & computed
            if expert_x[padding].any() or expert_dy[padding].any():
                raise AssertionError("padding read by expert VJP was not zero")
            if not (expert_x[~computed].eq(7).all() and expert_dy[~computed].eq(7).all()):
                raise AssertionError("unread expert slab rows were modified")
            expert_dx = torch.zeros_like(expert_x)
            for row in range(count):
                source_packet = source_begin + row
                receive_packet = receive_begin + row
                slab = int(packet_map[receive_packet, 0])
                token, slot = sent_meta[source_packet, 1:3].tolist()
                expert_dx[slab] = expected_route_dx[token * TOPK + slot]
            upload(expert_dx_h, expert_dx)
            run(
                "return_pack", expert_dx_h, recv_counts_h,
                packet_map_h, returned_owner_h,
            )
            returned_owner = download(
                returned_owner_h, (PACKET_ROWS, HIDDEN), torch.bfloat16,
            )
            if returned_owner[:receive_begin].any() or returned_owner[receive_begin + count:].any():
                raise AssertionError("inactive return packet rows were not zero")
            for row in range(count):
                source_packet = source_begin + row
                receive_packet = receive_begin + row
                value = returned_owner[receive_packet]
                token, slot = sent_meta[source_packet, 1:3].tolist()
                torch.testing.assert_close(
                    value, expected_route_dx[token * TOPK + slot], atol=0, rtol=0,
                )
                assembled_return[source_packet] = value
            owner_reports.append({"owner": owner, "routes": count,
                                  "expert_input_rows_written": int(computed.sum())})

        returned_h = allocate((PACKET_ROWS, HIDDEN), torch.bfloat16)
        upload(returned_h, assembled_return)
        route_dx_h = allocate((ROUTES, HIDDEN), torch.bfloat16)
        token_dx_h = allocate((K32_ROWS, HIDDEN), torch.bfloat16)
        run(
            "source_restore", returned_h, sent_meta_h, sent_counts_h,
            route_dx_h, token_dx_h, active, source_rank,
        )
        route_dx = download(route_dx_h, (ROUTES, HIDDEN), torch.bfloat16)
        token_dx = download(token_dx_h, (K32_ROWS, HIDDEN), torch.bfloat16)
        torch.testing.assert_close(route_dx, expected_route_dx, atol=0, rtol=0)
        torch.testing.assert_close(token_dx, expected_token_dx, atol=0, rtol=0)
        report.update(
            passed=True, executed=True, active=active, source_rank=source_rank,
            simulator_execution=args.simulate,
            owners=owner_reports, routes=int(sent_counts[:, 0].sum()),
            route_max_abs=float((route_dx.float() - expected_route_dx.float()).abs().max()),
            token_max_abs=float((token_dx.float() - expected_token_dx.float()).abs().max()),
            inactive_route_nonzero=int(torch.count_nonzero(route_dx[active * TOPK:])),
            inactive_token_nonzero=int(torch.count_nonzero(token_dx[active:])),
        )
        if args.iterations:
            report["steady_kernel_ms"] = {
                name: {"median": median(samples), "calls": len(samples)}
                for name, samples in timings.items()
            }
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
