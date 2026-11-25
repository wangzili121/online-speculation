"""Check the cross-host rank layout and phase protocol without an accelerator."""

from concurrent.futures import ThreadPoolExecutor
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from blockspec.pypto_dsv41_cluster import ClusterDispatcher, ControlLink, RankPlan
from blockspec.pypto_dsv41_cluster_transport import ClusterCollectives
from blockspec.dsv41_online import read_workload


class ClusterTest(unittest.TestCase):
    def test_workload_reader_preserves_metadata_and_limits_rows(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (
                {"id": "a", "prompt": "one", "reference_answer": "#### 1"},
                {"id": "b", "prompt": "two", "reference_answer": "#### 2"},
            )) + "\n")
            rows = read_workload(path, 1)
            self.assertEqual(rows, [{"id": "a", "prompt": "one", "reference_answer": "#### 1"}])

    def test_workload_reader_rejects_duplicate_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            path.write_text('{"id":"same","prompt":"one"}\n{"id":"same","prompt":"two"}\n')
            with self.assertRaisesRegex(ValueError, "unique"):
                read_workload(path, None)

    def pair(self, task):
        left, right = socket.socketpair()
        for endpoint in (left, right):
            endpoint.settimeout(3)

        def run(node, endpoint):
            with ControlLink(endpoint, node, b"unique-test-job-key") as link:
                return task(node, link)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, node, endpoint) for node, endpoint in enumerate((left, right))]
            return [job.result() for job in futures]

    def test_ep16_and_tp4_ownership(self):
        a, b = RankPlan((8, 8), 4, 0), RankPlan((8, 8), 4, 1)
        self.assertEqual(list(a.local_ranks), list(range(8)))
        self.assertEqual(list(b.local_ranks), list(range(8, 16)))
        self.assertEqual(a.groups("tp"), ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)))
        self.assertEqual(a.groups("dp")[2], (2, 6, 10, 14))
        self.assertEqual(a.groups("ep"), (tuple(range(16)),))
        with self.assertRaises(ValueError):
            RankPlan((7, 9), 4, 0)

    def test_transport_accepts_serving_string_device_ids(self):
        plan = RankPlan((8, 8), 4, 1)
        link = SimpleNamespace(node=1)
        native = SimpleNamespace(CANNMemory=object, HCCLBytes=object, _check=None, _bind=lambda *_: None)
        with patch("blockspec.pypto_dsv41_cluster_transport.C.CDLL"):
            for devices in (tuple(map(str, range(8))), tuple(range(8))):
                with self.subTest(devices=devices):
                    runtime = SimpleNamespace(devices=devices)
                    with ClusterCollectives(runtime, plan, link, native=native) as transport:
                        self.assertIs(transport.runtime, runtime)
            for devices in (tuple(range(8, 16)), tuple(reversed(range(8))), (False, *range(1, 8))):
                with self.subTest(devices=devices), self.assertRaisesRegex(ValueError, "node-local logical"):
                    ClusterCollectives(SimpleNamespace(devices=devices), plan, link, native=native)

    def test_local_tp_rejects_mismatched_shapes_before_device_transfer(self):
        import torch

        plan = RankPlan((4, 4), 4, 1)

        def unexpected_control(*_):
            self.fail("Local TP should never contact the other host")

        link = SimpleNamespace(node=1, exchange=unexpected_control)
        native = SimpleNamespace(CANNMemory=object, HCCLBytes=object, _check=None, _bind=lambda *_: None)
        runtime = SimpleNamespace(devices=tuple(range(4)))
        packets = [{"value": {"hidden": torch.zeros((rows, 8))}} for rows in (4, 4, 8, 4)]
        with patch("blockspec.pypto_dsv41_cluster_transport.C.CDLL"):
            with ClusterCollectives(runtime, plan, link, native=native) as transport:
                with self.assertRaisesRegex(ValueError, r"HCCL tensor ABI differs in tp group .*4, 5, 6, 7"):
                    transport.exchange("draft", packets, "reduce_scatter", "tp")

    def test_local_tp_keeps_the_cross_host_collective_schedule(self):
        import torch

        def task(node, link):
            plan = RankPlan((2, 2), 2, node)
            local = list(plan.local_ranks)
            runtime = SimpleNamespace(
                devices=(0, 1),
                map_ranks=lambda fn: [fn(0), fn(1)],
                workspace=lambda shape, dtype, rank, fill: torch.empty(shape, dtype=dtype),
                tensor=lambda value, rank: value,
            )
            native = SimpleNamespace(CANNMemory=object, HCCLBytes=object, _check=None, _bind=lambda *_: None)
            packets = [{"wave": {"value": torch.zeros((4, 8))}} for _ in local]
            stages = []
            with ClusterCollectives(runtime, plan, link, native=native) as transport:
                def channels(group):
                    return {
                        i: SimpleNamespace(
                            world=len(members), group_ranks=list(members),
                            _validate_device_transfer=lambda *_: None,
                            exchange_device_batch=lambda *_: None,
                        )
                        for i, rank in enumerate(local) for members in plan.groups(group)
                        if rank in members
                    }

                transport._channels = channels
                dispatch = ClusterDispatcher(plan, link, build_id="tp-local")
                for phase, group in (("prefill", "tp"), ("draft", "ep"), ("verify", "dp"), ("accept", "tp")):
                    def execute(_):
                        start = link.step
                        _, receipts = transport.exchange(phase, packets, "all_gather", group)
                        self.assertEqual([row["rank"] for row in receipts], local)
                        stages.append(link.step - start)

                    dispatch.execute(phase, execute)
                return stages, dispatch.epoch

        with patch("blockspec.pypto_dsv41_cluster_transport.C.CDLL"):
            self.assertEqual(self.pair(task), [([0, 2, 2, 0], 2)] * 2)

    def test_wave_preserves_global_dp_and_empty_partitions(self):
        active = {"request_slot": 0, "generation": 7, "token_ids": [12, 13]}
        contract = dict(topology={"dp": 4, "tp": 4, "ep": 16},
                        DP_engines=[dict(dp_rank=dp, active=2 if dp == 2 else 0,
                                         requests=[active] if dp == 2 else []) for dp in range(4)])

        def task(node, link):
            dispatch = ClusterDispatcher(RankPlan((8, 8), 4, node), link, build_id="same")
            return dispatch.execute_wave("prefill", lambda wave: wave["DP_engines"],
                                         contract=contract if node == 0 else None)

        a, b = self.pair(task)
        self.assertEqual(a, contract["DP_engines"][:2])
        self.assertEqual(b, contract["DP_engines"][2:])
        self.assertEqual(b[0]["requests"][0]["generation"], 7)
        self.assertFalse(b[1]["requests"])

    def test_both_hosts_execute_the_leader_command_and_commit_once(self):
        def task(node, link):
            dispatch = ClusterDispatcher(RankPlan((8, 8), 4, node), link, build_id="same")
            seen = []
            for phase in ("prefill", "draft", "verify", "accept", "release"):
                dispatch.execute(phase, lambda command: seen.append((command["phase"], dispatch.epoch)),
                                 command={"phase": phase} if node == 0 else None)
            return seen, dispatch.epoch

        a, b = self.pair(task)
        expected = [("prefill", 0), ("draft", 1), ("verify", 1), ("accept", 1), ("release", 2)]
        self.assertEqual(a, (expected, 2))
        self.assertEqual(a, b)

    def test_failure_fences_both_hosts_before_accept(self):
        def task(node, link):
            dispatch = ClusterDispatcher(RankPlan((1, 1), 1, node), link, build_id="same")
            for phase in ("prefill", "draft"):
                dispatch.execute(phase, lambda _: None)

            def verify(_):
                if node == 1:
                    raise RuntimeError("verify failed")

            with self.assertRaisesRegex(RuntimeError, "verify failed"):
                dispatch.execute("verify", verify)
            with self.assertRaisesRegex(RuntimeError, "failed dispatch"):
                dispatch.execute("accept", lambda _: None)
            return dispatch.epoch

        self.assertEqual(self.pair(task), [1, 1])

    def test_nested_exchange_does_not_mask_the_original_dispatch_error(self):
        def task(node, link):
            dispatch = ClusterDispatcher(RankPlan((1, 1), 1, node), link, build_id="same")
            dispatch.execute("prefill", lambda _: None)

            def draft(_):
                if node == 0:
                    raise RuntimeError("online replay capture failed")
                link.exchange("greedy-output/draft", [])

            message = "online replay capture failed" if node == 0 else "Peer aborted draft"
            with self.assertRaisesRegex(RuntimeError, message):
                dispatch.execute("draft", draft)
            return dispatch.failed

        self.assertEqual(self.pair(task), [True, True])

    def test_token_rows_merge_in_scheduler_order(self):
        contract = dict(topology={"dp": 4, "tp": 4, "ep": 16},
                        DP_engines=[dict(dp_rank=dp, active=1,
                                         requests=[dict(request_slot=0, generation=1, token_ids=[dp])])
                                    for dp in range(4)])

        def task(node, link):
            dispatch = ClusterDispatcher(RankPlan((8, 8), 4, node), link, build_id="same")
            return dispatch.execute_wave("prefill", lambda wave: [[100 + dp["dp_rank"]] for dp in wave["DP_engines"]],
                                         contract=contract if node == 0 else None, collect_tokens=True)

        expected = [[100], [101], [102], [103]]
        self.assertEqual(self.pair(task), [expected, expected])

    def test_different_builds_fail_before_device_work(self):
        def task(node, link):
            with self.assertRaisesRegex(ValueError, "software build"):
                ClusterDispatcher(RankPlan((8, 8), 4, node), link, build_id=str(node))

        self.pair(task)


if __name__ == "__main__":
    unittest.main()
