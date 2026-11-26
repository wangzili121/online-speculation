from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch

from blockspec.dsv41_native_rank_dispatch import NativeRankGraph, RankLocalStack


HOST_SOURCE = """
from types import SimpleNamespace

def _alloc_intermediates(tensors, world_size=1):
    tensors["__comm_d0_ord"] = [0] * world_size

def entry(orch, _args, config, *, tensors, callables, sub_ids, _keep, world_size,
          _domain_provider=None, _window_sources=None):
    with (_domain_provider or orch.allocate_domain)(
        name="comm_d0", workers=[*range(world_size)], window_size=128,
        buffers=[SimpleNamespace(name="mailbox", nbytes=128, load_from_host=False, store_to_host=False)],
    ) as domain:
        size = world_size
        for rank in range(0, size, 1):
            weights = tensors["weights"][rank * 2:rank * 2 + 2, 0:4]
            state = tensors["state"][rank]
            args = (rank, domain[rank].device_ctx, weights, state, tensors["epoch"], tensors["__comm_d0_ord"][rank])
            _keep.append(args)
            _submit_chip(orch, callables["chip"], args, config, rank)

entry._pypto_distributed_entry = True
"""


class FakeStack:
    def __init__(self, global_ranks):
        self.global_ranks = global_ranks
        self.shape = (len(global_ranks), 4)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            physical, tail = key
            assert tail == slice(0, 4)
            assert physical.start % 2 == 0 and physical.stop == physical.start + 2
            return self.global_ranks[physical.start // 2], "NZ"
        return self.global_ranks[key], "ND"


def test_native_graph_uses_global_rank_and_local_device(tmp_path: Path):
    host = tmp_path / "orchestration"
    host.mkdir()
    (host / "host_orch.py").write_text(HOST_SOURCE)
    graph = NativeRankGraph(tmp_path, 16)
    allocated = []
    orch = SimpleNamespace(get_or_allocate_domain=lambda **spec: allocated.append(spec))
    graph.allocate_domain(orch, "decode", ((rank // 8, rank % 8) for rank in range(16)))
    assert allocated[0]["members"] == tuple((rank // 8, rank % 8) for rank in range(16))
    assert allocated[0]["buffers"][0].name == "mailbox"

    submitted = []
    graph.entry.__globals__["_submit_chip"] = lambda orch, chip, args, cfg, worker: submitted.append((worker, args))
    for ranks in (range(8), range(8, 16)):
        local = tuple(ranks)
        stack = FakeStack(local)
        view = [SimpleNamespace(domain_rank=rank, domain_size=16, device_ctx=100 + rank) for rank in local]
        keepalive = []
        graph.submit(
            SimpleNamespace(get_global_domain=lambda domain_id: view), None, domain_id=7, ranks=local,
            tensors={**graph.host_intermediates(), "weights": stack, "state": stack, "epoch": 9},
            callables={"chip": 3}, keepalive=keepalive,
        )
        assert len(keepalive) == 9

    assert submitted == [(rank % 8, (rank, 100 + rank, (rank, "NZ"), (rank, "ND"), 9, 0))
                         for rank in range(16)]


def test_sampled_output_views_replace_only_the_dp_owners():
    ranks = tuple(range(8, 16))
    local = RankLocalStack(FakeStack(ranks), ranks, 16, {8: "owner2", 12: "owner3"})
    assert [local[rank] for rank in ranks] == [
        "owner2", (9, "ND"), (10, "ND"), (11, "ND"),
        "owner3", (13, "ND"), (14, "ND"), (15, "ND"),
    ]


def test_daemon_scratch_uses_worker_owned_buffers(tmp_path, monkeypatch):
    host = tmp_path / "orchestration"
    host.mkdir()
    (host / "host_orch.py").write_text(HOST_SOURCE)
    graph = NativeRankGraph(tmp_path, 16)
    graph.allocate_intermediates = lambda tensors, world_size: tensors.update(
        __comm_d0_ord=[torch.zeros(1, dtype=torch.int32) for _ in range(world_size)]
    )
    torch_interop = ModuleType("simpler_setup.torch_interop")
    torch_interop.torch_dtype_to_datatype = lambda dtype: dtype
    monkeypatch.setitem(sys.modules, torch_interop.__name__, torch_interop)

    def new_buffer(nbytes):
        data = bytearray([99] * nbytes)
        return SimpleNamespace(host_view=lambda: memoryview(data),
                               tensor=lambda **kwargs: (data, kwargs))

    tensors, buffers = graph.bind_remote_intermediates(SimpleNamespace(create_buffer=new_buffer), (8, 9))
    assert len(buffers) == 2
    assert tuple(tensors["__comm_d0_ord"]) == (8, 9)
    for storage, view in tensors["__comm_d0_ord"].values():
        assert storage == bytes(4)
        assert view == {"shapes": (1,), "dtype": torch.int32}


def test_owner_only_graph_skips_the_peer_and_uses_global_dp_index(tmp_path):
    host = tmp_path / "orchestration"
    host.mkdir()
    (host / "host_orch.py").write_text("""
def _alloc_intermediates(tensors, world_size=1):
    pass

def entry(orch, _args, config, *, tensors, callables, sub_ids, _keep, world_size,
          _domain_provider=None, _window_sources=None):
    owner = tensors["owner"]
    replica = int(owner) // 4
    rows = tensors["student"][replica]
    _submit_chip(orch, callables["loss"], (rows, owner), config, owner)

entry._pypto_distributed_entry = True
""")
    graph = NativeRankGraph(tmp_path, 16)
    submitted = []
    graph.entry.__globals__["_submit_chip"] = lambda orch, chip, args, cfg, worker: submitted.append((worker, args))
    assert graph.allocate_domain(None, "loss", ()) is None
    for first in (0, 8):
        replicas = (first // 4, first // 4 + 1)
        graph.submit(None, None, domain_id=0, ranks=tuple(range(first, first + 8)),
                     tensors={"owner": 12, "student": RankLocalStack(FakeStack(replicas), replicas, 4)},
                     callables={"loss": 0}, keepalive=[])
    assert submitted == [(4, ((3, "ND"), 12))]


def test_static_adam_loop_keeps_its_four_canonical_ranks(tmp_path):
    host = tmp_path / "orchestration"
    host.mkdir()
    (host / "host_orch.py").write_text("""
def _alloc_intermediates(tensors, world_size=1):
    pass

def entry(orch, _args, config, *, tensors, callables, sub_ids, _keep, world_size,
          _domain_provider=None, _window_sources=None):
    for rank in range(0, 4, 1):
        state = tensors["state"][rank]
        if rank == 0:
            _submit_chip(orch, callables["first"], state, config, rank)
        else:
            _submit_chip(orch, callables["peer"], state, config, rank)

entry._pypto_distributed_entry = True
""")
    graph = NativeRankGraph(tmp_path, 16)
    submitted = []
    graph.entry.__globals__["_submit_chip"] = lambda orch, chip, args, cfg, worker: submitted.append((worker, chip))
    for first in (0, 8):
        indices = tuple(range(4)) if first == 0 else ()
        graph.submit(None, None, domain_id=0, ranks=tuple(range(first, first + 8)),
                     tensors={"state": RankLocalStack(FakeStack(indices), indices, 4)},
                     callables={"first": 0, "peer": 1}, keepalive=[])
    assert submitted == [(0, 0), (1, 1), (2, 1), (3, 1)]
