"""Run the pinned 1122 Prefill and Decode phases through the dual-host fabric."""

from contextlib import contextmanager
from hashlib import sha256
import json
from math import prod
from multiprocessing import RawArray
from pathlib import Path
from struct import Struct
from types import SimpleNamespace

from .dsv41_native_graph_pair import (
    NativeChipGraph, bind_chip, native_peer_mailbox, open_native_graph_pair, remote_chip, remote_mailbox_view,
)
from .dsv41_native_rank_dispatch import NativeRankGraph
from .dsv41_native_resident import NativeRankMemory, NativeResidentForward, forward_inputs, forward_templates


PREFLIGHT, LOAD, PREFILL, DECODE, READ, CLEAR = range(6)
PHASES = ("prefill", "decode")
PHASE_OPS = (PREFILL, DECODE)
OUTPUTS = ("sampled_ids", "logits")
ROWS_AND_MASK = Struct("<4iB")
READ_ROWS = Struct("<4i")
RELEASE = Struct("<2i")

_local_node = None
_remote_node = None
_local_mailbox = None


class ForwardArtifacts:
    def __init__(self, directory, checkpoint, slots):
        from models.deepseek_v4_1_flash_a3 import config
        from models.deepseek_v4_1_flash_a3.driver import load_manifest
        from models.deepseek_v4_1_flash_a3.engram import HostEngram
        from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights
        from pypto.ir import DistributedCompiledProgram
        from tokenizers import Tokenizer

        self.root = Path(directory)
        self.slots = slots
        manifests = tuple(load_manifest(self.root / phase, mode=phase, request_slots=slots) for phase in PHASES)
        self.capacities = tuple(item["capacity"] for item in manifests)
        self.context = manifests[0]["history_capacity"]
        if self.context != manifests[1]["history_capacity"]:
            raise ValueError("Prefill and Decode compiled with different history capacities")
        self.slab = manifests[0]["moe_slab_rows"]
        config.set_moe_tokens(self.slab)
        self.programs = {phase: DistributedCompiledProgram.from_dir(self.root / phase) for phase in PHASES}
        self.graphs = {phase: NativeRankGraph(self.root / phase, 16) for phase in PHASES}
        checkpoint = Checkpoint(Path(checkpoint))
        self.weights = ResidentWeights(checkpoint)
        self.tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
        self.engram = HostEngram(checkpoint, self.tokenizer)
        digest = sha256(json.dumps(checkpoint.identity, sort_keys=True).encode())
        for phase in PHASES:
            for name in ("distributed_meta.json", "orchestration/host_orch.py"):
                digest.update((self.root / phase / name).read_bytes())
        self.identity = digest.digest()


class ForwardNode:
    def __init__(self, node, local_callables=None):
        self.ranks = tuple(range(node * 8, node * 8 + 8))
        self.dps = range(node * 2, node * 2 + 2)
        self.callables = local_callables
        self.keepalive = []
        self.bank = None

    def dispatch(self, orch, args, config, io):
        opcode, size, reply = (args.scalar(i) for i in range(3))
        payload = io[:size]
        if opcode == PREFLIGHT:
            launch = json.loads(bytes(payload))
            self.artifacts = ForwardArtifacts(launch["graphs"], launch["checkpoint"], launch["slots"])
            self.chip_abi = launch["chip_abi"]
            io[:reply] = self.artifacts.identity
        elif opcode == LOAD:
            a = self.artifacts
            self.bank = NativeResidentForward(
                NativeRankMemory(orch._worker), a.programs, a.graphs, self.ranks, a.weights, a.engram,
                capacities=a.capacities, context=a.context, slots=a.slots,
            )
        elif opcode in PHASE_OPS:
            self.forward(orch, args, config, payload, PHASES[PHASE_OPS.index(opcode)])
        elif opcode == READ:
            if reply:
                self.read(payload, io[:reply])
        elif opcode == CLEAR:
            if size:
                dp, slot = RELEASE.unpack_from(payload)
                from array import array

                pages = array("i")
                pages.frombytes(payload[RELEASE.size:])
                self.bank.clear(dp, slot, pages)
        else:
            raise ValueError(f"Unknown native target command {opcode}")

    def forward(self, orch, args, config, payload, phase):
        bank = self.bank
        *counts, mask = ROWS_AND_MASK.unpack_from(payload)
        image_bytes = bank.inputs[phase].images[2].nbytes
        active = tuple(dp for dp in self.dps if mask & (1 << dp))
        if len(payload) != ROWS_AND_MASK.size + len(active) * image_bytes:
            raise ValueError(f"{phase} request does not match the negotiated packed image size")
        start = ROWS_AND_MASK.size
        for dp in active:
            bank.stage_image(phase, dp, payload[start:start + image_bytes])
            start += image_bytes
        if phase == "decode":
            bank.stage_decode_counts(counts)
        if self.callables is None:
            name, host_indices = self.chip_abi[phase]
            callables = bind_chip(name, remote_chip(args, 4), host_indices)
        else:
            callables = self.callables[phase]
        self.keepalive.clear()
        self.artifacts.graphs[phase].submit(
            orch, config, domain_id=args.scalar(3), ranks=self.ranks, tensors=bank.arguments[phase],
            callables=callables, keepalive=self.keepalive,
        )

    def read(self, payload, output):
        import torch
        from models.deepseek_v4_1_flash_a3.lm_head import SAMPLED_IDS_PAD

        phase, dp, field, row = READ_ROWS.unpack(payload)
        phase, name = PHASES[phase], OUTPUTS[field]
        if dp == -1 and name == "sampled_ids":
            values = torch.frombuffer(output, dtype=torch.int32).reshape(len(self.dps), -1, SAMPLED_IDS_PAD)
            for owner, value in zip(self.dps, values, strict=True):
                self.bank.read(phase, owner, name, value)
            return
        handle = self.bank.buffers[phase][self.ranks.index(dp * self.bank.tp)][name]
        row_bytes = prod(handle.shape[1:]) * torch.empty((), dtype=handle.dtype).element_size()
        rows, remainder = divmod(len(output), row_bytes)
        if remainder or not 0 <= row < row + rows <= handle.shape[0]:
            raise ValueError("Native target readback range exceeds its bound output")
        value = torch.frombuffer(output, dtype=handle.dtype).reshape(rows, *handle.shape[1:])
        self.bank.read(phase, dp, name, value, first_row=row)


def local_forward(orch, args, config):
    with memoryview(_local_mailbox).cast("B") as io:
        _local_node.dispatch(orch, args, config, io)


def remote_forward(orch, args, config):
    global _remote_node
    if _remote_node is None:
        _remote_node = ForwardNode(1)
    with remote_mailbox_view(args) as io:
        _remote_node.dispatch(orch, args, config, io)


class NativeForwardTarget:
    """Translate Serving input images and output reads into both hosts' commands."""

    link = None

    def __init__(self, mailbox, config, artifacts, images, templates, ropes):
        self.mailbox = mailbox
        self.config = config
        self.artifacts = artifacts
        self.images = images
        self.templates = tuple({name: templates[phase][name] for name in images[phase].fields[0]} for phase in PHASES)
        self.ropes = ropes
        self.staged = ({}, {})
        self.sampled = [None, None]
        self.prefill = SimpleNamespace(step=lambda _stage, _fresh, replay: self.run(0, replay))
        self.decode = SimpleNamespace(step=lambda _stage, counts: self.run(1, counts))

    def owns(self, dp):
        return dp in range(4)

    def stage(self, index, dp, values, _workers):
        phase = PHASES[index]
        io = self.images[phase]
        for name, field in io.fields[2].items():
            field.copy_(values[name])
        self.staged[index][dp] = io.images[2].numpy().tobytes()

    def run(self, index, counts):
        phase = PHASES[index]
        pending = self.staged[index]
        def pack_node(dps):
            active = tuple(dp for dp in dps if dp in pending)
            header = ROWS_AND_MASK.pack(*counts, sum(1 << dp for dp in active))
            return b"".join((header, *(pending[dp] for dp in active)))

        self.mailbox.exchange(self.config, PHASE_OPS[index], (pack_node(range(2)), pack_node(range(2, 4))),
                              (0, 0), phase=phase)
        pending.clear()
        self.sampled[index] = None

    def read(self, index, dp, name, out, *, row=0):
        if name == "sampled_ids":
            out.copy_(self.read_sampled(index)[dp][row:row + out.shape[0]])
            return
        owner = dp // 2
        request = READ_ROWS.pack(index, dp, OUTPUTS.index(name), row)
        payloads = (request, b"") if owner == 0 else (b"", request)
        sizes = (out.nbytes, 0) if owner == 0 else (0, out.nbytes)
        result = self.mailbox.exchange(self.config, READ, payloads, sizes)[owner]
        memoryview(out.numpy()).cast("B")[:] = result

    def read_sampled(self, index):
        import torch
        from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS, SAMPLED_IDS_PAD

        if self.sampled[index] is None:
            request = READ_ROWS.pack(index, -1, OUTPUTS.index("sampled_ids"), 0)
            size = 2 * MAX_LOGIT_ROWS * SAMPLED_IDS_PAD * 4
            outputs = self.mailbox.exchange(self.config, READ, (request, request), (size, size))
            self.sampled[index] = torch.frombuffer(
                bytearray(b"".join(outputs)), dtype=torch.int32,
            ).reshape(4, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD)
        return self.sampled[index]

    def clear(self, dp, slot, pages, _workers):
        from array import array

        request = RELEASE.pack(dp, slot) + array("i", pages).tobytes()
        payloads = (request, b"") if dp < 2 else (b"", request)
        self.mailbox.exchange(self.config, CLEAR, payloads, (0, 0))

    def session(self):
        from models.deepseek_v4_1_flash_a3.packed import PackedSession

        a = self.artifacts
        graphs = SimpleNamespace(staged_target=self, weights=a.weights, engram=a.engram, rope_tables=self.ropes,
                                 host_io={"idle_inputs": self.templates})
        return PackedSession(graphs, a.slots, a.context, a.capacities)


@contextmanager
def open_native_forward(artifacts, *, endpoint, peer_listen_host, devices, timeout):
    from pypto.runtime import RunConfig
    from pypto.runtime.runner import _apply_ring_overrides
    from simpler.task_interface import CallConfig

    slots = artifacts.slots
    phases = {name: NativeChipGraph.from_program(artifacts.graphs[name], program)
              for name, program in artifacts.programs.items()}
    templates, initial_cache, ropes = forward_templates(
        artifacts.weights, artifacts.engram, artifacts.capacities, artifacts.context, slots,
    )
    _, images = forward_inputs(artifacts.programs, artifacts.weights, templates, initial_cache)
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS

    capacity = max(2 * max(io.images[2].nbytes for io in images.values()) + ROWS_AND_MASK.size,
                   MAX_LOGIT_ROWS * FLASH.vocab_size * 4, 4096)
    global _local_mailbox
    _local_mailbox = RawArray("B", capacity)

    def bind_local(callables):
        global _local_node
        _local_node = ForwardNode(0, callables)
        return local_forward

    config = CallConfig()
    _apply_ring_overrides(config, RunConfig(
        ring_heap=(4 * 1024**3, 4 * 1024**3, (4 if artifacts.slab >= 128 else 2) * 1024**3, 1024**3),
        ring_dep_pool=131072,
    ))
    launch = json.dumps(dict(
        graphs=str(artifacts.root.resolve()), checkpoint=str(artifacts.weights.checkpoint.root.resolve()), slots=slots,
        chip_abi={phase: (item.name, item.host_tensor_indices) for phase, item in phases.items()},
    )).encode()
    with open_native_graph_pair(
        phases, endpoint=endpoint, peer_listen_host=peer_listen_host, devices=devices, platform="a2a3",
        comm_profile="a3-fabric-v1", timeout=timeout, local_endpoint=bind_local,
        remote_endpoint=f"{__name__}:remote_forward",
    ) as pair, native_peer_mailbox(pair, _local_mailbox) as mailbox:
        identities = mailbox.exchange(config, PREFLIGHT, (launch, launch), (32, 32))
        if identities != (artifacts.identity, artifacts.identity):
            raise ValueError("The native hosts resolved different weights or compiled graph metadata")
        pair.reserve_windows(config)
        mailbox.exchange(config, LOAD, (b"", b""), (0, 0))
        yield NativeForwardTarget(mailbox, config, artifacts, images, templates, ropes)
        pair.release_windows(config)
