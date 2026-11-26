"""Run the pinned 1122 Target and Orthrus forward graphs through the dual-host fabric."""

from array import array
from contextlib import contextmanager
from hashlib import sha256
import json
from math import prod
from multiprocessing import RawArray
from pathlib import Path
from struct import Struct
from time import perf_counter
from types import SimpleNamespace

from .dsv41_native_graph_pair import (
    NativeChipGraph, bind_chip, native_peer_mailbox, open_native_graph_pair, remote_chip, remote_mailbox_view,
)
from .dsv41_native_rank_dispatch import NativeRankGraph
from .dsv41_native_resident import (
    NativeRankMemory, NativeResidentForward, bind_native_draft, forward_inputs, forward_templates,
)
from .dsv41_packed_proposer import (
    DraftRoundStats, draft_geometry, draft_graph_manifest, make_draft_input_images, read_packed_head,
    stage_draft_request,
)


PREFLIGHT, LOAD, PREFILL, DECODE, READ, CLEAR, DRAFT = range(7)
PHASES = ("prefill", "decode")
FORWARD_PHASES = (*PHASES, "draft")
PHASE_OPS = (PREFILL, DECODE, DRAFT)
OUTPUTS = ("sampled_ids", "logits")
ROWS_AND_MASK = Struct("<4iB")
READ_ROWS = Struct("<4i")
RELEASE = Struct("<2i")

_local_node = None
_remote_node = None
_local_mailbox = None


class ForwardArtifacts:
    def __init__(self, directory, checkpoint, slots, *, draft_directory=None, head=None):
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
        self.draft = None
        if draft_directory is not None:
            if slots != 1 or head is None:
                raise ValueError("Native Orthrus requires a head and one request slot per DP")
            path, head = Path(draft_directory), Path(head)
            manifest = json.loads((path / "orthrus_draft_manifest.json").read_text())
            block, last_layers = manifest["block"], manifest["last_layers"]
            if manifest != draft_graph_manifest(self.context, block, last_layers=last_layers):
                raise ValueError("Compiled native draft does not match this model or context")
            head_artifact = read_packed_head(checkpoint, head, block)
            self.draft = SimpleNamespace(
                root=path, head=head, block=block, last_layers=last_layers, head_artifact=head_artifact,
                program=DistributedCompiledProgram.from_dir(path), graph=NativeRankGraph(path, 16),
            )
        digest = sha256(json.dumps(checkpoint.identity, sort_keys=True).encode())
        paths = [self.root / phase for phase in PHASES]
        if self.draft is not None:
            paths.append(self.draft.root)
            digest.update((self.draft.head / "manifest.json").read_bytes())
        for path in paths:
            for name in ("distributed_meta.json", "orchestration/host_orch.py"):
                digest.update((path / name).read_bytes())
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
            self.artifacts = ForwardArtifacts(
                launch["graphs"], launch["checkpoint"], launch["slots"], draft_directory=launch["draft_graph"],
                head=launch["head"],
            )
            self.chip_abi = launch["chip_abi"]
            io[:reply] = self.artifacts.identity
        elif opcode == LOAD:
            a = self.artifacts
            self.bank = NativeResidentForward(
                NativeRankMemory(orch._worker), a.programs, a.graphs, self.ranks, a.weights, a.engram,
                capacities=a.capacities, context=a.context, slots=a.slots,
            )
            if a.draft is not None:
                d = a.draft
                self.draft = bind_native_draft(
                    self.bank, d.graph, d.program, a, d.head, d.block, d.last_layers, d.head_artifact,
                )
        elif opcode in PHASE_OPS:
            self.forward(orch, args, config, payload, FORWARD_PHASES[PHASE_OPS.index(opcode)])
        elif opcode == READ:
            if reply:
                self.read(payload, io[:reply])
        elif opcode == CLEAR:
            if size:
                dp, slot = RELEASE.unpack_from(payload)
                pages = array("i")
                pages.frombytes(payload[RELEASE.size:])
                self.bank.clear(dp, slot, pages)
        else:
            raise ValueError(f"Unknown native target command {opcode}")

    def forward(self, orch, args, config, payload, phase):
        bank = self.bank
        *counts, mask = ROWS_AND_MASK.unpack_from(payload)
        is_draft = phase == "draft"
        images = self.draft.draft.inputs if is_draft else bank.inputs[phase]
        image_bytes = images.images[2].nbytes
        active = tuple(dp for dp in self.dps if mask & (1 << dp))
        if len(payload) != ROWS_AND_MASK.size + len(active) * image_bytes:
            raise ValueError(f"{phase} request does not match the negotiated packed image size")
        start = ROWS_AND_MASK.size
        for dp in active:
            image = payload[start:start + image_bytes]
            if is_draft:
                self.stage_draft(dp, image)
            else:
                bank.stage_image(phase, dp, image)
            start += image_bytes
        if phase == "decode":
            bank.stage_decode_counts(counts)
        if is_draft:
            with self.draft.count_buffer.host_view() as view:
                view[:] = array("i", (n for n in counts for _ in range(bank.tp))).tobytes()
        if self.callables is None:
            name, host_indices = self.chip_abi[phase]
            callables = bind_chip(name, remote_chip(args, 4), host_indices)
        else:
            callables = self.callables[phase]
        self.keepalive.clear()
        graph = self.artifacts.draft.graph if is_draft else self.artifacts.graphs[phase]
        tensors = self.draft.arguments if is_draft else bank.arguments[phase]
        graph.submit(
            orch, config, domain_id=args.scalar(3), ranks=self.ranks, tensors=tensors,
            callables=callables, keepalive=self.keepalive,
        )

    def stage_draft(self, dp, image):
        bank = self.bank
        inputs = self.draft.draft.inputs
        memoryview(inputs.images[2].numpy())[:] = image
        inputs.images[1].copy_(inputs.images[2])
        inputs.fields[1]["logit_row_indices"].fill_(-1)
        workers = bank._workers(dp)
        for values, peers in zip(inputs.images[1:], (workers[1:], workers[:1]), strict=True):
            bank.memory.upload_to_ranks(
                values.data_ptr(), values.nbytes,
                ((worker, inputs.devices[worker].data_ptr, 0) for worker in peers),
            )

    def read(self, payload, output):
        import torch
        from models.deepseek_v4_1_flash_a3.lm_head import SAMPLED_IDS_PAD

        phase, dp, field, row = READ_ROWS.unpack(payload)
        phase, name = FORWARD_PHASES[phase], OUTPUTS[field]
        if dp == -1 and name == "sampled_ids":
            values = torch.frombuffer(output, dtype=torch.int32).reshape(len(self.dps), -1, SAMPLED_IDS_PAD)
            for owner, value in zip(self.dps, values, strict=True):
                if phase == "draft":
                    worker = self.ranks.index(owner * self.bank.tp)
                    tensor = self.draft.draft.ranks[worker][name]
                    self.bank.memory.copy_from(value.data_ptr(), tensor.data_ptr, value.nbytes, worker_id=worker)
                else:
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
        self.staged = ({}, {}, {})
        self.sampled = [None, None, None]
        self.sample_checks = None
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
        phase = FORWARD_PHASES[index]
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
            if self.sample_checks is not None and index not in self.sample_checks:
                self.check_sample(index, dp, row, int(out[0, 0]))
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

    def check_sample(self, index, dp, row, token):
        import torch
        from models.deepseek_v4_1_flash_a3.config import FLASH

        logits = torch.empty((1, FLASH.vocab_size), dtype=torch.float32)
        self.read(index, dp, "logits", logits, row=row)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError(f"Non-finite {PHASES[index]} logits on DP {dp}, row {row}")
        maximum = logits.max()
        if not 0 <= token < FLASH.vocab_size or logits[0, token] != maximum:
            raise ValueError(f"{PHASES[index]} greedy output disagrees with logits on DP {dp}, row {row}")
        self.sample_checks[index] = dict(dp=dp, row=row, token=token, maximum=float(maximum))

    def clear(self, dp, slot, pages, _workers):
        request = RELEASE.pack(dp, slot) + array("i", pages).tobytes()
        payloads = (request, b"") if dp < 2 else (b"", request)
        self.mailbox.exchange(self.config, CLEAR, payloads, (0, 0))

    def session(self):
        from models.deepseek_v4_1_flash_a3.packed import PackedSession

        a = self.artifacts
        graphs = SimpleNamespace(staged_target=self, weights=a.weights, engram=a.engram, rope_tables=self.ropes,
                                 host_io={"idle_inputs": self.templates})
        draft = NativeDraftProposer(self) if a.draft is not None else None
        return PackedSession(graphs, a.slots, a.context, a.capacities, draft=draft)


class NativeDraftProposer:
    """Prepare one owner image per active DP and run the intact Orthrus graph."""

    def __init__(self, target):
        from models.deepseek_v4_1_flash_a3.config import TP_SIZE

        self.target = target
        self.tp = TP_SIZE
        a = target.artifacts
        d = a.draft
        self.block = d.block
        self.head_role = d.head_artifact[1]["artifact_role"]
        self.stats = DraftRoundStats()
        self.student = {}
        self.previous_dps = set()
        pages, *buckets = draft_geometry(a.context)
        self.stager = SimpleNamespace(
            inputs=make_draft_input_images(d.block, buckets[0]), pages=pages, buckets=buckets,
            weights=a.weights, engram=a.engram, rope_tables=target.ropes, block=d.block,
            mask_token_id=d.head_artifact[1]["mask_token_id"],
            page_tables=tuple(target.images["decode"].fields[0][name][0] for name in
                              ("c1a_index_block_table", "c2a_index_block_table")),
        )

    def propose_batch(self, requests, *, uploads, capture=None):
        if capture is not None:
            raise ValueError("The native draft path currently accepts frozen-head inference only")
        if not requests:
            return {}
        prepared = perf_counter()
        active = {row.owner // self.tp for row in requests}
        pending = self.target.staged[2]
        images = self.stager.inputs.images
        for row in requests:
            stage_draft_request(self.stager, row)
            pending[row.owner // self.tp] = images[2].numpy().tobytes()
        for dp in self.previous_dps - active:
            pending[dp] = images[0].numpy().tobytes()
        self.previous_dps = active
        self.stats.prepare_seconds += perf_counter() - prepared
        self.stats.input_bytes += sum(map(len, pending.values()))
        now = perf_counter()
        self.target.run(2, tuple(self.block if dp in active else 0 for dp in range(4)))
        self.stats.graph_seconds += perf_counter() - now
        now = perf_counter()
        values = self.target.read_sampled(2)
        self.stats.readback_seconds += perf_counter() - now
        self.stats.sampled_id_bytes += values.nbytes
        self.stats.calls += 1
        return {row.owner: values[row.owner // self.tp, :row.candidates, 0].tolist() for row in requests}

    def release_student(self, handle):
        pass


@contextmanager
def open_native_forward(artifacts, *, endpoint, peer_listen_host, devices, timeout):
    from pypto.runtime import RunConfig
    from pypto.runtime.runner import _apply_ring_overrides
    from simpler.task_interface import CallConfig

    slots = artifacts.slots
    phases = {name: NativeChipGraph.from_program(artifacts.graphs[name], program)
              for name, program in artifacts.programs.items()}
    if artifacts.draft is not None:
        draft = artifacts.draft
        phases["draft"] = NativeChipGraph.from_program(draft.graph, draft.program)
    templates, initial_cache, ropes = forward_templates(
        artifacts.weights, artifacts.engram, artifacts.capacities, artifacts.context, slots,
    )
    _, images = forward_inputs(artifacts.programs, artifacts.weights, templates, initial_cache)
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS

    phase_image_sizes = [io.images[2].nbytes for io in images.values()]
    if artifacts.draft is not None:
        _, bucket, _ = draft_geometry(artifacts.context)
        phase_image_sizes.append(make_draft_input_images(artifacts.draft.block, bucket).images[2].nbytes)
    capacity = max(2 * max(phase_image_sizes) + ROWS_AND_MASK.size,
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
        draft_graph=str(artifacts.draft.root.resolve()) if artifacts.draft else None,
        head=str(artifacts.draft.head.resolve()) if artifacts.draft else None,
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
