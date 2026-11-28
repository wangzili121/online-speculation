# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bind the single-host A3 draft to the resident target worker."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from .. import config as C
from ..device_views import resident_graph_argument
from ..driver import manifest as target_manifest
from .attention_history import packed_swa_controls
from .forward import make_packed_draft_graph
from .head import bind_head
from .history import packed_history_lookup
from .indexer import INDEX_CAPACITIES
from .inputs import draft_engram_prefix_rows
from .packed_indexer import packed_index_controls


@dataclass(frozen=True)
class DraftGeometry:
    block: int
    context: int
    slots: int
    last_layers: int = 0

    def __post_init__(self):
        if self.block not in (8, 16, 32) or not self.block <= self.context <= INDEX_CAPACITIES[-1] or self.slots <= 0:
            raise ValueError("Invalid A3 draft geometry")
        if self.block == 8 and self.last_layers:
            raise ValueError("K8 Draft currently supports frozen-head inference")
        if not 0 <= self.last_layers <= C.FLASH.num_hidden_layers - max(C.FLASH.engram_layer_ids):
            raise ValueError("The suffix must stop at or after the last Engram layer")

    @property
    def pages(self):
        return ((self.context + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE) * self.slots + 1

    @property
    def buckets(self):
        return tuple(next(bucket for bucket in INDEX_CAPACITIES if bucket >= self.context // ratio)
                     for ratio in (1, 2))

    def manifest(self):
        draft = Path(__file__).parent
        base = target_manifest("decode", 16, self.context, slab_rows=16, request_slots=self.slots)
        return {
            "format": "a3-orthrus-draft-v2", "block": self.block, "context": self.context, "slots": self.slots,
            "target": base,
            "last_layers": self.last_layers,
            "draft_sources": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(draft.glob("*.py"))},
        }

    def compile(self, directory, config):
        from dataclasses import replace

        entry = make_packed_draft_graph(self.pages, *self.buckets, self.block, last_layers=self.last_layers)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        program = entry.compile(config=replace(config, save_kernels=True, save_kernels_dir=str(directory.resolve())))
        (directory / "orthrus_draft_manifest.json").write_text(json.dumps(self.manifest(), indent=2) + "\n")
        return program

    def validate(self, directory):
        directory = Path(directory)
        if json.loads((directory / "orthrus_draft_manifest.json").read_text()) != self.manifest():
            raise ValueError("Draft artifact differs from the A3 source or resident cache geometry")


def make_draft_input_images(block, bucket):
    """Allocate the shared host image layout used by the native and staged proposers."""
    from ..config import BLOCK_SIZE, FLASH
    from ..engram import ENGRAM_K
    from ..config import MAX_LOGIT_ROWS
    from ..replay import DecodeInputs

    capacity = max(16, block)
    template = {
        "draft_embeddings": torch.zeros((2, FLASH.hidden_size), dtype=torch.bfloat16),
        "pre_mix": torch.zeros((capacity, FLASH.hc_mult), dtype=torch.float32),
        "engram_rows": torch.zeros((2, draft_engram_prefix_rows(block), ENGRAM_K), dtype=torch.bfloat16),
        "draft_cos": torch.ones((2, capacity, FLASH.qk_rope_head_dim // 2), dtype=torch.float32),
        "draft_sin": torch.zeros((2, capacity, FLASH.qk_rope_head_dim // 2), dtype=torch.float32),
        "window_rows": torch.full((8, 16), -1, dtype=torch.int32),
        "window_starts": torch.full((8,), -2, dtype=torch.int32),
        "window_spans": torch.tensor(((0, 0, BLOCK_SIZE), (0, 0, 0)), dtype=torch.int32),
        "window_bias": torch.full((1, BLOCK_SIZE), -1e30, dtype=torch.float32),
        "c1a_table": torch.full((1, bucket // BLOCK_SIZE), -1, dtype=torch.int32),
        "c2a_table": torch.full((1, bucket // BLOCK_SIZE), -1, dtype=torch.int32),
        "request_ids": torch.zeros((capacity,), dtype=torch.int32),
        "c1a_visible": torch.zeros((capacity,), dtype=torch.int32),
        "c2a_visible": torch.zeros((capacity,), dtype=torch.int32),
        "logit_row_indices": torch.full((MAX_LOGIT_ROWS,), -1, dtype=torch.int32),
    }
    template["pre_mix"][:, 0] = 1
    return DecodeInputs(template)


@dataclass(frozen=True)
class StudentCapture:
    owner: int
    position: int
    rows: int
    logits: object
    final_state: object
    ffn_tape: dict
    pages: tuple


@dataclass(frozen=True)
class FinalState:
    streams: object
    premix: object


class ResidentDraft:
    """Propose one pending block per DP from the target's committed pages."""

    def __init__(self, graphs, program, geometry, head, artifact):
        from .attention_tape import (
            packed_attention_tape_shapes, packed_core_tape_shapes, packed_qkv_tape_shapes, packed_wo_tape_shapes,
        )
        from .ffn_tape import packed_ffn_tape_shapes

        g = graphs
        self.graphs, self.program, self.geometry = g, program, geometry
        self.runtime, self.config = g.runtime, g.config
        self.block, self.pages, self.last_layers = geometry.block, geometry.pages, geometry.last_layers
        self.tp, self.world = C.TP_SIZE, C.EP_SIZE
        self.student = None
        self.last_propose_seconds = {}
        self.inputs = make_draft_input_images(geometry.block, geometry.buckets[0])
        self.counts = torch.zeros(C.EP_SIZE, dtype=torch.int32).share_memory_()
        self.previous_owners = set()
        self.sampled = torch.empty((C.MAX_LOGIT_ROWS, C.SAMPLED_IDS_PAD), dtype=torch.int32)
        model, block = C.FLASH, geometry.block
        self.ranks = []
        self.capture_inputs = make_draft_input_images(block, geometry.buckets[0]) if self.last_layers else None
        self.capture_banks = {}
        head_banks, manifest, self.resident_head_bytes = bind_head(g.runtime, head, C.EP_SIZE, C.TP_SIZE, artifact)
        self.mask_token_id = manifest["mask_token_id"]
        tapes = {
            **packed_ffn_tape_shapes(block, self.last_layers, C.TP_SIZE, model.hidden_size, model.hc_mult),
            **packed_attention_tape_shapes(block, self.last_layers, model.hidden_size, model.hc_mult),
            **packed_wo_tape_shapes(block, self.last_layers, model, C.TP_SIZE),
            **packed_core_tape_shapes(block, self.last_layers, model, C.TP_SIZE),
            **packed_qkv_tape_shapes(block, self.last_layers, model),
        }
        constants = dict(
            window_lookup=packed_history_lookup("fp8"),
            compressed_lookup=packed_history_lookup("fp4"),
            zero_kv=torch.zeros((C.BLOCK_SIZE, model.head_dim), dtype=torch.bfloat16),
            draft_embedding_ids=torch.tensor([0] + [1] * (max(16, block) - 1), dtype=torch.int64),
        )
        residual_shape = (max(16, block), model.hc_mult, model.hidden_size)
        for rank in range(C.EP_SIZE):
            target = g.ranks[0][rank] | g.ranks[1][rank]
            if g.host_cache[rank]["window_cache"].shape[1] != geometry.pages:
                raise ValueError("Draft and target disagree on resident cache pages")
            banks = target | head_banks[rank] | self.inputs.bind(g.runtime, rank)
            banks["x_hc"] = g.runtime.alloc_tensor(residual_shape, torch.float32, worker_id=rank)
            for name, tensor in constants.items():
                banks[name] = g.runtime.alloc_tensor(tensor.shape, tensor.dtype, init=tensor, worker_id=rank)
            for name, (shape, dtype) in tapes.items():
                banks[name] = g.runtime.alloc_tensor(shape, getattr(torch, dtype), worker_id=rank)
            # Inference draft and target dispatch sequentially and have identical output shapes.
            self.ranks.append(banks)
            if self.capture_inputs is not None and rank % C.TP_SIZE == 0:
                capture = self.capture_inputs.bind(g.runtime, rank)
                capture["x_hc"] = g.runtime.alloc_tensor(residual_shape, torch.float32, worker_id=rank)
                capture["logits"] = g.runtime.alloc_tensor(
                    (C.MAX_LOGIT_ROWS, model.vocab_size), torch.float32, worker_id=rank,
                )
                self.capture_banks[rank] = capture

        self.arguments = []
        self.capture_parameters = {}
        self.capture_owner_index = None
        for parameter in program._get_metadata()[0]:
            name = parameter.name.removesuffix("__ssa_v0")
            if name == "ffn_capture_owner":
                value = -1
                self.capture_owner_index = len(self.arguments)
            elif name == "moe_num_tokens":
                value = self.counts
            elif name == "execute_layers":
                continue
            else:
                value = resident_graph_argument([rank[name] for rank in self.ranks], tuple(range(C.EP_SIZE)), parameter)
                if self.capture_banks and name in next(iter(self.capture_banks.values())):
                    self.capture_parameters[name] = (len(self.arguments), parameter)
            self.arguments.append(value)
        g.runtime.run(program, *self.arguments, 0, config=g.config)

    def _stage(self, handle, position, pages, anchor, candidates):
        block = self.geometry.block
        g = self.graphs
        embeddings = g.weights.lookup_rows(torch.tensor([anchor, self.mask_token_id], dtype=torch.int64))
        prefix = draft_engram_prefix_rows(block)
        provisional = [anchor] + [self.mask_token_id] * (prefix - 1)
        engram = g.engram.lookup(provisional, prefix, position, request_id=handle, commit=False)
        window = packed_swa_controls(pages, position, self.geometry.pages)
        index = [packed_index_controls(pages, position, ratio, block, bucket=self.geometry.buckets[0])
                 for ratio in (1, 2)]
        for image in self.inputs.fields[1:]:
            image["draft_embeddings"].copy_(embeddings)
            image["pre_mix"].copy_(self.inputs.fields[0]["pre_mix"])
            image["engram_rows"].copy_(engram)
            for name, table in zip(("draft_cos", "draft_sin"), g.rope_tables, strict=True):
                image[name][:, :block].copy_(table[:, position:position + block])
            for name, value in zip(("window_rows", "window_starts", "window_spans", "window_bias"), window, strict=True):
                image[name].copy_(value)
            for family, (_, table, _, visible) in zip(("c1a", "c2a"), index, strict=True):
                image[f"{family}_table"].copy_(table)
                image[f"{family}_visible"].copy_(visible)
        indices = self.inputs.fields[2]["logit_row_indices"].fill_(-1)
        indices[:candidates] = torch.arange(candidates, dtype=torch.int32)

    def propose(self, session, handle, anchor, candidates, *, capture_rows=0):
        capture = (handle, capture_rows) if capture_rows else None
        return self.propose_batch(session, [(handle, anchor, candidates)], capture=capture)[handle]

    def propose_batch(self, session, queries, *, capture=None):
        """Draft one query per active DP group in one distributed forward."""
        start = perf_counter()
        block = self.geometry.block
        g = self.graphs
        groups = {}
        for handle, anchor, candidates in queries:
            record = session._lookup(handle)
            owner = handle.dp * C.TP_SIZE
            if owner in groups:
                raise ValueError("Draft expects at most one request per DP group")
            if not (0 < candidates < block and candidates <= C.MAX_LOGIT_ROWS
                    and record.prompt_length <= record.committed <= self.geometry.context - block):
                raise ValueError("Draft block or candidate count exceeds the resident request geometry")
            groups[owner] = (record, anchor, candidates)
        if not groups:
            raise ValueError("Empty draft dispatch")

        capture_owner, capture_rows = -1, 0
        if capture is not None:
            handle, capture_rows = capture
            capture_owner = handle.dp * C.TP_SIZE
            if (capture_owner not in groups or groups[capture_owner][0].handle is not handle
                    or self.student is not None or not self.last_layers
                    or not 1 <= capture_rows < block or capture_rows > C.MAX_LOGIT_ROWS):
                raise ValueError("Draft capture requires a vacant online tape and valid student row count")

        stage_seconds = perf_counter() - start
        owners = set(groups)
        with ThreadPoolExecutor(max_workers=C.TP_SIZE) as workers:
            active = next(iter(groups))
            idle = (rank for owner in self.previous_owners - owners for rank in range(owner, owner + C.TP_SIZE))
            list(workers.map(lambda rank: self.inputs.upload(g.runtime, rank, active), idle))
            self.counts.zero_()
            for owner, (record, anchor, candidates) in groups.items():
                mark = perf_counter()
                rows = max(candidates, capture_rows) if owner == capture_owner else candidates
                self._stage(record.handle, record.committed, record.pages, anchor, rows)
                stage_seconds += perf_counter() - mark
                ranks = range(owner, owner + C.TP_SIZE)
                list(workers.map(lambda rank: self.inputs.upload(g.runtime, rank, owner), ranks))
                self.counts[owner:owner + C.TP_SIZE] = block
                if owner == capture_owner:
                    self.capture_inputs.images[2].copy_(self.inputs.images[2])
                    self.capture_inputs.upload(g.runtime, owner, owner)
        self.previous_owners = owners

        arguments = self.arguments
        if capture is not None:
            record = groups[capture_owner][0]
            output = self.capture_banks[capture_owner]
            arguments = list(self.arguments)
            arguments[self.capture_owner_index] = capture_owner
            for name, (idx, parameter) in self.capture_parameters.items():
                shards = [output[name] if rank == capture_owner else values[name]
                          for rank, values in enumerate(self.ranks)]
                arguments[idx] = resident_graph_argument(shards, tuple(range(C.EP_SIZE)), parameter)
            tapes = {rank: {name: tensor for name, tensor in self.ranks[rank].items()
                            if name.startswith(("attention_tape_", "ffn_tape_"))}
                     for rank in range(capture_owner, capture_owner + C.TP_SIZE)}
            self.student = StudentCapture(
                capture_owner, record.committed, capture_rows, output["logits"],
                FinalState(output["x_hc"], output["pre_mix"]), tapes, tuple(record.pages),
            )
        submitted = perf_counter()
        g.runtime.run(self.program, *arguments, 1, config=g.config)
        completed = perf_counter()
        proposals = {}
        for owner, (record, _, candidates) in groups.items():
            rows = max(candidates, capture_rows) if owner == capture_owner else candidates
            sampled = self.sampled[:rows]
            source = self.ranks[owner]["sampled_ids"]
            g.runtime.copy_from(sampled.data_ptr(), source.data_ptr, sampled.nbytes, worker_id=owner)
            if not bool((sampled[:, C.SAMPLED_FINITE_COLUMN] == 1).all()):
                raise FloatingPointError("Nonfinite Orthrus draft logits")
            proposals[record.handle] = sampled[:candidates, 0].tolist()
        self.last_propose_seconds = dict(
            draft_stage=stage_seconds, draft_upload_and_bind=submitted - start - stage_seconds,
            draft_graph=completed - submitted, draft_readback=perf_counter() - completed,
        )
        return proposals
