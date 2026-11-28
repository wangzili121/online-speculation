# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bind a long-lived service to the native A3 Prefill and Decode programs."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from tokenizers import Tokenizer

from .encoding import encode_messages


@dataclass(frozen=True)
class LaunchConfig:
    model_root: Path
    devices: tuple[int, ...]
    context: int
    chunk_size: int
    slab_rows: int
    graph_dir: Path | None
    cache_dir: Path
    orthrus_head: Path | None = None
    draft_graph: Path | None = None
    draft_block: int = 16
    verify_width: int = 8
    online_last_layers: int = 0
    online_loss: str = "tv"
    online_graphs: Path | None = None
    update_every: int = 16
    teacher_rows: int = 7
    learning_rate: float = 1e-5
    warmup_updates: int = 8


class NativeSession:
    """Own one request slot and reuse the same device resources across requests."""

    def __init__(self, graphs, tokenizer, eos_id, config, metadata):
        self.graphs = graphs
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.config = config
        self.metadata = metadata
        self.memory = SimpleNamespace(phase="ready")
        self.needs_reset = False
        self.orthrus = None

    def encode(self, prompt, *, chat=False):
        if chat:
            return self.tokenizer.encode(encode_messages(prompt), add_special_tokens=False).ids
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt).ids
        return prompt

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate(self, request_id, tokens, max_tokens, ignore_eos, on_token, cancelled):
        from concurrent.futures import CancelledError

        if cancelled():
            raise CancelledError()
        if self.orthrus is not None:
            if self.needs_reset:
                self.graphs.reset()
                self.needs_reset = False
            return self.orthrus.generate(request_id, tokens, max_tokens, ignore_eos, on_token, cancelled)
        if self.needs_reset:
            self.graphs.reset()
        self.needs_reset = True
        request = {"request_id": request_id, "dp_rank": 0, "input_ids": tokens}
        options = SimpleNamespace(
            capacity=self.metadata["prefill_capacity"], decode_capacity=16,
            chunk_size=self.config.chunk_size, context_capacity=self.config.context,
            max_new_tokens=max_tokens, ignore_eos=ignore_eos,
        )
        return self.graphs.replay(
            self.tokenizer, request, self.memory, {}, options=options, label=request_id,
            on_token=on_token, cancelled=cancelled, collect_routing=False,
        )


@contextmanager
def open_session(settings, *, prepare_only=False):
    """Compile or validate artifacts, then bind the checkpoint once."""
    from models import LIB_ROOT

    if settings.orthrus_head is not None and not LIB_ROOT:
        raise ValueError("Set PYPTO_LIB_ROOT to a pypto-lib checkout with A3 Orthrus support")
    from models.deepseek_v4_1_flash_a3 import config as model
    from models.deepseek_v4_1_flash_a3.driver import compile_generation_graphs, load_manifest
    from models.deepseek_v4_1_flash_a3.engram import HostEngram
    from models.deepseek_v4_1_flash_a3.generation import open_resident_graphs
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights

    checkpoint = Checkpoint(settings.model_root)
    if settings.orthrus_head is not None:
        from models.deepseek_v4_1_flash_a3.orthrus_draft.head import read_head
        from models.deepseek_v4_1_flash_a3.orthrus_draft.resident import DraftGeometry

        geometry = DraftGeometry(settings.draft_block, settings.context, 1, settings.online_last_layers)
        head = read_head(checkpoint, settings.orthrus_head, geometry.block)
    prefill_capacity = ((settings.chunk_size + 15) // 16) * 16
    graph_dir = settings.graph_dir
    if graph_dir is None:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        graph_dir = Path(tempfile.mkdtemp(prefix="graphs-", dir=settings.cache_dir))
        print(f"A3_SERVING_GRAPH_DIR {graph_dir}", flush=True)
        compile_generation_graphs(
            graph_dir / "prefill", graph_dir / "decode", devices=",".join(map(str, settings.devices)),
            context=settings.context, prefill_capacity=prefill_capacity, prefill_slab_rows=settings.slab_rows,
        )
    prefill_dir, decode_dir = graph_dir / "prefill", graph_dir / "decode"
    prefill = load_manifest(prefill_dir, mode="prefill", capacity=prefill_capacity)
    decode = load_manifest(decode_dir, mode="decode", capacity=16)
    if prefill["history_capacity"] != settings.context or decode["history_capacity"] != settings.context:
        raise ValueError("Both graphs must match --max-model-len")
    if prefill["moe_slab_rows"] != settings.slab_rows:
        raise ValueError("Prefill graph must match --prefill-moe-slab-rows")

    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokenizer_config = json.loads((checkpoint.root / "tokenizer_config.json").read_text())
    eos = tokenizer_config["eos_token"]
    eos_id = tokenizer.token_to_id(eos["content"] if isinstance(eos, dict) else eos)
    if eos_id is None:
        raise ValueError("Checkpoint EOS token is missing from the tokenizer")
    options = SimpleNamespace(
        runtime_dir=prefill_dir, decode_runtime_dir=decode_dir, profile_all_steps=None,
        capacity=prefill_capacity, decode_capacity=16, context_capacity=settings.context, max_new_tokens=256,
    )
    metadata = {
        "checkpoint_identity": checkpoint.identity,
        "graph_dir": str(graph_dir),
        "context_capacity": settings.context,
        "prefill_capacity": prefill_capacity,
        "prefill_chunk_size": settings.chunk_size,
        "prefill_moe_slab_rows": settings.slab_rows,
        "vocab_size": model.FLASH.vocab_size,
        "topology": prefill["topology"],
        "cache_layout": decode["cache_layout"],
    }
    extra_program_dirs = ()
    online_paths = {}
    if settings.orthrus_head is not None:
        from pypto.backend import BackendType
        from pypto.ir import DistributedConfig
        from pypto.runtime import ExecutionMode, RunConfig

        draft_dir = settings.draft_graph or graph_dir / f"orthrus-k{geometry.block}-last{geometry.last_layers}"
        config = RunConfig(
            backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
            distributed_config=DistributedConfig(device_ids=list(settings.devices), num_sub_workers=0),
        )
        if settings.draft_graph is None and not draft_dir.exists():
            geometry.compile(draft_dir, config)
        geometry.validate(draft_dir)
        if settings.online_last_layers:
            from models.deepseek_v4_1_flash_a3.orthrus_draft.online.programs import prepare_online_programs

            online_dir = settings.online_graphs or draft_dir.with_name(draft_dir.name + "-" + settings.online_loss)
            online_paths = prepare_online_programs(online_dir, geometry, settings.online_loss, config,
                                                   compile_missing=settings.online_graphs is None)
        extra_program_dirs = (draft_dir, *online_paths.values())
        metadata["orthrus"] = dict(head_role=head[1]["artifact_role"], block=geometry.block,
                                   verify_width=settings.verify_width, graph=str(draft_dir),
                                   online_last_layers=settings.online_last_layers)
        if online_paths:
            metadata["orthrus"].update(loss=settings.online_loss, update_every=settings.update_every,
                                       teacher_rows=settings.teacher_rows, learning_rate=settings.learning_rate)

    if prepare_only:
        if extra_program_dirs:
            from models.deepseek_v4_1_flash_a3.generation import preflight_programs

            metadata["callables"] = preflight_programs((prefill_dir, decode_dir, *extra_program_dirs), settings.devices)
        yield metadata
        return
    weights = ResidentWeights(checkpoint)
    engram = HostEngram(checkpoint, tokenizer)
    memory = SimpleNamespace(phase="prepare", sample=lambda: None)
    with open_resident_graphs(
        options, list(settings.devices), weights, engram, memory, metadata,
        prefill_slab_rows=prefill["moe_slab_rows"], extra_program_dirs=extra_program_dirs,
    ) as graphs:
        session = NativeSession(graphs, tokenizer, eos_id, settings, metadata)
        if extra_program_dirs:
            from models.deepseek_v4_1_flash_a3.orthrus_draft.resident import ResidentDraft
            from models.deepseek_v4_1_flash_a3.packed import PackedSession
            from .orthrus import SpeculativeSession

            draft = ResidentDraft(graphs, graphs.extra_programs[0], geometry, settings.orthrus_head, head)
            packed = PackedSession(graphs, 1, settings.context, (prefill_capacity, 16))
            learner = None
            if online_paths:
                from models.deepseek_v4_1_flash_a3.orthrus_draft.online.learner import ResidentLearner

                learner = ResidentLearner(draft, dict(zip(online_paths, graphs.extra_programs[1:], strict=True)))
            session.orthrus = SpeculativeSession(
                packed, draft, eos_id, settings.chunk_size, settings.verify_width,
                learner=learner, update_every=settings.update_every, teacher_rows=settings.teacher_rows,
                learning_rate=settings.learning_rate, warmup_updates=settings.warmup_updates,
            )
        yield session
