# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the rank-local reverse and optimizer programs for one worker."""

from .attention_vjp import make_packed_attention_vjp
from .attention_input_vjp import make_packed_attn_input_vjp
from .attention_post_vjp import make_packed_attn_post_vjp
from .ffn_input_vjp import make_packed_ffn_input_vjp
from .ffn_mhc_vjp import make_packed_ffn_mhc_vjp
from ..ffn_tape import make_packed_ffn_post_vjp
from .final_vjp import make_packed_final_vjp
from .lm_head_vjp import make_packed_lm_head_vjp
from .loss import make_packed_loss_graph
from .qkv_vjp import make_packed_qkv_vjp
from .wo_vjp import make_packed_wo_input_vjp


from .adam import make_packed_adam_programs, packed_parameters
from .expert_dispatch import make_packed_ep_dispatch
from .expert_reverse import make_packed_ep_reverse


def make_ranked_reverse_graphs(block, layers, *, physical_pages, ranks, loss_kind):
    from ...config import FLASH, TP_SIZE
    from ...lm_head import MAX_LOGIT_ROWS

    model, tp, rows = FLASH, TP_SIZE, MAX_LOGIT_ROWS
    dp = ranks // tp
    hidden, copies, vocab = model.hidden_size, model.hc_mult, model.vocab_size
    graphs = {
        "loss": make_packed_loss_graph(rows, vocab, dp, tp, loss_kind),
        "lm-head": make_packed_lm_head_vjp(rows, vocab, hidden, dp, tp),
        "final-norm": make_packed_final_vjp(rows, hidden, copies, dp, tp, model.rms_norm_eps, output_rows=block),
        "ffn-output": make_packed_ffn_post_vjp(block, layers, block, hidden, copies, dp, tp),
        "ffn-input": make_packed_ffn_input_vjp(block, layers, model, dp, tp),
        "ffn-mhc": make_packed_ffn_mhc_vjp(block, layers, model, dp, tp),
        "attention-output": make_packed_attn_post_vjp(block, layers, model, dp, tp),
        "wo": make_packed_wo_input_vjp(block, model, tp, world=ranks, parameters=True),
        "attention": make_packed_attention_vjp(block, world=ranks, tp=tp, physical_pages=physical_pages),
        "qkv": make_packed_qkv_vjp(block, model, world=ranks, tp=tp),
    }
    if layers > 1:
        graphs["attention-input"] = make_packed_attn_input_vjp(block, layers, model, dp, tp)
    return graphs


def _adam_entries(model, tp):
    return {parameter.program_key: parameter.name for parameter in reversed(packed_parameters(model, tp))}


def bundle_program_names(layers):
    from ...config import FLASH, TP_SIZE

    names = ["loss", "lm-head", "final-norm", "ffn-output", "ffn-input", "ffn-mhc",
             "attention-output", "wo", "attention", "qkv", "expert-dispatch", "expert-reverse"]
    if layers > 1:
        names.append("attention-input")
    return tuple(names) + tuple(
        f"adam/{key}/{phase}"
        for key in _adam_entries(FLASH, TP_SIZE).values()
        for phase in ("stage", "apply", "publish")
    )


def online_programs(geometry, layers, loss):
    from ...config import DP_SIZE, EP_SIZE, FLASH, TP_SIZE

    block, pages = geometry.block, geometry.pages
    model, tp, dp = FLASH, TP_SIZE, DP_SIZE
    graphs = make_ranked_reverse_graphs(block, layers, physical_pages=pages, ranks=EP_SIZE, loss_kind=loss)
    graphs["expert-dispatch"] = make_packed_ep_dispatch(
        block, layers, model.hidden_size, model.hc_mult, model.num_experts_per_tok, model.n_routed_experts, dp, tp,
    )
    graphs["expert-reverse"] = make_packed_ep_reverse(
        block, model.hidden_size, model.moe_intermediate_size, model.swiglu_limit,
        model.num_experts_per_tok, model.n_routed_experts, dp, tp,
    )
    for (size, replicated, fp32), name in _adam_entries(model, tp).items():
        programs = make_packed_adam_programs(size, replicated, fp32, world=EP_SIZE, tp=tp)
        for operation, program in zip(("stage", "apply", "publish"), programs, strict=True):
            graphs[f"adam/{name}/{operation}"] = program
    return graphs

def prepare_online_programs(directory, geometry, loss, config, *, compile_missing):
    """Load a matched complete set of online graphs, compiling an empty directory."""
    from dataclasses import replace
    from pathlib import Path
    from hashlib import sha256
    import json

    directory = Path(directory)
    manifest = directory / "orthrus_online_manifest.json"
    names = bundle_program_names(geometry.last_layers)
    online = Path(__file__).parent
    expected = dict(geometry=geometry.manifest(), loss=loss, phases=list(names),
                    sources={p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(online.glob("*.py"))})
    if compile_missing and not directory.exists():
        directory.mkdir(parents=True)
        for name, entry in online_programs(geometry, geometry.last_layers, loss).items():
            output = directory / name
            output.mkdir(parents=True)
            entry.compile(config=replace(config, save_kernels=True, save_kernels_dir=str(output.resolve())))
        manifest.write_text(json.dumps(expected, indent=2) + "\n")
    if json.loads(manifest.read_text()) != expected:
        raise ValueError("Online graphs differ from the A3 suffix or draft ABI")
    paths = {name: directory / name for name in names}
    for path in paths.values():
        if not (path / "distributed_meta.json").is_file():
            raise ValueError(f"Missing distributed graph metadata at {path}")
    return paths
