"""Compile and load the native EP16 learner against an existing draft artifact."""

from hashlib import sha256
import json
from pathlib import Path

from .dsv41_native_rank_dispatch import NativeRankGraph
from .dsv41_packed_adam import make_packed_adam_programs, packed_parameters
from .dsv41_packed_ep_dispatch import make_packed_ep_dispatch
from .dsv41_packed_ep_reverse import make_packed_ep_reverse
from .dsv41_packed_proposer import draft_geometry, draft_graph_manifest
from .dsv41_staged_reverse import make_ranked_reverse_graphs


def learner_geometry(draft_dir):
    from models.deepseek_v4_1_flash_a3.config import FLASH

    manifest_path = draft_dir / "orthrus_draft_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    block, context, layers = (manifest[key] for key in ("block", "context", "last_layers"))
    if not 1 <= layers <= FLASH.num_hidden_layers - max(FLASH.engram_layer_ids):
        raise ValueError("Native online last-n must fit the suffix above the last Engram input boundary")
    if manifest != draft_graph_manifest(context, block, last_layers=layers):
        raise ValueError("Online artifacts require a matching native draft graph with a last-n tape")
    return block, context, layers


def _adam_entries(model, tp):
    return {parameter.program_key: parameter.name for parameter in reversed(packed_parameters(model, tp))}


def bundle_program_names(layers):
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE

    names = ["loss", "teacher-copy", "lm-head", "final-norm", "ffn-output", "ffn-input", "ffn-mhc",
             "attention-output", "wo", "attention", "qkv", "expert-dispatch", "expert-reverse"]
    if layers > 1:
        names.append("attention-input")
    return tuple(names) + tuple(
        f"adam/{key}/{phase}"
        for key in _adam_entries(FLASH, TP_SIZE).values()
        for phase in ("stage", "apply", "publish")
    )


def make_native_online_graphs(block, context, layers, loss):
    from models.deepseek_v4_1_flash_a3.config import DP_SIZE, EP_SIZE, FLASH, TP_SIZE

    pages, *_ = draft_geometry(context)
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


def _program_digest(directory):
    digest = sha256()
    for path in ("distributed_meta.json", "orchestration/host_orch.py"):
        digest.update((directory / path).read_bytes())
    return digest.hexdigest()


def bundle_manifest(directory, draft_dir, loss):
    block, context, layers = learner_geometry(draft_dir)
    return dict(
        format="orthrus-dsv41-native-online-v1", block=block, context=context, last_layers=layers, loss=loss,
        programs={name: _program_digest(directory / name) for name in bundle_program_names(layers)},
    )


def read_native_online_programs(directory, names):
    from models.deepseek_v4_1_flash_a3.config import EP_SIZE
    from pypto.ir import DistributedCompiledProgram

    return {
        name: (DistributedCompiledProgram.from_dir(directory / name), NativeRankGraph(directory / name, EP_SIZE))
        for name in names
    }


def load_native_online_bundle(directory, draft_dir, loss):
    directory = Path(directory)
    recorded = json.loads((directory / "online_manifest.json").read_text())
    if recorded != bundle_manifest(directory, draft_dir, loss):
        raise ValueError("Online reverse and optimizer artifacts do not match the native draft")
    return read_native_online_programs(directory, recorded["programs"])
