"""Bind the packed last-n replay and optimizer to the native graph runtime."""

from .dsv41_native_online_artifacts import _adam_entries
from .dsv41_packed_adam import PackedAdam
from .dsv41_packed_attention_vjp import PackedAttentionVJP
from .dsv41_packed_attn_input_vjp import PackedAttentionInputVJP
from .dsv41_packed_attn_post_vjp import PackedAttentionPostVJP
from .dsv41_packed_ep_dispatch import PackedExpertDispatch
from .dsv41_packed_ep_reverse import PackedExpertReverse
from .dsv41_packed_ffn_input_vjp import PackedFFNInputVJP
from .dsv41_packed_ffn_mhc_vjp import PackedFFNMHCVJP
from .dsv41_packed_ffn_tape import PackedFFNPostVJP
from .dsv41_packed_final_vjp import PackedFinalVJP
from .dsv41_packed_lm_head_vjp import PackedLMHeadVJP
from .dsv41_packed_loss import PackedLoss, PackedTeacherCopy
from .dsv41_packed_qkv_vjp import PackedQKVVJP
from .dsv41_packed_wo_vjp import PackedWoInputVJP


def bind_native_learner(draft, runtime, programs, last_layers):
    from models.deepseek_v4_1_flash_a3.config import EP_SIZE, FLASH, N_LOCAL_EXPERTS, TP_SIZE

    draft.world, draft.last_layers, draft.pages = EP_SIZE, last_layers, draft.stager.pages
    draft.runtime, draft.config = runtime, None
    banks = runtime.load_banks()
    draft.ranks = tuple(banks["draft"][rank] for rank in range(EP_SIZE))
    draft.resident_captures = banks["capture"]
    draft.tape_specs = tuple(name for name in draft.ranks[0] if name.startswith(("ffn_tape_", "attention_tape_")))
    if set(draft.resident_captures) != set(range(0, EP_SIZE, TP_SIZE)):
        raise ValueError("Native online updates require one capture bank per DP owner")

    graphs = {name: program for name, (program, _) in programs.items()}
    model, ranks = FLASH, draft.ranks
    loss = PackedLoss(draft, graphs["loss"])
    sources = tuple(banks["target"][rank]["logits"] for rank in loss.owners)
    draft.teacher_copy = PackedTeacherCopy(loss, graphs["teacher-copy"], sources, single_capture=True)
    lm_head = PackedLMHeadVJP(draft, loss, graphs["lm-head"])
    final = PackedFinalVJP(draft, lm_head, graphs["final-norm"], output_rows=draft.block)
    ffn_post = PackedFFNPostVJP(draft, final, graphs["ffn-output"], model.num_hidden_layers)
    expert_dispatch = PackedExpertDispatch(draft, ffn_post, graphs["expert-dispatch"],
                                           model.num_experts_per_tok, model.hidden_size)
    expert_reverse = PackedExpertReverse(runtime, None, graphs["expert-reverse"], ranks, draft.block,
                                         model.hidden_size, model.moe_intermediate_size, model.num_experts_per_tok,
                                         N_LOCAL_EXPERTS, TP_SIZE)
    ffn_input = PackedFFNInputVJP(draft, graphs["ffn-input"], ranks, model)
    ffn_mhc = PackedFFNMHCVJP(draft, ffn_post, ffn_input, graphs["ffn-mhc"], ranks, model)
    attention_post = PackedAttentionPostVJP(draft, ffn_mhc, graphs["attention-output"], ranks, model)
    wo = PackedWoInputVJP(draft, graphs["wo"], model, parameters=True)
    attention = PackedAttentionVJP(draft, graphs["attention"], model)
    qkv = PackedQKVVJP(draft, graphs["qkv"], model)
    attention_input = None
    if last_layers > 1:
        attention_input = PackedAttentionInputVJP(draft, attention_post, qkv, graphs["attention-input"], model)
    adam_graphs = {
        key: tuple(graphs[f"adam/{name}/{phase}"] for phase in ("stage", "apply", "publish"))
        for key, name in _adam_entries(model, TP_SIZE).items()
    }
    optimizer = PackedAdam(draft, model, adam_graphs)
    runtime.prepare()
    return dict(
        draft_loss=loss, draft_lm_head_vjp=lm_head, draft_final_vjp=final, draft_ffn_post_vjp=ffn_post,
        draft_expert_dispatch=expert_dispatch, draft_expert_reverse=expert_reverse, draft_ffn_input=ffn_input,
        draft_ffn_mhc=ffn_mhc, draft_attn_post=attention_post, draft_wo_input=wo, draft_attention=attention,
        draft_qkv=qkv, draft_attn_input=attention_input, draft_optimizer=optimizer,
    )
