"""Connect node-local draft VJPs around the cross-host expert exchange."""

from time import perf_counter

from .dsv41_online_profile import timed_stage
from .dsv41_packed_attention_vjp import PackedAttentionVJP, make_packed_attention_vjp
from .dsv41_packed_attn_input_vjp import PackedAttentionInputVJP, make_packed_attn_input_vjp
from .dsv41_packed_attn_post_vjp import PackedAttentionPostVJP, make_packed_attn_post_vjp
from .dsv41_packed_ffn_input_vjp import PackedFFNInputVJP, make_packed_ffn_input_vjp
from .dsv41_packed_ffn_mhc_vjp import PackedFFNMHCVJP, make_packed_ffn_mhc_vjp
from .dsv41_packed_ffn_tape import PackedFFNPostVJP, make_packed_ffn_post_vjp
from .dsv41_packed_final_vjp import PackedFinalVJP, make_packed_final_vjp
from .dsv41_packed_lm_head_vjp import PackedLMHeadVJP, make_packed_lm_head_vjp
from .dsv41_packed_loss import PackedLoss, PackedTeacherCopy, make_packed_loss_graph, make_packed_teacher_copy
from .dsv41_packed_qkv_vjp import PackedQKVVJP, make_packed_qkv_vjp
from .dsv41_packed_wo_vjp import PackedWoInputVJP, make_packed_wo_input_vjp
from .dsv41_staged_ep_vjp import StagedExpertVJP, make_staged_expert_vjp


_EXPERT_STAGES = ("expert-pack", "expert-compute", "expert-combine")
_LOCAL_STAGES = ("loss", "lm-head", "final-norm", "ffn-output", "ffn-input", "ffn-mhc",
                 "attention-output", "wo", "attention", "qkv")


def make_staged_reverse_graphs(block, layers, *, physical_pages, local_ranks):
    from models.deepseek_v4_1_flash_a3.config import EP_SIZE, FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS

    model, tp, rows = FLASH, TP_SIZE, MAX_LOGIT_ROWS
    dp = local_ranks // tp
    hidden, copies, vocab = model.hidden_size, model.hc_mult, model.vocab_size
    graphs = {
        "loss": make_packed_loss_graph(rows, vocab, dp, tp, "tv"),
        "teacher-copy": make_packed_teacher_copy(rows, vocab, dp, tp),
        "lm-head": make_packed_lm_head_vjp(rows, vocab, hidden, dp, tp),
        "final-norm": make_packed_final_vjp(rows, hidden, copies, dp, tp, model.rms_norm_eps, output_rows=block),
        "ffn-output": make_packed_ffn_post_vjp(block, layers, block, hidden, copies, dp, tp),
        "ffn-input": make_packed_ffn_input_vjp(block, layers, model, dp, tp),
        "ffn-mhc": make_packed_ffn_mhc_vjp(block, layers, model, dp, tp),
        "attention-output": make_packed_attn_post_vjp(block, layers, model, dp, tp),
        "wo": make_packed_wo_input_vjp(block, model, tp, world=local_ranks, parameters=True),
        "attention": make_packed_attention_vjp(block, world=local_ranks, tp=tp, physical_pages=physical_pages),
        "qkv": make_packed_qkv_vjp(block, model, world=local_ranks, tp=tp),
    }
    if layers > 1:
        graphs["attention-input"] = make_packed_attn_input_vjp(block, layers, model, dp, tp)
    graphs.update(zip(_EXPERT_STAGES, make_staged_expert_vjp(
        model, block, layers, world=EP_SIZE, local_ranks=local_ranks, tp=tp,
    ), strict=True))
    return graphs


def load_staged_reverse_programs(directory, layers, *, resident_teacher=False):
    from pypto.ir import DistributedCompiledProgram

    names = (*_LOCAL_STAGES, *_EXPERT_STAGES)
    if resident_teacher:
        names += ("teacher-copy",)
    if layers > 1:
        names += ("attention-input",)
    return {name: DistributedCompiledProgram.from_dir(directory / "reverse" / name) for name in names}


class StagedDraftReverse:
    """Compute one retained draft suffix while both hosts participate in expert VJP."""

    def __init__(self, draft, config, programs):
        from models.deepseek_v4_1_flash_a3.config import FLASH

        draft.config = config
        self.draft = draft
        self.layers = tuple(range(FLASH.num_hidden_layers - 1, FLASH.num_hidden_layers - draft.last_layers - 1, -1))
        self.loss = PackedLoss(draft, programs["loss"])
        if "teacher-copy" in programs:
            sources = [draft.target.handles[1][rank]["logits"] for rank in self.loss.owners]
            draft.teacher_copy = PackedTeacherCopy(self.loss, programs["teacher-copy"], sources)
        self.lm_head = PackedLMHeadVJP(draft, self.loss, programs["lm-head"])
        self.final = PackedFinalVJP(draft, self.lm_head, programs["final-norm"], output_rows=draft.block)
        self.ffn_post = PackedFFNPostVJP(draft, self.final, programs["ffn-output"], FLASH.num_hidden_layers)
        self.ffn_input = PackedFFNInputVJP(draft, programs["ffn-input"], draft.ranks, FLASH)
        self.ffn_mhc = PackedFFNMHCVJP(draft, self.ffn_post, self.ffn_input, programs["ffn-mhc"], draft.ranks, FLASH)
        self.attn_post = PackedAttentionPostVJP(draft, self.ffn_mhc, programs["attention-output"],
                                               draft.ranks, FLASH)
        self.wo = PackedWoInputVJP(draft, programs["wo"], FLASH, parameters=True)
        self.attention = PackedAttentionVJP(draft, programs["attention"], FLASH)
        self.qkv = PackedQKVVJP(draft, programs["qkv"], FLASH)
        if draft.last_layers > 1:
            self.attn_input = PackedAttentionInputVJP(draft, self.attn_post, self.qkv,
                                                      programs["attention-input"], FLASH)
        self.experts = StagedExpertVJP(draft.runtime, draft.ffn.moe.transport,
                                       tuple(programs[name] for name in _EXPERT_STAGES),
                                       draft.ranks, FLASH, block=draft.block, last_layers=draft.last_layers)

    def run(self, stage, *, owner, feedback, finished, consume):
        """Deliver each layer's gradients before scratch is reused for the preceding layer."""
        draft = self.draft
        local = owner in draft.target.global_ranks
        if local != (feedback is not None):
            raise ValueError("The request owner must hold the only teacher feedback")
        link = draft.target.link
        teacher = feedback.logits(finished=finished) if local else None
        local_rows = len(teacher) if teacher is not None else 0
        peer_rows = link.exchange(f"{stage}/teacher-rows", local_rows)
        rows = local_rows if local else peer_rows
        if not rows:
            return None

        start = perf_counter()
        loss, student, boundary = None, None, None
        if local:
            student = feedback.student
            with timed_stage("online_loss"):
                loss = self.loss(student, teacher)
            with timed_stage("online_lm_head_vjp"):
                head_gradient = self.lm_head(loss)
            with timed_stage("online_final_vjp"):
                boundary = self.final(student, head_gradient)

        for index, layer in enumerate(self.layers):
            count = rows if index == 0 else draft.block
            post = None
            if local:
                with timed_stage("online_ffn_post_vjp"):
                    post = self.ffn_post(student, boundary, layer=layer)
            result = self.experts.run_layer(
                f"{stage}/layer-{layer}", owner=owner, layer=layer, rows=count,
                weights=draft.layer_weights[layer][1], student=student, gradient=post,
            )
            projection = None
            if local:
                with timed_stage("online_ffn_input_vjp"):
                    normed = self.ffn_input(student, result, result.gradients)
                    mixed = self.ffn_mhc(student, post, normed)
                with timed_stage("online_attention_output_vjp"):
                    seed = self.attn_post(student, mixed)
                    projected = self.wo(student, seed)
                with timed_stage("online_attention_vjp"):
                    attention_gradient = self.attention(student, projected)
                with timed_stage("online_qkv_vjp"):
                    projection = self.qkv(student, attention_gradient)
            consume(layer, projection)
            if local and layer != self.layers[-1]:
                with timed_stage("online_attention_input_vjp"):
                    boundary = self.attn_input(student, projection)

        local_loss = loss.value if loss is not None else None
        peer_loss = link.exchange(f"{stage}/complete", local_loss)
        return dict(owner=owner, teacher_rows=rows, loss=local_loss if local else peer_loss,
                    layers=list(self.layers), reverse_seconds=perf_counter() - start)
