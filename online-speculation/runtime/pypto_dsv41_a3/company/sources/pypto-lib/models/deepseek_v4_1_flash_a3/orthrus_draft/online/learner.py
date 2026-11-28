# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Train the captured suffix and publish updated device buffers."""

from .adam import PackedAdam
from .attention_input_vjp import PackedAttentionInputVJP
from .attention_post_vjp import PackedAttentionPostVJP
from .attention_vjp import PackedAttentionVJP
from .expert_dispatch import PackedExpertDispatch
from .expert_reverse import PackedExpertReverse
from .ffn_input_vjp import PackedFFNInputVJP
from .ffn_mhc_vjp import PackedFFNMHCVJP
from .final_vjp import PackedFinalVJP
from .lm_head_vjp import PackedLMHeadVJP
from .loss import PackedLoss
from .programs import _adam_entries
from .qkv_vjp import PackedQKVVJP
from .wo_vjp import PackedWoInputVJP
from ..ffn_tape import PackedFFNPostVJP
from ... import config as C


class ResidentLearner:
    def __init__(self, draft, graphs):
        self.draft = draft
        model = C.FLASH
        ranks = draft.ranks
        self.loss = PackedLoss(draft, graphs["loss"])
        self.lm_head = PackedLMHeadVJP(draft, self.loss, graphs["lm-head"])
        self.final = PackedFinalVJP(draft, self.lm_head, graphs["final-norm"], output_rows=draft.block)
        self.ffn_post = PackedFFNPostVJP(draft, self.final, graphs["ffn-output"], model.num_hidden_layers)
        self.dispatch = PackedExpertDispatch(draft, self.ffn_post, graphs["expert-dispatch"],
                                             model.num_experts_per_tok, model.hidden_size)
        self.experts = PackedExpertReverse(draft.runtime, draft.config, graphs["expert-reverse"], ranks, draft.block,
                                           model.hidden_size, model.moe_intermediate_size, model.num_experts_per_tok,
                                           C.N_LOCAL_EXPERTS, C.TP_SIZE)
        self.ffn_input = PackedFFNInputVJP(draft, graphs["ffn-input"], ranks, model)
        self.ffn_mhc = PackedFFNMHCVJP(draft, self.ffn_post, self.ffn_input, graphs["ffn-mhc"], ranks, model)
        self.attention_post = PackedAttentionPostVJP(draft, self.ffn_mhc, graphs["attention-output"], ranks, model)
        self.wo = PackedWoInputVJP(draft, graphs["wo"], model, parameters=True)
        self.attention = PackedAttentionVJP(draft, graphs["attention"], model)
        self.qkv = PackedQKVVJP(draft, graphs["qkv"], model)
        self.attention_input = (PackedAttentionInputVJP(draft, self.attention_post, self.qkv,
                                                       graphs["attention-input"], model)
                                if draft.last_layers > 1 else None)
        adam_graphs = {
            key: tuple(graphs[f"adam/{name}/{phase}"] for phase in ("stage", "apply", "publish"))
            for key, name in _adam_entries(model, C.TP_SIZE).items()
        }
        self.optimizer = PackedAdam(draft, model, adam_graphs)

    def update(self, teacher, *, rate):
        student = self.draft.student
        if student is None:
            raise ValueError("An online update requires a captured draft suffix")

        def differentiate(consume):
            loss = self.loss(student, teacher)
            lm_grad = self.lm_head(loss)
            boundary = self.final(student, lm_grad)
            stop = C.FLASH.num_hidden_layers - self.draft.last_layers
            for layer in range(C.FLASH.num_hidden_layers - 1, stop - 1, -1):
                post = self.ffn_post(student, boundary, layer=layer)
                packets = self.dispatch(student, post)
                expert = self.experts(packets)
                collapsed = self.ffn_input(student, packets, expert)
                mixed = self.ffn_mhc(student, post, collapsed)
                post = self.attention_post(student, mixed)
                wo = self.wo(student, post)
                attention = self.attention(student, wo)
                projection = self.qkv(student, attention)
                consume(projection)
                if layer > stop:
                    boundary = self.attention_input(student, projection)
            return loss

        try:
            return self.optimizer.update(differentiate, rate=rate)
        finally:
            self.draft.student = None
