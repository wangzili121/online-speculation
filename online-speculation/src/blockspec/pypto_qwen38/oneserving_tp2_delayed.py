"""Reuse verified future rows to supervise an earlier masked block.

Only the feedback schedule changes. Draft, verifier and target caches keep
their original execution. All teacher rows follow the corrected AR prefix.
"""

from dataclasses import replace

from .delayed_teacher import DelayedTeacherBlock
from .oneserving_tp2_online import TP2Online


class TP2DelayedOnline(TP2Online):
    def __init__(self, *args, teacher_tokens=31, feedback_timing="early", **kwargs):
        if type(teacher_tokens) is not int or not 1 <= teacher_tokens <= 31:
            raise ValueError("teacher_tokens must be in 1..31")
        self.teacher_tokens = teacher_tokens
        if feedback_timing not in {"early", "near_update"}:
            raise ValueError("feedback timing must be early or near_update")
        self.feedback_timing = feedback_timing
        self.window = self.request = None
        self.captures = 0
        super().__init__(*args, **kwargs)

    def restore(self):
        super().restore()
        self.window = self.request = None
        self.captures = 0

    def prepare_batch(self):
        super().prepare_batch()
        if self.evaluation_paused:
            return  # 固定组不消费反馈；在线组的窗口与 replay 跨题保留。
        if self.stride is None:
            # Unfinished feedback must not leak into a frozen evaluation or reset.
            self.window = self.request = self.feedback = None
            self.replay.clear()
        elif self.window is not None and (self.request.finish_reason is not None
                                          or self.request.cancelled.is_set()):
            self.window.finish()
            if self.window.ready:
                self.feedback["complete"] = True
            else:
                self.window = self.request = self.feedback = None

    def feedback_index(self, active):
        if (self.stride is None or self.update_limit_reached() or self.pending_update is not None
                or self.queued_update is not None
                or self.window is not None or len(self.replay) >= self.replay_blocks):
            return None
        if self.feedback_timing == "near_update":
            interval = self.update_interval()
            # 最多两个活跃请求；为每个未填反馈块预留逐 token 验证的轮数。
            # 只推迟采集起点，不补算教师、不缩短监督，也不跨过 EOS。
            lead = min(interval, 2 * self.teacher_tokens * (self.replay_blocks - len(self.replay)) + 2)
            if interval - self.rounds % interval > lead:
                return None
        index = self.captures % active
        self.captures += 1
        return index

    def capture_request(self, slot):
        super().capture_request(slot)
        request = slot.request
        if request.stop_strings:
            raise ValueError("delayed teacher feedback currently requires token-based stopping")
        prefix = tuple(request.prompt_ids) + tuple(request.generated)
        if len(prefix) - 1 != slot.position or prefix[-1] != slot.token_ids[0]:
            raise RuntimeError("the draft boundary must follow the actual delivered prefix")
        targets = min(self.feedback["width"] - 1, self.teacher_tokens)
        self.window = DelayedTeacherBlock(request.request_id, prefix, width=targets + 1)
        self.request = request
        # Inference may propose only seven tokens; its K32 block can still be
        # trained on 31 later, genuinely verified AR positions.
        self.feedback["candidates"] = targets
        self.feedback["teacher_tokens"] = targets
        self.feedback["request_id"] = request.request_id

    def observe_teacher(self, slot, batch_slot, emitted, previous, offset, count, *, final,
                        selected, source_row=0):
        if (self.stride is None or self.window is None or self.window.ready
                or slot.request.request_id != self.window.request_id):
            return
        request = slot.request
        prefix = tuple(request.prompt_ids) + tuple(request.generated) + tuple(previous)
        remaining = request.max_tokens - len(request.generated) - len(previous)
        delivered = list(emitted[:remaining])
        stopped = False
        for i, token in enumerate(delivered):
            if token in request.stop_token_ids:
                delivered = delivered[:i + 1]
                stopped = True
                break
        if not delivered:
            return
        if slot.position != len(prefix) - 1 or len(delivered) > count:
            raise RuntimeError("teacher rows and committed output positions differ")
        first = self.feedback["valid"]
        # The CPU window validates that the verifier's input rows really follow
        # the emitted correction before any of their distributions are uploaded.
        self.window.append_verified(request.request_id, prefix, slot.token_ids[:len(delivered)],
                                    delivered, list(range(slot.position, slot.position + len(delivered))))
        extra = {"target_ids": delivered} if self.teacher_targets == "greedy" else {}
        super().capture_teacher(batch_slot, first, len(delivered), final=False, source_row=source_row, **extra)
        if stopped or len(delivered) == remaining:
            self.window.finish()
        self.feedback["full_feedback"] = self.window.full
        self.feedback["complete"] = self.window.ready
        if self.feedback["valid"] != len(self.window.rows):
            raise RuntimeError("teacher snapshot count differs from the verified prefix")

    def observe_native(self, slots, outputs):
        if self.window is None or self.stride is None:
            return
        for index, (slot, emitted) in enumerate(zip(slots, outputs, strict=True)):
            if emitted is None or slot.request.request_id != self.window.request_id:
                continue
            row = len(slot.token_ids) - 1 if slot.phase == "prefill" else 0
            current = replace(slot, token_ids=(slot.token_ids[row],),
                              position=slot.position + row, phase="decode")
            self.observe_teacher(current, index, emitted, [], 0, 1, final=True,
                                 selected=False, source_row=row)
        self.finish_batch(0)

    def take_feedback(self):
        if self.feedback is None or not self.feedback["complete"]:
            return None
        state = super().take_feedback()
        self.window = self.request = None
        return state

    def ready_for_update(self):
        # A standalone transfer cannot run on the learner worker while a graph
        # owns its queue. Finish an active collector before launching training.
        return self.window is None and bool(self.replay)
