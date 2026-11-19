"""完整块教师反馈的前缀校验；由服务端采集与更新调度调用。

调用方只传已交付 token 对应的验证行，并在设备输出复用前完成快照。
后续轮次沿修正后的真实前缀补齐教师分布，不使用拒绝候选后面的行。
"""

from copy import deepcopy


class DelayedTeacherBlock:
    def __init__(self, request_id, prefix, *, width=32):
        if not request_id or not prefix or type(width) is not int or not 2 <= width <= 32:
            raise ValueError("request, known anchor prefix and block width 2..32 required")
        self.request_id = request_id
        self.anchor = len(prefix) - 1
        self.width = width
        self.prefix = tuple(prefix)
        self.rows = []
        self.closed = False

    @property
    def full(self):
        return len(self.rows) == self.width - 1

    @property
    def ready(self):
        return self.full or self.closed and bool(self.rows)

    def append_verified(self, request_id, prefix, inputs, delivered, teacher_rows):
        """inputs 是确实被验证的输入行；delivered 包含最终接受的修正 token。"""
        if self.closed or self.full:
            raise ValueError("teacher block is already finished")
        prefix, inputs, delivered = tuple(prefix), tuple(inputs), tuple(delivered)
        if request_id != self.request_id or prefix != self.prefix:
            raise ValueError("teacher rows must continue this request's corrected prefix")
        if (not delivered or len(inputs) != len(delivered) or len(teacher_rows) != len(delivered)
                or inputs != (prefix[-1], *delivered[:-1])):
            raise ValueError("unverified or wrong-branch teacher rows cannot be replayed")
        count = min(len(delivered), self.width - 1 - len(self.rows))
        saved = deepcopy(list(teacher_rows[:count]))
        self.rows.extend(saved)
        self.prefix += delivered

    def finish(self):
        """EOS 或预算耗尽时保留已经真实触及的短块，不填充虚构教师行。"""
        self.closed = True
        return self.ready
