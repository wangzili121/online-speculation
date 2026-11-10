"""干净回答的整块重放；训练不使用投机拒绝之后的错误前缀。"""

import copy
from dataclasses import dataclass
import random

from .target import NativeTarget


@dataclass(frozen=True)
class ReplayJob:
    record_id: str
    tokens: tuple[int, ...]
    anchors: tuple[int, ...]
    width: int

    def __post_init__(self):
        if (not 2 <= self.width <= 32 or not self.anchors or tuple(sorted(self.anchors)) != self.anchors
                or len(set(self.anchors)) != len(self.anchors)
                or any(a < 1 or a+self.width > len(self.tokens) for a in self.anchors)):
            raise ValueError("sorted distinct anchors and complete clean blocks with nonempty history required")


def sample_jobs(records, rng: random.Random, *, width, records_per_update=2, anchors_per_record=4):
    if records_per_update > len(records):
        raise ValueError("not enough distinct response records")
    jobs = []
    for record in rng.sample(list(records), records_per_update):
        candidates = range(1, len(record["tokens"])-width+1)
        if len(candidates) < anchors_per_record:
            raise ValueError("record too short for distinct anchors")
        anchors = tuple(sorted(rng.sample(candidates, anchors_per_record)))
        jobs.append(ReplayJob(record["id"], tuple(record["tokens"]), anchors, width))
    return jobs


class ReplayRunner:
    """由现有单请求调度器执行重放；持久状态只用于本次内部训练请求。"""

    def __init__(self, runner, learner, *, target_factory=NativeTarget):
        self.runner, self.learner, self.target_factory = runner, learner, target_factory
        self.job = None
        self.samples = []

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def run_batch(self, slots):
        if self.job is None or len(slots) != 1 or slots[0].phase != "prefill":
            raise RuntimeError("clean replay requires one explicitly assigned prefill request")
        slot, job = slots[0], self.job
        output = self.runner.native.run_batch(slots)
        if not slot.completes_prefill:
            return output
        if tuple(slot.request.prompt_ids) != job.tokens[:job.anchors[0]] or slot.request.max_tokens != 1:
            raise RuntimeError("replay request must prefill exactly the first anchor's clean history")
        request = copy.copy(slot.request)
        target = self.target_factory(self.runner.native, request, self.runner.slot_type)
        try:
            for anchor in job.anchors:
                past = target.resident.pages.length
                if past > anchor:
                    raise RuntimeError("teacher history passed the next anchor")
                if past < anchor:
                    target.forward(job.tokens[past:anchor])
                request.prompt_ids, request.generated = job.tokens[:anchor], [job.tokens[anchor]]
                snapshot = target.snapshot()
                try:
                    # q 只看 anchor 历史和 MASK；在此之后才向教师输入真实的后续 token。
                    candidates = self.runner.draft.propose(request, job.tokens[anchor], job.width)
                    version = self.runner.draft.weights.version
                    block = self.learner.current
                    if block is None or block.past != anchor or block.width != job.width:
                        raise RuntimeError("clean replay failed to capture the draft inputs")
                    target.collector = block
                    teacher_tokens = target.forward(job.tokens[anchor:anchor+job.width])
                    target.collector = None
                    self.learner.observe_clean_block()
                    self.samples.append(dict(record_id=job.record_id, anchor=anchor, width=job.width,
                        supervised_positions=job.width-1,
                        head_version=version,
                        matched_positions=[a == b for a,b in zip(candidates, teacher_tokens[:-1], strict=True)],
                        teacher_forced_argmax_matches=sum(a == b for a,b in zip(candidates, teacher_tokens[:-1], strict=True))))
                finally:
                    target.collector = None
                    target.restore(snapshot)
                    del snapshot
            return output  # 内部请求只交付一个 token 后结束，调度器释放其全部临时历史。
        except BaseException:
            self.learner.abort()
            raise
