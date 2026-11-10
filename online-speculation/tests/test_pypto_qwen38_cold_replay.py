"""教师始终消费真实序列，draft 永远不能看到当前块答案。"""

import random
from types import SimpleNamespace as NS

import pytest

from blockspec.pypto_qwen38.cold_replay import ReplayJob, ReplayRunner, sample_jobs


def test_overlapping_anchors_restore_before_advancing_the_clean_history():
    events=[]
    job=ReplayJob("q",tuple(range(24)),(2,5,9),8)
    native=NS(history=[],requests={})
    resident=NS(pages=NS(length=0))
    native.requests["r"]=resident
    def prefill(slots):
        native.history.extend(slots[0].token_ids)
        resident.pages.length=len(native.history)
        return [99]
    native.run_batch=prefill
    learner=NS(current=None,rounds=0)
    def observe():
        assert learner.current.rows==8
        learner.rounds+=1
        learner.current=None
    learner.observe_clean_block=observe
    learner.abort=lambda:events.append("abort")
    def propose(request,anchor,width):
        assert native.history==list(job.tokens[:anchor])
        assert tuple(request.prompt_ids)==job.tokens[:anchor] and request.generated==[anchor]
        learner.current=NS(past=anchor,width=width,rows=0)
        events.append(("draft",anchor))
        return [999]*(width-1)
    class Target:
        def __init__(self,native,request,slot_type):
            self.resident=resident;self.collector=None
        def snapshot(self):return tuple(native.history)
        def restore(self,state):
            native.history[:]=state;resident.pages.length=len(state)
        def forward(self,inputs):
            start=len(native.history)
            assert tuple(inputs)==job.tokens[start:start+len(inputs)]
            native.history.extend(inputs);resident.pages.length=len(native.history)
            if self.collector is not None:
                assert events[-1]==("draft",start)
                self.collector.rows+=len(inputs)
                events.append(("teacher",start))
            return list(inputs)
    runner=NS(native=native,draft=NS(propose=propose,weights=NS(version=0)),slot_type=object)
    replay=ReplayRunner(runner,learner,target_factory=Target);replay.job=job
    request=NS(request_id="r",prompt_ids=job.tokens[:2],max_tokens=1,generated=[])
    result=replay.run_batch([NS(request=request,token_ids=job.tokens[:2],position=0,phase="prefill",completes_prefill=True)])
    assert result==[99] and request.generated==[] and request.prompt_ids==job.tokens[:2]
    assert native.history==list(job.tokens[:9]) and learner.rounds==3
    assert [s["anchor"] for s in replay.samples]==[2,5,9]
    assert all(s["supervised_positions"]==7 and s["teacher_forced_argmax_matches"]==0 for s in replay.samples)


def test_sampling_uses_distinct_records_and_reproducible_complete_anchors():
    records=[dict(id=str(i),tokens=list(range(40))) for i in range(5)]
    a=sample_jobs(records,random.Random(42),width=8)
    assert a==sample_jobs(records,random.Random(42),width=8)
    assert len({job.record_id for job in a})==2
    assert all(len(job.anchors)==4 and min(job.anchors)>=1 and max(job.anchors)+8<=40 for job in a)
    with pytest.raises(ValueError):ReplayJob("q",tuple(range(10)),(5,),8)
