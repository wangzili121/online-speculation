import asyncio
from threading import RLock
from types import SimpleNamespace

import pytest

from blockspec import pypto_glm53_online_serving

async_engine = pytest.importorskip("pypto_serving.serving.engine.async_engine")
ipc = pytest.importorskip("pypto_serving.serving.server.ipc")
serving_worker = pytest.importorskip("pypto_serving.serving.server.serving_worker")
npu_runner = pytest.importorskip("pypto_serving.model.glm53.npu_runner")

if "orthrus_replays" not in ipc.StepCommand.__struct_fields__:
    pytest.skip("requires online_http_serving.patch", allow_module_level=True)


@pytest.mark.parametrize(
    "lr,reason,count,release_worker,expected",
    [
        (1e-4, "FINISHED_LENGTH", 6, True, {"req": [10, 11, 12, 13, 14, 15]}),
        (1e-4, "FINISHED_EOS", 7, True, {"req": [10, 11, 12, 13, 14, 15]}),
        (1e-4, "FINISHED_STOP", 5, True, {}),
        (1e-4, "FINISHED_ABORTED", 7, True, {}),
        (None, "FINISHED_LENGTH", 7, True, {}),
        (1e-4, "FINISHED_LENGTH", 7, False, {}),
    ],
)
def test_only_final_scheduler_tokens_are_sent_for_replay(lr, reason, count, release_worker, expected):
    from pypto_serving.serving.sched.scheduler import Request, RequestOutput, SchedulerOutput

    engine = async_engine.ReplicaEngineCore.__new__(async_engine.ReplicaEngineCore)
    engine.config = SimpleNamespace(executor_kwargs={"orthrus_online_learning_rate": lr})
    engine._runtime = SimpleNamespace(num_speculative_tokens=5)
    engine._pending_free_ids = []
    engine._worker_known_req_ids = {"req"}
    req = Request(request_id="req", prompt_token_ids=[1, 2], max_new_tokens=8)
    # The worker may have put more candidates into the result than the scheduler consumed.
    req.output_token_ids.extend(range(10, 18))
    engine._request_contexts = {"req": async_engine._RequestContext(req)}
    engine._detokenize_incrementally = lambda ctx, end_offset: "output"
    engine._finalize_detokenization = lambda ctx: "output"
    engine._publish_outputs([
        RequestOutput(request_id="req", new_token_id=10 + count - 1, finished=True,
                      finish_reason=reason, num_output_tokens=count),
    ], release_worker=release_worker)
    command = engine._build_step_command(SchedulerOutput(scheduled_requests=[]), engine._pending_free_ids)
    wire = ipc.decode_command(ipc.encode_command(command))
    assert wire.orthrus_replays == expected


def test_cleanup_only_step_delivers_replay_and_waits_for_worker_ack():
    engine = async_engine.ReplicaEngineCore.__new__(async_engine.ReplicaEngineCore)
    context = async_engine._RequestContext(SimpleNamespace())
    context.orthrus_replay_tokens = tuple(range(6))
    engine._request_contexts = {"req": context}
    engine._pending_free_ids = ["req"]
    engine._worker_known_req_ids = {"req"}
    engine._batch_queue = []
    engine._step_counter = 12
    submitted = []
    engine._input_queue = SimpleNamespace(put=submitted.append)

    async def worker_result():
        assert not context.worker_release_done.is_set()
        command = ipc.decode_command(submitted[0])
        assert command.prefill_requests == command.decode_requests == []
        assert command.orthrus_replays == {"req": list(range(6))}
        return ipc.encode_result(ipc.StepResult(new_tokens={}, step_id=command.step_id))

    engine._get_live_result = worker_result
    asyncio.run(engine._flush_pending_frees())
    assert context.worker_release_done.is_set()


def test_worker_replays_after_release_and_before_reregistration_once():
    worker = serving_worker.WorkerProcess.__new__(serving_worker.WorkerProcess)
    worker._last_tokens = {"req": [15]}
    old = ipc.NewRequestData("req", [1, 2], 0.0, 1.0, None)
    replacement = ipc.NewRequestData("req", [3, 4], 0.0, 1.0, None)
    worker._req_cache = {"req": old}
    worker.model_record = SimpleNamespace(config=SimpleNamespace(model_id="glm"))
    events = []

    def update(model_id, prompt, completion):
        assert "req" not in worker._req_cache
        assert "req" not in worker._last_tokens
        events.append((model_id, prompt, completion))

    worker.executor = SimpleNamespace(
        release_finished_requests=lambda ids: events.append(tuple(ids)), run_orthrus_update=update,
    )
    worker._apply_command_lifecycle(ipc.StepCommand(
        [replacement], [], [], ["req", "req"], orthrus_replays={"req": [10, 11, 12, 13, 14, 15]},
    ))
    assert events == [("req", "req"), ("glm", [1, 2], [10, 11, 12, 13, 14, 15])]
    assert worker._req_cache["req"] is replacement


def test_runner_reuses_one_learner_and_only_free_slot(monkeypatch):
    calls = []
    runner = npu_runner.Glm53ModelRunner.__new__(npu_runner.Glm53ModelRunner)
    runner._orthrus_online_learning_rate = 1e-4
    runner._orthrus_learner = None
    runner._orthrus_mask_id = 99
    runner._api = SimpleNamespace(VERIFY_QUERY_WIDTH=6)
    runner._state_lock = RLock()
    runner._max_seq_len = 8
    runner._requests = {"busy": object()}
    runner._free_slots = [1]

    class Learner:
        def __init__(self, target, *, learning_rate):
            assert target is runner
            calls.append(("init", learning_rate))
            self.mla_update = SimpleNamespace(step_count=0)

        def __enter__(self):
            return self

        def step(self, tokens, anchor, slot, mask):
            calls.append((tokens, anchor, slot, mask))
            self.mla_update.step_count += 1

    monkeypatch.setattr(pypto_glm53_online_serving, "ResidentDistillation", Learner)
    output = list(range(10, 16))
    runner.run_orthrus_update([1, 2], output)
    assert calls == []
    runner._requests.clear()
    runner.run_orthrus_update([1, 2, 3], output)
    runner.run_orthrus_update([1, 2], output[:-1])
    assert calls == []
    runner.run_orthrus_update([1, 2], output)
    runner.run_orthrus_update([3, 4], output)
    assert calls == [("init", 1e-4), ([1, 2, *output], 2, 1, 99), ([3, 4, *output], 2, 1, 99)]
