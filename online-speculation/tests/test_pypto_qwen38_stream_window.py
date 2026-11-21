"""提交窗口与执行并发分开；旧两题路径保持原格式，三组请求覆盖必须一致。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blockspec.pypto_qwen38.online_study_metrics import matched_request_timelines


spec = importlib.util.spec_from_file_location("tp2_window_benchmark",
    Path(__file__).parents[1] / "scripts/benchmark_qwen38_oneserving_tp2_online.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def timeline(lengths):
    rows = [dict(seconds=1., tokens=[[1]] * n, stats={}, learner={}) for n in lengths]
    return bench.request_timeline(rows)


@pytest.mark.parametrize("window,sizes", [(2, [2, 2, 2]), (4, [4, 2]), (8, [6])])
def test_groups_preserve_exact_order_and_a_short_final_window(tmp_path, window, sizes):
    path = tmp_path / "prompts.json"
    texts = [f"question {i}" for i in range(6)]
    path.write_text(json.dumps(texts))
    groups = bench.load_pairs(path, window=window)
    assert [len(g) for g in groups] == sizes
    assert [q for g in groups for q in g] == texts
    assert matched_request_timelines(6, window, *[timeline(sizes)] * 3)
    if window == 2:
        assert bench.load_pairs(path) == [(texts[0], texts[1]), (texts[2], texts[3]), (texts[4], texts[5])]


@pytest.mark.parametrize("window", [True, 1, 3, 64])
def test_bad_window_fails_before_reading_prompts(tmp_path, window):
    with pytest.raises(ValueError, match="request window"):
        bench.load_pairs(tmp_path / "missing.json", window=window)


def test_all_requests_enter_the_queue_before_waiting_for_completion():
    requests = []

    class Events:
        def get(self, timeout):
            assert len(requests) == 4  # 没有在第二题结束后才提交第三、四题。
            return "done", None

    def submit(messages, tokens, sampling, ignore_eos):
        requests.append((messages, sampling))
        return SimpleNamespace(events=Events(), generated=[len(requests)], metrics=lambda: {"tpot_ms": 1.})

    engine = SimpleNamespace(runner=SimpleNamespace(spec_stats={"rounds": 0}),
                             scheduler=SimpleNamespace(error=None), submit_chat=submit)
    row = bench.run_pair(engine, ["a", "b", "c", "d"], 1)
    assert row["tokens"] == [[1], [2], [3], [4]]
    assert [request[0][0]["content"] for request in requests] == ["a", "b", "c", "d"]
    assert all(request[1] is None for request in requests)


def test_sampling_mismatch_cannot_submit_a_partial_window():
    engine = SimpleNamespace(submit_chat=lambda *_a, **_k: pytest.fail("request leaked before validation"))
    with pytest.raises(ValueError, match="sampling configuration"):
        bench.run_pair(engine, ["a", "b", "c", "d"], 1, sampling=[None, None])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "different_groups", "zero_output", "nan_time", "bool_id"])
def test_coverage_rejects_incomplete_or_mismatched_arm_windows(mutation):
    left, middle, right = [timeline([4, 4]) for _ in range(3)]
    if mutation == "missing":
        right.pop()
    elif mutation == "duplicate":
        right[1]["requests"][0] = 4
    elif mutation == "different_groups":
        right = timeline([2, 2, 2, 2])
    elif mutation == "zero_output":
        right[0]["output_tokens"][0] = 0
    elif mutation == "nan_time":
        right[0]["seconds"] = float("nan")
    else:
        right[0]["requests"][0] = True
    assert not matched_request_timelines(8, 4, left, middle, right)


def test_coverage_requires_all_three_arms_and_preserves_original_pair_contract():
    rows = timeline([2, 2])
    assert matched_request_timelines(4, 2, rows, rows, rows)
    assert not matched_request_timelines(4, 2, rows, rows)
    assert rows[1]["requests"] == [3, 4]


def test_trace_distinguishes_initial_fill_from_refill_and_checks_actual_concurrency():
    events = [dict(event="admitted", request_id="a"), dict(event="admitted", request_id="b"),
              dict(event="batch_submit", slots=[dict(phase="decode"), dict(phase="decode")]),
              dict(event="finished", request_id="a"), dict(event="admitted", request_id="c"),
              dict(event="batch_submit", slots=[dict(phase="decode"), dict(phase="prefill")]),
              dict(event="finished", request_id="b"), dict(event="finished", request_id="c")]
    assert bench.admission_summary(events) == dict(maximum_active_requests=2,
        refill_with_unfinished_request=1, batches=2, mixed_prefill_decode_batches=1, remaining_active=0)
