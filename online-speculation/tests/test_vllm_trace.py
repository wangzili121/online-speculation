import json

from blockspec.vllm_ascend import trace


def test_trace_is_disabled_without_environment(monkeypatch):
    monkeypatch.delenv("ORTHRUS_TRACE_FILE", raising=False)
    monkeypatch.setattr(trace, "_WRITER", None)
    with trace.span("disabled"):
        pass
    assert trace.writer() is None


def test_trace_writes_chrome_events(tmp_path, monkeypatch):
    path = tmp_path / "trace-{pid}.jsonl"
    monkeypatch.setenv("ORTHRUS_TRACE_FILE", str(path))
    monkeypatch.setattr(trace, "_WRITER", None)
    trace.instant("ready", {"request_ids": ["r1"]})
    with trace.span("run", {"batch_requests": 1}):
        pass
    output = list(tmp_path.glob("trace-*.jsonl"))[0]
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert [event["name"] for event in events] == [
        "orthrus.clock_sync", "ready", "run"]
    assert events[0]["args"]["wall_time_us"] > 0
    assert events[2]["ph"] == "X" and events[2]["dur"] >= 0
    merged = tmp_path / "merged.json"
    assert trace.merge_events([str(output)], merged) == 3
    assert len(json.loads(merged.read_text())["traceEvents"]) == 3
