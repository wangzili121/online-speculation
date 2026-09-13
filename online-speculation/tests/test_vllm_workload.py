from pathlib import Path

from blockspec.vllm_ascend.workload import (build_records, collect_code,
                                            read_manifest, write_manifest)
from blockspec.vllm_ascend.benchmark import parse_metrics


class Tokens:
    def __init__(self, values):
        self.input_ids = values


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return Tokens(list(text.encode()))

    def decode(self, values, skip_special_tokens=False):
        del skip_special_tokens
        return bytes(values).decode()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        return list((messages[0]["content"] + "<assistant>").encode())


def test_private_workload_is_deterministic_and_preserves_buckets(tmp_path):
    source = "def f(x): return x + 1\n" * 400
    buckets = ((200, 220, 0.5), (300, 320, 0.5))
    first = build_records(FakeTokenizer(), source, count=8, buckets=buckets,
                          max_new_tokens=16, seed=7)
    second = build_records(FakeTokenizer(), source, count=8, buckets=buckets,
                           max_new_tokens=16, seed=7)
    assert first == second
    assert sum(row["bucket"] == 0 for row in first) == 4
    assert all(buckets[row["bucket"]][0] <= row["actual_prompt_tokens"] <=
               buckets[row["bucket"]][1] for row in first)
    path = tmp_path / "workload.jsonl"
    assert write_manifest(path, first) == write_manifest(
        tmp_path / "copy.jsonl", second)
    assert read_manifest(path) == first


def test_collect_code_uses_only_supported_text_sources(tmp_path):
    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "b.md").write_text("ignored")
    text, count = collect_code([Path(tmp_path)])
    assert count == 1 and "print('a')" in text and "ignored" not in text


def test_prometheus_parser_keeps_only_scheduler_and_spec_metrics():
    metrics = parse_metrics("""
# ignored comment
vllm:num_requests_running{engine="0",model_name="orthrus"} 7.0
vllm:kv_cache_usage_perc{engine="0",model_name="orthrus"} 9.8e-1
vllm:spec_decode_num_draft_tokens_total{engine="0"} 310.0
python_gc_objects_collected_total{generation="0"} 4
""")
    assert metrics == {
        "vllm:num_requests_running": 7.0,
        "vllm:kv_cache_usage_perc": 0.98,
        "vllm:spec_decode_num_draft_tokens_total": 310.0,
    }
