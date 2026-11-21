"""三组共用输入长度预算；不截断输入、不跳题，也不放宽逐请求的输出检查。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("tp2_budget_benchmark",
    Path(__file__).parents[1] / "scripts/benchmark_qwen38_oneserving_tp2_online.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("prompt,expected", [(453, 256), (847, 177), (896, 128), (1023, 1)])
def test_complete_prompt_and_output_stay_within_context(prompt, expected):
    assert bench.context_output_budget(256, prompt, 1024) == expected


@pytest.mark.parametrize("values", [(256, 1024, 1024), (256, 1025, 1024),
                                    (True, 4, 1024), (0, 4, 1024), (256, 0, 1024)])
def test_unusable_inputs_fail_without_removing_any_question(values):
    with pytest.raises(ValueError):
        bench.context_output_budget(*values)


@pytest.mark.parametrize("mapping", [False, True])
def test_chat_budget_counts_the_same_rendered_template_as_engine(mapping):
    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert options == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
            ids = list(range({"short": 453, "long": 847}[messages[0]["content"]]))
            return {"input_ids": ids} if mapping else ids

    budgets, lengths = bench.chat_output_budgets(Tokenizer(), ["short", "long"], 256, 1024)
    assert budgets == {"short": 256, "long": 177}
    assert lengths == {"short": 453, "long": 847}


def engine_for_limits(submitted, extra_first_token=False):
    def submit(messages, limit, **options):
        submitted.append(limit)
        count = limit + int(extra_first_token and len(submitted) == 1)
        return SimpleNamespace(events=SimpleNamespace(get=lambda **_: ("done", None)),
                               generated=[1] * count, metrics=lambda: {"tpot_ms": 1.})

    return SimpleNamespace(runner=SimpleNamespace(spec_stats={"rounds": 0}),
                           scheduler=SimpleNamespace(error=None), submit_chat=submit)


def test_each_request_receives_its_own_shared_budget_and_timeline_records_it():
    submitted = []
    row = bench.run_pair(engine_for_limits(submitted), ["a", "b"], 4, budgets=[1, 3])
    assert submitted == [1, 3]
    assert list(map(len, row["tokens"])) == [1, 3]
    row["learner"] = {}
    assert bench.request_timeline([row])[0]["output_budgets"] == [1, 3]


def test_eos_does_not_allow_a_response_to_exceed_its_own_budget():
    with pytest.raises(ValueError, match="generation budget"):
        bench.run_pair(engine_for_limits([], extra_first_token=True), ["a", "b"], 4,
                       budgets=[1, 3], stop_on_eos=True)


@pytest.mark.parametrize("limits", [[1], [1, 2, 3], [True, 2], [0, 2], [5, 2], [1.5, 2]])
def test_invalid_budgets_cannot_partially_submit_a_window(limits):
    submitted = []
    with pytest.raises(ValueError, match="output budget"):
        bench.run_pair(engine_for_limits(submitted), ["a", "b"], 4, budgets=limits)
    assert submitted == []


def test_fixed_budget_keeps_original_record_format():
    row = bench.run_pair(engine_for_limits([]), ["a", "b"], 1)
    row["learner"] = {}
    assert "output_budgets" not in row
    assert "output_budgets" not in bench.request_timeline([row])[0]
