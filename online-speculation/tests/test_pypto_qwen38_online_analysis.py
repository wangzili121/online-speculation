"""接受长度按总轮数汇总，配对区间不能掩盖题目变化。"""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("online_analysis",
    Path(__file__).parents[1] / "scripts/analyze_qwen38_pypto_online.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prefix_change_weights_rounds_and_pairs_by_question():
    before = [dict(id="a", rounds=1, accepted=1), dict(id="b", rounds=100, accepted=0)]
    after = [dict(id="b", rounds=100, accepted=0), dict(id="a", rounds=1, accepted=2)]
    result = module.paired_prefix_change(before, after, samples=200)
    assert result["delta"] == pytest.approx(1/101)
    assert result["ci95"] == [0, 1]
    assert result["improved_questions"] == 1


def test_uniform_improvement_has_exact_bootstrap_interval():
    before = [dict(id=str(i), rounds=4, accepted=4) for i in range(12)]
    after = [{**row, "accepted": 6} for row in before]
    result = module.paired_prefix_change(before, after, samples=200)
    assert result["delta"] == .5 and result["ci95"] == [.5, .5]


def test_duplicate_or_changed_question_sets_are_rejected():
    rows = [dict(id="a", rounds=1, accepted=1)]
    for other in (rows*2, [{**rows[0], "id": "b"}]):
        with pytest.raises(ValueError, match="paired"):
            module.paired_rows(rows, other)


def test_unfinished_runs_are_not_presented_as_final_results():
    with pytest.raises(ValueError, match="complete run"):
        module.analyze(dict(completed=False))


def test_stream_audit_checks_raw_outputs_and_all_update_ownership():
    report = dict(arguments={}, dataset=dict(splits=dict(train=[dict(id="train")])),
        train=[dict(id="train", tokens=[1], updates=1)], updates=[{}],
        online_net=dict(target_rows=[dict(id="train", tokens=[1])]))
    module.validate_stream(report)
    report["train"][0]["updates"] = 0
    with pytest.raises(ValueError, match="outside"):
        module.validate_stream(report)
    report["train"][0]["updates"] = 1
    report["train"][0]["tokens"] = [2]
    with pytest.raises(ValueError, match="paired target"):
        module.validate_stream(report)
    report["train"][0]["id"] = "dev"
    with pytest.raises(ValueError, match="partition"):
        module.validate_stream(report)


def test_feedback_coverage_distinguishes_token_count_from_prediction_horizon():
    # 100 块各监督前两位，只有 1 块看到了第三位；不是每个位置都监督了 201 次。
    assert module.supervision_coverage({2: 100, 3: 1}, 5) == [101, 101, 1, 0]
