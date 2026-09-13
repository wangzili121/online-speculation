from pathlib import Path

import pytest

from experiments.shared.math_benchmark import MathJudge, MathProblem, load_math500, stratified_subset
from experiments.shared.artifacts import write_json_atomic


def test_subset_is_disjoint_reproducible_and_answer_independent():
    problems = [MathProblem(str(i), "question", str(i), str(i % 3), 3 + i % 2) for i in range(50)]
    options = dict(seed=17, minimum_level=3, development_count=8, test_count=24)
    selected = stratified_subset(problems, **options)
    assert selected == stratified_subset(list(reversed(problems)), **options)
    assert not {p.identifier for p in selected["development"]} & {p.identifier for p in selected["test"]}
    changed = [MathProblem(p.identifier, p.question, "different", p.subject, p.level) for p in problems]
    assert [p.identifier for p in selected["test"]] == [p.identifier for p in stratified_subset(changed, **options)["test"]]
    assert len({(p.subject, p.level) for p in selected["test"]}) == 6
    excluded = selected["test"][0].identifier
    filtered = stratified_subset(problems, excluded_ids=[excluded], **options)
    assert excluded not in {p.identifier for values in filtered.values() for p in values}


def test_loader_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    row = '{"unique_id":"x","problem":"x","answer":"1","subject":"a","level":3}\n'
    path.write_text(row + row)
    with pytest.raises(ValueError, match="duplicate"):
        load_math500(path)


def test_atomic_json_retains_previous_artifact_after_serialization_failure(tmp_path):
    path = tmp_path / "pool.json"
    write_json_atomic(path, {"completed": 2})
    previous = path.read_bytes()
    with pytest.raises(TypeError):
        write_json_atomic(path, {"bad": object()})
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_math_judge_handles_equivalence_missing_answers_and_cleanup():
    pytest.importorskip("math_verify")
    judge = MathJudge()
    try:
        assert judge.grade(r"Therefore $\boxed{\frac12}$.", "0.5")["correct"]
        assert not judge.grade(r"$\boxed{3}$", "4")["correct"]
        assert not judge.grade("unfinished thought", "4")["parseable"]
        assert judge.equivalent(r"\boxed{2/4}", r"\boxed{0.5}")
        assert judge.answer_key(r"\boxed{2}") == "2"
    finally:
        judge.close()
    assert judge._process is None
