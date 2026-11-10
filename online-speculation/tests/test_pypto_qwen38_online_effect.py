"""在线效果报告必须隔离数据、配对请求，并拒绝把不同输出当作无损加速。"""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("pypto_online_effect", scripts / "benchmark_qwen38_pypto_online.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_questions_never_include_answers_and_all_partitions_are_disjoint(tmp_path):
    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            seen.append(messages[0]["content"])
            assert options == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
            return [1, 2, 3]

    for split in ("train", "test"):
        rows = [dict(question=f"{split} question {i}", answer="SECRET ANSWER") for i in range(20)]
        if split == "test":
            rows.append(dict(question="train question 0", answer="SECRET ANSWER"))
        (tmp_path / f"{split}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    first = module.prepare_questions(tmp_path, Tokenizer(), train_count=8, dev_count=3, test_count=4)
    second = module.prepare_questions(tmp_path, Tokenizer(), train_count=8, dev_count=3, test_count=4)
    assert first == second
    hashes = [row["question_sha256"] for rows in first["splits"].values() for row in rows]
    assert len(hashes) == len(set(hashes)) == 16
    assert all("SECRET" not in text for text in seen)


def test_insufficient_distinct_questions_fails_instead_of_recycling(tmp_path):
    (tmp_path / "train.jsonl").write_text(json.dumps(dict(question="same", answer="secret"))+"\n")
    tokenizer = type("Tokenizer", (), {"apply_chat_template": lambda *args, **kwargs: [1]})()
    with pytest.raises(ValueError, match="not enough disjoint"):
        module.prepare_questions(tmp_path, tokenizer, train_count=2, dev_count=1, test_count=1)


def test_speedup_uses_paired_total_time_and_requires_exact_outputs():
    target = [dict(id="a", tokens=[1, 2], output_tokens=2, seconds=1),
              dict(id="b", tokens=[3], output_tokens=1, seconds=2)]
    rows = [{**target[0], "seconds": .5, "rounds": 1, "accepted": 1, "proposed": 3},
            {**target[1], "seconds": .25, "rounds": 1, "accepted": 0, "proposed": 1}]
    result = module.summarize(rows, target=target)
    assert result["tokens_per_second"] == result["lossless_speedup"] == 4
    assert result["candidate_acceptance"] == .25
    assert result["accepted_per_round"] == .5
    rows[1]["tokens"] = [4]
    result = module.summarize(rows, target=target)
    assert result["lossless_speedup"] is None and not result["greedy_equal_all"]
    with pytest.raises(ValueError, match="paired questions"):
        module.summarize(rows[::-1], target=target)


@pytest.mark.parametrize("layers", (1, 4))
def test_initial_and_trained_evaluation_use_the_same_resident_suffix_without_an_adam_step(layers):
    calls = []

    class Publisher:
        version = 0

        def publish(self, layer, values, *, expected_version):
            calls.append((layer, values, expected_version))
            self.version += 1

        def publish_flat(self, layout, values, *, expected_version):
            calls.append((layout, values, expected_version))
            self.version += 1

    names = ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight")
    optimizer = SimpleNamespace(steps=0, tensors=lambda: dict(zip(names, range(6))), layout="last4", execution=[[1,2]])
    learner = SimpleNamespace(updates=0, optimizer=optimizer, publisher=Publisher(),
                              options=SimpleNamespace(last_layers=layers))
    module.publish_initial_suffix(learner)
    expected = (63, list(range(6)), 0) if layers == 1 else ("last4", [1,2], 0)
    assert calls == [expected] and learner.publisher.version == optimizer.steps == 0
    learner.updates = 1
    with pytest.raises(ValueError, match="fresh learner"):
        module.publish_initial_suffix(learner)


def test_continuation_keeps_heldout_and_hypers_and_moves_to_new_questions():
    import copy
    arguments = dict.fromkeys(("last_layers", "width", "stride", "replay_blocks", "learning_rate", "warmup_steps",
                              "tokens", "seed", "max_prompt", "dev_count", "test_count", "share_ar_prefix"), 1)
    dataset = dict(files_sha256={"train": "a", "test": "b"}, splits=dict(
        calibration=[{"id": "c"}], dev=[{"id": "d"}], test=[{"id": "t"}],
        train=[{"id": "0"}, {"id": "1"}, {"id": "2"}]))
    previous = dict(completed=True, updates=[{"step": 32}], arguments=arguments,
                    dataset=copy.deepcopy(dataset), train=[{"id": "0"}, {"id": "1"}])
    assert module.check_continuation(previous, dataset, arguments) == 2
    with pytest.raises(ValueError, match="preserve learning"):
        module.check_continuation(previous, dataset, {**arguments, "learning_rate": 2})
    changed = copy.deepcopy(dataset)
    changed["splits"]["dev"] = [{"id": "other"}]
    with pytest.raises(ValueError, match="held-out"):
        module.check_continuation(previous, changed, arguments)
    changed = copy.deepcopy(dataset)
    changed["splits"]["train"].reverse()
    with pytest.raises(ValueError, match="used training prefix"):
        module.check_continuation(previous, changed, arguments)


@pytest.mark.parametrize("fail", (False, True))
def test_frozen_stream_control_restores_trained_head_and_keeps_version(monkeypatch, fail):
    from blockspec.pypto_qwen38 import ar_prefix
    trained, frozen, owned = {63: ("trained",)}, {63: ("initial",)}, [object()]
    weights = SimpleNamespace(overrides=trained, version=7)
    draft = SimpleNamespace(learner=None, weights=weights, handles=[])
    learner = SimpleNamespace(updates=7, optimizer=SimpleNamespace(steps=7))
    engine = SimpleNamespace(runner=SimpleNamespace(draft=draft, native=object()))
    monkeypatch.setattr(module, "idle", lambda engine: None)
    monkeypatch.setattr(ar_prefix, "borrow_prefix", lambda native, head, stop: (frozen, owned) if stop == 64 else None)
    try:
        with module.initial_head_control(engine, learner, object()):
            assert weights.overrides is frozen and weights.version == learner.updates == 7
            if fail:
                raise ValueError("request failure")
    except ValueError:
        assert fail
    assert weights.overrides is trained and weights.version == learner.updates == 7
    assert draft.handles == owned


def test_frozen_stream_control_refuses_active_learning(monkeypatch):
    engine = SimpleNamespace(runner=SimpleNamespace(draft=SimpleNamespace(learner=object())))
    monkeypatch.setattr(module, "idle", lambda engine: None)
    with pytest.raises(RuntimeError, match="paused learning"):
        with module.initial_head_control(engine, object(), object()):
            pytest.fail("active learner must not enter the control")
