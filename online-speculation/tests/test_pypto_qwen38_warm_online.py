"""预热与在线更新的数据和状态必须隔离。"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

scripts = Path(__file__).parents[1]/"scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("warm_online", scripts/"benchmark_qwen38_pypto_online.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    def row(key):
        return dict(id=key, question_sha256=key, prompt_ids=[1, 2])
    splits = dict(calibration=[row("c")], dev=[row("d")],
                  test=[row("t")], train=[row("a"), row("b"), row("e")])
    old = dict(completed=True, clean_shutdown=True, records=[dict(id="a")],
               dataset=dict(splits=splits, files_sha256=dict(train="x", test="y")))
    candidate = deepcopy(old["dataset"])
    candidate["splits"]["test"] += [row("f"), row("g")]
    return old, candidate


def test_warm_online_uses_new_train_questions_and_untouched_confirmation_set():
    old, candidate = inputs()
    result = module.warm_dataset(old, candidate, train_count=2, confirm_count=2)
    assert [row["id"] for row in result["splits"]["train"]] == ["b", "e"]
    assert [row["id"] for row in result["splits"]["confirm"]] == ["f", "g"]
    assert result["excluded_prewarm_record_ids"] == ["a"]
    for name in ("calibration", "dev", "test"):
        assert result["splits"][name] == old["dataset"]["splits"][name]
    assert candidate["splits"]["train"][0]["id"] == "a"


@pytest.mark.parametrize("problem", ("unclean", "data", "prompt", "test", "recycle", "overlap"))
def test_warm_online_rejects_changed_or_overlapping_sources(problem):
    old, candidate = inputs()
    if problem == "unclean": old["clean_shutdown"] = False
    if problem == "data": candidate["files_sha256"]["train"] = "bad"
    if problem == "prompt": candidate["splits"]["dev"][0]["prompt_ids"] = [9]
    if problem == "test": candidate["splits"]["test"][0]["prompt_ids"] = [9]
    if problem == "recycle": old["records"].append(dict(id="b"))
    if problem == "overlap": candidate["splits"]["test"][1]["question_sha256"] = "b"
    with pytest.raises(ValueError):
        module.warm_dataset(old, candidate, train_count=2, confirm_count=2)


@pytest.mark.parametrize("problem", (None, "master", "moment", "counter", "publication"))
def test_initialization_audit_reads_actual_device_states(tmp_path, monkeypatch, problem):
    values = dict(master=torch.arange(8, dtype=torch.float32), first=torch.ones(8), second=torch.ones(8))
    save_file(values, str(tmp_path/"optimizer.safetensors"))
    device = [values["master"].clone(), torch.zeros(8), torch.zeros(8)]
    if problem == "master": device[0][0] = 9
    if problem == "moment": device[1][0] = 9
    execution = torch.ones(8)
    served = execution.clone()
    if problem == "publication": served[0] = 9
    optimizer = NS(state=device, steps=int(problem == "counter"), execution=[(execution,)], index=0)
    learner = NS(optimizer=optimizer, trainer=object(), updates=0, draft=NS(weights=NS(version=0)),
                 options=NS(last_layers=4), serving_banks=[(served,)], publisher=NS(active=0),
                 native=NS(worker=object()))
    import diagnose_qwen38_pypto_training
    monkeypatch.setattr(diagnose_qwen38_pypto_training, "read_tensor", lambda worker, handle: handle)
    if problem:
        with pytest.raises(AssertionError): module.audit_warm_initialization(learner, tmp_path)
    else:
        result = module.audit_warm_initialization(learner, tmp_path)
        assert result["master_equal"] and result["adam_moments_zero"] and result["serving_equal"]
        assert result["counters"] == [0, 0, 0]


def test_width_selection_uses_dev_throughput_not_acceptance_count():
    rows = [dict(width=32, split="dev", summary=dict(greedy_equal_all=True, tokens_per_second=4,
                                                   accepted_per_round=2)),
            dict(width=4, split="dev", summary=dict(greedy_equal_all=True, tokens_per_second=6,
                                                  accepted_per_round=1))]
    assert module.choose_width(rows) == 4
    rows[1]["summary"]["greedy_equal_all"] = False
    with pytest.raises(ValueError, match="lossless"): module.choose_width(rows)
    rows[1]["summary"]["greedy_equal_all"] = True
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="development"): module.choose_width(rows)


def test_continuation_keeps_warm_exclusions_and_grows_the_fresh_stream():
    old, candidate = inputs()
    dataset = module.warm_dataset(old, candidate, train_count=2, confirm_count=2)
    candidate["splits"]["train"].append(dict(id="h", question_sha256="h", prompt_ids=[1]))
    result = module.continued_dataset(dict(dataset=dataset), candidate, train_count=3)
    assert [row["id"] for row in result["splits"]["train"]] == ["b", "e", "h"]
    assert result["splits"]["train"][:2] == dataset["splits"]["train"]
    assert result["excluded_prewarm_record_ids"] == ["a"]
    assert result["splits"]["confirm"] == dataset["splits"]["confirm"]


@pytest.mark.parametrize("problem", ("data", "exclusion", "prompt", "overlap"))
def test_continuation_refuses_changed_data_or_reintroduced_prewarm_questions(problem):
    old, candidate = inputs()
    dataset = deepcopy(module.warm_dataset(old, candidate, train_count=2, confirm_count=2))
    if problem == "data": candidate["files_sha256"]["test"] = "changed"
    if problem == "exclusion": candidate["splits"]["train"].pop(0)
    if problem == "prompt": candidate["splits"]["calibration"][0]["prompt_ids"] = [99]
    if problem == "overlap": candidate["splits"]["train"][1]["question_sha256"] = "f"
    with pytest.raises(ValueError):
        module.continued_dataset(dict(dataset=dataset), candidate, train_count=2)
