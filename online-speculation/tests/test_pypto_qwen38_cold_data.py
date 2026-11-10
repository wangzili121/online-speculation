"""整块预热仅消费当前训练划分的 AR 回答，保留回答起点。"""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("cold_data", scripts / "benchmark_qwen38_pypto_cold.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    dataset = dict(files_sha256=dict(train="a", test="b"), splits=dict(
        train=[dict(id="train:1", prompt_ids=[1, 2, 3])], dev=[], test=[], calibration=[]))
    source = dict(completed=True, dataset=copy.deepcopy(dataset), online_net=dict(target_rows=[
        dict(id="train:1", tokens=[4, 5, 6], output_tokens=3, seconds=2.)]))
    return dataset, source


def test_records_keep_the_exact_prompt_answer_boundary():
    dataset, source = fixture()
    records, seconds = module.response_records(source, dataset)
    assert records == [dict(id="train:1", tokens=[1, 2, 3, 4, 5, 6], answer_start=3)]
    assert seconds == 2.


def test_old_training_questions_cannot_leak_into_a_new_dev_partition():
    dataset, source = fixture()
    dataset["splits"]["dev"] = dataset["splits"]["train"]
    dataset["splits"]["train"] = []
    with pytest.raises(ValueError, match="partition"):
        module.response_records(source, dataset)


def test_truncated_or_invalid_delivered_tokens_are_rejected():
    dataset, source = fixture()
    source["online_net"]["target_rows"][0]["output_tokens"] = 4
    with pytest.raises(ValueError, match="incomplete"):
        module.response_records(source, dataset)
    source["online_net"]["target_rows"][0].update(output_tokens=3, tokens=[4, 5, 248320])
    with pytest.raises(ValueError, match="token ID"):
        module.response_records(source, dataset)
