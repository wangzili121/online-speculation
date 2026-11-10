"""小窗口复测不能继续使用旧确认集来选参数。"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

scripts=Path(__file__).parents[1]/"scripts"
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location("width_refinement",scripts/"benchmark_qwen38_pypto_verifier.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def inputs():
    def row(key):return dict(id=key,question_sha256=key,prompt_ids=[1])
    old=dict(files_sha256=dict(test="sha"),splits=dict(dev=[row("dev")],train=[row("train")],
             calibration=[row("cal")],test=[row("test")],confirm=[row("old-confirm")]))
    candidate=dict(files_sha256=dict(test="sha"),splits=dict(dev=deepcopy(old["splits"]["dev"]),
                   test=[row("test"),row("old-confirm"),row("new1"),row("new2")]))
    return dict(dataset=old),candidate


def test_refinement_reserves_the_next_confirmation_questions():
    old,new=inputs()
    assert [row["id"] for row in module.refinement_questions(old,new,2)]==["new1","new2"]


@pytest.mark.parametrize("problem",("data","prompt","old_test","duplicate","leak"))
def test_refinement_rejects_changed_or_reused_questions(problem):
    old,new=inputs()
    if problem=="data":new["files_sha256"]["test"]="changed"
    if problem=="prompt":new["splits"]["dev"][0]["prompt_ids"]=[9]
    if problem=="old_test":new["splits"]["test"][0]["id"]="changed"
    if problem=="duplicate":new["splits"]["test"][-1]["question_sha256"]="new1"
    if problem=="leak":new["splits"]["test"][-1]["question_sha256"]="train"
    with pytest.raises(ValueError):module.refinement_questions(old,new,2)
