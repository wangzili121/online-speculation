"""全局统计合并只减少调用，不改变归约树、裁剪或溢出保护。"""

from types import SimpleNamespace

import pytest
import torch

from blockspec.pypto_qwen38.training_statistics import statistics_source
from blockspec.pypto_qwen38.online_suffix_optimizer import SuffixOptimizer


@pytest.mark.parametrize("elements,counts", [(64, [1]), (4096, [1]), (8192, [2, 1]),
    (4096 * 64, [64, 1]), (4096 * 65, [65, 2, 1]),
    (104_857_600, [25600, 400, 7, 1]), (454_946_816, [111071, 1736, 28, 1])])
def test_statistics_preserve_the_original_tree_and_buffer_order(elements, counts):
    text, actual = statistics_source((1, elements))
    assert actual == counts
    calls = []
    def stat(*args):
        calls.append(("statistics", args))
        return args[1:]
    def reduce(index):
        def run(*args):
            calls.append((index, args))
            return args[2:]
        return run
    def summarize(*args):
        calls.append(("summary", args))
        return args[-1]
    namespace = dict(pl=SimpleNamespace(Tensor=object, Out=list), statistics=stat, summarize=summarize)
    namespace.update({f"reduce_{i}": reduce(i) for i in range(1, len(counts))})
    exec(compile(text, "statistics_generated", "exec"), namespace)
    buffers = [f"{kind}{i}" for i in range(len(counts)) for kind in ("sums", "flags")]
    assert namespace["statistics_chip"]("value", *buffers, "output") == "output"
    assert calls[0] == ("statistics", ("value", "sums0", "flags0"))
    assert calls[-1] == ("summary", (f"sums{len(counts)-1}", f"flags{len(counts)-1}", "output"))
    assert [c[0] for c in calls] == ["statistics", *range(1, len(counts)), "summary"]
    for i in range(1, len(counts)):
        assert calls[i] == (i, (f"sums{i-1}", f"flags{i-1}", f"sums{i}", f"flags{i}"))


@pytest.mark.parametrize("shape", [(), (True,), (-1,), (0,), (1.5,), (4097,)])
def test_bad_shape_fails_before_importing_the_compiler(shape):
    with pytest.raises(ValueError):
        statistics_source(shape)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("summary,valid", [((2., .5, 1.), True), ((0., 1., 1.), True),
    ((float("inf"), 0., 0.), False), ((float("nan"), 1., 1.), False),
    ((2., float("nan"), 1.), False), ((2., .5, 0.), False)])
def test_one_submission_retains_all_nonfinite_guards(fused, summary, valid):
    opt = object.__new__(SuffixOptimizer)
    opt.programs = SimpleNamespace(fused_statistics="fused" if fused else None,
        stat="stat", reductions=["r0", "r1"], summary="summary")
    opt.stats = [["s0", "f0"], ["s1", "f1"], ["s2", "f2"]]
    opt.summary, opt.summary_host = SimpleNamespace(data_ptr=1), torch.empty((3, 8))
    calls = []
    opt.run = lambda program, *args: calls.append((program, args))
    def copy_from(*args):
        assert args[-1] == 96
        opt.summary_host.copy_(torch.tensor(summary)[:, None])
    opt.worker = SimpleNamespace(copy_from=copy_from)
    if valid:
        assert opt.statistics("value") == summary[:2]
    else:
        with pytest.raises(FloatingPointError, match="serving head remains unchanged"):
            opt.statistics("value")
    if fused:
        assert len(calls) == 1
        assert calls[0][0] == "fused"
        assert calls[0][1] == ("value", "s0", "f0", "s1", "f1", "s2", "f2", opt.summary)
    else:
        assert [call[0] for call in calls] == ["stat", "r0", "r1", "summary"]
