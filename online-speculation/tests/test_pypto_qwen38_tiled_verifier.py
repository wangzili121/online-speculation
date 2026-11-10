"""分块早停与完整 verifier 的输出和提交状态等价。"""

from itertools import product

import pytest

from blockspec.pypto_qwen38.greedy import verify_tiled, verify_transaction


class Target:
    collector = None

    def __init__(self):
        self.inputs = [100]
        self.calls = []

    def snapshot(self): return tuple(self.inputs)
    def restore(self, state): self.inputs[:] = state

    def forward(self, tokens):
        self.inputs.extend(tokens)
        self.calls.append(tuple(tokens))
        return [value+1 for value in tokens]


def test_tiled_verifier_matches_full_transactions_at_all_rejections_and_boundaries():
    # 包含 tile 边界的前后、EOS、输出预算和全接受；比较提交的 AR 输入而非只看输出。
    for reject, budget, stop, rows in product(range(32), (1, 2, 16, 17, 31, 32, 40),
                                             (None, 1, 16, 17, 31, 32), (1, 8, 16, 32)):
        candidates = list(range(1, 32))
        if reject < 31: candidates[reject] = 300
        full, tiled = Target(), Target()
        stops = () if stop is None else (stop,)
        expected = verify_transaction(full, 0, candidates, budget=budget, stop_tokens=stops)
        actual = verify_tiled(tiled, 0, candidates, budget=budget, stop_tokens=stops, rows=rows)
        assert expected == actual, (reject, budget, stop, rows)
        assert full.inputs == tiled.inputs, (reject, budget, stop, rows)


def test_rejection_skips_unused_tiles_and_teacher_capture_retains_full_path():
    candidates = list(range(1, 32)); candidates[1] = 300
    target = Target()
    verify_tiled(target, 0, candidates, budget=32)
    assert [len(call) for call in target.calls] == [16, 2]
    target = Target(); target.collector = object()
    verify_tiled(target, 0, candidates, budget=32)
    assert [len(call) for call in target.calls] == [32, 2]


def test_failure_in_replay_restores_original_history():
    target = Target()
    forward = target.forward

    def fail(tokens):
        result = forward(tokens)
        if len(target.calls) == 2: raise RuntimeError("device replay failed")
        return result

    target.forward = fail
    candidates = list(range(1, 32)); candidates[1] = 300
    with pytest.raises(RuntimeError, match="device replay failed"):
        verify_tiled(target, 0, candidates, budget=32)
    assert target.inputs == [100]
