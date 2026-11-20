"""独立 GDN 参考的可解析边界；不依赖设备或主模型。"""

import importlib.util
from pathlib import Path

import pytest
import torch

from blockspec.pypto_qwen38.tp2_gdn_rows import make_recurrence


@pytest.mark.parametrize('kwargs', [dict(batch_size=0), dict(batch_size=3), dict(batch_size=True),
    dict(batch_size=2, history_slots=1), dict(batch_size=1, valid_rows='yes')])
def test_bad_request_shapes_are_rejected_before_importing_compiler(kwargs):
    with pytest.raises(ValueError): make_recurrence(**kwargs)


@pytest.mark.parametrize('widths', [(8,), (8, 6)])
@pytest.mark.parametrize('replace', [False, True])
def test_golden_matches_analytic_overwrite_or_history_decay(widths, replace):
    path = Path(__file__).resolve().parents[1] / 'scripts/validate_qwen38_pypto_gdn_rows.py'
    spec = importlib.util.spec_from_file_location('gdn_rows_reference', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    n = len(widths) * 32
    q = torch.zeros(n, 24, 128); q[:, :, 0] = 1
    k = q.clone()
    v = torch.full((n, 5120), 999., dtype=torch.bfloat16)
    for row in range(n): v[row, 2048:] = row / 8
    history = torch.full((2, 1, 24, 128, 128), .125).expand(2, 48, 24, 128, 128)
    decay = torch.full((n, 32), 0. if replace else -0.6931471805599453)
    beta = torch.full((n, 32), 1. if replace else 0.)
    actual = module.golden(q.reshape(n, 3072), k.reshape(n, 3072), v, decay, beta, history, widths)
    for seq, width in enumerate(widths):
        for t in range(width):
            expected = ((seq * 32 + t) / 8 if replace else .125 * .5 ** (t + 1))
            torch.testing.assert_close(actual[seq * 32 + t].float(),
                torch.full((3072,), expected), atol=1e-6, rtol=0)
        assert torch.count_nonzero(actual[seq * 32 + width:(seq + 1) * 32]) == 0
    assert (history == .125).all()


def test_legacy_recurrence_experiments_retain_their_keyword_interface():
    from inspect import signature
    from blockspec.pypto_qwen38.tp2_gdn_recurrence import make_entry as old_entry
    from blockspec.pypto_qwen38.tp2_gdn_recurrence import make_recurrence as old_recurrence
    for factory in [old_entry, old_recurrence]:
        signature(factory).bind(2, prefetch=True, state_tile=128, precompute_decay=False)
        signature(factory).bind(1, prefetch=False, state_tile=64, precompute_decay=True)
