"""长词表行重排必须完整覆盖，不截断词表尾部。"""

import pytest
from blockspec.pypto_qwen38.training_long_linear import make_long_linear, plan


@pytest.mark.parametrize("k", (65536, 66048, 248320))
def test_chunks_cover_the_complete_vocabulary(k):
    chunk, parts = plan(k)
    assert chunk * parts == k and chunk % 512 == 0 and chunk <= 4096
    assert chunk < 65536


def test_real_qwen38_plan_has_no_padding_or_dropped_tokens():
    assert plan(248320) == (2560, 97)


def test_unsupported_row_buffer_is_rejected_before_runtime_import():
    with pytest.raises(ValueError, match="pinned K32"):
        make_long_linear(64, 248320, 5120)
