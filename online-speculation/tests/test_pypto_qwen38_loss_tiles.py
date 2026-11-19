"""损失分块必须完整覆盖词表，不截断尾部，也不改变数学目标。"""

import pytest

from blockspec.pypto_qwen38.training_programs import loss_vocabulary_tile


@pytest.mark.parametrize("vocabulary,expected", [(64, 64), (128, 128), (512, 512),
    (1024, 512), (2560, 2560), (248320, 2560), (151552, 512)])
def test_loss_tiles_cover_vocabulary(vocabulary, expected):
    tile = loss_vocabulary_tile(vocabulary)
    assert tile == expected
    assert vocabulary % tile == 0
    assert tile % 64 == 0


@pytest.mark.parametrize("vocabulary", [0, -64, 63, 768, 64.0, True])
def test_loss_tiles_reject_invalid_or_incomplete_shapes(vocabulary):
    with pytest.raises(ValueError):
        loss_vocabulary_tile(vocabulary)
