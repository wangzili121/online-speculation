import pytest

from blockspec.pypto_dsv41_attention_vjp import C1AAttentionVJPGeometry, c1a_vjp_factories


def test_c1a_vjp_geometry_covers_swa_compressed_block_and_sink():
    shape = C1AAttentionVJPGeometry()
    assert shape.rows == 512
    assert shape.token_columns == 672
    assert shape.block_begin == 640
    assert shape.sink_column == 672
    assert shape.padded_columns == 704


@pytest.mark.parametrize("columns", [672, 673, 700])
def test_c1a_vjp_rejects_unaligned_or_short_softmax_capacity(columns):
    with pytest.raises(ValueError):
        C1AAttentionVJPGeometry(padded_columns=columns)


def test_c1a_vjp_includes_per_query_replay_and_shared_block_gradients():
    assert set(c1a_vjp_factories()) == {
        "joined_history", "probability_replay",
        "probability_vjp", "softmax_block_vjp", "sink_reduce", "score_block_extract",
        "probability_block_extract",
        "d_probability", "d_query", "d_key_block", "d_value_block",
    }
