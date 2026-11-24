"""Parallel active-row validation for the wide Orthrus target and draft phases."""

STATUS_WORDS = 16


def make_speculative_finite_check(capacity):
    import pypto.language as pl

    width, tile, lanes = 4 * 5120, 2048, STATUS_WORDS
    finite_max, expected_count = 3.4028234663852886e38, float(width)

    @pl.jit
    def finite_check(
        hidden_state: pl.Tensor[[capacity, 4, 5120], pl.BF16],
        pre_mix: pl.Tensor[[capacity, 4], pl.FP32],
        status: pl.Out[pl.Tensor[[capacity, lanes], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        flattened = pl.reshape(hidden_state, [capacity, width])
        for token in pl.spmd(active, name_hint="orthrus_dsv41_finite_rows"):
            finite_count = pl.cast(0.0, pl.FP32)
            for column in pl.range(0, width, tile):
                values = pl.cast(pl.slice(flattened, [1, tile], [token, column]), pl.FP32)
                finite = pl.cmp(pl.abs(pl.reshape(values, [8, tile // 8])), finite_max, cmp_type=3)
                counts = pl.row_sum(finite)
                for group in pl.unroll(8):
                    finite_count = finite_count + pl.read(counts, [group, 0])
            valid = pl.cast(1, pl.INT32)
            if finite_count != expected_count:
                valid = pl.cast(0, pl.INT32)
            for lane in pl.unroll(4):
                value = pl.read(pre_mix, [token, lane])
                if not (value <= finite_max and value >= -finite_max):
                    valid = pl.cast(0, pl.INT32)
            # Separate 64-byte rows prevent simultaneous writers sharing a cache line.
            pl.write(status, [token, 0], valid)
        return status

    return finite_check
