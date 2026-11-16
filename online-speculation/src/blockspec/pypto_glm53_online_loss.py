"""Device loss for GLM Orthrus online distillation."""

from math import gcd


def make_kl_program(rows, vocabulary):
    """Compute full-vocabulary KL(teacher || draft) and the draft logit gradient."""
    if rows < 1 or vocabulary < 64 or vocabulary % 64:
        raise ValueError("positive rows and a 64-aligned vocabulary required")
    import pypto.language as pl
    T = gcd(512, vocabulary)

    @pl.jit
    def kl(student: pl.Tensor, teacher: pl.Tensor, gradient: pl.Out[pl.Tensor],
           losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        with pl.spmd(rows, name_hint="orthrus_kl_backward"):
            row = pl.tile.get_block_idx()
            if row < valid:
                sm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                tm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                for tile in pl.range(vocabulary // T):
                    s = pl.reshape(pl.slice(student, [1, T], [row, tile * T]), [8, T // 8])
                    t = pl.reshape(pl.slice(teacher, [1, T], [row, tile * T]), [8, T // 8])
                    sr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(s))
                    tr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(t))
                    sx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(tr, 0, 1)))
                    sm = pl.maximum(sm, sx)
                    tm = pl.maximum(tm, tx)
                sz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                tz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    s = pl.reshape(pl.slice(student, [1, T], [row, tile * T]), [8, T // 8])
                    t = pl.reshape(pl.slice(teacher, [1, T], [row, tile * T]), [8, T // 8])
                    se = pl.exp(pl.row_expand_sub(s, pl.row_max(sm)))
                    te = pl.exp(pl.row_expand_sub(t, pl.row_max(tm)))
                    sr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(se))
                    tr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(te))
                    sx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(tr, 0, 1)))
                    sz = pl.add(sz, sx)
                    tz = pl.add(tz, tx)
                # log-softmax 由 logits 直接计算，避免概率下溢后再取 log。
                lsz = pl.log(pl.mul(pl.row_sum(sz), .125))
                ltz = pl.log(pl.mul(pl.row_sum(tz), .125))
                loss = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    s = pl.reshape(pl.slice(student, [1, T], [row, tile * T]), [8, T // 8])
                    t = pl.reshape(pl.slice(teacher, [1, T], [row, tile * T]), [8, T // 8])
                    log_student = pl.row_expand_sub(pl.row_expand_sub(s, pl.row_max(sm)), lsz)
                    log_teacher = pl.row_expand_sub(pl.row_expand_sub(t, pl.row_max(tm)), ltz)
                    p = pl.exp(log_student)
                    q = pl.exp(log_teacher)
                    ds = pl.div(pl.sub(p, q), pl.cast(normalizer, pl.FP32))
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.reshape(ds, [1, T])
                    term = pl.mul(q, pl.sub(log_teacher, log_student))
                    partial = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(term))
                    total = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(partial, 0, 1)))
                    loss = pl.add(loss, total)
                losses[row:row + 1, 0:8] = pl.div(pl.slice(loss, [1, 8], [0, 0]), pl.cast(normalizer, pl.FP32))
            else:
                losses[row:row + 1, 0:8] = pl.full([1, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.full([1, T], dtype=pl.FP32, value=0.)
        return gradient, losses

    @pl.jit.host
    def entry(student: pl.Tensor, teacher: pl.Tensor, gradient: pl.Out[pl.Tensor],
              losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        kl(student, teacher, gradient, losses, valid, normalizer, device=0)

    return entry
