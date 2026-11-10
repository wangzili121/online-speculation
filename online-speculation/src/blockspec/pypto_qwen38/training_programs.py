"""在线训练的显式梯度 kernel；模型值和优化器状态均在设备计算。"""


def make_tv(rows, vocabulary, *, inline=False):
    """完整词表 TV 及 d(logits)，只监督 verifier 真正走到的行。"""
    if rows < 1 or vocabulary % 64:
        raise ValueError("positive rows and a 64-aligned vocabulary required")
    import pypto.language as pl
    T = min(512, vocabulary)
    if vocabulary % T:
        raise ValueError("complete vocabulary tiles required")

    kernel = pl.jit.inline if inline else pl.jit
    @kernel
    def tv(student: pl.Tensor, teacher: pl.Tensor, gradient: pl.Out[pl.Tensor],
           losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        with pl.spmd(rows, name_hint="orthrus_tv_backward"):
            row = pl.tile.get_block_idx()
            if row < valid:
                sm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                tm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                for tile in pl.range(vocabulary // T):
                    s = pl.reshape(pl.cast(pl.slice(student, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    t = pl.reshape(pl.cast(pl.slice(teacher, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    sr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(s))
                    tr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(t))
                    sx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(tr, 0, 1)))
                    sm = pl.maximum(sm, sx)
                    tm = pl.maximum(tm, tx)
                sz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                tz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    s = pl.reshape(pl.cast(pl.slice(student, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    t = pl.reshape(pl.cast(pl.slice(teacher, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    se = pl.exp(pl.row_expand_sub(s, pl.row_max(sm)))
                    te = pl.exp(pl.row_expand_sub(t, pl.row_max(tm)))
                    sr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(se))
                    tr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(te))
                    sx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(tr, 0, 1)))
                    sz = pl.add(sz, sx)
                    tz = pl.add(tz, tx)
                dot = pl.full([8, 8], dtype=pl.FP32, value=0.)
                loss = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    s = pl.reshape(pl.cast(pl.slice(student, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    t = pl.reshape(pl.cast(pl.slice(teacher, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    p = pl.row_expand_div(pl.exp(pl.row_expand_sub(s, pl.row_max(sm))), pl.mul(pl.row_sum(sz), .125))
                    q = pl.row_expand_div(pl.exp(pl.row_expand_sub(t, pl.row_max(tm))), pl.mul(pl.row_sum(tz), .125))
                    diff = pl.sub(p, q)
                    sign = pl.mul(pl.sub(pl.cmp(diff, 0., cmp_type=4), pl.cmp(diff, 0., cmp_type=2)), .5)
                    weighted = pl.mul(p, sign)
                    absolute = pl.mul(pl.abs(diff), .5)
                    sr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(weighted))
                    tr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(absolute))
                    sx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(tr, 0, 1)))
                    dot = pl.add(dot, sx)
                    loss = pl.add(loss, tx)
                losses[row:row + 1, 0:8] = pl.div(pl.slice(loss, [1, 8], [0, 0]), pl.cast(normalizer, pl.FP32))
                for tile in pl.range(vocabulary // T):
                    s = pl.reshape(pl.cast(pl.slice(student, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    t = pl.reshape(pl.cast(pl.slice(teacher, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    p = pl.row_expand_div(pl.exp(pl.row_expand_sub(s, pl.row_max(sm))), pl.mul(pl.row_sum(sz), .125))
                    q = pl.row_expand_div(pl.exp(pl.row_expand_sub(t, pl.row_max(tm))), pl.mul(pl.row_sum(tz), .125))
                    diff = pl.sub(p, q)
                    sign = pl.mul(pl.sub(pl.cmp(diff, 0., cmp_type=4), pl.cmp(diff, 0., cmp_type=2)), .5)
                    ds = pl.div(pl.mul(p, pl.row_expand_sub(sign, pl.mul(pl.row_sum(dot), .125))), pl.cast(normalizer, pl.FP32))
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.cast(pl.reshape(ds, [1, T]), pl.BF16, mode="rint")
            else:
                losses[row:row + 1, 0:8] = pl.full([1, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.full([1, T], dtype=pl.BF16, value=0.)
        return gradient, losses

    @pl.jit.host
    def entry(student: pl.Tensor, teacher: pl.Tensor, gradient: pl.Out[pl.Tensor],
              losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        tv(student, teacher, gradient, losses, valid, normalizer, device=0)

    return tv if inline else entry


def make_kl(rows, vocabulary, *, inline=False):
    """完整词表 KL(teacher || draft) 及 d(logits)，用于干净序列的全块监督。"""
    if rows < 1 or vocabulary % 64:
        raise ValueError("positive rows and a 64-aligned vocabulary required")
    import pypto.language as pl
    T = min(512, vocabulary)
    if vocabulary % T:
        raise ValueError("complete vocabulary tiles required")

    kernel = pl.jit.inline if inline else pl.jit
    @kernel
    def kl(student: pl.Tensor, teacher: pl.Tensor, gradient: pl.Out[pl.Tensor],
           losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        with pl.spmd(rows, name_hint="orthrus_kl_backward"):
            row = pl.tile.get_block_idx()
            if row < valid:
                sm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                tm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                for tile in pl.range(vocabulary // T):
                    s = pl.reshape(pl.cast(pl.slice(student, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    t = pl.reshape(pl.cast(pl.slice(teacher, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    sr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(s))
                    tr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(t))
                    sx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(tr, 0, 1)))
                    sm = pl.maximum(sm, sx)
                    tm = pl.maximum(tm, tx)
                sz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                tz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    s = pl.reshape(pl.cast(pl.slice(student, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    t = pl.reshape(pl.cast(pl.slice(teacher, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
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
                    s = pl.reshape(pl.cast(pl.slice(student, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    t = pl.reshape(pl.cast(pl.slice(teacher, [1, T], [row, tile * T]), pl.FP32), [8, T // 8])
                    log_student = pl.row_expand_sub(pl.row_expand_sub(s, pl.row_max(sm)), lsz)
                    log_teacher = pl.row_expand_sub(pl.row_expand_sub(t, pl.row_max(tm)), ltz)
                    p = pl.exp(log_student)
                    q = pl.exp(log_teacher)
                    ds = pl.div(pl.sub(p, q), pl.cast(normalizer, pl.FP32))
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.cast(pl.reshape(ds, [1, T]), pl.BF16, mode="rint")
                    term = pl.mul(q, pl.sub(log_teacher, log_student))
                    partial = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(term))
                    total = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(partial, 0, 1)))
                    loss = pl.add(loss, total)
                losses[row:row + 1, 0:8] = pl.div(pl.slice(loss, [1, 8], [0, 0]), pl.cast(normalizer, pl.FP32))
            else:
                losses[row:row + 1, 0:8] = pl.full([1, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.full([1, T], dtype=pl.BF16, value=0.)
        return gradient, losses

    @pl.jit.host
    def entry(student: pl.Tensor, teacher: pl.Tensor, gradient: pl.Out[pl.Tensor],
              losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        kl(student, teacher, gradient, losses, valid, normalizer, device=0)

    return kl if inline else entry


def make_adam(shape, *, beta1=.9, beta2=.999, epsilon=1e-8, inline=False):
    """FP32 master/moments；写候选状态，调用方确认后才发布执行权重。"""
    from math import prod
    N = prod(shape)
    T = min(2048, N)
    if N % T:
        raise ValueError("complete T-element parameter tiles required")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    @kernel
    def adam(master: pl.Tensor, gradient: pl.Tensor, first: pl.Tensor, second: pl.Tensor,
             next_master: pl.Out[pl.Tensor], next_first: pl.Out[pl.Tensor], next_second: pl.Out[pl.Tensor],
             rate: pl.Scalar[pl.FP32], correction1: pl.Scalar[pl.FP32], correction2: pl.Scalar[pl.FP32],
             clip: pl.Scalar[pl.FP32]):
        p = pl.reshape(master, [N // T, T])
        g = pl.reshape(gradient, [N // T, T])
        m = pl.reshape(first, [N // T, T])
        v = pl.reshape(second, [N // T, T])
        op = pl.reshape(next_master, [N // T, T])
        om = pl.reshape(next_first, [N // T, T])
        ov = pl.reshape(next_second, [N // T, T])
        with pl.spmd((N // T + 15) // 16, name_hint="orthrus_adam_candidate"):
            base = pl.tile.get_block_idx() * 16
            for part in pl.range(16):
                row = base + part
                if row < N // T:
                    grad = pl.mul(pl.slice(g, [1, T], [row, 0]), clip)
                    mean = pl.add(pl.mul(pl.slice(m, [1, T], [row, 0]), beta1), pl.mul(grad, 1. - beta1))
                    variance = pl.add(pl.mul(pl.slice(v, [1, T], [row, 0]), beta2), pl.mul(pl.mul(grad, grad), 1. - beta2))
                    denominator = pl.add(pl.sqrt(pl.div(variance, correction2)), epsilon)
                    update = pl.div(pl.mul(mean, rate), pl.mul(denominator, correction1))
                    op[row:row + 1, 0:T] = pl.sub(pl.slice(p, [1, T], [row, 0]), update)
                    om[row:row + 1, 0:T] = mean
                    ov[row:row + 1, 0:T] = variance
        return next_master, next_first, next_second

    @pl.jit.host
    def entry(master: pl.Tensor, gradient: pl.Tensor, first: pl.Tensor, second: pl.Tensor,
              next_master: pl.Out[pl.Tensor], next_first: pl.Out[pl.Tensor], next_second: pl.Out[pl.Tensor],
              rate: pl.Scalar[pl.FP32], correction1: pl.Scalar[pl.FP32], correction2: pl.Scalar[pl.FP32],
              clip: pl.Scalar[pl.FP32]):
        adam(master, gradient, first, second, next_master, next_first, next_second,
             rate, correction1, correction2, clip, device=0)

    return adam if inline else entry


def reduction_tile(k, limit):
    """归约 tile 必须整除 K；形状不齐整时退到较小的硬件对齐 tile。"""
    if type(limit) is not int or limit not in (32, 64, 128, 256, 512):
        raise ValueError("reduction tile must be 32, 64, 128, 256 or 512")
    return max(t for t in (32, 64, 128, 256, 512) if t <= limit and k % t == 0)


def make_linear(a_shape, b_shape, *, transpose_a=False, transpose_b=True, gradient_weight=False,
                batches=1, b_repeats=1, inline=False, accumulate=False, reduction_limit=None):
    """前向与显式线性 VJP 共用矩阵乘；激活/参数梯度按 BF16 执行语义舍入。"""
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit
    M, K = a_shape[::-1] if transpose_a else a_shape
    kb, N = b_shape[::-1] if transpose_b else b_shape
    if K != kb or M % 32 or N % 64 or K % 32:
        raise ValueError("aligned M/N/K required")
    if batches < 1 or b_repeats < 1 or batches % b_repeats:
        raise ValueError("whole GQA groups required")
    # 真实词表的 dLogits 行步长很大；先重排，避免长行二维读取错误。
    if (not gradient_weight and not transpose_a and not transpose_b and batches==1
            and K>=65536 and K%512==0):
        from .training_long_linear import make_long_linear
        return make_long_linear(M,K,N,inline=inline,
                                reduction_limit=512 if reduction_limit is None else reduction_limit)
    MT = 64 if M % 64 == 0 else 32
    NT = 64
    KT = reduction_tile(K, 64 if reduction_limit is None else reduction_limit)
    AR, AC = (KT, MT) if transpose_a else (MT, KT)
    BR, BC = (NT, KT) if transpose_b else (KT, NT)
    TA = int(transpose_a)
    TB = int(transpose_b)
    if gradient_weight:
        tasks = batches * (M // MT) * (N // NT)
        @kernel
        def linear(a: pl.Tensor, b: pl.Tensor, output: pl.InOut[pl.Tensor]):
            # 多个输出 tile 复用同一组 Cube→Vector 缓冲，避免临时量随整张 dW 矩阵放大。
            with pl.spmd((tasks + 15) // 16, name_hint="orthrus_training_linear"):
                base = pl.tile.get_block_idx() * 16
                for part in pl.range(16):
                    task = base + part
                    if task < tasks:
                        head = task // ((M // MT) * (N // NT))
                        local = task % ((M // MT) * (N // NT))
                        row = local // (N // NT) * MT
                        col = local % (N // NT) * NT
                        if K == KT:
                            # dW 沿 K32 token 维归约，一次 Cube 即可。
                            x = pl.slice(a, [AR, AC], [head * a_shape[0] + (1 - TA) * row, TA * row])
                            w = pl.slice(b, [BR, BC], [head // b_repeats * b_shape[0] + TB * col, (1 - TB) * col])
                            acc = pl.matmul(x, w, a_trans=transpose_a, b_trans=transpose_b, out_dtype=pl.FP32)
                        else:
                            acc = pl.create_tensor([MT, NT], dtype=pl.FP32)
                            for step in pl.pipeline(K // KT, stage=2):
                                begin = step * KT
                                x = pl.slice(a, [AR, AC], [head * a_shape[0] + TA * begin + (1 - TA) * row,
                                                           TA * row + (1 - TA) * begin])
                                w = pl.slice(b, [BR, BC], [head // b_repeats * b_shape[0] + TB * col + (1 - TB) * begin,
                                                           TB * begin + (1 - TB) * col])
                                acc = pl.matmul_acc(acc, x, w, a_trans=transpose_a, b_trans=transpose_b, init_cond=(step == 0))
                        rounded = pl.cast(acc, pl.BF16, mode="rint")
                        result = pl.cast(rounded, pl.FP32)
                        if accumulate:
                            result = pl.add(result, pl.slice(output, [MT, NT], [head * M + row, col]))
                        output[head * M + row:head * M + row + MT, col:col + NT] = result
            return output

    else:
        @kernel
        def linear(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
            with pl.spmd(batches * (M // MT) * (N // NT), name_hint="orthrus_training_linear"):
                task = pl.tile.get_block_idx()
                head = task // ((M // MT) * (N // NT))
                task = task % ((M // MT) * (N // NT))
                row = task // (N // NT) * MT
                col = task % (N // NT) * NT
                acc = pl.create_tensor([MT, NT], dtype=pl.FP32)
                for step in pl.pipeline(K // KT, stage=2):
                    begin = step * KT
                    x = pl.slice(a, [AR, AC], [head * a_shape[0] + TA * begin + (1 - TA) * row,
                                               TA * row + (1 - TA) * begin])
                    w = pl.slice(b, [BR, BC], [head // b_repeats * b_shape[0] + TB * col + (1 - TB) * begin,
                                               TB * begin + (1 - TB) * col])
                    acc = pl.matmul_acc(acc, x, w, a_trans=transpose_a, b_trans=transpose_b, init_cond=(step == 0))
                rounded = pl.cast(acc, pl.BF16, mode="rint")
                output[head * M + row:head * M + row + MT, col:col + NT] = rounded
            return output

    @pl.jit.host
    def entry(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        linear(a, b, output, device=0)

    if inline:
        return linear
    return entry


def make_element(shape, kind, *, scale=1., inline=False):
    """BF16 激活的逐元素前向/VJP，保持运算边界的舍入。"""
    names = ("add", "mul", "silu", "sigmoid", "silu_grad", "sigmoid_grad", "scale", "copy")
    op = names.index(kind)
    from math import prod
    N = prod(shape)
    if N % 128:
        raise ValueError("aligned elementwise tensor required")
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def element(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        x = pl.reshape(a, [N // 128, 128])
        y = pl.reshape(b, [N // 128, 128])
        out = pl.reshape(output, [N // 128, 128])
        with pl.spmd(N // 128, name_hint="orthrus_training_element"):
            row = pl.tile.get_block_idx()
            u = pl.cast(pl.slice(x, [1, 128], [row, 0]), pl.FP32)
            v = pl.cast(pl.slice(y, [1, 128], [row, 0]), pl.FP32)
            if op == 0:
                value = pl.add(u, v)
            elif op == 1:
                value = pl.mul(u, v)
            elif op == 2:
                value = pl.div(u, pl.add(pl.exp(pl.neg(u)), 1.))
            elif op == 3:
                value = pl.recip(pl.add(pl.exp(pl.neg(u)), 1.))
            elif op == 4:
                s = pl.recip(pl.add(pl.exp(pl.neg(u)), 1.))
                value = pl.mul(v, pl.mul(s, pl.add(pl.mul(u, pl.add(pl.neg(s), 1.)), 1.)))
            elif op == 5:
                s = pl.recip(pl.add(pl.exp(pl.neg(u)), 1.))
                value = pl.mul(v, pl.mul(s, pl.add(pl.neg(s), 1.)))
            elif op == 6:
                value = pl.mul(u, scale)
            else:
                value = u
            out[row:row + 1, 0:128] = pl.cast(value, pl.BF16, mode="rint")
        return output

    @pl.jit.host
    def entry(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        element(a, b, output, device=0)

    if inline:
        return element
    return entry


def make_rms(rows, dimension, *, mode="forward", inline=False, accumulate=False):
    """RMSNorm 前向、输入梯度或共享 Q/K norm 参数梯度。gamma 已包含 1。"""
    if dimension % 64 or rows % 32 or mode not in {"forward", "input", "weight"}:
        raise ValueError("aligned RMSNorm shape and a supported mode required")
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit
    D = dimension
    INV_D = 1. / D

    @kernel
    def forward(x: pl.Tensor, gamma: pl.Tensor, y: pl.Out[pl.Tensor], inv: pl.Out[pl.Tensor]):
        with pl.spmd(rows, name_hint="orthrus_rms_saved_forward"):
            row = pl.tile.get_block_idx()
            z = pl.reshape(pl.cast(pl.slice(x, [1, D], [row, 0]), pl.FP32), [8, D // 8])
            partial = pl.row_sum(pl.mul(z, z))
            repeated = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), partial)
            total = pl.row_sum(pl.transpose(repeated, 0, 1))
            r = pl.rsqrt(pl.add(pl.mul(total, 1. / D), 1e-6), high_precision=True)
            g = pl.reshape(pl.slice(gamma, [1, D], [0, 0]), [8, D // 8])
            normalized = pl.reshape(pl.mul(pl.row_expand_mul(z, r), g), [1, D])
            y[row:row + 1, 0:D] = pl.cast(normalized, pl.BF16, mode="rint")
            inv[row:row + 1, 0:8] = pl.reshape(r, [1, 8])
        return y, inv

    @kernel
    def input_grad(x: pl.Tensor, gamma: pl.Tensor, inv: pl.Tensor, dy: pl.Tensor, dx: pl.Out[pl.Tensor]):
        with pl.spmd(rows, name_hint="orthrus_rms_input_backward"):
            row = pl.tile.get_block_idx()
            z = pl.reshape(pl.cast(pl.slice(x, [1, D], [row, 0]), pl.FP32), [8, D // 8])
            g = pl.reshape(pl.slice(gamma, [1, D], [0, 0]), [8, D // 8])
            delta = pl.mul(pl.reshape(pl.cast(pl.slice(dy, [1, D], [row, 0]), pl.FP32), [8, D // 8]), g)
            r = pl.read(inv, [row, 0])
            partial = pl.row_sum(pl.mul(z, delta))
            repeated = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), partial)
            dot = pl.row_sum(pl.transpose(repeated, 0, 1))
            correction = pl.mul(dot, r * r * r * INV_D)
            grad = pl.sub(pl.mul(delta, r), pl.row_expand_mul(z, correction))
            dx[row:row + 1, 0:D] = pl.cast(pl.reshape(grad, [1, D]), pl.BF16, mode="rint")
        return dx

    @kernel
    def weight_grad(x: pl.Tensor, inv: pl.Tensor, dy: pl.Tensor, dg: pl.InOut[pl.Tensor]):
        with pl.spmd(D // 64, name_hint="orthrus_rms_weight_backward"):
            col = pl.tile.get_block_idx() * 64
            acc = pl.full([1, 64], dtype=pl.FP32, value=0.)
            for block in pl.range(rows // 32):
                row = block * 32
                z = pl.cast(pl.slice(x, [32, 64], [row, col]), pl.FP32)
                d = pl.cast(pl.slice(dy, [32, 64], [row, col]), pl.FP32)
                r = pl.mul(pl.row_sum(pl.slice(inv, [32, 8], [row, 0])), .125)
                acc = pl.add(acc, pl.col_sum(pl.row_expand_mul(pl.mul(z, d), r)))
            # 训练 raw norm 是 BF16；转回 master FP32 前保留该梯度舍入。
            result = pl.cast(pl.cast(acc, pl.BF16, mode="rint"), pl.FP32)
            if accumulate:
                result = pl.add(result, pl.slice(dg, [1, 64], [0, col]))
            dg[0:1, col:col + 64] = result
        return dg

    if mode == "forward":
        @pl.jit.host
        def entry(x: pl.Tensor, gamma: pl.Tensor, y: pl.Out[pl.Tensor], inv: pl.Out[pl.Tensor]):
            forward(x, gamma, y, inv, device=0)
    elif mode == "input":
        @pl.jit.host
        def entry(x: pl.Tensor, gamma: pl.Tensor, inv: pl.Tensor, dy: pl.Tensor, dx: pl.Out[pl.Tensor]):
            input_grad(x, gamma, inv, dy, dx, device=0)
    else:
        @pl.jit.host
        def entry(x: pl.Tensor, inv: pl.Tensor, dy: pl.Tensor, dg: pl.Out[pl.Tensor]):
            weight_grad(x, inv, dy, dg, device=0)
    if inline:
        return {"forward": forward, "input": input_grad, "weight": weight_grad}[mode]
    return entry
