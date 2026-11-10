"""采集在线蒸馏所需的私有历史/教师分布，不持有可回收的目标页引用。"""


def make_history(pages):
    import pypto.language as pl
    ROWS = 16 * 4 * pages * 16

    @pl.jit
    def history(key: pl.Tensor, value: pl.Tensor, table: pl.Tensor,
                out_key: pl.Out[pl.Tensor], out_value: pl.Out[pl.Tensor], past: pl.Scalar[pl.INT32]):
        k = pl.reshape(key, [ROWS, 256])
        v = pl.reshape(value, [ROWS, 256])
        with pl.spmd(4 * 512, name_hint="orthrus_online_private_history"):
            task = pl.tile.get_block_idx()
            head = task // 512
            row = task % 512
            if row < past:
                physical = pl.read(table, [row // 16])
                source = ((15 * 4 + head) * pages + physical) * 16 + row % 16
                out_key[task:task + 1, 0:256] = pl.slice(k, [1, 256], [source, 0])
                out_value[task:task + 1, 0:256] = pl.slice(v, [1, 256], [source, 0])
            else:
                out_key[task:task + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.)
                out_value[task:task + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.)
        return out_key, out_value

    @pl.jit.host
    def entry(key: pl.Tensor, value: pl.Tensor, table: pl.Tensor,
              out_key: pl.Out[pl.Tensor], out_value: pl.Out[pl.Tensor], past: pl.Scalar[pl.INT32]):
        history(key, value, table, out_key, out_value, past, device=0)
    return entry


def make_teacher():
    import pypto.language as pl

    @pl.jit
    def capture(source: pl.Tensor, output: pl.Out[pl.Tensor], source_row: pl.Scalar[pl.INT32],
                target_row: pl.Scalar[pl.INT32], count: pl.Scalar[pl.INT32]):
        x = pl.reshape(source, [128, 248320])
        with pl.spmd(16, name_hint="orthrus_online_teacher_rows"):
            row = pl.tile.get_block_idx()
            if row < count:
                for col in pl.range(485):
                    output[target_row + row:target_row + row + 1, col * 512:col * 512 + 512] = pl.slice(x, [1, 512], [source_row + row, col * 512])
        return output

    @pl.jit.host
    def entry(source: pl.Tensor, output: pl.Out[pl.Tensor], source_row: pl.Scalar[pl.INT32],
              target_row: pl.Scalar[pl.INT32], count: pl.Scalar[pl.INT32]):
        capture(source, output, source_row, target_row, count, device=0)
    return entry


def make_copy(rows, columns):
    import pypto.language as pl

    @pl.jit
    def copy(source: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.spmd(rows, name_hint="orthrus_online_rope_copy"):
            row = pl.tile.get_block_idx()
            output[row:row + 1, 0:columns] = pl.slice(source, [1, columns], [row, 0])
        return output

    @pl.jit.host
    def entry(source: pl.Tensor, output: pl.Out[pl.Tensor]):
        copy(source, output, device=0)
    return entry


def make_gdn_history(layer):
    """复制该层在 anchor 前的 AR 卷积和递推状态，反馈不持有可回收状态的引用。"""
    import pypto.language as pl
    index = layer-layer//4

    @pl.jit
    def capture(conv: pl.Tensor, recurrent: pl.Tensor, private_conv: pl.Out[pl.Tensor],
                private_recurrent: pl.Out[pl.Tensor], slot: pl.Scalar[pl.INT32]):
        c = pl.reshape(conv,[8*48*4,10240])
        r = pl.reshape(recurrent,[8*48*6144,128])
        with pl.spmd(4*80,name_hint="orthrus_online_private_gdn_conv"):
            task = pl.tile.get_block_idx()
            row = task//80
            col = task%80*128
            block = pl.slice(c,[1,128],[(slot*48+index)*4+row,col])
            private_conv[row:row+1,col:col+128] = block
        with pl.spmd(6144//16,name_hint="orthrus_online_private_gdn_state"):
            row = pl.tile.get_block_idx()*16
            block = pl.slice(r,[16,128],[(slot*48+index)*6144+row,0])
            private_recurrent[row:row+16,0:128] = block
        return private_conv,private_recurrent

    @pl.jit.host
    def entry(conv: pl.Tensor, recurrent: pl.Tensor, private_conv: pl.Out[pl.Tensor],
              private_recurrent: pl.Out[pl.Tensor], slot: pl.Scalar[pl.INT32]):
        capture(conv,recurrent,private_conv,private_recurrent,slot,device=0)
    return entry


def make_loss_sum():
    import pypto.language as pl

    @pl.jit
    def reduce(losses: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.spmd(1, name_hint="orthrus_online_loss_summary"):
            row = pl.tile.get_block_idx()
            output[row:row + 1, 0:8] = pl.col_sum(pl.slice(losses, [32, 8], [0, 0]))
        return output

    @pl.jit.host
    def entry(losses: pl.Tensor, output: pl.Out[pl.Tensor]):
        reduce(losses, output, device=0)
    return entry
