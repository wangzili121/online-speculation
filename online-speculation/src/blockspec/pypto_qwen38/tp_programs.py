"""TP component programs assembled from native Linear/residual and PyPTO Sum."""

from .baseline import native_module
from .tp import TPLayout


def validate_linear_tiles(input_width, output_width, *, k_tile=512, output_tile=128):
    if any(type(n) is not int or n < 1 for n in (input_width, output_width, k_tile, output_tile)):
        raise ValueError("positive native Linear dimensions and tile sizes required")
    if input_width % k_tile or output_width % output_tile:
        raise ValueError(f"native Linear needs K%{k_tile}=0 and N%{output_tile}=0; "
                         "pad weights/activations or use a tail-aware native implementation")


def make_row_projection(source, tp, input_width):
    import pypto.language as pl

    TPLayout(tp)
    model = native_module(source, "model")
    validate_linear_tiles(input_width, 5120, k_tile=model.linear_K_TILE,
                          output_tile=model.linear_OUTPUT_TILE)
    linear = model.linear_hidden_projection_inline

    @pl.jit
    def project(x: pl.Tensor, weight: pl.Tensor, result: pl.Out[pl.Tensor]):
        for half in pl.range(2):
            row = half * 16
            tile = pl.slice(x, [16, input_width], [row, 0])
            output = pl.create_tensor([16, 5120], dtype=pl.BF16)
            output = linear(tile, weight, output)
            result[row:row + 16, 0:5120] = output
        return result

    @pl.jit.host
    def project_all(x: pl.Tensor, weight: pl.Tensor, result: pl.Out[pl.Tensor]):
        for rank in pl.range(tp):
            project(x[rank], weight[rank], result[rank], device=rank)

    return project_all


def make_output_sum(tp, *, iterations=1):
    import pypto.language as pl
    import pypto.language.distributed as pld

    TPLayout(tp)
    if type(iterations) is not int or iterations < 1:
        raise ValueError("positive collective iteration count required")

    @pl.program
    class OutputSum:
        @pl.function(type=pl.FunctionType.InCore)
        def sum_rows(self, x: pl.Tensor[[32, 5120], pl.BF16],
                     y: pl.Out[pl.Tensor[[32, 5120], pl.BF16]],
                     data: pl.InOut[pld.DistributedTensor[[32, 5120], pl.FP32]],
                     signal: pl.InOut[pld.DistributedTensor[[tp, 1], pl.INT32]]) -> pl.Tensor[[32, 5120], pl.BF16]:
            # 当前 pinned collective 不接收 BF16；使用已有 cast 和 FP32 Sum。
            # InCore Sum 的信号协议支持在循环内重复使用同一 signal。
            for iteration in pl.range(iterations):
                for row in pl.range(32):
                    value = pl.cast(pl.load(x, [row, 0], [1, 5120]), pl.FP32)
                    data = pl.store(value, [row, 0], data)
                # 整个 K32 输出一次归约；builtin 在内部处理块大小和尾部。
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
                for row in pl.range(32):
                    total = pl.cast(pl.load(data, [row, 0], [1, 5120]), pl.BF16, mode="rint")
                    y = pl.store(total, [row, 0], y)
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def rank_sum(self, x: pl.Tensor[[32, 5120], pl.BF16],
                     y: pl.Out[pl.Tensor[[32, 5120], pl.BF16]],
                     data: pl.InOut[pld.DistributedTensor[[32, 5120], pl.FP32]],
                     signal: pl.InOut[pld.DistributedTensor[[tp, 1], pl.INT32]]) -> pl.Tensor[[32, 5120], pl.BF16]:
            return self.sum_rows(x, y, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def sum_all(self, x: pl.Tensor[[tp, 32, 5120], pl.BF16],
                    y: pl.Out[pl.Tensor[[tp, 32, 5120], pl.BF16]]) -> pl.Tensor[[tp, 32, 5120], pl.BF16]:
            data_buffer = pld.alloc_window_buffer([32, 5120], dtype=pl.FP32)
            signal_buffer = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
            for rank in pl.range(tp):
                data = pld.window(data_buffer, [32, 5120], dtype=pl.FP32)
                signal = pld.window(signal_buffer, [tp, 1], dtype=pl.INT32)
                self.rank_sum(x[rank], y[rank], data, signal, device=rank)
            return y

    return OutputSum


def make_residual(source, tp):
    import pypto.language as pl

    TPLayout(tp)
    residual = native_module(source, "model").residual_residual_add_inline

    @pl.jit
    def add(x: pl.Tensor, summed: pl.Tensor, result: pl.Out[pl.Tensor]):
        for half in pl.range(2):
            row = half * 16
            left = pl.slice(x, [16, 5120], [row, 0])
            right = pl.slice(summed, [16, 5120], [row, 0])
            output = pl.create_tensor([16, 5120], dtype=pl.BF16)
            output = residual(left, right, output)
            result[row:row + 16, 0:5120] = output
        return result

    @pl.jit.host
    def add_all(x: pl.Tensor, summed: pl.Tensor, result: pl.Out[pl.Tensor]):
        for rank in pl.range(tp):
            add(x[rank], summed[rank], result[rank], device=rank)

    return add_all
