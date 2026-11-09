"""将原始训练分片在 NPU 上转换为推理 bank，不在 CPU 上计算权重值。"""

from dataclasses import dataclass
import ctypes
from math import prod
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open

from .loading import DeviceLoader, LoadingPrograms, parameter_pieces


def bank_specs(layer, tp=1):
    from .head import tensor_schema
    schema = tensor_schema(layer, tp)
    bf, fp = torch.bfloat16, torch.float32
    if layer % 4 == 3:
        return [(name, schema[name], bf, "copy") for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight")] + [
            (name, (1, schema[name][0]), fp, "gamma") for name in ("q_norm.weight", "k_norm.weight")] + [
            ("o_proj.weight", schema["o_proj.weight"], bf, "copy")]
    return [("in_proj_qkv.weight", schema["in_proj_qkv.weight"], bf, "copy"),
            ("in_proj_z.weight", schema["in_proj_z.weight"], bf, "copy"),
            *[(name, (128, schema[name][1]), bf, "copy") for name in ("in_proj_a.weight", "in_proj_b.weight")],
            ("conv1d.weight", (4, schema["conv1d.weight"][0]), bf, "conv"),
            ("A_log", (1, 64), fp, "constant"), ("dt_bias", (1, 64), bf, "constant"),
            ("norm.weight", (1, schema["norm.weight"][0]), bf, "copy"),
            ("out_proj.weight", schema["out_proj.weight"], bf, "copy")]


def host_layout(head):
    specs = [bank_specs(layer, head.layout.size) for layer in range(64)]
    cursor, offsets = 0, []
    for layer in specs:
        row = []
        for _, shape, dtype, _ in layer:
            cursor = (cursor + 63) // 64 * 64
            size = prod(shape) * (4 if dtype == torch.float32 else 2)
            row.append((cursor, size))
            cursor += size
        offsets.append(row)
    return specs, offsets, cursor


def allocate_host_banks(head):
    specs, offsets, cursor = host_layout(head)
    if Path("/dev/shm").is_dir() and shutil.disk_usage("/dev/shm").free < cursor + (64 << 20):
        raise RuntimeError("insufficient shared memory for prepared head")
    storage = torch.empty(cursor, dtype=torch.uint8).share_memory_()
    end = 0
    for row in offsets:
        for offset, size in row:
            if offset > end:
                ctypes.memset(storage.data_ptr() + end, 0, offset - end)
            end = offset + size
    banks = tuple(tuple(storage[offset:offset + size].view(dtype).reshape(shape)
                        for (_, shape, dtype, _), (offset, size) in zip(layer, row, strict=True))
                  for layer, row in zip(specs, offsets, strict=True))
    return storage, banks


def raw_location(head, layer, name, rank):
    if head.directory is not None:
        return head.directory / f"rank{rank}.safetensors", f"layers.{layer}.attention.draft.{name}"
    kind = "self_attn" if layer % 4 == 3 else "linear_attn"
    key = f"model.language_model.layers.{layer}.{kind}.{name}"
    filename = Path(head.index[key])
    if filename.is_absolute() or len(filename.parts) != 1:
        raise ValueError("checkpoint shard names must be local basenames")
    return head.model_root / filename, key


@dataclass
class Bank:
    name: str
    shape: tuple
    dtype: object
    operation: str
    source_dtype: object
    source_shape: tuple
    axis: int
    pieces: list
    keys: list


class HeadLoading:
    def __init__(self, head, config):
        from .head import tensor_schema
        self.head, self.registry, self.layers = head, LoadingPrograms(config), []
        source_tp = head.tp if head.directory is not None else 1
        self.source_tp = source_tp
        for layer in range(64):
            banks = []
            for name, shape, dtype, operation in bank_specs(layer, head.layout.size):
                filename, key = raw_location(head, layer, name, 0)
                with safe_open(filename, framework="pt", device="cpu") as file:
                    source_dtype = {"BF16": torch.bfloat16, "F32": torch.float32}.get(file.get_slice(key).get_dtype())
                if source_dtype not in ({torch.bfloat16, torch.float32} if name == "A_log" else {torch.bfloat16}):
                    raise ValueError(f"invalid checkpoint dtype for {name}")
                source = tensor_schema(layer, source_tp)[name]
                source_shape = (source[0], 4) if operation == "conv" else source if len(source) == 2 else (1, source[0])
                axis, pieces = parameter_pieces(layer, name, source_tp, head.layout)
                keys = []
                if operation == "constant":
                    self.registry.zero((1, 64), source_dtype)
                    keys = [self.registry.copy((1, 64), shape, source_dtype, dtype)]
                else:
                    for piece in pieces:
                        if name.endswith("norm.weight"):
                            size = source_shape
                        elif operation == "conv":
                            size = (4, piece.width)
                        else:
                            size = (source_shape[0], piece.width) if axis else (piece.width, source_shape[1])
                        keys.append(self.registry.copy(source_shape, shape, source_dtype, dtype, size=size,
                                                       conv=operation == "conv", gamma=operation == "gamma"))
                if name.endswith("norm.weight") and len(pieces) > 1:
                    self.registry.equal(prod(source_shape))
                self.registry.zero(shape, dtype)
                banks.append(Bank(name, shape, dtype, operation, source_dtype, source_shape, axis, pieces, keys))
            self.layers.append(banks)

    def read(self, layer, bank, rank):
        from .head import tensor_schema
        filename, key = raw_location(self.head, layer, bank.name, rank)
        with safe_open(filename, framework="pt", device="cpu") as file:
            value = file.get_tensor(key)
        if (not value.is_contiguous() or value.dtype != bank.source_dtype
                or tuple(value.shape) != tensor_schema(layer, self.source_tp)[bank.name]):
            raise ValueError(f"raw checkpoint layout changed at layer {layer}: {bank.name}")
        return value.reshape(bank.source_shape)

    def convert_bank(self, loader, layer, bank, host):
        worker = loader.worker
        target = loader.allocate(bank.shape, bank.dtype)
        loader.zero(target)
        sources, stage = {}, None
        try:
            if bank.operation == "constant":
                stage = loader.allocate((1, 64), bank.source_dtype)
                loader.zero(stage)
                size = 4 if bank.source_dtype == torch.float32 else 2
                for piece in bank.pieces:
                    value = self.read(layer, bank, piece.rank)
                    worker.copy_to(stage.data_ptr, value.data_ptr() + piece.source * size,
                                   piece.width * size, dst_offset=piece.destination * size)
                loader.copy(bank.keys[0], stage, target)
            else:
                for piece, key in zip(bank.pieces, bank.keys, strict=True):
                    if piece.rank not in sources:
                        value = self.read(layer, bank, piece.rank)
                        sources[piece.rank] = loader.allocate(bank.source_shape, bank.source_dtype, init=value)
                    source = sources[piece.rank]
                    if bank.name.endswith("norm.weight"):
                        if piece.rank != bank.pieces[0].rank:
                            loader.equal(sources[bank.pieces[0].rank], source)
                            continue
                        src, dst = (0, 0), (0, 0)
                    elif bank.operation == "conv":
                        src, dst = (piece.source, 0), (0, piece.destination)
                    else:
                        src = (0, piece.source) if bank.axis else (piece.source, 0)
                        dst = (0, piece.destination) if bank.axis else (piece.destination, 0)
                    loader.copy(key, source, target, source_offset=src, target_offset=dst)
            worker.copy_from(host.data_ptr(), target.data_ptr, host.numel() * host.element_size())
        except ValueError as error:
            raise ValueError(f"head layer {layer}, {bank.name}: {error}") from error
        finally:
            for source in reversed(list(sources.values())):
                loader.free(source)
            if stage is not None:
                loader.free(stage)
            loader.free(target)


def prepare_head(head, config):
    if head._prepared is not None:
        return
    from pypto.runtime import DistributedWorker
    start = time.monotonic()
    loading = HeadLoading(head, config)
    storage, banks = allocate_host_banks(head)
    worker = loader = None
    try:
        worker = DistributedWorker(list(loading.registry.programs.values()), config=config, inherited_host_tensors=[storage])
        loader = DeviceLoader(loading.registry, worker)
        for layer, row in enumerate(loading.layers):
            for bank, host in zip(row, banks[layer], strict=True):
                loading.convert_bank(loader, layer, bank, host)
            print("prepared head on NPU", layer, flush=True)
    finally:
        try:
            if loader is not None:
                loader.close()
        finally:
            if worker is not None:
                worker.close()
    head.shared_weights, head._prepared = storage, banks
    head.preparation_seconds = time.monotonic() - start
