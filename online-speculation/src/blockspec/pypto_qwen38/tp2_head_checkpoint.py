"""将已发布的 TP2 bank 导出为 BF16 head，不使用未发布的 Adam 参数。"""

import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from .oneserving_tp2 import LAYERS, SHAPES, checkpoint_manifest, digest, rank_weights


def restore_tensor_layout(tensors, banks):
    """将运行时 bank 还原为训练交换格式，保留未更新层的原始参数。"""
    def key(layer, name):
        return f"layers.{layer}.attention.draft.{name}"

    for index, layer in enumerate(LAYERS[:-1]):
        if banks[0] is None:
            break
        # GDN 的同类投影按层堆叠，a/b 与 decay 在运行时带有零 padding。
        for bank, name in ((0, "in_proj_qkv.weight"), (1, "in_proj_z.weight"),
                           (2, "out_proj.weight"), (3, "in_proj_a.weight"),
                           (4, "in_proj_b.weight")):
            original = tensors[key(layer, name)]
            width = banks[bank].shape[0] // len(LAYERS[:-1])
            tensors[key(layer, name)] = banks[bank][index * width:index * width + original.shape[0]].reshape_as(original).clone()
        original = tensors[key(layer, "conv1d.weight")]
        width = original.shape[-1]
        tensors[key(layer, "conv1d.weight")] = banks[5][index * width:(index + 1) * width].T.reshape_as(original).contiguous()
        for bank, name in ((6, "A_log"), (7, "dt_bias"), (8, "norm.weight")):
            original = tensors[key(layer, name)]
            tensors[key(layer, name)] = banks[bank][index, :original.numel()].reshape_as(original).clone()

    layer = LAYERS[-1]
    for bank, name in ((9, "q_proj.weight"), (10, "k_proj.weight"),
                       (11, "v_proj.weight"), (14, "o_proj.weight")):
        tensors[key(layer, name)] = banks[bank].reshape_as(tensors[key(layer, name)]).clone()
    for bank, name in ((12, "q_norm.weight"), (13, "k_norm.weight")):
        original = tensors[key(layer, name)]
        # render 的 gamma = BF16(raw) + 1。去掉偏置后恢复交换格式，
        # 导出时还会重建全部 bank，逐元素核对与当前发布值一致。
        tensors[key(layer, name)] = (banks[bank].reshape_as(original) - 1).to(original.dtype)
    return tensors


def export_served_head(trainer, directory):
    """在无在途请求和更新的阶段导出，用于后续独立学习流的共同起点。"""
    runner = trainer.runner
    if (trainer.stride is not None or trainer.pending_update is not None
            or runner.requests or trainer.published_version <= trainer.initial_step):
        raise RuntimeError("export requires idle serving and a committed published head")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        shards, rounding = [], []
        for rank in range(2):
            banks = [None] * trainer.first_bank
            for handle, (shape, dtype) in zip(runner.suffix[trainer.first_bank:], SHAPES[trainer.first_bank:], strict=True):
                value = torch.empty(shape, dtype=dtype)
                trainer.read(rank, handle, value)
                banks.append(value)
            tensors = load_file(str(runner.head_directory / f"rank{rank}.safetensors"), device="cpu")
            tensors = restore_tensor_layout(tensors, banks)
            rounding.append(max((1 + tensors[f"layers.{LAYERS[-1]}.attention.draft.{name}"].float()
                                - banks[bank].reshape(-1)).abs().max().item()
                                for bank, name in ((12, "q_norm.weight"), (13, "k_norm.weight"))))
            path = directory / f"rank{rank}.safetensors"
            save_file(tensors, str(path))
            for actual, rebuilt in zip(banks, rank_weights(path), strict=True):
                if actual is not None and not torch.equal(actual, rebuilt):
                    raise ValueError("exported head does not reproduce the published bank")
            shards.append(dict(name=path.name, sha256=digest(path)))
        manifest = dict(runner.head)
        manifest.update(rank_files=shards, parent_payload_sha256=runner.head["source_payload_sha256"],
                        step=runner.head["step"] + trainer.published_version - trainer.initial_step,
                        published_updates=trainer.published_version - trainer.initial_step,
                        export_kind="bf16_training_exchange", norm_rounding_max_abs=rounding,
                        published_banks_exact=True,
                        online_parameter_scope=getattr(trainer, "parameter_scope", "full"),
                        source_payload_sha256=hashlib.sha256("".join(s["sha256"] for s in shards).encode()).hexdigest())
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        checkpoint_manifest(directory, runner.settings.model_root)
        return manifest
    except BaseException:
        shutil.rmtree(directory)
        raise
