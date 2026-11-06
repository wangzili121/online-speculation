#!/usr/bin/env python3
"""Repartition an exported draft head without changing its weights or training semantics."""

import argparse
import json
from pathlib import Path
import shutil


def export_head(model, source, output, tp):
    from safetensors.torch import save_file
    from blockspec.pypto_qwen38.head import HeadSource

    output = Path(output)
    readers = [HeadSource(model, source, runtime_tp=tp, runtime_rank=rank) for rank in range(tp)]
    output.mkdir(parents=True, exist_ok=False)
    try:
        for rank, reader in enumerate(readers):
            tensors = {f"layers.{layer}.attention.draft.{name}": value
                       for layer in range(64) for name, value in reader.layer(layer).items()}
            save_file(tensors, str(output / f"rank{rank}.safetensors"))
            del tensors
        manifest = json.loads((Path(source) / "manifest.json").read_text())
        manifest["tp_size"] = tp
        # 完整写出所有 rank 后才发布 manifest，避免半成品被当成可加载 head。
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        HeadSource(model, output)
    except BaseException:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    export_head(args.model, args.head_dir, args.output, args.tp)


if __name__ == "__main__":
    main()
