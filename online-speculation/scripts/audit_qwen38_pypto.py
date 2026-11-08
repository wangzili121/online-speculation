#!/usr/bin/env python3
"""Check a pinned OneServing environment or inspect a real training head on CPU."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="OneServing git checkout")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--mask-token-id", type=int, default=1)
    parser.add_argument("--runtime-tp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--runtime-rank", type=int, default=0)
    parser.add_argument("--mode", choices=("environment", "head", "compile"), default="environment")
    parser.add_argument("--attention-backend", choices=("native", "fused", "paged"), default="native")
    parser.add_argument("--draft-group-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--gdn-backend", choices=("native", "vector"), default="native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    if not 0 <= args.runtime_rank < args.runtime_tp or args.mode != "head" and args.runtime_tp != 1:
        parser.error("rank-local TP layouts are currently available only for head inspection")
    import torch
    from blockspec.pypto_qwen38.baseline import native_module, runtime_identity, ONESERVING_COMMIT, PYPTO_COMMIT
    from blockspec.pypto_qwen38.head import HeadSource

    torch.set_num_threads(4)
    report = dict(scope=args.mode, oneserving=ONESERVING_COMMIT, required_pypto=PYPTO_COMMIT,
                  status="failed", npu_execution=False)
    try:
        weights = native_module(args.source, "weight")
        weights.verify_model_identity(args.model)
        if args.mode == "environment":
            import pypto
            report.update(runtime_identity(pypto))
            report["status"] = "environment_ready"
        else:
            head = HeadSource(args.model, args.head_dir, block_size=args.block_size,
                              mask_token_id=args.mask_token_id, runtime_tp=args.runtime_tp,
                              runtime_rank=args.runtime_rank)
            report.update(head_source="trained" if args.head_dir else "copied_target_initialization",
                          source_tp=head.tp, runtime_tp=head.layout.size, runtime_rank=head.layout.rank,
                          head_ranges=head.layout.head_ranges,
                          block_size=head.block_size, hybrid_policy="orthrus_qwen35_reference")
            if args.mode == "head":
                layers = []
                for layer in range(64):
                    banks = head.banks(layer)
                    layers.append(dict(layer=layer, shapes=[list(value.shape) for value in banks],
                                       dtype=[str(value.dtype) for value in banks],
                                       sha256=[hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
                                               for value in banks]))
                    print(f"checked head layer {layer}", flush=True)
                report.update(status="head_layout_checked", layers=layers)
            else:
                import pypto
                from pypto.ir import DistributedConfig
                from pypto.runtime import RunConfig
                from blockspec.pypto_qwen38.draft import NeuralDraft
                report.update(runtime_identity(pypto))
                report["attention_backend"] = args.attention_backend
                report["gdn_backend"] = args.gdn_backend
                draft = NeuralDraft(args.source, head, attention_backend=args.attention_backend,
                                    group_size=args.draft_group_size,
                                    head_transfer="packed" if args.draft_group_size == 4 else "individual",
                                    gdn_backend=args.gdn_backend)
                config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(
                    device_ids=[0], num_sub_workers=0))
                draft.compile(native_module(args.source, "runtime").single_meta_samples(), config)
                artifacts = {name: len(list(Path(program.output_dir).rglob("kernel_config.py")))
                             for name, program in draft.programs.items()}
                if any(count < 1 for count in artifacts.values()):
                    raise RuntimeError("a draft entry has only compile-only output, without PTOAS kernel metadata")
                report.update(status="draft_frontend_and_ptoas_checked", kernel_configs=artifacts)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
