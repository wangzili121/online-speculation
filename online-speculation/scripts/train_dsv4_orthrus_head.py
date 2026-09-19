#!/usr/bin/env python3
"""Train, resume, or continually adapt the BF16 DSV4 Orthrus diffusion head."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from blockspec.dsv4_fitting import DSV4FitConfig, DSV4Trainer
from blockspec.dsv4_training import (
    OrthrusDSV4TrainingModel,
    export_bf16_head,
    load_bf16_head,
)
from blockspec.parallel.fitting import TokenDataset
from blockspec import reporting as report
from blockspec.vllm_ascend.head_contract import BASE_MODEL_ID, target_identity


def _resolve_device(name: str) -> torch.device:
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as error:
            raise SystemExit("NPU training requires torch_npu") from error
    return torch.device(name)


def _reject_quantized_training_base(path: Path) -> None:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    markers = (
        "quant_model_description.json",
        "quantization_config.json",
    )
    if config.get("quantization_config") is not None or any((path / name).exists() for name in markers):
        raise SystemExit(
            "the trainer needs a dequantized BF16 DSV4 checkpoint; W8A8/FP8 serving "
            "weights do not provide the backward path"
        )


def _load_target(path: Path, device: torch.device, precision: str):
    _reject_quantized_training_base(path)
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4ForCausalLM,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise SystemExit("DSV4 training requires Transformers with DeepseekV4ForCausalLM support") from error
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if not isinstance(target, DeepseekV4ForCausalLM):
        raise SystemExit(f"expected DeepseekV4ForCausalLM, got {type(target).__name__}")
    target.config.use_cache = False
    return target.to(device)


def _config(args) -> DSV4FitConfig:
    return DSV4FitConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        anchors_per_sequence=args.anchors_per_sequence,
        accumulate=args.accumulate,
        chunk_rows=args.chunk_rows,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        minimum_lr_ratio=args.minimum_lr_ratio,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        precision=args.precision,
        objective=args.objective,
        indexer_weight=args.indexer_weight,
        seed=args.seed,
    )


def _digest(module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tiny_model():
    try:
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    except ImportError as error:
        raise SystemExit("smoke test requires Transformers DSV4 support") from error
    config = DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=128,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        mlp_layer_types=["moe", "moe", "moe"],
        hc_mult=2,
        hc_sinkhorn_iters=2,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        partial_rotary_factor=0.5,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return DeepseekV4ForCausalLM(config)


def smoke(args) -> dict:
    torch.manual_seed(args.seed)
    if args.npu_devices:
        npu_devices = [
            _resolve_device(f"npu:{int(value)}")
            for value in args.npu_devices.split(",")
            if value.strip()
        ]
        if not npu_devices:
            raise SystemExit("--npu-devices did not contain a device")
        target = _tiny_model()
        model = OrthrusDSV4TrainingModel(target, block_size=4, mask_token_id=63)
        placement = [npu_devices[index % len(npu_devices)] for index in range(len(model.head))]
        model.place_layerwise(
            placement, input_device=npu_devices[0], output_device=npu_devices[-1]
        )
        device = npu_devices[0]
    else:
        device = _resolve_device(args.device)
        target = _tiny_model().to(device)
        model = OrthrusDSV4TrainingModel(target, block_size=4, mask_token_id=63).to(device)
    before_target = _digest(target)
    before_head = _digest(model.head)
    with tempfile.TemporaryDirectory(prefix="orthrus-dsv4-train-") as temporary:
        data_path = Path(temporary) / "tokens.jsonl"
        rows = [
            {"input_ids": [2 + ((offset + index) % 48) for index in range(24)]}
            for offset in range(8)
        ]
        encoder = json.JSONEncoder(separators=(",", ":"))
        data_path.write_text(
            "\n".join(encoder.encode(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        data = TokenDataset(data_path, 64, 16)
        config = DSV4FitConfig(
            steps=args.smoke_steps,
            batch_size=1,
            sequence_length=16,
            anchors_per_sequence=2,
            accumulate=1,
            chunk_rows=4,
            learning_rate=1e-3,
            warmup_steps=0,
            precision="fp32" if device.type == "cpu" else args.precision,
            objective="forward_kl",
            indexer_weight=0.01,
            seed=args.seed,
        )
        trainer = DSV4Trainer(model, data, config)
        records = trainer.run()
    result = {
        "mode": "smoke",
        "layers": model.compress_ratios,
        "steps": len(records),
        "losses": [record["loss"] for record in records],
        "head_changed": _digest(model.head) != before_head,
        "target_unchanged": _digest(target) == before_target,
        "target_gradients_absent": all(parameter.grad is None for parameter in target.parameters()),
        "pass": False,
    }
    result["pass"] = result["head_changed"] and result["target_unchanged"] and result["target_gradients_absent"]
    return result


def run(args) -> dict:
    if args.bf16_base is None or args.data is None:
        raise SystemExit("train/resume/continue require --bf16-base and --data")
    if args.mask_token_id is None or args.mask_token_id < 0:
        raise SystemExit("train/resume/continue require an explicit --mask-token-id")
    if args.block_size != 32 and not args.allow_nonproduction_block_size:
        raise SystemExit("production DSV4 Orthrus training requires --block-size 32")
    layer_devices = None
    if args.npu_devices:
        if args.layers_per_npu < 1:
            raise SystemExit("--layers-per-npu must be positive")
        npu_devices = [
            _resolve_device(f"npu:{int(value)}")
            for value in args.npu_devices.split(",")
            if value.strip()
        ]
        if not npu_devices:
            raise SystemExit("--npu-devices did not contain a device")
        device = torch.device("cpu")
    else:
        npu_devices = []
        device = _resolve_device(args.device)
    target = _load_target(args.bf16_base.resolve(), device, args.precision)
    base_identity = target_identity(args.bf16_base.resolve(), args.base_model_id)
    model = OrthrusDSV4TrainingModel(
        target, block_size=args.block_size, mask_token_id=args.mask_token_id
    )
    if npu_devices:
        layer_devices = [
            npu
            for npu in npu_devices
            for _ in range(args.layers_per_npu)
        ][: len(model.head)]
        layer_devices.extend([torch.device("cpu")] * (len(model.head) - len(layer_devices)))
        model.place_layerwise(
            layer_devices,
            input_device=npu_devices[0],
            output_device=npu_devices[-1],
        )
    else:
        model.to(device)
    if args.initial_head is not None:
        load_bf16_head(model, args.initial_head)
    data = TokenDataset(args.data, model.config.vocab_size, args.sequence_length)
    if args.mode == "resume":
        if args.checkpoint is None:
            raise SystemExit("resume requires --checkpoint")
        trainer = DSV4Trainer.resume(
            args.checkpoint, model, data, base_identity=base_identity
        )
    else:
        trainer = DSV4Trainer(
            model, data, _config(args), base_identity=base_identity
        )
        if args.mode == "continue":
            if args.checkpoint is None and args.initial_head is None:
                raise SystemExit("continue requires --checkpoint or --initial-head")
            if args.checkpoint is not None:
                trainer.load_continual_weights(args.checkpoint)

    def progress(record):
        if record["step"] % args.log_every == 0:
            print(report.dumps(record, sort_keys=True), flush=True)

    records = trainer.run(args.stop_after, progress=progress)
    trainer.save(args.output_checkpoint)
    exported = None
    if args.export_head is not None:
        if args.serving_target is None:
            raise SystemExit("--export-head requires --serving-target for the W8A8 identity")
        exported = export_bf16_head(
            model,
            args.export_head,
            serving_target=args.serving_target,
            training_base=args.bf16_base,
            mask_token_id=args.mask_token_id,
            base_model_id=args.base_model_id,
            training_metadata={
                "mode": args.mode,
                "step": trainer.step,
                "config": asdict(trainer.config),
                "data_sha256": data.fingerprint,
            },
        )
    return {
        "mode": args.mode,
        "step": trainer.step,
        "updates_this_run": len(records),
        "last": records[-1] if records else None,
        "checkpoint": str(args.output_checkpoint),
        "exported_head": None if exported is None else str(exported),
        "compress_ratios": model.compress_ratios,
        "layer_devices": None if layer_devices is None else [str(value) for value in layer_devices],
        "target_frozen": all(not parameter.requires_grad for parameter in target.parameters()),
        "pass": all(not parameter.requires_grad for parameter in target.parameters()),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("mode", choices=("smoke", "train", "resume", "continue"))
    value.add_argument("--bf16-base", type=Path)
    value.add_argument("--serving-target", type=Path)
    value.add_argument("--initial-head", type=Path)
    value.add_argument("--checkpoint", type=Path)
    value.add_argument("--data", type=Path)
    value.add_argument("--output-checkpoint", type=Path)
    value.add_argument("--export-head", type=Path)
    value.add_argument("--base-model-id", default=BASE_MODEL_ID)
    value.add_argument("--mask-token-id", type=int)
    value.add_argument("--block-size", type=int, default=32)
    value.add_argument("--allow-nonproduction-block-size", action="store_true")
    value.add_argument("--steps", type=int, default=1000)
    value.add_argument("--stop-after", type=int)
    value.add_argument("--batch-size", type=int, default=1)
    value.add_argument("--sequence-length", type=int, default=4096)
    value.add_argument("--anchors-per-sequence", type=int, default=16)
    value.add_argument("--accumulate", type=int, default=1)
    value.add_argument("--chunk-rows", type=int, default=8)
    value.add_argument("--learning-rate", type=float, default=2e-4)
    value.add_argument("--warmup-steps", type=int, default=20)
    value.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    value.add_argument("--weight-decay", type=float, default=0.0)
    value.add_argument("--clip-grad", type=float, default=1.0)
    value.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    value.add_argument("--objective", choices=("forward_kl", "cross_entropy"), default="forward_kl")
    value.add_argument("--indexer-weight", type=float, default=0.01)
    value.add_argument("--device", default="npu:0")
    value.add_argument(
        "--npu-devices",
        help="comma-separated devices for sequential model parallelism; remaining layers stay on CPU",
    )
    value.add_argument("--layers-per-npu", type=int, default=1)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--log-every", type=int, default=1)
    value.add_argument("--smoke-steps", type=int, default=2)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.mode != "smoke" and args.output_checkpoint is None:
        raise SystemExit("train/resume/continue require --output-checkpoint")
    if args.log_every < 1 or args.smoke_steps < 1:
        raise SystemExit("positive log and smoke step counts required")
    result = smoke(args) if args.mode == "smoke" else run(args)
    print(report.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
