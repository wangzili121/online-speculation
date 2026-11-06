#!/usr/bin/env python3
# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：离线全层训练入口：读取 tokenized JSONL，抽样 anchor，通过 worker RPC 驱动蒸馏与导出；不是对话请求训练。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Offline all-layer Qwen draft distillation on the native TP-sharded backbone."""

# 依赖：导入 argparse。
import argparse
# 依赖：导入 hashlib。
import hashlib
# 依赖：导入 json。
import json
# 依赖：从 pathlib 导入 Path；供本模块调用。
from pathlib import Path
# 依赖：导入 time。
import time

# 依赖：从 run_qwen38_vllm 导入 arguments, native_options, prepare_runtime_environment, speculative_options；供本模块调用。
from run_qwen38_vllm import arguments, native_options, prepare_runtime_environment, speculative_options


# ── main ──
# 职责：读取 tokenized 数据，创建 offline LLM worker，逐步 RPC 全层蒸馏，保存 loss.jsonl 并导出 head。
def main():
    # 调用：补齐 CANN Python 搜索路径，让后续进程能导入运行时绑定。
    prepare_runtime_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="tokenized JSONL: input_ids, optional anchor_ranges")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--audit", action="store_true", help="hash frozen target and draft before/after fitting")
    parser.add_argument("--timing", action="store_true", help="measure synchronized update phases in validation workers")
    args = arguments(parser, defaults=dict(optimizer_device="cpu", checkpoint_layers=True))
    if args.online or args.steps < 1 or not args.block_size <= args.sequence_length <= args.max_model_len:
        parser.error("offline fitting requires positive steps and block_size <= sequence_length <= max_model_len")
    args.output.mkdir(parents=True, exist_ok=False)

    # 依赖：导入 torch。
    import torch
    # 依赖：从 transformers 导入 AutoConfig；供本模块调用。
    from transformers import AutoConfig
    # 依赖：从 vllm 导入 LLM；供本模块调用。
    from vllm import LLM
    # 依赖：从 blockspec.parallel.fitting 导入 TokenDataset, BatchStream；供本模块调用。
    from blockspec.parallel.fitting import TokenDataset, BatchStream

    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    text = getattr(config, "text_config", config)
    dataset = TokenDataset(args.data, text.vocab_size, args.sequence_length)
    stream = BatchStream(dataset, args.seed)
    args.online, args.all_draft_layers = False, True
    args.checkpoint_dir = str(args.output / "head")
    settings = native_options(args) | speculative_options(args)
    settings["additional_config"]["qwen38_orthrus"]["offline"] = True
    if args.audit or args.timing:
        settings["worker_extension_cls"] = "qwen38_validation_worker.ValidationWorker"
    llm = LLM(**settings)
    if args.gdn_prefill_backend == "cann":
        llm.collective_rpc("orthrus_use_cann_gdn_prefill")
    records = []
    frozen_before = llm.collective_rpc("orthrus_target_fingerprint") if args.audit else None
    head_before = llm.collective_rpc("orthrus_head_fingerprint") if args.audit else None
    with (args.output / "loss.jsonl").open("w") as log:
        for step in range(args.steps):
            # 数据：待主模型验证的 token ID，一维长度最多 K−1。
            tokens, candidates = stream.batch_with_anchor_candidates(1, args.block_size)
            choices = candidates[0]
            # 数据：锚点：起草时指已知 token ID，离线训练函数的 anchor 参数则是文本中的位置索引，注意所在函数。
            anchor = int(choices[int(torch.randint(len(choices), (), generator=stream.rng))])
            start = time.perf_counter()
            method = "orthrus_timed_offline_step" if args.timing else "orthrus_offline_step"
            ranks = llm.collective_rpc(method, args=(tokens[0].tolist(), anchor))
            record = {"step": step + 1, "anchor": anchor, "seconds": time.perf_counter() - start,
                      "sequence_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(), "ranks": ranks}
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：offline learner did not select every draft layer。
            if any(row["layers"] != text.num_hidden_layers
                   or row["gradient_layers"] != list(range(text.num_hidden_layers)) for row in ranks):
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise RuntimeError("offline learner did not select every draft layer")
            # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
            log.write(json.dumps(record) + "\n")
            # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
            log.flush()
            # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
            print(json.dumps(record), flush=True)
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            records.append(record)
    head = llm.collective_rpc("orthrus_checkpoint")
    report = {"mode": "offline_all_draft", "data_sha256": dataset.fingerprint, "steps": args.steps,
              "eligible_records": len(dataset), "skipped_records": dataset.skipped,
              "seed": args.seed, "loss": "forward_kl", "optimizer_device": args.optimizer_device,
              "checkpoint_layers": args.checkpoint_layers, "head": head, "records": records,
              "stats": llm.collective_rpc("orthrus_stats")}
    if args.audit:
        frozen_after = llm.collective_rpc("orthrus_target_fingerprint")
        head_after = llm.collective_rpc("orthrus_head_fingerprint")
        report["audit"] = dict(target_unchanged=frozen_before == frozen_after,
                               draft_changed=head_before != head_after,
                               target_before=frozen_before, target_after=frozen_after,
                               head_before=head_before, head_after=head_after)
        # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：offline fitting changed target weights or failed to publish draft updates。
        if frozen_before != frozen_after or head_before == head_after:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise AssertionError("offline fitting changed target weights or failed to publish draft updates")
    # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    # 调用：组织本文件入口流程；具体模型数学在被调用的模型/算子模块中。
    main()
