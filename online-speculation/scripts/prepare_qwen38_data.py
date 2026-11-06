#!/usr/bin/env python3
# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：数据准备入口：将语料编码为训练脚本读取的 token ID 记录；数据准备与模型参数更新分开。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Tokenize local text/chat JSONL or Parquet for Qwen draft distillation."""

# 依赖：导入 argparse。
import argparse
# 依赖：导入 hashlib。
import hashlib
# 依赖：导入 json。
import json
# 依赖：从 pathlib 导入 Path；供本模块调用。
from pathlib import Path


# ── sha256 ──
# 职责：计算文件内容 SHA256，用于记录数据身份；不是模型前向或 loss。
def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# ── rows ──
# 职责：按批次读取 Parquet，或逐行解析 JSONL，yield 原始语料记录；不会一次读入全部数据。
def rows(path):
    if path.suffix == ".parquet":
        # 依赖：导入 pyarrow.parquet。
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            # 逐项产出：调用者迭代时才取出这条记录/结果，不一次性装入整个数据集。
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    # 逐项产出：调用者迭代时才取出这条记录/结果，不一次性装入整个数据集。
                    yield json.loads(line)


# ── tokenize ──
# 职责：文本使用 tokenizer.encode，对话使用 apply_chat_template；输出 token ID 序列，不是 embedding。
# 输入提示：row：某层工作缓存行，或输入数据记录，具体由所在模块决定。
def tokenize(row, tokenizer):
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：each source row must be an object。
    if not isinstance(row, dict):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("each source row must be an object")
    if "messages" in row:
        return tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：source rows require text or messages。
    if not isinstance(row.get("text"), str):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("source rows require text or messages")
    return tokenizer.encode(row["text"], add_special_tokens=False)


# ── prepare ──
# 职责：逐条读取语料、分词、截断并过滤短样本，写入新的 JSONL 和来源摘要。
# 输入提示：output：前向结果包或算子输出；若为 BackboneOutput，可访问 logits/cache/boundary。
def prepare(source, output, tokenizer, *, min_tokens, max_tokens, max_records):
    selected, skipped = [], 0
    with output.open("x", encoding="utf-8") as stream:
        # 保护区：执行可能失败的步骤；异常处理或 finally 负责恢复状态/释放资源。
        try:
            for index, row in enumerate(rows(source)):
                ids = tokenize(row, tokenizer)[:max_tokens]
                if len(ids) < min_tokens:
                    # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
                    skipped += 1
                    # 跳过本次迭代的后续步骤，进入下一项。
                    continue
                # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
                stream.write(json.dumps(dict(input_ids=ids, source_row=index)) + "\n")
                # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
                selected.append(dict(source_row=index, tokens=len(ids)))
                if max_records and len(selected) >= max_records:
                    # 结束当前循环，继续循环之后的流程。
                    break
        except BaseException:
            output.unlink()
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise
    # 约束检查：下方条件表示不支持的输入/状态；触发时报错：no source rows meet min_tokens。
    if not selected:
        output.unlink()
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise ValueError("no source rows meet min_tokens")
    return dict(records=selected, skipped_short=skipped, source_sha256=sha256(source),
                tokens_sha256=sha256(output))


# ── main ──
# 职责：解析数据/模型路径与长度限制，加载本地 tokenizer，调用 prepare 生成训练输入。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="dataset ID, revision, config and split for provenance")
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=0, help="0 processes the complete file")
    args = parser.parse_args()
    if not 0 < args.min_tokens <= args.max_tokens or args.max_records < 0:
        parser.error("require 0 < min_tokens <= max_tokens and max_records >= 0")
    # 依赖：从 transformers 导入 AutoTokenizer；供本模块调用。
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = prepare(args.input, args.output, tokenizer, min_tokens=args.min_tokens,
                       max_tokens=args.max_tokens, max_records=args.max_records)
    manifest.update(source=args.source, model=str(args.model), min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens, chat_template_sha256=hashlib.sha256(
                        json.dumps(tokenizer.chat_template, sort_keys=True).encode()).hexdigest())
    # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
    print(json.dumps(manifest | {"records": len(manifest["records"])}), flush=True)


if __name__ == "__main__":
    # 调用：组织本文件入口流程；具体模型数学在被调用的模型/算子模块中。
    main()
