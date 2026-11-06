#!/usr/bin/env python3
# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：服务入口：解析命令行，拼出原生模型与 custom_class 配置，再执行 vllm serve。它不实现模型 forward。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Launch native vLLM OpenAI serving with the Qwen3.8 Orthrus plugin."""

# 依赖：导入 argparse。
import argparse
# 依赖：导入 json。
import json
# 依赖：导入 os。
import os
# 依赖：从 pathlib 导入 Path；供本模块调用。
from pathlib import Path
# 依赖：导入 sys。
import sys


# ── prepare_runtime_environment ──
# 职责：补齐 CANN Python 搜索路径，让后续进程能导入运行时绑定。
def prepare_runtime_environment():
    """Keep CANN Python bindings visible when Docker overrides PYTHONPATH."""
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if home:
        site = str(Path(home) / "python" / "site-packages")
        if Path(site).is_dir():
            if site not in sys.path:
                # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
                sys.path.append(site)
            paths = [path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path]
            if site not in paths:
                os.environ["PYTHONPATH"] = os.pathsep.join([*paths, site])


# ── arguments ──
# 职责：读取命令行选项并做基础合法性检查；返回 argparse.Namespace。
def arguments(parser=None, *, defaults=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--mask-token-id", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--last-layers", type=int, default=1)
    parser.add_argument("--all-draft-layers", action="store_true")
    parser.add_argument("--optimizer-device", choices=("model", "cpu"), default="model")
    parser.add_argument("--checkpoint-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--training-gdn-backend", choices=("torch", "fla"), default="torch")
    parser.add_argument("--training-loss-backend", choices=("gathered", "vocab_parallel"), default="gathered")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--update-stride", type=int, default=16)
    parser.add_argument("--head-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.65)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gdn-prefill-backend", choices=("native", "cann"), default="cann")
    parser.set_defaults(**(defaults or {}))
    args = parser.parse_args()
    if args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes < 1:
        parser.error("--kv-cache-memory-bytes must be positive")
    return args


# ── native_options ──
# 职责：生成原生 vLLM 主模型配置：TP、eager、单请求、KV 预算与 worker 扩展类。
def native_options(args):
    options = dict(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
                enforce_eager=True, max_num_seqs=1, max_model_len=args.max_model_len,
                max_num_batched_tokens=256, gpu_memory_utilization=args.gpu_memory_utilization,
                enable_prefix_caching=False, async_scheduling=False, language_model_only=True,
                additional_config={"qwen38_gdn_prefill_backend": args.gdn_prefill_backend},
                worker_extension_cls="blockspec.vllm_ascend.qwen38_proposer.OrthrusWorkerExtension")
    if getattr(args, "kv_cache_memory_bytes", None) is not None:
        options["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    return options


# ── speculative_options ──
# 职责：声明 custom_class proposer 路径及 K−1 个候选数，把训练/head 选项放入 additional_config。
def speculative_options(args):
    options = dict(mask_token_id=args.mask_token_id, online=args.online, last_layers=args.last_layers,
                   update_stride=args.update_stride, checkpoint_every=args.checkpoint_every)
    for name, default in (("all_draft_layers", False), ("optimizer_device", "model"),
                          ("checkpoint_layers", False), ("learning_rate", 1e-5),
                          ("training_gdn_backend", "torch"), ("training_loss_backend", "gathered")):
        options[name] = getattr(args, name, default)
    for name in ("head_dir", "checkpoint_dir"):
        if getattr(args, name, None):
            options[name] = getattr(args, name)
    return dict(speculative_config={
        "method": "custom_class", "model": "blockspec.vllm_ascend.qwen38_proposer.Qwen38OrthrusProposer",
        "num_speculative_tokens": args.block_size - 1, "disable_padded_drafter_batch": True,
        "draft_sample_method": "probabilistic",
    }, additional_config={"qwen38_orthrus": options, "qwen38_gdn_prefill_backend": args.gdn_prefill_backend})


# ── main ──
# 职责：准备 CANN 环境，解析参数，组合原生/投机配置，构造 vllm serve 命令并 exec 启动服务进程。
def main():
    # 调用：补齐 CANN Python 搜索路径，让后续进程能导入运行时绑定。
    prepare_runtime_environment()
    args = arguments()
    settings = native_options(args) | speculative_options(args)
    model = settings.pop("model")
    command = ["vllm", "serve", model, "--host", args.host, "--port", str(args.port)]
    for name, value in settings.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            command.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            command.extend([flag, json.dumps(value) if isinstance(value, dict) else str(value)])
    os.execvp(command[0], command)


if __name__ == "__main__":
    # 调用：组织本文件入口流程；具体模型数学在被调用的模型/算子模块中。
    main()
