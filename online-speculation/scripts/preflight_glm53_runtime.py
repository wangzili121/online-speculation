#!/usr/bin/env python3
"""Check model/runner/KDA interfaces before arming a full-model smoke run."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = dict(status="failed", image=args.image)
    try:
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM
        from blockspec.vllm_ascend.qwen38_runtime import install_runner_hooks
        from probe_glm53_ops import initialize_npu_runtime, main as probe

        report["versions"] = {name: importlib.metadata.version(name)
                              for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers")}
        initialize_npu_runtime()
        importlib.import_module("acl.rt")
        if not callable(getattr(Glm5NextForCausalLM, "compute_logits", None)):
            raise RuntimeError("native GLM causal LM interface is unavailable")
        install_runner_hooks()
        # 全部原生模块测试必须在同一运行镜像内通过，才会发布 ready 文件。
        import sys
        original = sys.argv
        try:
            sys.argv = ["probe_glm53_ops", "--output", str(args.output.with_name("operators.json"))]
            probe()
        finally:
            sys.argv = original
        names = ("src/blockspec/vllm_ascend/glm53_ops.py", "src/blockspec/vllm_ascend/glm53_native.py",
                 "src/blockspec/vllm_ascend/glm53_proposer.py", "src/blockspec/vllm_ascend/qwen38_runtime.py",
                 "scripts/run_glm53_vllm.py", "scripts/run_in_runtime.py")
        report["sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
        report["status"] = "passed"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
