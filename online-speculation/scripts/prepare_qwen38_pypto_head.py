#!/usr/bin/env python3
"""在 NPU 上整理训练 head，并保存可直接加载的 PyPTO 推理 bank。"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--head-dir", type=Path)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--runtime-tp", type=int, choices=(1, 2, 4), default=1)
    p.add_argument("--runtime-rank", type=int, default=0)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--mask-token-id", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() or a.device < 0:
        p.error("new output directory and assigned device required")
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.prepared_head import save
    runtime_identity(pypto)
    head = HeadSource(a.model, a.head_dir, block_size=a.block_size, mask_token_id=a.mask_token_id,
                      runtime_tp=a.runtime_tp, runtime_rank=a.runtime_rank)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
    head.prepare_on_device(config)
    manifest = save(head, a.output_dir)
    print(json.dumps(dict(output=str(a.output_dir), seconds=head.preparation_seconds, manifest=manifest), indent=2))


if __name__ == "__main__":
    main()
