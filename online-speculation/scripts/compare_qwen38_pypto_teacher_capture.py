#!/usr/bin/env python3
"""核对完整/前缀教师采集是否产生同一训练轨迹和 FP32 Adam checkpoint。"""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_qwen38_pypto_online import analyze


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(16 << 20): digest.update(block)
    return digest.hexdigest()


def checkpoint_equal(left, right):
    directories = [Path(report["checkpoint"]["path"]) for report in (left, right)]
    manifests = [json.loads((directory/"manifest.json").read_text()) for directory in directories]
    if ({k: v for k, v in manifests[0].items() if k != "payload_sha256"} !=
            {k: v for k, v in manifests[1].items() if k != "payload_sha256"}):
        raise ValueError("optimizer checkpoint contracts differ")
    paths = [directory/"optimizer.safetensors" for directory in directories]
    hashes = [sha256(path) for path in paths]
    if any(value != manifest["payload_sha256"] for value, manifest in zip(hashes, manifests, strict=True)):
        raise ValueError("checkpoint payload SHA256 mismatch")
    if hashes[0] == hashes[1]:
        return dict(bitwise_equal=True, whole_payload_equal=True, payload_sha256=hashes)
    # safetensors 文件头的键序不同不代表 tensor 不同；必要时逐 tensor 比较实际字节。
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    values = [load_file(str(path), device="cpu") for path in paths]
    if values[0].keys() != values[1].keys(): raise ValueError("optimizer tensor sets differ")
    equal = {}
    for name in values[0]:
        a, b = values[0][name], values[1][name]
        equal[name] = (a.shape == b.shape and a.dtype == b.dtype and
                       torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
    return dict(bitwise_equal=all(equal.values()), whole_payload_equal=False,
                payload_sha256=hashes, tensors_bitwise_equal=equal)


def compare(left, right):
    summaries = [analyze(report) for report in (left, right)]
    if left["arguments"].get("partial_teacher_capture", False) or not right["arguments"].get("partial_teacher_capture"):
        raise ValueError("expected full teacher control and partial teacher candidate")
    common = ("last_layers", "steps", "width", "tokens", "stride", "replay_blocks", "learning_rate",
              "warmup_steps", "seed", "max_prompt", "train_count", "dev_count", "test_count",
              "share_ar_prefix", "verification_backend")
    if any(left["arguments"][key] != right["arguments"][key] for key in common):
        raise ValueError("training/inference recipe changed")
    if left["head_identity"] != right["head_identity"] or left["dataset"]["files_sha256"] != right["dataset"]["files_sha256"]:
        raise ValueError("base/head/data changed")
    if left["warm_start"]["payload_sha256"] != right["warm_start"]["payload_sha256"]:
        raise ValueError("different prewarmed initialization")
    # 确认集不参与训练，允许仅一组做最终宽度选择/确认；其余数据必须完全一致。
    for split in ("calibration", "train", "dev", "test"):
        if left["dataset"]["splits"][split] != right["dataset"]["splits"][split]:
            raise ValueError("training or paired held-out questions changed")
    request_keys = ("id", "tokens", "updates", "rounds", "accepted", "proposed",
                    "head_version_before", "head_version_after")
    if len(left["train"]) != len(right["train"]) or any(
            any(a[key] != b[key] for key in request_keys) for a, b in zip(left["train"], right["train"], strict=True)):
        raise ValueError("online request or acceptance trajectory changed")
    keys = ("step", "version", "learning_rate", "reached_positions", "feedback_valid_rows")
    if any(any(a[key] != b[key] for key in keys) for a, b in zip(left["updates"], right["updates"], strict=True)):
        raise ValueError("update schedule or supervised rows changed")
    deltas = {key: max(abs(a[key]-b[key]) for a, b in zip(left["updates"], right["updates"], strict=True))
              for key in ("loss", "gradient_norm", "clip_scale")}
    checkpoint = checkpoint_equal(left, right)
    net = [report["online_net"]["summary"] for report in (left, right)]
    return dict(passed=checkpoint["bitwise_equal"], training_trajectory_equal=True,
                updates=len(left["updates"]), update_statistics_max_abs_delta=deltas,
                checkpoint=checkpoint, online_tokens_per_second=[row["tokens_per_second"] for row in net],
                online_tps_ratio=net[1]["tokens_per_second"]/net[0]["tokens_per_second"],
                paired_ar_normalized_ratio=(net[1]["measured_tps_ratio"]/net[0]["measured_tps_ratio"]),
                captured_teacher_rows=sum(sum(row["feedback_captured_rows"]) for row in right["updates"]),
                supervised_positions=[row["supervised_positions"] for row in summaries])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = compare(json.loads(a.control.read_text()), json.loads(a.candidate.read_text()))
    result["reports_sha256"] = [sha256(path) for path in (a.control, a.candidate)]
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]: raise SystemExit("teacher capture changed optimizer state")


if __name__ == "__main__":
    main()
