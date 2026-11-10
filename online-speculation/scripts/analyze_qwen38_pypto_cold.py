#!/usr/bin/env python3
"""独立核对整块预热的数据、监督范围和留出评测。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_qwen38_pypto_online import paired_prefix_change, paired_rows
from benchmark_qwen38_pypto_cold import response_records


def analyze(report, source):
    if not report.get("completed") or not report.get("clean_shutdown") or report.get("error"):
        raise ValueError("completed, clean prewarm run required")
    args = report["arguments"]
    records, _ = response_records(source, report["dataset"])
    if records != report["records"] or report["head_identity"] != source["head_identity"]:
        raise ValueError("prewarm records/base differ from the verified AR response source")
    rows = {row["id"]: row for row in records}
    batch = args["records_per_update"]*args["anchors_per_record"]
    valid = args["width"]-1
    if len(report["updates"]) != args["steps"] or len(report["samples"]) != args["steps"]*batch:
        raise ValueError("incomplete update or supervised-block budget")
    for index, row in enumerate(report["updates"], 1):
        if (row["step"] != index or row["version"] != index or
                row["feedback_valid_rows"] != [valid]*batch or row["reached_positions"] != valid*batch):
            raise ValueError("prewarm updates must supervise the full block and publish once per batch")
        if any(not math.isfinite(row[key]) for key in ("loss", "gradient_norm", "clip_scale", "seconds")):
            raise ValueError("nonfinite prewarm update")
    for index, sample in enumerate(report["samples"]):
        record = rows[sample["record_id"]]
        start = record["answer_start"] if args.get("answer_only") else 1
        if (not start <= sample["anchor"] <= len(record["tokens"])-args["width"] or
                sample["width"] != args["width"] or sample["supervised_positions"] != valid or
                len(sample["matched_positions"]) != valid or sample["head_version"] != index//batch):
            raise ValueError("invalid anchor, teacher coverage or sample head version")
    for entry in report["evaluations"]:
        for name in ("dev", "test"):
            if not entry.get(name):
                continue
            actual, targets = paired_rows(entry[name]["rows"], entry[name]["target_rows"])
            for row, target in zip(actual, targets, strict=True):
                if row["tokens"] != target["tokens"]:
                    raise ValueError("held-out output differs from AR")
                if any(item["head_version_before"] != entry["step"] or
                       item["head_version_after"] != entry["step"] for item in (row, target)):
                    raise ValueError("evaluation changed the head version")
    first, final = report["evaluations"][0], report["evaluations"][-1]
    if (first["step"], final["step"]) != (0, args["steps"]):
        raise ValueError("initial and final held-out evaluations required")
    result = dict(updates=args["steps"], records=len(records), supervised_blocks=len(report["samples"]),
                  supervised_positions=valid*len(report["samples"]), cost=report["cost"], audits=report["audits"],
                  dev_curve=[dict(step=entry["step"], **entry["dev"]["summary"])
                             for entry in report["evaluations"]])
    for name in ("dev", "test"):
        result[name] = paired_prefix_change(first[name]["rows"], final[name]["rows"])
        result[name]["final_summary"] = final[name]["summary"]
    # 教师强制下的逐位置准确率是训练诊断，不能替代真实连续接受长度。
    for name, samples in (("first64", report["samples"][:64]), ("last64", report["samples"][-64:])):
        result[name+"_teacher_forced_accuracy"] = [sum(row["matched_positions"][i] for row in samples)/len(samples)
                                                 for i in range(valid)]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--records-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.report.read_text())
    payload = a.records_report.read_bytes()
    if hashlib.sha256(payload).hexdigest() != report["records_source_sha256"]:
        p.error("AR response source SHA256 differs from the recorded input")
    result = analyze(report, json.loads(payload))
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
