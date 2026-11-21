"""从已审核的完整双卡记录提取接受数、执行次数与开销；不读取中途窗口或启动模型。"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math


def read_raw(path):
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_suffix(".json.gz").read_bytes())


def summarize(root):
    groups = [
        ("gsm8k", "gsm254/confirmations", "gsm254-mixed271-confirmation-audit.json"),
        ("longproc_pseudo2code", "pseudo310", "three-confirmation-audit.json"),
        ("spoc_supplementary", "spoc312", "three-confirmation-audit.json"),
    ]
    results = []
    for name, folder, manifest in groups:
        directory = root / folder
        # SPoC 尚未收齐时，只使用已经单独审核完成的记录，不拼接运行中的进度。
        partial = not (directory / manifest).exists()
        proof_path = directory / ("first-two-complete-audit.json" if partial else manifest)
        proof = json.loads(proof_path.read_text())
        assert not partial or name == "spoc_supplementary"
        if not partial:
            assert proof["completed"] and proof["full_audit_passed"]
        cases = []
        for case in proof["cases"]:
            candidates = list(directory.glob("*confirm" + str(case["seed"]) + ".json"))
            candidates += list(directory.glob("*confirm" + str(case["seed"]) + ".json.gz"))
            matched = []
            for path in candidates:
                raw = read_raw(path.with_suffix("") if path.suffix == ".gz" else path)
                if hashlib.sha256(raw).hexdigest() == case["raw_sha256"]:
                    matched.append((path, raw))
            assert len(matched) == 1, candidates
            path, raw = matched[0]
            data = json.loads(raw)
            assert data["completed"] and data["clean_shutdown"]
            assert data["physical_device_count"] == 2 and not data["background"]
            online_keys = [key for key in data["results"] if key.isdecimal()]
            assert len(online_keys) == 1, online_keys
            online_key = online_keys[0]
            assert data["learner_device"] == 0 and data["results"][online_key]["exact_greedy"]
            arms = {}
            for label, key in (("ar", "ar"), ("fixed", "frozen"), ("online", online_key)):
                stream = data["results"][key]["stream"]
                stat = stream["stats"]
                assert math.isclose(stream["tokens"] / stream["seconds"], stream["tok_s"])
                # 回读、TP 整理在 update_ns 内；publish_ns 又包含它们，不能重复相加。
                direct_seconds = sum(stat[k] for k in ("feedback_ns", "update_ns", "publish_upload_ns")) / 1e9
                arms[label] = dict(
                    tokens=stream["tokens"], seconds=stream["seconds"], tok_s=stream["tok_s"],
                    accepted_per_round=stream["accepted_per_round"],
                    draft_rounds=stat["rounds"], draft_batches=stat["draft_batches"],
                    verify_calls=stat["verify_calls"],
                    draft_seconds=stat["draft_ns"] / 1e9,
                    verify_seconds=stat["verify_ns"] / 1e9,
                    direct_learning_seconds=direct_seconds,
                    direct_learning_fraction=direct_seconds / stream["seconds"],
                    updates=stat["updates"], publications=stat["publications"],
                )
            fixed, online = arms["fixed"], arms["online"]
            assert len({arm["tokens"] for arm in arms.values()}) == 1
            assert not arms["ar"]["updates"] and not fixed["updates"]
            cases.append(dict(seed=case["seed"], requests=data["requests"], raw_sha256=case["raw_sha256"],
                              raw_path=str(path.relative_to(root)), arms=arms,
                              online_vs_ar=online["tok_s"] / arms["ar"]["tok_s"] - 1,
                              online_vs_fixed=online["tok_s"] / fixed["tok_s"] - 1,
                              fewer_draft_rounds=1 - online["draft_rounds"] / fixed["draft_rounds"],
                              fewer_verify_calls=1 - online["verify_calls"] / fixed["verify_calls"]))
        results.append(dict(dataset=name, complete_confirmations=len(cases), required_confirmations=3,
                            full_group_complete=not partial, audit_path=str(proof_path.relative_to(root)),
                            cases=cases))
    return dict(
        completed_only=True, no_NN_dispatch=True, groups=results,
        scope="Full two-card streams only. Counters explain observed work, not a controlled causal ablation. "
              "Different datasets, request lengths and recipes are not compared as one-factor experiments. "
              "AR verify_calls=0 is an uninstrumented AR counter, not zero target execution. "
              "Direct learning cost is feedback+update+upload; read/pack is already within update. "
              "Device/host phase timers are not pure kernel time and are not subtracted to predict speedup.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.validation_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({group["dataset"]: group["complete_confirmations"] for group in result["groups"]}))
