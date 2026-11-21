"""只读汇总三卡固定配方重复；不派发模型、不补跑失败或负结果。"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def read(path):
    if path.exists():
        return path.read_bytes()
    compressed = path.with_name(path.name + ".gz")
    return gzip.decompress(compressed.read_bytes()) if compressed.exists() else None


def summarize(root):
    q = root / "resource338"
    accounting = module(root / "rows255/audit_accounting.py", "accounting_resource_summary")
    schedule = module(root / "schedule326/audit_schedule.py", "schedule_resource_summary")
    registration = json.loads((q / "registration.json").read_text())
    groups = {
        "spoc": [(root / "aux333", registration["existing_spoc743"])],
        "gsm": [],
    }
    for name in ("A", "B_spoc", "B_gsm"):
        for case in json.loads((q / f"queue-{name}.json").read_text())["cases"]:
            group = case.get("resource_repeat_group")
            if group:
                groups[group.removeprefix("threecard-")].append((q, case["tag"]))
    result = dict(
        completed=False, target_met=False, groups={},
        physical_cards_per_arm=3, tp=2, learner="independent third card",
        numerical_retuning=False, original_two_card_results_unchanged=True,
        extension_registered_after_first_three_card_partial_observation=True,
        new_problem_cohort=False,
        scope="Frozen resource-layout repeats on already assigned cohorts. Separate-stream timing; "
              "do not infer a causal two-versus-three-card advantage or original LongProc success.",
    )
    for name, cases in groups.items():
        reference_path = (root / "spoc312/a111-v292-spoc312-confirm743.json" if name == "spoc"
                          else root / "gsm254/confirmations/a111-v248-gsm254-confirm743.json")
        reference = json.loads(read(reference_path))
        keys = list(registration["gsm_numeric_contract"])
        if name == "spoc":
            keys += ["training_rows", "max_model_len"]
        rows, pending, errors = [], [], []
        for directory, tag in cases:
            raw = read(directory / (tag + ".json"))
            proof_raw = read(directory / (tag + ".audit.json"))
            binding_raw = read(q / (tag + ".resources.json"))
            if raw is None or proof_raw is None or binding_raw is None:
                pending.append(tag)
                continue
            try:
                d, proof, binding = map(json.loads, (raw, proof_raw, binding_raw))
                assert proof["completed"] and proof["accounting_passed"]
                assert proof["raw_sha256"] == hashlib.sha256(raw).hexdigest()
                assert d["completed"] and d["clean_shutdown"] and not d.get("error")
                assert d["physical_device_count"] == 3 and d["background"] is True
                assert d["device_ids"] == [0, 1] and d["learner_device"] == 2
                assert binding["captured"] and len(set(binding["physical_cards"])) == 3
                assert binding["physical_cards"] == list(map(int, d["visible_devices"].split(",")))
                assert {w["physical_npu"] for w in binding["workers"]} == set(binding["physical_cards"])
                assert all(d[key] == reference[key] for key in keys)
                assert d["requests"] == reference["requests"] and d["optimizer_init"] == "fresh"
                stride = 64 if name == "spoc" else 128
                on = d["results"][str(stride)]
                manifest = on["exported_head"]["manifest"]
                assert manifest["published_banks_exact"]
                assert proof["actual_published_shards"] == {v["name"]: v["sha256"] for v in manifest["rank_files"]}
                source = json.loads(read(directory / (tag + ".source.json")))["files"]
                if directory.name == "aux333":
                    frozen = json.loads((q / "stage-A.json").read_text())["source_manifests"]["online-studies"]["spoc312-frozen-source.json"]
                    assert source == frozen
                    assert proof["source_initial_head_config_assets_checked"]
                else:
                    assert proof["frozen_source_head_assets_base_checked"]
                row = accounting.audit(d, stride, layout="separate_streams")
                events = schedule.audit(d, stride)
                assert events["completed"] and events["actual_schedule_checked"]
                row.update(tag=tag, seed=d["seed"], raw_sha256=proof["raw_sha256"],
                           physical_cards=binding["physical_cards"], event_audit=events,
                           final_published_step=manifest["step"])
                rows.append(row)
            except (AssertionError, KeyError, ValueError, TypeError) as error:
                errors.append(dict(tag=tag, error=repr(error)))
        group = dict(completed=len(rows) == 3 and not errors, target_met=False,
                     completed_cases=rows, pending=pending, errors=errors)
        if len(rows) == 3 and not errors:
            assert sorted(row["seed"] for row in rows) == [743, 744, 745]
            pooled = {arm: sum(r["arms"][arm]["tokens"] for r in rows)
                      / sum(r["arms"][arm]["seconds"] for r in rows)
                      for arm in ("ar", "fixed", "online")}
            gains = {arm: pooled["online"] / pooled[arm] - 1 for arm in ("ar", "fixed")}
            group.update(pooled_tok_s=pooled, pooled_gain=gains,
                         target_met=all(row["online_vs_ar"] > .05 and row["online_vs_fixed"] > .05 for row in rows)
                         and all(value > .05 for value in gains.values()))
        result["groups"][name] = group
    result["completed"] = all(g["completed"] for g in result["groups"].values())
    result["target_met"] = result["completed"] and all(g["target_met"] for g in result["groups"].values())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    result = summarize(args.validation_root)
    (args.validation_root / "resource338/three-card-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": result["completed"], "target_met": result["target_met"],
                      "groups": {k: {"completed": v["completed"], "cases": len(v["completed_cases"]),
                                     "pending": len(v["pending"]), "errors": v["errors"]}
                                 for k, v in result["groups"].items()}}))
