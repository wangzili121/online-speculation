"""只读审核新题三次确认；单次失败不得被合并收益覆盖。"""
import json
from pathlib import Path
import random

from common import check_runtime_contract, comparable, module, read, read_bytes, sha


def summarize(root):
    q = root / "confirm350"
    if not (q / "frozen-recipe.json").exists():
        return dict(completed=False, target_met=False, reason="Recipe not frozen; no new confirmation evidence")
    frozen = read(q / "frozen-recipe.json")
    assert sha((q / "registration.json").read_bytes()) == frozen["registration_sha256"]
    assert all(sha((q / name).read_bytes()) == value for name, value in frozen["files_sha256"].items())
    assert sha((q / frozen["freshness_report"]).read_bytes()) == frozen["freshness_sha256"]
    assignment = read(q / "assignment.json")
    ids, hold = assignment["confirmation_indices"], assignment["independent_holdout_indices"]
    assert len(set(ids)) == 384 and len(set(hold)) == 32 and not set(ids) & set(hold)
    freshness = read(q / frozen["freshness_report"])
    assert freshness["completed"] and set(ids + hold) <= set(freshness["snapshot_eligible_indices"])
    metadata = read(root / "gsm-reserved1024-manifest.json")
    assert sha((root / "gsm-reserved1024.json").read_bytes()) == metadata["sha256"] == assignment["parent_pool_sha256"]
    texts = dict(zip(metadata["indices"], read(root / "gsm-reserved1024.json")))
    plan = read(q / "queue-B.json")
    assert len(plan["cases"]) == 3
    accounting = module(root / "rows255/audit_accounting.py", "accounting_confirm350")
    schedule = module(root / "schedule326/audit_schedule.py", "schedule_confirm350")
    qualifier = read(root / "sync345" / (frozen["qualification"]["template"]["tag"] + ".json"))
    contract = read(root / "resource338/registration.json")["gsm_numeric_contract"]
    totals, pending, errors = [], [], []
    for case in plan["cases"]:
        tag = case["tag"]
        path = q / (tag + ".audit.json")
        if not path.exists():
            pending.append(tag)
            continue
        proof = read(path)
        if not proof.get("completed"):
            errors.append(dict(tag=tag, error=proof.get("error", "Incomplete or invalid run")))
            continue
        try:
            raw = read_bytes(q / (tag + ".json"))
            d = json.loads(raw)
            assert sha(raw) == proof["raw_sha256"]
            assert proof["fresh_reserved_questions_used"] and proof["resources_checked"]
            assert proof["frozen_source_head_assets_base_checked"]
            assert comparable(case["args"]) == comparable(frozen["qualification"]["template"]["args"])
            assert d["requests"] == d["ar_checked_requests"] == 384
            assert d["physical_device_count"] == 3 and d["learner_device"] == 2 and d["device_ids"] == [0, 1]
            assert d["background"] is False and d["optimizer_init"] == "fresh"
            args = case["args"]
            order = list(ids)
            random.Random(d["seed"]).shuffle(order)
            prompt_path = q / "assets" / args[args.index("--request-file") + 1]
            hold_path = q / "assets" / args[args.index("--holdout-file") + 1]
            assert read(prompt_path) == [texts[i] for i in order]
            assert read(hold_path) == [texts[i] for i in hold]
            assert d["prompts_sha256"] == sha(prompt_path.read_bytes())
            assert d["holdout_sha256"] == sha(hold_path.read_bytes())
            assert all(d[key] == qualifier[key] for key in contract)
            check_runtime_contract(d, qualifier)
            binding = read(q / (tag + ".resources.json"))
            assert binding["captured"] and len(set(binding["physical_cards"])) == 3
            assert binding["physical_cards"] == list(map(int, d["visible_devices"].split(",")))
            assert {v["physical_npu"] for v in binding["workers"]} == set(binding["physical_cards"])
            on = d["results"]["128"]
            assert proof["actual_published_shards"] == {
                p["name"]: p["sha256"] for p in on["exported_head"]["manifest"]["rank_files"]}
            result = accounting.audit(d, 128)
            events = schedule.audit(d, 128)
            assert events == proof["actual_events"] and events["actual_schedule_checked"]
            result.update(tag=tag, seed=d["seed"], raw_sha256=sha(raw),
                          final_published_step=on["exported_head"]["manifest"]["step"],
                          physical_cards=binding["physical_cards"], actual_event_audit=events)
            totals.append(result)
        except (AssertionError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(tag=tag, error=repr(error)))
    out = dict(completed=len(totals) == 3 and not errors, target_met=False,
               completed_cases=totals, pending=pending, errors=errors,
               physical_cards_per_arm=3, target_TP=2, learner="independent third card",
               scope="New preassignment GSM384/holdout32; three independently reset/reordered streams on B. "
                     "Original assigned-cohort three-card negative remains unchanged.")
    if out["completed"]:
        assert sorted(v["seed"] for v in totals) == [743, 744, 745]
        pooled = {arm: sum(v["arms"][arm]["tokens"] for v in totals) /
                  sum(v["arms"][arm]["seconds"] for v in totals) for arm in ("ar", "fixed", "online")}
        gains = {arm: pooled["online"] / pooled[arm] - 1 for arm in ("ar", "fixed")}
        out.update(pooled_tok_s=pooled, pooled_gain=gains,
                   target_met=all(v["online_vs_ar"] > .05 and v["online_vs_fixed"] > .05 for v in totals)
                   and all(g > .05 for g in gains.values()))
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    value = summarize(root)
    (root / "confirm350/three-confirmation-audit.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))
