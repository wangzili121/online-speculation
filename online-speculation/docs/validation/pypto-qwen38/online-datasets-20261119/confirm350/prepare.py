"""完整旧流资格通过后，重新审计并冻结唯一新题配方；本程序不派发模型。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys

from common import change, comparable, qualified, read, sha


def prepare():
    q = Path(__file__).resolve().parent
    root = q.parent
    frozen = q / "frozen-recipe.json"
    if frozen.exists():
        d = read(frozen)
        assert sha((q / "registration.json").read_bytes()) == d["registration_sha256"]
        assert sha((q / d["freshness_report"]).read_bytes()) == d["freshness_sha256"]
        assert all(sha((q / name).read_bytes()) == value for name, value in d["files_sha256"].items())
        return d
    qualification = qualified(root)
    registration = read(q / "registration.json")
    assert not (q / "queue-B.json").exists() and not (q / "protocol.json").exists(), "Inspect partial preparation"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    audit_path = q / ("freshness-" + stamp + ".json")
    subprocess.run([sys.executable, str(root / "reserve349/audit_usage.py"),
                    "--via-A-for-C", "--host-timeout", "120", "--output", str(audit_path)], check=True)
    audit = read(audit_path)
    assert audit["completed"] and set(audit["hosts"]) == {"159.138.5.111", "189.1.227.167", "159.138.54.61"}
    now = datetime.now(timezone.utc)
    for observation in audit["hosts"].values():
        assert not observation["unknown_assets"] and not observation["read_errors"]
        age = (now - datetime.fromisoformat(observation["observed_UTC"])).total_seconds()
        assert -5 <= age <= registration["maximum_usage_observation_age_seconds"], "Usage observation stale"
    available = audit["snapshot_eligible_indices"]
    assert len(set(available)) == len(available) and len(available) >= 416
    metadata = read(root / "gsm-reserved1024-manifest.json")
    pool = read(root / "gsm-reserved1024.json")
    texts = dict(zip(metadata["indices"], pool))
    confirm, holdout = available[:384], available[384:416]
    assert not set(confirm) & set(holdout)
    prior = set(read(root / "gsm254/gsm254-confirm384-manifest.json")["indices"])
    prior.update(read(root / "gsm254/gsm254-holdout32-manifest.json")["indices"])
    head_ids = set(read(root / "gsm254/gsm254-unused-reserve-audit.json")["parent128_conservatively_excluded_IDs"])
    assert not set(confirm + holdout) & prior
    assert not {"train:" + str(i) for i in confirm + holdout} & head_ids
    assets = q / "assets"
    assets.mkdir(exist_ok=True)
    assert not list(assets.iterdir()), "Inspect partially written assignment"
    hashes = {}

    def save(path, value):
        raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
        with path.open("xb") as handle:
            handle.write(raw)
        hashes[str(path.relative_to(q))] = sha(raw)

    hold_name = "confirm350-gsm-holdout32.json"
    save(assets / hold_name, [texts[i] for i in holdout])
    template = qualification["template"]
    cases = []
    for seed in registration["seeds"]:
        order = list(confirm)
        random.Random(seed).shuffle(order)
        name = "confirm350-gsm-order" + str(seed) + ".json"
        save(assets / name, [texts[i] for i in order])
        args = list(template["args"])
        for flag, value in (("--devices", "0,1,2,3,4,5,6,7"), ("--request-file", name),
                            ("--holdout-file", hold_name), ("--seed", seed)):
            args = change(args, flag, value)
        assert comparable(args) == comparable(template["args"])
        case = dict(template, tag="b167-v308-confirm350-gsm384-seed" + str(seed), args=args,
                    requires=[dict(file=registration["qualification_tag"] + ".json", greedy=True)]
                             if not cases else [dict(file=cases[-1]["tag"] + ".json", greedy=True)],
                    max_attempts=1, prompts_sha256=hashes["assets/" + name],
                    purpose="New reserved GSM384, independently reset head128/freshAdam. Exactly3physicalA2/TP2+learner2; "
                            "synchronous uncapped periodicupdates, fullAR/fixed/online interleaving andallcosts, "
                            "actualpublished finalseen+independent32. Frozen recipe; all3 validflows run evenifnegative.")
        for key in ("confirm", "confirmation_group", "resource_repeat_group", "unless_achieved"):
            case.pop(key, None)
        cases.append(case)
    manifest = dict(source=metadata["source"], source_sha256=metadata["source_sha256"], split="train",
                    parent_pool_sha256=metadata["sha256"], selection="Registered pool order after recorded/head128/current exclusions; no labels or speed filters.",
                    confirmation_indices=confirm, independent_holdout_indices=holdout,
                    old_confirm254_indices_disjoint=True, recorded_head128_indices_disjoint=True,
                    freshness_report=audit_path.name, freshness_sha256=sha(audit_path.read_bytes()),
                    prepared_UTC=now.isoformat(), fresh_cohort_assigned=True, NN_dispatched=False)
    save(q / "assignment.json", manifest)
    save(q / "queue-B.json", dict(cases=cases))
    protocol = dict(registered_UTC=registration["registered_UTC"], frozen_UTC=now.isoformat(),
                    fresh_reserved_questions_used=True, background=False, physical_cards_per_arm=3, target_TP=2,
                    learner_logical_device=2, numerical_recipe_unchanged=True,
                    files_sha256={Path(name).name: value for name, value in hashes.items() if name.startswith("assets/")},
                    evidence_scope="Prospectively frozen three-card confirmation on new preassignment GSM384/holdout32. "
                                   "Three orders, independently reset head128/Adam, same3physicalcards/TP2/config perarm, allcontinuouscosts. "
                                   "B host/runtime scoped; not original failed assigned-cohort resource repeat.")
    save(q / "protocol.json", protocol)
    result = dict(registration_sha256=sha((q / "registration.json").read_bytes()),
                  frozen_UTC=now.isoformat(), qualification=qualification,
                  freshness_report=audit_path.name, freshness_sha256=sha(audit_path.read_bytes()),
                  files_sha256=hashes, fresh_cohort_assigned=True, NN_dispatched=False,
                  old_negative_results_preserved=True)
    with frozen.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


if __name__ == "__main__":
    value = prepare()
    print(json.dumps(dict(frozen=True, fresh_cohort_assigned=value["fresh_cohort_assigned"], NN_dispatched=False)))
