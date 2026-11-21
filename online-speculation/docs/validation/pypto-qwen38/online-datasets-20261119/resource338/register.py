"""固定三卡资源对照的重复计划；只整理已冻结配方与题池，不运行模型。"""
from pathlib import Path
import argparse
import copy
import datetime
import hashlib
import json
import random


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace(args, key, value):
    if key in args:
        args[args.index(key) + 1] = str(value)
    else:
        args.extend([key, str(value)])


def build(root):
    read = lambda name: json.loads((root / name).read_text())
    spoc_freeze = read("spoc312/frozen-recipe.json")
    a_spoc = read("spoc312/queue-spoc312-confirm745.json")["cases"][0]
    b_spoc_plan = read("relocate318/queue-spoc323B.json")["cases"]
    old_three = read("aux308/queue-aux308-threecard.json")["cases"]
    old_gsm = old_three[-1]
    primary_gsm = read("gsm254/confirmations/a111-v248-gsm254-confirm743.json")
    auxiliary_gsm = read("aux308/b167-v308-aux3card-compact16-step128-gsm384-TV.json")
    contract_keys = ["head_step", "head_shard_sha256", "last_layers", "parameter_scope",
                     "parameterization", "draft_width", "candidates", "teacher_tokens", "objective",
                     "learning_rate", "warmup_updates", "warm_update_stride", "warm_update_steps",
                     "lr_schedule", "replay_blocks", "publish_every", "inner_steps", "teacher_targets",
                     "teacher_temperature", "teacher_scaling", "feedback_mode", "feedback_timing",
                     "publication_backend", "temperature", "stop_on_eos", "paired_physical_rows",
                     "draft_convolution", "gdn_valid_rows", "output_tokens_per_request"]
    assert all(primary_gsm[k] == auxiliary_gsm[k] for k in contract_keys)
    gsm_a_source = read("gsm254/confirmations/a111-v248-gsm254-confirm743.source.json")["files"]
    gsm_b_source = read("aux308/b167-v308-aux3card-compact16-step128-gsm384-TV.source.json")["files"]
    nn = lambda files: {"blockspec/" + k.split("blockspec/", 1)[1]: v
                        for k, v in files.items() if "blockspec/" in k}
    assert nn(gsm_a_source) == nn(gsm_b_source) and len(nn(gsm_a_source)) == 145
    assert gsm_a_source["orthrus_bench_dense_v248.py"] == gsm_b_source["orthrus_bench_dense_v308.py"]
    assets = {}

    def asset(name, value):
        data = (json.dumps(value, ensure_ascii=False) + "\n").encode()
        assets[name] = data
        return name

    spoc = read("context273/spoc283/spoc283-confirm256.json")
    gsm = read("gsm254/gsm254-confirm384.json")
    assert len(spoc) == 256 and len(gsm) == 384
    sh = asset("resource338-spoc-holdout32.json", read("context273/spoc283/spoc283-holdout32.json"))
    gh = asset("resource339-gsm-holdout32.json", read("gsm254/gsm254-holdout32.json"))

    def resource_case(base, tag, host, preflight, dataset=None, seed=None):
        case = copy.deepcopy(base)
        for key in ("confirm", "confirmation_group", "unless_achieved"):
            case.pop(key, None)
        case.update(tag=tag, max_attempts=1, minimum_available_host_gib=96,
                    allow_parallel_startup=False, requires=[dict(file=preflight)])
        args = case["args"]
        for flag in ("--interleaved", "--interleave-ar"):
            if flag in args:
                args.remove(flag)
        replace(args, "--learner-device", 2)
        replace(args, "--devices", "0,1,2,3" if host == "B" else "2,3,5,6,7")
        replace(args, "--wait", 28800)
        if "--background" not in args:
            args.append("--background")
        if dataset is not None:
            texts = list(spoc if dataset == "spoc" else gsm)
            random.Random(seed).shuffle(texts)
            name = asset(f"resource{338 if dataset == 'spoc' else 339}-{dataset}-order{seed}.json", texts)
            for key, value in (("--request-file", name), ("--holdout-file", sh if dataset == "spoc" else gh),
                               ("--seed", seed), ("--ar-check-requests", len(texts))):
                replace(args, key, value)
            case.update(seed=seed, prompts_sha256=sha(assets[name]),
                        resource_repeat_group="threecard-" + dataset)
        case["purpose"] = ("Fixed numeric recipe, head128/fresh Adam, three physical A2: TP2 target/draft plus "
                           "independent learner. AR/fixed/continuous reserve the same three cards; final published "
                           "seen replay and heldout retained. Same previously reserved cohort, not new problems or "
                           "a substitute for two-card proof. No tuning, no favorable-noise retry or performance-based stop.")
        return case

    bgate = resource_case(b_spoc_plan[0], "b167-v292-resource338-spoc-c1024-3card-gate", "B", "resource338B-preflight.json")
    b744 = resource_case(b_spoc_plan[1], "b167-v292-resource338-spoc256-seed744", "B", "resource338B-preflight.json", "spoc", 744)
    b744["requires"].append(dict(file=bgate["tag"] + ".json", greedy=True, audit=True))
    a745 = resource_case(a_spoc, "a111-v292-resource338-spoc256-seed745", "A", "resource338A-preflight.json", "spoc", 745)
    a745["requires"].append(dict(file="a111-v292-spoc333-aux3card-confirmed256-order743.json", greedy=True))

    agate = resource_case(old_three[0], "a111-v248-resource339-gsm3card-gate", "A", "resource339A-preflight.json")
    agate["launcher"] = "run_dense_v248_docker.py"
    for key, name in (("--request-file", "resource339-gsm-smoke6.json"),
                      ("--holdout-file", "resource339-gsm-smoke-holdout.json")):
        replace(agate["args"], key, name)
    agate["requires"].append(dict(file=a745["tag"] + ".json", greedy=True))
    a_gsm = resource_case(old_gsm, "a111-v248-resource339-gsm384-seed744", "A", "resource339A-preflight.json", "gsm", 744)
    a_gsm["launcher"] = "run_dense_v248_docker.py"
    a_gsm["requires"].append(dict(file=agate["tag"] + ".json", greedy=True, audit=True))
    b_gsm = []
    for seed in (743, 745):
        case = resource_case(old_gsm, f"b167-v308-resource339-gsm384-seed{seed}", "B", "resource339B-preflight.json", "gsm", seed)
        case["launcher"] = "run_dense_v308.py"
        case["requires"].append(dict(file="../online-studies/" + b744["tag"] + ".json", greedy=True))
        case["requires"].append(dict(file=old_three[0]["tag"] + ".json", greedy=True, audit=True))
        if b_gsm:
            case["requires"].append(dict(file=b_gsm[-1]["tag"] + ".json", greedy=True))
        b_gsm.append(case)
    for case in (agate, a_gsm, *b_gsm):
        host = "A" if case["tag"].startswith("a111") else "B"
        case.update(expected_source_manifest=f"resource339-gsm-{host}-source.json",
                    expected_base_manifest="spoc312-base-files.json" if host == "A" else "resource339-gsm-B-base-files.json",
                    expected_initial_head="kl-prewarm-suffix-tp2",
                    expected_initial_head_shards=spoc_freeze["initial_head_shards"])
        if host == "A":
            case.update(expected_image_tag="quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc",
                        expected_image_id=spoc_freeze["expected_image_id"])
    return dict(A=dict(cases=[a745, agate, a_gsm]), B_spoc=dict(cases=[bgate, b744]),
                B_gsm=dict(cases=b_gsm)), assets, dict(
        registered_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="registered_not_dispatched", model_code_changed=False, numerical_recipe_changed=False,
        existing_spoc743="a111-v292-spoc333-aux3card-confirmed256-order743",
        extension_registered_after_first_threecard_case_started=True,
        first_case_was_already_observed_partially=True,
        original_primary_reserved_cohorts_already_assigned=True,
        new_problem_cohort=False, independent_head_and_Adam_reset_per_trial=True,
        GSM145_NN_modules_and_benchmark_equal_to_primary=True,
        gsm_numeric_contract={k: primary_gsm[k] for k in contract_keys},
        desired_seeds=dict(spoc=[743, 744, 745], gsm=[743, 744, 745]),
        criterion="Every complete trial and pooled throughput > matched AR and initial fixed by5%; all costs, no capped updates.",
        scope="Resource-layout repeats of the same frozen numerical recipes and previously reserved cohorts. "
              "Record three-card separately from primary two-card confirmations; never claim new globally unseen problems. "
              "First three-card SPoC case is already running; extension timing is explicit, not retrospectively hidden.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    output = args.validation_root / "resource338"
    assert not (output / "registration.json").exists(), "Already registered; do not reselect or overwrite"
    plans, assets, registration = build(args.validation_root)
    (output / "assets").mkdir(exist_ok=True)
    for name, content in assets.items():
        (output / "assets" / name).write_bytes(content)
    for host, plan in plans.items():
        (output / f"queue-{host}.json").write_text(json.dumps(plan, indent=2) + "\n")
    registration["assets"] = {name: sha(content) for name, content in assets.items()}
    registration["plans"] = {host: sha((json.dumps(plan, indent=2) + "\n").encode()) for host, plan in plans.items()}
    (output / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    print(json.dumps(registration))
