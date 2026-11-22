"""仅检查确认流程的拒绝条件和分配文件；不运行NPU或模型。"""
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from common import change, check_runtime_contract, comparable, module, qualified, read, sha


def main():
    q = Path(__file__).resolve().parent
    root = q.parent
    for p in q.glob("*.py"):
        compile(p.read_text(), str(p), "exec")
    tree = ast.parse((q / "dispatch.py").read_text())
    prefix = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "prefix" for t in n.targets))
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        assigned = dict(schema="orthrus-reserved-requests-v1", assigned_input_sha256={"test.json": "123"})
        for _ in range(2):
            exec(prefix, {"P": dict(root=str(temp), assignment=assigned)})
            assert read(temp / "confirm350-assignment.json") == assigned
        try:
            exec(prefix, {"P": dict(root=str(temp), assignment=assigned | dict(changed=True))})
        except AssertionError:
            pass
        else:
            raise AssertionError("A changed assignment overwrote the frozen one")
        fixture = temp / "fixture"
        (fixture / "confirm350").mkdir(parents=True)
        (fixture / "sync345").mkdir()
        for name in ("rows255", "schedule326"):
            (fixture / name).symlink_to(root / name, target_is_directory=True)
        template_raw = (root / "sync345/queue-B.json").read_bytes()
        (fixture / "sync345/queue-B.json").write_bytes(template_raw)
        tag = "guard-fixture"
        (fixture / "confirm350/registration.json").write_text(json.dumps(
            dict(template_sha256=sha(template_raw), qualification_tag=tag)))
        original = read(root / "gsm254/confirmations/a111-v248-gsm254-confirm743.json")
        schedule = module(root / "schedule326/audit_schedule.py", "selfcheck_schedule350")
        events = schedule.audit(original, 128)

        def reject(data, expected_text=None):
            raw = json.dumps(data).encode()
            (fixture / "sync345" / (tag + ".json")).write_bytes(raw)
            proof = dict(completed=True, accounting_passed=True, resources_checked=True,
                         frozen_source_head_assets_base_checked=True, raw_sha256=sha(raw),
                         functional_only=False, performance_qualification=True, actual_events=events)
            (fixture / "sync345" / (tag + ".audit.json")).write_text(json.dumps(proof))
            try:
                qualified(fixture)
            except (AssertionError, ValueError) as error:
                if expected_text is not None:
                    assert expected_text in str(error), error
            else:
                raise AssertionError("A false qualification was accepted")

        reject(original)  # 真实历史双卡流不能冒充三卡资格。
        reject(original | dict(completed=False), "run incomplete")
        reject(original | dict(max_updates=64), "updates were capped")
    args = read(root / "sync345/queue-B.json")["cases"][0]["args"]
    changed = list(args)
    for flag, value in (("--devices", "0,1,2,3,4,5,6,7"), ("--seed", 744),
                        ("--request-file", "new.json"), ("--holdout-file", "new-holdout.json")):
        changed = change(changed, flag, value)
    assert comparable(changed) == comparable(args)
    assert comparable(change(changed, "--learning-rate", 1e-4)) != comparable(args)
    # 真实冻结版本的完整报告没有两个较新版本才增加的顶层字段。
    reference = read(root / "sync345/b167-v308-sync345-gsm384-explore.json")
    assert "max_model_len" not in reference and "training_rows" not in reference
    check_runtime_contract(reference, reference)
    for field in ("kv_pages", "trainable_parameters", "training_dispatches", "paired_physical_rows"):
        mutated = copy.deepcopy(reference)
        mutated[field] += 1
        try:
            check_runtime_contract(mutated, reference)
        except AssertionError as error:
            assert str(error) == field
        else:
            raise AssertionError(f"Changed runtime layout accepted: {field}")
    mutated = copy.deepcopy(reference)
    mutated["resource_policy"]["max_rows"] += 1
    try:
        check_runtime_contract(mutated, reference)
    except AssertionError as error:
        assert str(error) == "max_rows"
    else:
        raise AssertionError("Changed maximum rows accepted")
    # 只在临时目录验证数据编排；假资格/语料均不用于任何真实实验。
    builder = module(q / "prepare.py", "prepare_fixture350")
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        fq = fixture / "confirm350"
        fq.mkdir()
        (fixture / "gsm254").mkdir()

        def save(path, value):
            path.write_text(json.dumps(value) + "\n")

        save(fq / "registration.json", read(q / "registration.json"))
        questions = ["Synthetic metadata fixture question " + str(i) for i in range(1024)]
        save(fixture / "gsm-reserved1024.json", questions)
        save(fixture / "gsm-reserved1024-manifest.json", dict(
            source="synthetic fixture", source_sha256="no real corpus", indices=list(range(1024)),
            sha256=sha((fixture / "gsm-reserved1024.json").read_bytes())))
        save(fixture / "gsm254/gsm254-confirm384-manifest.json", dict(indices=list(range(2000, 2384))))
        save(fixture / "gsm254/gsm254-holdout32-manifest.json", dict(indices=list(range(2384, 2416))))
        save(fixture / "gsm254/gsm254-unused-reserve-audit.json", dict(parent128_conservatively_excluded_IDs=[]))
        fake_qualification = dict(template=read(root / "sync345/queue-B.json")["cases"][0], scope="CPU fixture only")

        def fake_audit(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            save(output, dict(completed=True, snapshot_eligible_indices=list(range(592)),
                              hosts={h: dict(unknown_assets=[], read_errors=[], observed_UTC=datetime.now(timezone.utc).isoformat())
                                     for h in ("159.138.5.111", "189.1.227.167", "159.138.54.61")}))

        with patch.object(builder, "__file__", str(fq / "prepare.py")), \
             patch.object(builder, "qualified", return_value=fake_qualification) as guard, \
             patch.object(builder.subprocess, "run", side_effect=fake_audit) as audit:
            frozen = builder.prepare()
            assert builder.prepare() == frozen
            assert guard.call_count == audit.call_count == 1
        cases = read(fq / "queue-B.json")["cases"]
        assert len(cases) == 3 and all(c["max_attempts"] == 1 for c in cases)
        for i, case in enumerate(cases):
            assert comparable(case["args"]) == comparable(args)
            assert not any(k in case for k in ("confirm", "unless_achieved", "confirmation_group"))
            if i:
                assert case["requires"] == [dict(file=cases[i-1]["tag"] + ".json", greedy=True)]
        assignment = read(fq / "assignment.json")
        assert len(assignment["confirmation_indices"]) == 384 and len(assignment["independent_holdout_indices"]) == 32
        assert not set(assignment["confirmation_indices"]) & set(assignment["independent_holdout_indices"])
    proof = dict(passed=True, observed_UTC=datetime.now(timezone.utc).isoformat(),
                 assignment_JSON_roundtrip_and_idempotence=True, changed_assignment_rejected=True,
                 partial_stream_rejected=True, capped_update_strategy_rejected=True,
                 two_card_proof_cannot_qualify_three_cards=True,
                 temporary_synthetic_preparation_and_reinvocation_checked=True,
                 all_three_valid_performance_trials_have_no_skip_or_retry=True,
                 numerical_recipe_mutation_detected=True, NPU_or_model_calls=False,
                 actual_v308_report_schema_checked=True, runtime_layout_mutations_rejected=True,
                 scope="CPU guard checks only; does not prove new NPU performance or confirmation success.")
    (q / "selfcheck.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
