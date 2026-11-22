"""收集同步独立学习卡功能测试；只读，不启动或重启模型。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import traceback
import zlib


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def collect(root, host, suite="sync344", case_tag=None):
    q = root / suite
    watch = module(root / "resource338/watch.py", "resource_watch_sync344")
    accounting = module(root / "rows255/audit_accounting.py", "accounting_sync344")
    schedule = module(root / "schedule326/audit_schedule.py", "schedule_sync344")
    launch = json.loads((q / f"launch-{host}.json").read_text())
    cases = json.loads((q / f"queue-{host}.json").read_text())["cases"]
    case = next(c for c in cases if c["tag"] == case_tag) if case_tag is not None else cases[0]
    tag = case["tag"]
    proof_path = q / (tag + ".audit.json")
    # 不重复读取已归档的大权重；失败证据也保留，不能自动重试模型。
    if proof_path.exists():
        return json.loads(proof_path.read_text())
    address = "159.138.5.111" if host == "A" else "189.1.227.167"
    binding_path = q / (tag + ".resources.json")
    if not binding_path.exists():
        capture = module(root / "resource338/capture_resources.py", "capture_sync344")
        binding_payload = dict(root=launch["root"], tag=tag, host=address, launcher=case["launcher"])
        observation = subprocess.run(
            ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=10", "wangzili@" + address, "python3 -"],
            input=("P=" + repr(binding_payload) + "\n" + capture.REMOTE).encode(),
            capture_output=True, timeout=40,
        )
        if observation.returncode:
            raise RuntimeError(observation.stderr.decode()[-2000:])
        binding = json.loads(observation.stdout)
        with (q / (tag + ".resource-observations.jsonl")).open("a") as history:
            history.write(json.dumps(binding) + "\n")
        if binding.get("captured"):
            write_json(binding_path, binding)
    payload = launch | dict(cases=[case])
    call = subprocess.run(
        ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=10", "wangzili@" + address, "python3 -"],
        input=("P=" + repr(payload) + "\n" + watch.REMOTE).encode(),
        capture_output=True, timeout=60,
    )
    if call.returncode:
        raise RuntimeError(call.stderr.decode()[-2000:])
    response = json.loads(zlib.decompress(call.stdout))
    item = response["cases"][0]
    if not item["terminal"]:
        return dict(host=host, tag=tag, terminal=False,
                    controller_live=response["controller_live"], stage=item["stage"])
    for name, text in item.get("files", {}).items():
        raw = text.encode()
        if name.endswith(".log") or len(raw) > 2 << 20:
            (q / (name + ".gz")).write_bytes(gzip.compress(raw, mtime=0))
        else:
            (q / name).write_bytes(raw)
    try:
        stage = json.loads((root / "resource338" / f"stage-{host}.json").read_text())
        source = stage["source_manifests"][Path(launch["root"]).name][case["expected_source_manifest"]]
        base = stage["public_base"] if host == "A" else stage["host_model_view"]
        protocol = json.loads((q / "protocol.json").read_text())
        proof = watch.audit(case, item, source, base, protocol["files_sha256"],
                            case["expected_initial_head_shards"], accounting, schedule,
                            layout="interleaved_ar_frozen_online", background=False)
        gate = "--audit-replay" in case["args"]
        binding = json.loads(binding_path.read_text()) if binding_path.exists() else None
        if binding:
            raw = json.loads(item["files"][tag + ".json"])
            assert binding["captured"] and binding["physical_cards"] == list(map(int, raw["visible_devices"].split(",")))
            assert {v["physical_npu"] for v in binding["workers"]} == set(binding["physical_cards"])
        assert gate or binding, "No live physical worker binding captured for performance run"
        proof.update(functional_only=gate,
                     performance_qualification=not gate and proof["online_vs_ar"] > .05 and proof["online_vs_fixed"] > .05,
                     resources_checked=binding is not None,
                     fresh_reserved_questions_used=protocol.get("fresh_reserved_questions_used", False), terminal=True)
        if "evidence_scope" in protocol:
            proof["scope"] = protocol["evidence_scope"]
    except Exception as error:
        proof = dict(completed=False, terminal=True, tag=tag, error=repr(error),
                     traceback=traceback.format_exc(), model_restarted=False,
                     controller_tail=item.get("controller_tail"))
    write_json(proof_path, proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    q = args.suite_root.resolve()
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = []
        futures = [(h, pool.submit(collect, q.parent, h, q.name)) for h in ("A", "B")]
        for host, future in futures:
            try:
                rows.append(future.result())
            except (subprocess.TimeoutExpired, RuntimeError, ValueError, zlib.error) as error:
                rows.append(dict(host=host, observation_error=repr(error), terminal=False,
                                 model_restarted=False))
    out = dict(observed_UTC=datetime.now(timezone.utc).isoformat(), results=rows,
               completed=all(v.get("terminal") for v in rows),
               all_functional_passed=all(v.get("completed") and v.get("functional_only") for v in rows),
               all_performance_qualified=all(v.get("completed") and v.get("performance_qualification") for v in rows),
               no_NN_dispatch=True)
    (q / "collection-state.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
