"""新题确认共用的文件与完整资格检查。"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


def read_bytes(path):
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def read(path):
    return json.loads(read_bytes(path))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def qualified(root):
    q = root / "confirm350"
    registration = read(q / "registration.json")
    template_raw = (root / "sync345/queue-B.json").read_bytes()
    assert sha(template_raw) == registration["template_sha256"]
    template = json.loads(template_raw)["cases"][0]
    tag = registration["qualification_tag"]
    raw = read_bytes(root / "sync345" / (tag + ".json"))
    d = json.loads(raw)
    proof = read(root / "sync345" / (tag + ".audit.json"))
    assert proof["completed"] and proof["accounting_passed"] and proof["resources_checked"]
    assert proof["frozen_source_head_assets_base_checked"] and proof["raw_sha256"] == sha(raw)
    assert not proof["functional_only"] and proof["performance_qualification"]
    accounting = module(root / "rows255/audit_accounting.py", "accounting350")
    schedule = module(root / "schedule326/audit_schedule.py", "schedule350")
    totals = accounting.audit(d, 128)
    events = schedule.audit(d, 128)
    assert events["completed"] and events["actual_schedule_checked"] and events == proof["actual_events"]
    assert totals["online_vs_ar"] > .05 and totals["online_vs_fixed"] > .05
    assert d["requests"] == d["ar_checked_requests"] == 384
    assert d["head_step"] == 128 and d["optimizer_init"] == "fresh"
    assert d["head_shard_sha256"] == template["expected_initial_head_shards"]
    assert d["physical_device_count"] == 3 and d["device_ids"] == [0, 1]
    assert d["learner_device"] == 2 and d["background"] is False
    on = d["results"]["128"]
    assert proof["actual_published_shards"] == {
        v["name"]: v["sha256"] for v in on["exported_head"]["manifest"]["rank_files"]}
    assert len(proof["actual_published_shards"]) == 2
    return dict(template=template, raw_sha256=sha(raw), accounting=totals,
                actual_event_audit=events, actual_published_shards=proof["actual_published_shards"],
                qualification_audit_sha256=sha((root / "sync345" / (tag + ".audit.json")).read_bytes()))


def change(args, name, value):
    result = list(args)
    result[result.index(name) + 1] = str(value)
    return result


def comparable(args):
    result = list(args)
    for name in ("--devices", "--request-file", "--holdout-file", "--seed"):
        result[result.index(name) + 1] = "<trial placement or data>"
    return result


def check_runtime_contract(actual, reference):
    """核对冻结 v308 实际输出的形状/布局字段，源码默认值另由哈希审核。"""
    # 该版本没有 max_model_len、training_rows 顶层字段，不能向报告索取虚构字段。
    fields = ("kv_pages", "trainable_parameters", "training_dispatches",
              "paired_physical_rows", "execution_max_num_seqs", "stream_window",
              "head_payload_sha256")
    for key in fields:
        assert actual[key] == reference[key], key
    assert actual["resource_policy"]["max_rows"] == reference["resource_policy"]["max_rows"], "max_rows"
