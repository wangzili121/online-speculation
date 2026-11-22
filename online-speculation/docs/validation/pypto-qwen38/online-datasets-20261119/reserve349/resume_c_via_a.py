"""保留首次失败快照，经111补读61；不重跑模型，不分配确认题。"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zlib


def main():
    q = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", type=Path, default=q / "current-usage-snapshot.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old = json.loads(args.partial.read_text())
    assert not old["completed"]
    assert all(old["hosts"][h]["completed"] for h in ("159.138.5.111", "189.1.227.167"))
    source = q / "audit_usage.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == old["scanner_sha256"]
    spec = importlib.util.spec_from_file_location("usage349", source)
    usage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(usage)
    root = q.parent
    inventory_path = root / "reserve347/known-history-capacity.json"
    assert hashlib.sha256(inventory_path.read_bytes()).hexdigest() == old["recorded_inventory_sha256"]
    inventory = json.loads(inventory_path.read_text())
    for name, expected in inventory["source_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    pool = json.loads((root / "gsm-reserved1024.json").read_text())
    metadata = json.loads((root / "gsm-reserved1024-manifest.json").read_text())
    remaining = set(inventory["remaining_indices_in_registered_order"])
    candidates = [dict(index=i, normalized=" ".join(text.split()),
                       sha256=hashlib.sha256(text.encode()).hexdigest(),
                       normalized_sha256=hashlib.sha256(" ".join(text.split()).encode()).hexdigest())
                  for i, text in zip(metadata["indices"], pool) if i in remaining]
    assert len(candidates) == old["candidates"]
    payload = dict(base="/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118",
                   required_roots=["online-tp2-test"], candidates=candidates)
    code = "P=" + repr(payload) + "\n" + usage.REMOTE
    packed = base64.b64encode(zlib.compress(code.encode())).decode()
    wrapped = "import base64,zlib;exec(zlib.decompress(base64.b64decode(" + repr(packed) + ")))"
    call = subprocess.run(
        ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
         "-J", "wangzili@159.138.5.111", "wangzili@159.138.54.61", "python3 -"],
        input=wrapped.encode(), capture_output=True, timeout=180,
    )
    if call.returncode:
        failure = dict(completed=False, returncode=call.returncode,
                       stderr=call.stderr.decode(), NN_restarted=False, fresh_cohort_assigned=False)
        args.output.write_text(json.dumps(failure, indent=2) + "\n")
        raise RuntimeError(call.stderr.decode()[-2000:])
    value = json.loads(zlib.decompress(call.stdout))
    value.update(scan_progress=call.stderr.decode(), transport="SSH ProxyJump via 159.138.5.111",
                 remote_scanner_sha256=hashlib.sha256(usage.REMOTE.encode()).hexdigest())
    node = args.output.with_name(args.output.stem + "-159.138.54.61.json")
    with node.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    hosts = dict(old["hosts"])
    hosts["159.138.54.61"] = value
    complete = all(v["completed"] for v in hosts.values())
    exposed = {i for v in hosts.values() for i in v["candidate_hash_hits"]}
    exposed.update(row["index"] for v in hosts.values() for row in v["candidate_substring_hits"])
    out = old | dict(completed=complete, hosts=hosts, observed_UTC=datetime.now(timezone.utc).isoformat(),
                     exposed_candidate_indices=sorted(exposed),
                     snapshot_eligible_indices=[i for i in inventory["remaining_indices_in_registered_order"] if i not in exposed] if complete else None,
                     prior_incomplete_snapshot_sha256=hashlib.sha256(args.partial.read_bytes()).hexdigest(),
                     reused_A_B_observation_times={h: hosts[h]["observed_UTC"] for h in hosts if h != "159.138.54.61"},
                     all_host_observations_simultaneous=False,
                     recheck_before_assignment_required=True,
                     fresh_cohort_assigned=False, new_NN_requests=False)
    with args.output.open("x") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    print(json.dumps(dict(completed=complete, snapshot_eligible=len(out["snapshot_eligible_indices"]) if complete else None,
                          C_hash_hits=len(value["candidate_hash_hits"]), C_substring_hits=len(value["candidate_substring_hits"]),
                          unknown_inputs=len(value["unknown_assets"]), no_assignment=True)))


if __name__ == "__main__":
    main()
