"""收集固定三卡重复的完整证据；失败和负结果均保留，不派发或调参。"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib.util
import json
import math
import subprocess
import time
import traceback
import zlib


REMOTE = r'''
from pathlib import Path
import json,hashlib,subprocess,sys,zlib
r=Path(P['root']);p=Path('/proc')/str(P['pid'])/'cmdline';command=p.read_bytes().replace(bytes([0]),b' ').decode()if p.exists()else''
live=P['controller']in command and P['plan']in command
status_path=r/Path(P['plan']).with_suffix('.status.json');state=json.loads(status_path.read_text())if status_path.exists()else{}
out=dict(controller_live=live,status=state,cases=[])
for c in P['cases']:
 tag=c['tag'];p=r/(tag+'.json');d=json.loads(p.read_text())if p.exists()else{};entry=next((v for v in state.get('cases',[])if v['tag']==tag),{})
 terminal=bool(entry.get('completed')or entry.get('error')or(not live))
 item=dict(tag=tag,terminal=terminal,completed=bool(entry.get('completed')and d.get('completed')and d.get('clean_shutdown')),error=entry.get('error')or d.get('error'),stage=d.get('stage'))
 if terminal:
  item['files']={tag+ext:(r/(tag+ext)).read_text()for ext in ['.json','.source.json','.log']if(r/(tag+ext)).exists()}
  if not item['completed']:item['controller_tail']=(r/(P['plan']+'.controller.log')).read_text()[-3000:]
  else:
   on=next(v for v in d['results'].values()if'updates'in v);export=on['exported_head'];item['actual_shards']={}
   paths=[Path(export['path'])/part['name']for part in export['manifest']['rank_files']]
   try:
    for path in paths:
     h=hashlib.sha256()
     with path.open('rb')as f:
      for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
     item['actual_shards'][path.name]=h.hexdigest()
   except PermissionError:
    # Docker 导出的文件可能仅 root 可读；在无设备、无网络、只读挂载中核对。
    relative=[str(path.relative_to(r))for path in paths]
    code='import hashlib,json,pathlib,sys; p=[pathlib.Path("/evidence")/x for x in json.loads(sys.argv[1])]; print(json.dumps({x.name:hashlib.file_digest(x.open("rb"),"sha256").hexdigest() for x in p}))'
    item['actual_shards']=json.loads(subprocess.check_output(['docker','run','--rm','--network','none','--read-only','--user','0:0','--volume',str(r)+':/evidence:ro','--entrypoint','python3',c['expected_image_id'],'-c',code,json.dumps(relative)],text=True))
    item['shard_read_mode']='read-only container; no NPU devices or network'
   b=json.loads((r/c['expected_base_manifest']).read_text());model=Path(b['model'])
   item['current_base']=dict(model=str(model),metadata_sha256={n:hashlib.sha256((model/n).read_bytes()).hexdigest()for n in b['metadata_sha256']},weight_files_stat={})
   for n in b['weight_files_stat']:
    st=(model/n).stat();item['current_base']['weight_files_stat'][n]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
   if c.get('expected_image_id'):item['current_image_id']=json.loads(subprocess.check_output(['docker','image','inspect',c['expected_image_tag']],text=True))[0]['Id']
 out['cases'].append(item)
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(case, item, source, base, asset_hashes, head_shards, accounting, schedule):
    assert item["completed"] and not item.get("error"), item.get("error")
    tag = case["tag"]
    raw = item["files"][tag + ".json"].encode()
    d = json.loads(raw)
    args = case["args"]
    option = lambda flag, default=None: args[args.index(flag) + 1] if flag in args else default
    stride = int(option("--strides"))
    gate = "--audit-replay" in args
    result = accounting.audit(d, stride, layout="separate_streams", allow_replay=gate)
    assert d["physical_device_count"] == len(set(d["visible_devices"].split(","))) == 3
    assert d["device_ids"] == [0, 1] and d["learner_device"] == 2 and d["background"] is True
    assert d["head_step"] == 128 and d["head_shard_sha256"] == head_shards and d["optimizer_init"] == "fresh"
    assert d["requests"] == d["ar_checked_requests"] == int(option("--ar-check-requests"))
    fields = [("--tokens", "output_tokens_per_request", int, None),
              ("--last-layers", "last_layers", int, 1), ("--publish-every", "publish_every", int, 4),
              ("--warm-update-stride", "warm_update_stride", int, None),
              ("--warm-update-steps", "warm_update_steps", int, 0), ("--warmup-updates", "warmup_updates", int, 0),
              ("--learning-rate", "learning_rate", float, 1e-5), ("--objective", "objective", str, "tv"),
              ("--teacher-tokens", "teacher_tokens", int, 31), ("--draft-width", "draft_width", int, 32),
              ("--candidates", "candidates", int, 31), ("--seed", "seed", int, 743),
              ("--teacher-targets", "teacher_targets", str, "distribution"),
              ("--teacher-temperature", "teacher_temperature", float, 1),
              ("--teacher-scaling", "teacher_scaling", str, "host"), ("--feedback-mode", "feedback_mode", str, "delayed"),
              ("--feedback-timing", "feedback_timing", str, "early"),
              ("--draft-convolution", "draft_convolution", str, "sequential"),
              ("--replay-blocks", "replay_blocks", int, 4), ("--inner-steps", "inner_steps", int, 1),
              ("--lr-schedule", "lr_schedule", str, "constant"),
              ("--parameter-scope", "parameter_scope", str, "full"), ("--parameterization", "parameterization", str, "dense"),
              ("--temperature", "temperature", float, 0), ("--publication-backend", "publication_backend", str, "auto")]
    for flag, key, cast, default in fields:
        value = option(flag, default)
        assert d[key] == (cast(value) if value is not None else None), (tag, key, d.get(key), value)
    for flag, key in (("--training-rows", "training_rows"), ("--max-model-len", "max_model_len")):
        if flag in args:
            assert d[key] == int(option(flag))
    assert d["stop_on_eos"] == ("--stop-on-eos" in args) and d["gdn_valid_rows"] is True
    assert d["paired_physical_rows"] == 16
    for flag, key in (("--request-file", "prompts_sha256"), ("--holdout-file", "holdout_sha256")):
        assert d[key] == asset_hashes[option(flag)]
    assert json.loads(item["files"][tag + ".source.json"])["files"] == source
    assert item["current_base"] == {k: base[k] for k in ("model", "metadata_sha256", "weight_files_stat")}
    if case.get("expected_image_id"):
        assert item["current_image_id"] == case["expected_image_id"]
    for arm in ("ar", "frozen", str(stride)):
        assert all(0 < n <= d["output_tokens_per_request"] for row in d["results"][arm]["request_timeline"] for n in row["output_tokens"])
    on = d["results"][str(stride)]
    manifest = on["exported_head"]["manifest"]
    assert manifest["published_banks_exact"] and manifest["step"] == 128 + on["published_version"]
    assert on["optimizer_version"] == len(on["updates"])
    assert on["published_version"] == (len(on["updates"]) // d["publish_every"]) * d["publish_every"]
    assert item["actual_shards"] == {v["name"]: v["sha256"] for v in manifest["rank_files"]}
    replays = [v for u in on["updates"] for v in u.get("replay_audits", [])]
    if gate:
        assert replays and all(v["served_replay_tv"] <= .01 and v["relative_l2"] <= .01 for v in replays)
        event_proof = None
    else:
        event_proof = schedule.audit(d, stride)
    result.update(completed=True, tag=tag, raw_sha256=hashlib.sha256(raw).hexdigest(),
                  physical_cards=3, independent_learner=True, frozen_source_head_assets_base_checked=True,
                  actual_published_shards=item["actual_shards"], actual_events=event_proof,
                  first_feedback_replays=replays, primary_two_card_proof=False,
                  scope="Three-card resource repeat on an already assigned cohort. Separate-stream timing; "
                        "not new globally unseen problems or a substitute for primary two-card results.")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--artifacts", type=Path, required=True)
    args = p.parse_args()
    D, Q = args.validation_root, args.validation_root / "resource338"
    accounting = load_module(D / "rows255/audit_accounting.py", "accounting338")
    schedule = load_module(D / "schedule326/audit_schedule.py", "schedule338")
    reg = json.loads((Q / "registration.json").read_text())
    hashes = dict(reg["assets"])
    hashes.update({n: v["sha256"] for n, v in json.loads((Q / "fixture-bindings.json").read_text())["fixtures"].items()})
    old_gate = json.loads((D / "relocate318/a111-v292-spoc318B-boundary-mountfix322-metadata323.json").read_text())
    ga = json.loads((D / "relocate318/queue-spoc323B.json").read_text())["cases"][0]["args"]
    for flag, key in (("--request-file", "prompts_sha256"), ("--holdout-file", "holdout_sha256")):
        hashes[ga[ga.index(flag) + 1]] = old_gate[key]
    jobs = []
    for host, address in (("A", "159.138.5.111"), ("B", "189.1.227.167")):
        stage = json.loads((Q / f"stage-{host}.json").read_text())
        for launch in stage["launched"]:
            key = "A" if host == "A" else "B_gsm" if launch["root"].endswith("online-background") else "B_spoc"
            jobs.append(dict(host=address, launch=launch, stage=stage, cases=json.loads((Q / f"queue-{key}.json").read_text())["cases"]))
    cache = {}
    expected_count = sum(len(j["cases"]) for j in jobs)

    def read(job):
        pending = [c for c in job["cases"] if c["tag"] not in cache]
        if not pending:
            return job, None
        payload = job["launch"] | dict(cases=pending)
        call = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
                               "-o", "ConnectTimeout=10", "wangzili@" + job["host"], "python3 -"],
                              input=("P=" + repr(payload) + "\n" + REMOTE).encode(), capture_output=True, timeout=60)
        assert call.returncode == 0, call.stderr.decode()[-2000:]
        return job, json.loads(zlib.decompress(call.stdout))

    while len(cache) < expected_count:
        observations = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(read, job) for job in jobs]
            for future in futures:
                try:
                    job, response = future.result()
                    if response is None:
                        continue
                    observations.append(dict(host=job["host"], root=job["launch"]["root"], controller_live=response["controller_live"],
                                             cases=[{k: v for k, v in item.items() if k not in ("files", "actual_shards", "current_base")}
                                                    for item in response["cases"]]))
                    for item in response["cases"]:
                        if not item["terminal"]:
                            continue
                        tag = item["tag"]
                        for name, content in item.get("files", {}).items():
                            raw = content.encode()
                            if name.endswith(".log") or len(raw) > 2 << 20:
                                (Q / (name + ".gz")).write_bytes(gzip.compress(raw, mtime=0))
                            else:
                                (Q / name).write_bytes(raw)
                        try:
                            case = next(c for c in job["cases"] if c["tag"] == tag)
                            stage = job["stage"]
                            source = stage["source_manifests"][Path(job["launch"]["root"]).name][case["expected_source_manifest"]]
                            base = stage["host_model_view"] if job["launch"]["root"].endswith("online-background") else stage["public_base"]
                            proof = audit(case, item, source, base, hashes, reg["gsm_numeric_contract"]["head_shard_sha256"], accounting, schedule)
                        except Exception as error:
                            proof = dict(completed=False, tag=tag, error=repr(error), traceback=traceback.format_exc(), controller_tail=item.get("controller_tail"))
                        cache[tag] = proof
                        (Q / (tag + ".audit.json")).write_text(json.dumps(proof, indent=2) + "\n")
                        print(json.dumps(proof), flush=True)
                except (subprocess.TimeoutExpired, AssertionError, ValueError, zlib.error) as error:
                    observations.append(dict(observation_error=repr(error), terminal=False, NN_restarted=False))
        state = dict(completed=len(cache) == expected_count, results=list(cache.values()), observations=observations,
                     model_code_changed=False, no_retuning=True, original_two_card_confirmations_unchanged=True)
        (args.artifacts / "resource338-watcher-state.json").write_text(json.dumps(state, indent=2) + "\n")
        if len(cache) < expected_count:
            time.sleep(30)
    (Q / "completed-extension-audit.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
