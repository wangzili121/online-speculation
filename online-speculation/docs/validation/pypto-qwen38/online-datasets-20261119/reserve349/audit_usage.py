"""只读核对剩余GSM题的当前使用情况；不分配确认题，不调用模型。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zlib


REMOTE = r'''
from pathlib import Path
import datetime,hashlib,json,os,shlex,subprocess,sys,time,zlib
sha=lambda raw:hashlib.sha256(raw).hexdigest()
base=Path(P['base']);errors=[];unknown=[];roots=[];used=set();hits=[];seen_texts=set()
active_plans=set();active_launchers={}
def read(path,include_raw=False):
 for attempt in range(3):
  try:
   raw=path.read_bytes();value=json.loads(raw)
   return(value,raw)if include_raw else value
  except json.JSONDecodeError as error:
   if attempt==2:errors.append(dict(path=str(path),error=repr(error)))
   else:time.sleep(.03)
  except OSError as error:
   errors.append(dict(path=str(path),error=repr(error)));break
 return None
if not P.get('test_mode'):
 for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
  if not any(n in line for n in ('study_queue_','run_dense_v')):continue
  try:
   pid,command=line.split(None,1);words=shlex.split(command);cwd=(Path('/proc')/pid/'cwd').resolve()
   if not cwd.is_relative_to(base):continue
   if 'study_queue_'in command and '--plan'in words:
    value=Path(words[words.index('--plan')+1]);active_plans.add((cwd/value).resolve())
   if 'run_dense_v'in command and '--tag'in words:
    active_launchers.setdefault(str(cwd),[]).append((words[words.index('--tag')+1],words))
  except (ValueError,OSError,IndexError):continue
else:
 active_plans={Path(p).resolve()for p in P.get('active_plans',[])}
def consume(label,text):
 normal=' '.join(text.split());used.update((sha(text.encode()),sha(normal.encode())))
 if normal in seen_texts:return
 seen_texts.add(normal)
 for row in P['candidates']:
  if row['normalized']in normal:
   hits.append(dict(index=row['index'],asset=label,kind='normalized full question in executed or assigned input'))
for name in P['required_roots']:
 if not(base/name).is_dir():errors.append(dict(path=str(base/name),error='Required experiment root missing'))
for r in sorted(p for p in base.glob('online-*')if p.is_dir()and(p/'assets').is_dir()):
 print(json.dumps(dict(stage='reading-root',root=str(r))),file=sys.stderr,flush=True)
 assets={};by_hash={};names=set();tags=set();reports=[];plans=[]
 for path in sorted((r/'assets').glob('*.json')):
  record=read(path,True)
  if record is None:continue
  d,raw=record
  if isinstance(d,list)and d and all(isinstance(t,str)for t in d):
   item=dict(sha256=sha(raw),texts=d)
   assets[path.name]=item;by_hash.setdefault(item['sha256'],[]).append(path.name)
 def from_args(tag,args,origin):
  for flag in ('--request-file','--holdout-file','--prompts','--holdout'):
   if flag not in args:continue
   try:value=args[args.index(flag)+1]
   except IndexError:unknown.append(dict(tag=tag,flag=flag,origin=origin,reason='Missing value'));continue
   path=Path(value);relative=path.name
   if relative not in assets:
    unknown.append(dict(root=str(r),tag=tag,flag=flag,asset=value,origin=origin,reason='Referenced text asset missing'));continue
   names.add(relative)
 files={p:read(p)for p in sorted(r.glob('*.json'))}
 print(json.dumps(dict(stage='matching-inputs',root=str(r),assets=len(assets),metadata_files=len(files))),file=sys.stderr,flush=True)
 for path,d in files.items():
  if not isinstance(d,dict):continue
  if path.name.startswith('queue')and isinstance(d.get('cases'),list):
   for c in d['cases']:
    if not isinstance(c,dict)or not isinstance(c.get('args'),list)or not isinstance(c.get('tag'),str):continue
    tag=c['tag'];assigned=path.resolve()in active_plans
    if assigned or(r/(tag+'.source.json')).exists()or(r/(tag+'.json')).exists():
     tags.add(tag);plans.append(path.name);from_args(tag,c['args'],'live-assignment'if assigned else'launched-source-or-report')
  if isinstance(d.get('prompts_sha256'),str)and('stage'in d or'results'in d):
   reports.append(path.name);tags.add(path.stem)
   for key in ('prompts_sha256','holdout_sha256'):
    value=d.get(key)
    if not value:continue
    if value not in by_hash:unknown.append(dict(root=str(r),tag=path.stem,field=key,sha256=value,reason='Actual report input hash not found'))
    else:names.update(by_hash[value])
 for tag,args in active_launchers.get(str(r),[]):
  tags.add(tag);from_args(tag,args,'live-launcher')
 for name in sorted(names):
  for text in assets[name]['texts']:consume(str(r/'assets'/name),text)
 roots.append(dict(root=str(r),launched_or_assigned_tags=len(tags),actual_reports=len(reports),
                   used_assets=[dict(file=name,sha256=assets[name]['sha256'])for name in sorted(names)],
                   active_plan_files=sorted(str(p)for p in active_plans if p.parent==r),
                   source_or_live_plan_files=sorted(set(plans))))
parent=base.parent/'online-study-20261111/kl-answer-last4-k32-128-r1.json'
def strings(value):
 if isinstance(value,str):consume(str(parent),value)
 elif isinstance(value,list):
  for x in value:strings(x)
 elif isinstance(value,dict):
  for x in value.values():strings(x)
if parent.exists():strings(read(parent))
hash_hits=sorted({row['index']for row in P['candidates']if row['sha256']in used or row['normalized_sha256']in used})
out=dict(completed=not unknown and not errors,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         base=str(base),roots=roots,unknown_assets=unknown,read_errors=errors,
         used_question_sha256=sorted(used),candidate_substring_hits=hits,candidate_hash_hits=hash_hits,
         parent128_report_present=parent.exists(),parent128_report_sha256=sha(parent.read_bytes())if parent.exists()else None,
         no_NPU_calls=True,no_assignment=True)
sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-timeout", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parent.parent
    inventory_path = root / "reserve347/known-history-capacity.json"
    inventory = json.loads(inventory_path.read_text())
    for name, expected in inventory["source_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    pool_path = root / "gsm-reserved1024.json"
    metadata = json.loads((root / "gsm-reserved1024-manifest.json").read_text())
    pool = json.loads(pool_path.read_text())
    assert hashlib.sha256(pool_path.read_bytes()).hexdigest() == metadata["sha256"]
    assert len(pool) == len(metadata["indices"]) == 1024
    remaining = set(inventory["remaining_indices_in_registered_order"])
    candidates = [dict(index=i, normalized=" ".join(text.split()),
                       sha256=hashlib.sha256(text.encode()).hexdigest(),
                       normalized_sha256=hashlib.sha256(" ".join(text.split()).encode()).hexdigest())
                  for i, text in zip(metadata["indices"], pool) if i in remaining]
    assert len(candidates) == len(remaining) == inventory["remaining_known_history_capacity"]
    base = "/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118"
    jobs = [("159.138.5.111", ["online-studies"]),
            ("189.1.227.167", ["online-background", "online-studies"]),
            ("159.138.54.61", ["online-tp2-test"])]

    def scan(job):
        address, required = job
        payload = dict(base=base, required_roots=required, candidates=candidates)
        try:
            call = subprocess.run(
                ["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=10", "wangzili@" + address, "python3 -"],
                input=("P=" + repr(payload) + "\n" + REMOTE).encode(),
                capture_output=True, timeout=args.host_timeout,
            )
            if call.returncode:
                raise RuntimeError(call.stderr.decode()[-3000:])
            value = json.loads(zlib.decompress(call.stdout))
            value["scan_progress"] = call.stderr.decode()
        except (subprocess.SubprocessError, RuntimeError, ValueError, zlib.error) as error:
            value = dict(completed=False, observation_error=repr(error), terminal=False,
                         observed_UTC=datetime.now(timezone.utc).isoformat(), roots=[], unknown_assets=[],
                         read_errors=[], candidate_hash_hits=[], candidate_substring_hits=[],
                         NN_restarted=False)
            if isinstance(error, subprocess.TimeoutExpired):
                value["scan_progress"] = (error.stderr or b"").decode(errors="replace")
        node_path = args.output.with_name(args.output.stem + "-" + address + ".json")
        with node_path.open("x") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        print(json.dumps(dict(host=address, completed=value["completed"], roots=len(value["roots"]),
                              unknown=len(value["unknown_assets"]), read_errors=len(value["read_errors"]),
                              hash_hits=len(value["candidate_hash_hits"]),
                              substring_hits=len(value["candidate_substring_hits"]))), flush=True)
        return address, value

    with ThreadPoolExecutor(max_workers=3) as executor:
        hosts = dict(executor.map(scan, jobs))
    complete = all(value["completed"] for value in hosts.values())
    exposed = {i for value in hosts.values() for i in value["candidate_hash_hits"]}
    exposed.update(row["index"] for value in hosts.values() for row in value["candidate_substring_hits"])
    result = dict(observed_UTC=datetime.now(timezone.utc).isoformat(),
                   completed=complete, hosts=hosts,
                   recorded_inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                   scanner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   candidates=len(candidates), exposed_candidate_indices=sorted(exposed),
                   snapshot_eligible_indices=[i for i in inventory["remaining_indices_in_registered_order"] if i not in exposed] if complete else None,
                   initial_head128_exclusions_inherited_from_inventory=True,
                   fresh_cohort_assigned=False, new_NN_requests=False,
                   scope="Snapshot of this project's known OneServing study roots on three hosts, live assigned queues, "
                         "launched inputs/holdouts and recorded parent128. No claim about base pretraining. "
                         "Recheck before assignment after recipe qualification; unknown inputs invalidate completeness.")
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(dict(completed=result["completed"], candidates=len(candidates),
                          snapshot_eligible=len(result["snapshot_eligible_indices"]) if complete else None,
                          no_assignment=True)))


if __name__ == "__main__":
    main()
