"""校验冻结资源重复计划后派发；不修改现有模型代码或正在运行的控制器。"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


REMOTE = r'''
from pathlib import Path
import json,hashlib,subprocess,datetime
S=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def write_same(p,text):
 if p.exists():assert p.read_text()==text,('Existing file differs',str(p))
 else:
  with p.open('x')as f:f.write(text)
def binding(model,names,weights):
 return dict(model=str(model),metadata_sha256={n:digest(model/n)for n in names},weight_files_stat={n:dict(size=(st:=(model/n).stat()).st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)for n in weights})
record=S/P['record_name']
assert not record.exists(),'Already dispatched; inspect original handles instead of starting duplicates'
head=S/'kl-prewarm-suffix-tp2'
assert [digest(head/f'rank{i}.safetensors')for i in range(2)]==P['initial_head_shards']
public_binding=json.loads((S/'online-studies'/('spoc312-base-files.json'if P['host']=='A'else'spoc318B-base-files.json')).read_text())
public=Path(public_binding['model'])
actual_public=binding(public,public_binding['metadata_sha256'],public_binding['weight_files_stat'])
assert actual_public=={k:public_binding[k]for k in actual_public},'Public model payload changed'
view_binding=None
if P['host']=='B':
 view=S.parent/'inference-tuning-20261116/model-view'
 if not view.is_dir():view=Path('/data/disk/wangzili/models/Qwen3.8-27B')
 view_binding=binding(view,public_binding['metadata_sha256'],public_binding['weight_files_stat'])
 assert view_binding['metadata_sha256']==public_binding['metadata_sha256'],'GSM host model metadata differs from public primary model'
 for n in public_binding['weight_files_stat']:
  x,y=(view/n).stat(),(public/n).stat()
  if (x.st_dev,x.st_ino)!=(y.st_dev,y.st_ino):assert digest(view/n)==digest(public/n),n
bundles=[]
for bundle in P['bundles']:
 r=S/bundle['root'];assert r.is_dir();sources={}
 for name,files in bundle['source_manifests'].items():
  actual={k:digest((r/k)if'/'not in k else(S/k))for k in files}
  assert actual==files,('Frozen code changed',name)
  sources[name]=actual
 for c in bundle['plan']['cases']:
  if c.get('expected_image_id'):
   image=json.loads(subprocess.check_output(['docker','image','inspect',c['expected_image_tag']],text=True))[0]['Id']
   assert image==c['expected_image_id']
  for flag in ['--request-file','--holdout-file']:
   name=c['args'][c['args'].index(flag)+1]
   if name in P['assets']:assert hashlib.sha256(P['assets'][name].encode()).hexdigest()==P['asset_sha256'][name]
   else:assert digest(r/'assets'/name)==P['existing_asset_sha256'][name],name
 bundles.append((bundle,r,sources))
# 所有模型、head、代码和输入校验通过后才写入新文件；不改现有运行文件。
for bundle,r,sources in bundles:
 for name,files in sources.items():
  if(r/name).exists():assert json.loads((r/name).read_text())['files']==files
  else:write_same(r/name,json.dumps(dict(files=files),indent=2)+'\n')
 for name,text in P['assets'].items():write_same(r/'assets'/name,text)
 if bundle['root']=='online-background':write_same(r/'resource339-gsm-B-base-files.json',json.dumps(view_binding,indent=2)+'\n')
 write_same(r/'_study_metrics_resource338.py',P['metrics'])
 write_same(r/bundle['controller'],P['controller'])
 write_same(r/bundle['plan_name'],json.dumps(bundle['plan'],indent=2)+'\n')
 pre=dict(completed=True,registered_plan_sha256=P['registration_sha256'],source_byte_exact=True,
          initial_head_rehashed=True,public_model_binding_checked=True,host_model_view_equivalent=view_binding is not None,
          numeric_recipe_unchanged=True,physical_cards=3,previously_assigned_cohort=True)
 for name in bundle['preflights']:write_same(r/name,json.dumps(pre,indent=2)+'\n')
result=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=P['host'],
            phase='starting_controllers',registration_sha256=P['registration_sha256'],launched=[],
            source_manifests={b['root']:s for b,_,s in bundles},public_base=public_binding,host_model_view=view_binding)
with record.open('x')as f:json.dump(result,f,indent=2)
def persist():
 tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(record)
for bundle,r,_ in bundles:
 with(r/(bundle['plan_name']+'.controller.log')).open('x')as log:
  child=subprocess.Popen(['python3',bundle['controller'],'--plan',bundle['plan_name']],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result['launched'].append(dict(root=str(r),pid=child.pid,controller=bundle['controller'],plan=bundle['plan_name'],tags=[c['tag']for c in bundle['plan']['cases']]))
 persist()
result['phase']='dispatched';persist();print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("A", "B"), required=True)
    parser.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    D, host = args.validation_root, args.host
    Q = D / "resource338"
    registration_raw = (Q / "registration.json").read_bytes()
    registration = json.loads(registration_raw)
    read = lambda name: json.loads((D / name).read_text())
    controller = (D / "spoc312/study_queue_spoc312.py").read_text().replace(
        "from _study_metrics_windows209 import", "from _study_metrics_resource338 import")
    controller = controller.replace("state['primary_metric'] = 'continuous_online_vs_AR_and_initial_frozen'",
                                    "state['primary_metric'] = 'three_card_resource_repeat_NOT_two_card_primary'")
    plan_names = ["A"] if host == "A" else ["B_spoc", "B_gsm"]
    bundles = []
    for name in plan_names:
        raw = (Q / f"queue-{name}.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == registration["plans"][name]
        background = name == "B_gsm"
        gsm_source = read("gsm254/confirmations/a111-v248-gsm254-confirm743.source.json" if host == "A"
                          else "aux308/b167-v308-aux3card-compact16-step128-gsm384-TV.source.json")["files"]
        manifests = {}
        if name == "A":
            manifests["resource339-gsm-A-source.json"] = gsm_source
            manifests["spoc312-frozen-source.json"] = read("spoc312/frozen-source.json")["files"]
        elif background:
            manifests["resource339-gsm-B-source.json"] = gsm_source
        else:
            manifests["spoc322-frozen-source.json"] = read("relocate318/recovery323-stage.json")["source"]
        bundles.append(dict(root="online-background" if background else "online-studies",
                            plan=json.loads(raw), plan_name=f"queue-resource338-{name}.json",
                            controller="study_queue_resource338.py", source_manifests=manifests,
                            preflights=(["resource338A-preflight.json", "resource339A-preflight.json"] if name == "A"
                                        else ["resource339B-preflight.json" if background else "resource338B-preflight.json"])))
    assets = {p.name: p.read_text() for p in (Q / "assets").glob("*.json")}
    asset_hashes = dict(registration["assets"])
    asset_hashes.update({n: v["sha256"] for n, v in read("resource338/fixture-bindings.json")["fixtures"].items()})
    old_gate = read("relocate318/a111-v292-spoc318B-boundary-mountfix322-metadata323.json")
    gate_args = read("relocate318/queue-spoc323B.json")["cases"][0]["args"]
    existing_assets = {gate_args[gate_args.index(flag) + 1]: old_gate[key]
                       for flag, key in (("--request-file", "prompts_sha256"), ("--holdout-file", "holdout_sha256"))}
    payload = dict(host=host, record_name=f"resource338-{host}-dispatch.json", bundles=bundles,
                   initial_head_shards=registration["gsm_numeric_contract"]["head_shard_sha256"],
                   registration_sha256=hashlib.sha256(registration_raw).hexdigest(), assets=assets,
                   asset_sha256=asset_hashes, existing_asset_sha256=existing_assets,
                   controller=controller, metrics=(D / "windows209/_study_metrics_windows209.py").read_text())
    address = "159.138.5.111" if host == "A" else "189.1.227.167"
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
                             "-o", "ConnectTimeout=10", "wangzili@" + address, "python3 -"],
                            input=("P=" + repr(payload) + "\n" + REMOTE).encode(), capture_output=True, timeout=180)
    if result.returncode:
        (Q / f"dispatch-{host}-error.log").write_bytes(result.stderr)
        raise RuntimeError(result.stderr.decode()[-4000:])
    stage = json.loads(result.stdout)
    (Q / f"stage-{host}.json").write_text(json.dumps(stage, indent=2) + "\n")
    print(json.dumps(dict(host=host, phase=stage["phase"], launched=stage["launched"])))


if __name__ == "__main__":
    main()
