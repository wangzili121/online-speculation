from pathlib import Path
import copy
import hashlib
import json
import shlex
import subprocess

HERE = Path('/Users/li/Documents/Codex/2026-09-13/kan-xi')
A = HERE / 'artifacts/qwen38-online-datasets-20261119'
BASE = '/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118'
state = json.loads((A / 'ACTIVE_STUDY_STATE.json').read_text())

def ssh(host, code, data=None):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'wangzili@' + host, 'python3 -c ' + shlex.quote(code)],
                            input=data, capture_output=True, timeout=55)
    assert result.returncode == 0, result.stderr
    return result.stdout

def put(args, key, value):
    if key in args:
        args[args.index(key) + 1] = str(value)
    else:
        args.extend([key, str(value)])

code = r'''from pathlib import Path
import copy,hashlib,json,os,signal,subprocess
r=Path(ROOT);p=json.loads((r/OLDPLAN).read_text());extra=json.loads(EXTRA)
assets={};heads={}
for c in extra:
 args=c['args'];get=lambda k:args[args.index(k)+1]
 assert get('--learner-device')=='0' and '--background' not in args and '--interleave-ar' in args
 assert int(get('--teacher-tokens'))==31 and get('--draft-convolution')=='parallel128'
 rb=int(get('--replay-blocks'));ss=int(get('--strides'));ws=int(get('--warm-update-stride'));assert 2*rb<=ws<ss
 for flag in['--request-file','--holdout-file']:
  f=r/'assets'/get(flag);q=json.loads(f.read_text());assert len(q)>=2 and len(q)%2==0 and len(set(q))==len(q)
  assets[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 f=r.parent/get('--head')/'manifest.json';h=json.loads(f.read_text());assert h['tp']==2 and h['block_size']==32
 heads[get('--head')]=dict(step=h['step'],payload=h['source_payload_sha256'])
proof=dict(completed=True,scope='Input/head/cadence preflight only; real model gates required',assets=assets,heads=heads)
(r/'early131-preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
for c in extra:c['requires'].append({'file':'early131-preflight.json'})
started=[c for c in p['cases']if(r/(c['tag']+'.json')).exists()or(r/(c['tag']+'.source.json')).exists()]
future=[c for c in p['cases']if c not in started]
# Keep pending correctness gates before new performance runs, and keep the current NN untouched.
gates=[c for c in future if 'convolution-' in c['tag'] or c['tag'].endswith('dense31-last1-kl025-audit-r1')]
future=[c for c in future if c not in gates]
p.update(cases=started+gates+extra+future,note='Early starting points and longer GSM responses; same optimized computation for all arms. Fresh Adam, full periodic updates through end, dual >5% gate, no reserved questions. Preserve all live and historical model tasks.')
(r/NEWPLAN).write_text(json.dumps(p,indent=2)+'\n')
proc=Path(f'/proc/{PID}');assert proc.stat().st_uid==os.getuid()
assert 'study_queue_dual130.py --plan '+OLDPLAN in(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
os.kill(PID,signal.SIGTERM)
f=(r/NEWPLAN.replace('.json','.log')).open('x')
child=subprocess.Popen(['python3','study_queue_dual130.py','--plan',NEWPLAN],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,plan=NEWPLAN,new_cases=[c['tag']for c in extra],heads=heads)))
'''

for host, sub, suffix in [('189.1.227.167', 'online-background', '167'),
                          ('159.138.54.61', 'online-tp2-test', 'c23')]:
    field = 'ar_primary' if suffix == '167' else 'main'
    _, oldplan = state['controllers'][host][field + '_plan'].split('/')
    root = BASE + '/' + sub
    plan = json.loads(ssh(host, 'from pathlib import Path;print(Path(' + repr(root + '/' + oldplan) + ').read_text())'))
    extra = []
    if suffix == '167':
        for name in ['gsm-long384-explore128-early131.json', 'gsm-long384-dev8-early131.json',
                     'gsm-long384-early131-manifest.json']:
            data = (A / name).read_bytes()
            ssh(host, 'from pathlib import Path;import sys;p=Path(' + repr(root + '/assets/' + name) +
                ');s=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==s;p.write_bytes(s)', data)
        smoke = json.loads((A / 'gsm-long384-explore128-early131.json').read_text())[:6]
        name = 'gsm-long384-smoke6-early131.json'
        data = (json.dumps(smoke, ensure_ascii=False) + '\n').encode(); (A / name).write_bytes(data)
        ssh(host, 'from pathlib import Path;import sys;Path(' + repr(root + '/assets/' + name) + ').write_bytes(sys.stdin.buffer.read())', data)
        gate = copy.deepcopy(next(c for c in plan['cases'] if c['tag'] == 'b167-v114-dense31-last1-kl025-audit-r1'))
        gate['tag'] = 'b167-v114-long384-head128-audit-early131-r1'
        for key, val in [('--head', 'kl-prewarm-suffix-tp2'), ('--tokens', 384),
                         ('--request-file', name), ('--holdout-file', 'gsm-long384-dev8-early131.json'),
                         ('--candidates', 7), ('--learning-rate', 3e-5)]:
            put(gate['args'], key, val)
        gate['purpose'] = 'Diagnostic only: real27B K32/teacher31 with up to384 output tokens and512 total context. Replay, greedy/AR, export and tail checks before long performance runs.'
        extra.append(gate)
        template = copy.deepcopy(next(c for c in plan['cases'] if c['tag'] == 'b167-v114-triplet-gsm384-step572-kl025-dense31-stride64-r1'))
        for step, head, lr in [(128, 'kl-prewarm-suffix-tp2', 3e-5), (0, 'step0-suffix-tp2', 1e-4)]:
            c = copy.deepcopy(template); c['tag'] = f'b167-v114-triplet-gsm128-long384-head{step}-kl025-early131-r1'
            for key, val in [('--head', head), ('--tokens', 384), ('--candidates', 7),
                             ('--request-file', 'gsm-long384-explore128-early131.json'),
                             ('--holdout-file', 'gsm-long384-dev8-early131.json'),
                             ('--ar-check-requests', 128), ('--learning-rate', lr),
                             ('--strides', 64), ('--warm-update-steps', 128)]:
                put(c['args'], key, val)
            c['requires'] = [{'file': gate['tag'] + '.json', 'audit': True, 'greedy': True}]
            c['purpose'] = 'Exploration only: less-trained head, longer responses to amortize prefill and retain31 real teacher labels. Same physical2cards/TP2/learner0, target and generation config in AR/frozen/continuous online, fresh Adam; all costs plus final seen/dev checks.128 questions selected by input length only from prior train exploration, no reserved questions. Earlyhead128 and0 recipes differ in LR and are combined candidates, not isolated LR claims.'
            extra.append(c)
    else:
        gate = copy.deepcopy(next(c for c in plan['cases'] if c['tag'] == 'c23-v114-dense31-last1-kl025-audit-r1'))
        gate['tag'] = 'c23-v114-head64-last1-kl025-audit-early131-r1'
        put(gate['args'], '--head', 'step64-last1-suffix-tp2-cpu131')
        gate['requires'].append({'file': 'early64-export131.json'})
        gate['purpose'] = 'Diagnostic only:64-update early head exported from hashed last1 checkpoint; no Adam imported. Validate real27B replay/greedy and exact publication before using as common starting point.'
        extra.append(gate)
        template = copy.deepcopy(next(c for c in plan['cases'] if c['tag'] == 'c23-v114-triplet-pseudo56-step160-kl025-dense31-lr3e5-stride32-r1'))
        for step, head, lr in [(64, 'step64-last1-suffix-tp2-cpu131', 3e-5), (0, 'step0-suffix-tp2', 1e-4)]:
            c = copy.deepcopy(template); c['tag'] = f'c23-v114-triplet-pseudo56-head{step}-kl025-dense31-early131-r1'
            for key, val in [('--head', head), ('--learning-rate', lr), ('--strides', 64), ('--warm-update-steps', 64)]:
                put(c['args'], key, val)
            c['requires'] = [{'file': gate['tag'] + '.json', 'audit': True, 'greedy': True}]
            c['purpose'] = 'Exploration only: earlier same-weight starting point vs full continuous online, same optimized inference in all arms; K32/full31 candidates,31 genuine future teacher positions, KLteacher0.25. All costs, AR/frozen/online/finalseen/dev. step64 originated48 GSMtrain requests, no Pseudo training; step0 genuine AR-copy. No reserved confirmation consumption.'
            extra.append(c)
    newplan = 'queue-wave131-' + suffix + '.json'
    script = code
    for key, val in [('ROOT', root), ('OLDPLAN', oldplan), ('NEWPLAN', newplan),
                     ('EXTRA', json.dumps(extra)), ('PID', state['controllers'][host][field])]:
        script = script.replace(key, repr(val))
    result = json.loads(ssh(host, script))
    (A / newplan.replace('.json', '-launch.json')).write_text(json.dumps(result, indent=2) + '\n')
    state['controllers'][host].update({field: result['pid'], field + '_plan': sub + '/' + newplan})
    (A / 'ACTIVE_STUDY_STATE.json').write_text(json.dumps(state, indent=2) + '\n')
    print(host, result)
