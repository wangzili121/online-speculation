"""Execute registered continuous-update trials and confirm promising recipes on reserved streams."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--plan', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
plan = json.loads(a.plan.read_text())
out = a.plan.with_suffix('.status.json')
state = {'cases': [], 'execution_complete': False, 'confirmation_groups': {}}

def save():
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2)+'\n')
    tmp.replace(out)

def live(tag, launcher):
    rows = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'args='], text=True)
    return any(launcher+' --tag '+tag+' ' in row for row in rows.splitlines())

def capture_source(tag, launcher):
    files = {launcher: hashlib.sha256((root/launcher).read_bytes()).hexdigest()}
    if launcher.startswith('run_dense_v'):
        suffix = launcher.removeprefix('run_dense_').removesuffix('.py')
        directories = [root/('src-dense-'+suffix), root.parent/'pypto-serving-qwen3.8-27B'/'serving']
        bench = root/('orthrus_bench_dense_'+suffix+'.py')
        if bench.exists(): files[bench.name] = hashlib.sha256(bench.read_bytes()).hexdigest()
        for directory in directories:
            if directory.exists():
                for path in sorted(directory.rglob('*.py')):
                    files[str(path.relative_to(root.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = root/(tag+'.source.json')
    if path.exists() and json.loads(path.read_text())['files'] != files:
        raise RuntimeError('source snapshot changed: '+tag)
    if not path.exists():
        path.write_text(json.dumps(dict(files=files, recorded_at=time.time()),indent=2)+'\n')
    return str(path)

def require(spec):
    path = root/spec['file']
    deadline = time.monotonic()+36000
    while time.monotonic() < deadline:
        if path.exists():
            data = json.loads(path.read_text())
            if data.get('error'):
                return 'prerequisite failed: '+str(data['error'])
            if data.get('completed'):
                updates = [u for v in data.get('results', {}).values() for u in v.get('updates', [])]
                audits = [x for u in updates for x in u.get('replay_audits', [])]
                if spec.get('audit') and (not audits or any(
                        x['served_replay_tv'] > .01 or x['relative_l2'] > .01 for x in audits)):
                    return 'prerequisite replay mismatch'
                if spec.get('greedy') and not all(v.get('exact_greedy') is True for v in
                        data.get('results', {}).values() if 'updates' in v):
                    return 'prerequisite greedy equality not verified'
                return None
        time.sleep(10)
    return 'prerequisite timeout'

def replace_arg(args, key, value):
    if key in args: args[args.index(key)+1] = str(value)
    else: args.extend([key, str(value)])

def confirm(case, entry):
    spec = case.get('confirm')
    winners = [(v['net_increment'], k) for k, v in entry['comparisons'].items()
               if v['continuous'] and v['net_increment'] > .05]
    if not spec or not winners:
        return
    # A reserved cohort is not silently reused to tune another recipe after a failure.
    def option(name, default):
        args = case['args']
        return args[args.index(name)+1] if name in args else default
    cohort = [spec['requests'], spec['holdout'], option('--learner-device', '2'),
              option('--temperature', '0'), option('--tokens', '64')]
    lock = root.parent/('reserved-confirmation-'+hashlib.sha256(json.dumps(cohort).encode()).hexdigest()[:20]+'.json')
    try:
        with lock.open('x') as handle:
            json.dump(dict(recipe=case['tag'], cohort=cohort, args=case['args']), handle, indent=2)
    except FileExistsError:
        entry['confirmation_skipped'] = 'reserved cohort already assigned: '+str(lock)
        save()
        return
    stride = max(winners)[1]
    source = root/'assets'/spec['requests']
    texts = json.loads(source.read_text())
    for seed in (743, 744, 745):
        shuffled = texts.copy(); random.Random(seed).shuffle(shuffled)
        name = case['tag']+'-confirmation-order'+str(seed)+'.json'
        path = root/'assets'/name
        path.write_text(json.dumps(shuffled, ensure_ascii=False)+'\n')
        args = copy.deepcopy(case['args'])
        for flag in ('--skip-seen-replay', '--audit-replay'):
            if flag in args: args.remove(flag)
        for key, value in [('--strides', stride), ('--request-file', name), ('--seed', seed),
                           ('--holdout-file', spec['holdout']), ('--ar-check-requests', len(texts))]:
            replace_arg(args, key, value)
        plan['cases'].insert(plan['cases'].index(case) + 1 + (seed - 743), dict(tag=case['tag']+'-confirm'+str(seed), launcher=case.get('launcher'),
                                  args=args, confirmation_group=case['tag'], seed=seed,
                                  prompts_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

for case in plan['cases']:
    blocked = next((error for s in case.get('requires', []) if (error := require(s))), None)
    if blocked:
        state['cases'].append(dict(tag=case['tag'], completed=False, error=blocked));save();continue
    launcher = case.get('launcher') or 'run_dense_v4.py'
    for attempt in (1, 2):
        tag = case['tag'] if attempt == 1 else case['tag']+'-retry2'
        result = root/(tag+'.json')
        source_manifest = capture_source(tag, launcher)
        child = None
        if not result.exists() and not live(tag, launcher):
            with (root/(tag+'.launch.log')).open('ab') as log:
                child = subprocess.Popen(['python3', launcher, '--tag', tag, *case['args']], cwd=root,
                                         stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+36000
        data = {}
        while time.monotonic() < deadline:
            if result.exists():
                data = json.loads(result.read_text())
                if data.get('completed') or data.get('error'): break
            if child is not None and child.poll() is not None: break
            if child is None and not live(tag, launcher): break
            time.sleep(5)
        if child is not None and child.poll() is None: child.wait(timeout=120)
        entry = dict(tag=tag, completed=bool(data.get('completed')), error=data.get('error'),
                     source_result=str(result), source_manifest=source_manifest, comparisons={}, confirmation_group=case.get('confirmation_group'))
        frozen = data.get('results', {}).get('frozen', {}).get('stream', {})
        for key, value in data.get('results', {}).items():
            if 'updates' not in value or not frozen: continue
            stream = value['stream']
            entry['comparisons'][key] = dict(net_increment=stream['tok_s']/frozen['tok_s']-1,
                continuous=value.get('update_schedule', {}).get('policy')=='continuous_periodic',
                updates=len(value['updates']), frozen_tokens=frozen['tokens'], frozen_seconds=frozen['seconds'],
                online_tokens=stream['tokens'], online_seconds=stream['seconds'])
        state['cases'].append(entry); save(); print(json.dumps(entry), flush=True)
        if entry['completed']:
            confirm(case, entry)
            group = case.get('confirmation_group')
            if group:
                repeats = [x for x in state['cases'] if x.get('confirmation_group')==group and x['completed']]
                values = [next(iter(x['comparisons'].values())) for x in repeats]
                fixed = sum(x['frozen_tokens'] for x in values)/sum(x['frozen_seconds'] for x in values)
                online = sum(x['online_tokens'] for x in values)/sum(x['online_seconds'] for x in values)
                state['confirmation_groups'][group] = dict(repeats=len(values), pooled_gain=online/fixed-1,
                    individual_gains=[x['net_increment'] for x in values],
                    target_met=len(values)==3 and online/fixed>1.05 and all(x['net_increment']>.05 for x in values))
                save()
            break
        if not any(x in str(entry['error']) for x in ('lane is poisoned','LocalMailboxEndpoint','507018','507901')):
            break
state['execution_complete'] = True
save()
