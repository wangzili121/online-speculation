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

from _study_metrics_windows209 import confirmation_metrics, matched_request_timelines

p = argparse.ArgumentParser()
p.add_argument('--plan', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
plan = json.loads(a.plan.read_text())
out = a.plan.with_suffix('.status.json')
state = {'cases': [], 'execution_complete': False, 'confirmation_groups': {},
         'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'metrics_sha256': hashlib.sha256(Path(__file__).with_name('_study_metrics_windows209.py').read_bytes()).hexdigest()}

def ar_metrics(data, stream, timeline):
    results = data.get('results', {})
    ar, frozen = results.get('ar', {}), results.get('frozen', {})
    count = data.get('requests', 0)
    if (not data.get('completed') or data.get('clean_shutdown') is not True or count < 2
            or data.get('ar_checked_requests') != count
            or not matched_request_timelines(count, data.get('stream_window', 2),
                   ar.get('request_timeline', []), frozen.get('request_timeline', []), timeline)
            or not ar.get('stream')):
        return {}
    controls = all(all(results[arm]['stream'].get('stats', {}).get(k, -1) == 0 for k in
                       ('updates', 'publications', 'feedback_ns', 'update_ns', 'publish_ns'))
                   for arm in ('ar', 'frozen'))
    visible = data.get('visible_devices', '').split(',')
    resources = (data.get('physical_device_count') == 2 and len(set(visible)) == 2
                 and len(data.get('device_ids', [])) == 2 and data.get('learner_device') == 0
                 and data.get('background') is False and data.get('execution_max_num_seqs', 2) == 2)
    return dict(ar_tokens=ar['stream']['tokens'], ar_seconds=ar['stream']['seconds'],
                net_increment_vs_ar=stream['tok_s']/ar['stream']['tok_s']-1,
                ar_timing_layout=data.get('benchmark_layout'), full_cohort=True,
                resource_policy_ok=resources and controls,
                performance_run=not (data.get('diagnostic_only', False) or data.get('audit_replay', False)))


def ar_confirmation(values):
    return confirmation_metrics(values)


state['primary_metric'] = 'continuous_online_vs_AR_and_initial_frozen'


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
        suffix = launcher.removeprefix('run_dense_').removesuffix('.py').removesuffix('_docker')
        if launcher.endswith('_docker.py'):
            entry = root/('docker_entry_'+suffix+'.py')
            files[entry.name] = hashlib.sha256(entry.read_bytes()).hexdigest()
        directories = [root/('src-dense-'+suffix), root.parent/'pypto-serving-qwen3.8-27B'/'serving']
        bench = root/('orthrus_bench_dense_'+suffix+'.py')
        if bench.exists(): files[bench.name] = hashlib.sha256(bench.read_bytes()).hexdigest()
        probe = root/('orthrus_eos_probe_'+suffix+'.py')
        if suffix == 'v18' and probe.exists(): files[probe.name] = hashlib.sha256(probe.read_bytes()).hexdigest()
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
    prior_failure = next((x for x in state['cases'] if x['tag'] == path.stem and x.get('error')), None)
    if prior_failure:
        return 'prerequisite failed earlier: ' + str(prior_failure['error'])
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
                if spec.get('min_prefill_chunk', 0) > data.get('target_execution_totals', {}).get('max_prefill_chunk', 0):
                    return 'prerequisite did not exercise native long prefill'
                if spec.get('eos_banks') and (not data.get('diagnostic_only') or not data.get('eos_feedback_checks') or len(data.get('device_bank_readback', [])) < 2 or not all(x['initial_unchanged'] and x['published_matches_host'] for x in data['device_bank_readback'])):
                    return 'prerequisite EOS or device-bank readback missing'
                return None
        time.sleep(10)
    return 'prerequisite timeout'

def replace_arg(args, key, value):
    if key in args: args[args.index(key)+1] = str(value)
    else: args.extend([key, str(value)])

def confirm(case, entry):
    spec = case.get('confirm')
    winners = [(min(v.get('net_increment_vs_ar', float('-inf')), v.get('net_increment', float('-inf'))), k) for k, v in entry['comparisons'].items()
               if v['continuous'] and v.get('net_increment_vs_ar', float('-inf')) > .05
               and v.get('net_increment', float('-inf')) > .05 and v.get('full_cohort') is True
               and v.get('resource_policy_ok') is True and v.get('performance_run') is True]
    if not spec or not winners:
        return
    # A reserved cohort is not silently reused to tune another recipe after a failure.
    def option(name, default):
        args = case['args']
        return args[args.index(name)+1] if name in args else default
    cohort = [spec['requests'], spec['holdout'], option('--learner-device', '2'),
              option('--temperature', '0'), option('--tokens', '64')]
    lock = root.parent/('reserved-confirmation-'+hashlib.sha256(json.dumps(cohort).encode()).hexdigest()[:20]+'.json')
    source = root/'assets'/spec['requests']
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assignment = dict(recipe=case['tag'], cohort=cohort, args=case['args'], requests_sha256=source_hash)
    try:
        with lock.open('x') as handle:
            json.dump(assignment, handle, indent=2)
    except FileExistsError:
        prior = json.loads(lock.read_text())
        if prior != assignment:
            entry['confirmation_skipped'] = 'reserved cohort already assigned: '+str(lock)
            save()
            return
        # Same immutable recipe may resume after a dispatcher restart.
    stride = max(winners)[1]
    texts = json.loads(source.read_text())
    for seed in (743, 744, 745):
        tag = case['tag']+'-confirm'+str(seed)
        if any(c['tag'] == tag for c in plan['cases']):
            continue
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
        plan['cases'].insert(plan['cases'].index(case) + 1 + (seed - 743), dict(tag=tag, launcher=case.get('launcher'),
                                  args=args, confirmation_group=case['tag'], seed=seed,
                                  prompts_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    temporary = a.plan.with_suffix('.pending.json')
    temporary.write_text(json.dumps(plan, indent=2)+'\n')
    temporary.replace(a.plan)

for case in plan['cases']:
    if state['confirmation_groups'].get(case.get('unless_achieved'), {}).get('target_met'):
        state['cases'].append(dict(tag=case['tag'], skipped=True, reason='preceding reserved confirmations already met the target')); save(); continue
    blocked = next((error for s in case.get('requires', []) if (error := require(s))), None)
    if blocked:
        state['cases'].append(dict(tag=case['tag'], completed=False, error=blocked));save();continue
    launcher = case.get('launcher') or 'run_dense_v4.py'
    for attempt in range(1, case.get("max_attempts", 2) + 1):
        tag = case['tag'] if attempt == 1 else case['tag']+'-retry2'
        result = root/(tag+'.json')
        if not result.exists() and not live(tag, launcher):
            minimum = case.get('minimum_available_host_gib', 0) * (1 << 30)
            while minimum:
                info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
                available = int(info['MemAvailable'].split()[0]) * 1024
                starting = []
                for line in subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'args='], text=True).splitlines():
                    words = line.split()
                    if len(words) > 4 and words[0] == 'python3' and words[1].startswith('run_dense_v') and words[2] == '--tag' and words[3] != tag:
                        result_path = root/(words[3]+'.json')
                        if result_path.exists() and json.loads(result_path.read_text()).get('stage') == 'starting':
                            starting.append(words[3])
                required = minimum * (1 + len(starting)) if case.get('allow_parallel_startup', False) else minimum
                if available >= required and (not starting or case.get('allow_parallel_startup', False)):
                    break
                print(json.dumps(dict(waiting_host_memory=True, tag=tag, available_bytes=available, minimum_bytes=minimum, other_own_startups=starting)), flush=True)
                time.sleep(30)
        source_manifest = capture_source(tag, launcher)
        if case.get('expected_source_manifest'):
            expected=json.loads((root/case['expected_source_manifest']).read_text())['files']
            if json.loads(Path(source_manifest).read_text())['files']!=expected:
                raise RuntimeError('frozen model source mismatch before dispatch: '+tag)
            if case.get('expected_image_id'):
                image=json.loads(subprocess.check_output(['docker','image','inspect',case['expected_image_tag']],text=True))[0]['Id']
                if image!=case['expected_image_id']:raise RuntimeError('frozen image digest mismatch')
            model_binding=json.loads((root/case['expected_base_manifest']).read_text())
            model=Path(model_binding['model'])
            for name,expected_sha in model_binding['metadata_sha256'].items():
                if hashlib.sha256((model/name).read_bytes()).hexdigest()!=expected_sha:
                    raise RuntimeError('base metadata changed before dispatch: '+name)
            for name,expected_stat in model_binding['weight_files_stat'].items():
                st=(model/name).stat()
                if dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)!=expected_stat:
                    raise RuntimeError('base weights changed before dispatch: '+name)
            head=root.parent/case['expected_initial_head']
            for rank,expected_sha in enumerate(case['expected_initial_head_shards']):
                h=hashlib.sha256()
                with(head/f'rank{rank}.safetensors').open('rb')as handle:
                    for chunk in iter(lambda:handle.read(8<<20),b''):h.update(chunk)
                if h.hexdigest()!=expected_sha:raise RuntimeError('initial frozen head changed')

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
        if not (data.get('completed') and data.get('clean_shutdown')) and not data.get('error'):
            data['error'] = 'Launcher terminated without a complete clean result; inspect '+tag+'.log and .launch.log'
            data['launcher_exit_code'] = child.returncode if child is not None else None
            data['terminal'] = True
            temporary = result.with_suffix('.terminal.tmp')
            temporary.write_text(json.dumps(data, indent=2)+'\n')
            temporary.replace(result)
        entry = dict(tag=tag, completed=bool(data.get('completed')) and not data.get('error') and data.get('clean_shutdown') is True, error=data.get('error'), error_chain=data.get('error_chain', []),
                     source_result=str(result), source_manifest=source_manifest, comparisons={}, confirmation_group=case.get('confirmation_group'))
        frozen = data.get('results', {}).get('frozen', {}).get('stream', {})
        for key, value in data.get('results', {}).items():
            if 'updates' not in value or not frozen: continue
            stream = value['stream']
            entry['comparisons'][key] = dict(net_increment=stream['tok_s']/frozen['tok_s']-1,
                continuous=value.get('update_schedule', {}).get('policy') in {'continuous_periodic', 'continuous_two_stage'},
                updates=len(value['updates']), frozen_tokens=frozen['tokens'], frozen_seconds=frozen['seconds'],
                online_tokens=stream['tokens'], online_seconds=stream['seconds'])
            entry['comparisons'][key].update(ar_metrics(data, stream, value.get('request_timeline', [])))
        state['cases'].append(entry); save(); print(json.dumps(entry), flush=True)
        if entry['completed']:
            confirm(case, entry)
            group = case.get('confirmation_group')
            if group:
                repeats = [x for x in state['cases'] if x.get('confirmation_group')==group and x['completed']]
                values = [next(iter(x['comparisons'].values())) for x in repeats]
                state['confirmation_groups'][group] = ar_confirmation(values)
                save()
            break
        if not any(x in (str(entry['error']) + json.dumps(entry['error_chain'])) for x in ('lane is poisoned','LocalMailboxEndpoint','507018','507901')):
            break
state['execution_complete'] = True
save()
