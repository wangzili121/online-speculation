"""复核冻结题库与派发前隔离快照；不扫描当前使用记录，不启动模型。"""
from pathlib import Path
import argparse,hashlib,itertools,json


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();D=args.validation_root;files={}
    def read(name):
        raw=(D/name).read_bytes();files[name]=sha(raw);return json.loads(raw)
    g=read('gsm254/gsm254-unused-reserve-audit.json');assert g['completed']
    sets={};ids={};normalized={}
    for name,count in [('confirm384',384),('holdout32',32)]:
        path=f'gsm254/gsm254-{name}.json';questions=read(path)
        m=read(f'gsm254/gsm254-{name}-manifest.json')
        hashes=[sha(q.encode())for q in questions]
        assert m['completed']and len(questions)==m['requests']==count
        assert files[path]==m['sha256']and hashes==m['question_sha256']
        assert len(set(hashes))==len(set(m['indices']))==count
        assert not {'train:'+str(i)for i in m['indices']}&set(g['parent128_conservatively_excluded_IDs'])
        sets[name]=set(hashes);ids[name]=set(m['indices'])
        normalized[name]={sha(' '.join(q.split()).encode())for q in questions}
    assert not sets['confirm384']&sets['holdout32']and not ids['confirm384']&ids['holdout32']
    historical={}
    for host,old in g['hosts'].items():
        assert old['completed']and not old['unknown_assets']
        used=set(old['used_sha256'])|set(old['normalized_sha256'])
        for name in sets:assert not(sets[name]|normalized[name])&used
        historical[host]=dict(prior_tags=old['tag_count'],exact_and_whitespace_normalized_intersections=0)
    s=read('context273/spoc283/spoc283-selection-manifest.json');assert s['completed']
    pre=read('spoc312/freshness-audit.json');assert pre['completed']
    assert pre['source_selection_manifest_sha256']==files['context273/spoc283/spoc283-selection-manifest.json']
    spoc_ids={};spoc_hashes={}
    for name,count in [('explore64',64),('confirm256',256),('holdout32',32)]:
        key='spoc283-'+name;path='context273/spoc283/'+key+'.json';questions=read(path);m=s['sets'][key]
        hashes=[sha(q.encode())for q in questions]
        assert len(questions)==m['requests']==count and files[path]==m['sha256']
        assert hashes==[r['question_sha256']for r in m['rows']]
        assert [sha(' '.join(q.split()).encode())for q in questions]==[r['normalized_sha256']for r in m['rows']]
        spoc_ids[name]={r['problem_id']for r in m['rows']};spoc_hashes[name]=set(hashes)
        assert len(spoc_ids[name])==len(spoc_hashes[name])==count
        assert not spoc_ids[name]&set(s['excluded_all_LongProc_problem_ids'])
    for x,y in itertools.combinations(spoc_ids,2):
        assert not spoc_ids[x]&spoc_ids[y]and not spoc_hashes[x]&spoc_hashes[y]
    assert spoc_ids['confirm256']|spoc_ids['holdout32']==set(pre['problem_ids'])
    for old in pre['hosts'].values():
        assert old['completed']and not old['unknown_assets']
        assert not old['reserved_question_substring_hits']and not old['reserved_hash_intersections']
    proof=dict(completed=True,source_file_sha256=files,
        gsm8k=dict(confirmation=384,independent_holdout=32,split='train',
            initial_head128_recorded_IDs_excluded=len(g['parent128_conservatively_excluded_IDs']),
            exact_question_IDs_and_whitespace_normalized_texts_disjoint=True,
            prior_history_snapshot=historical),
        spoc=dict(exploration=64,confirmation=256,independent_holdout=32,
            one_example_per_problem=True,problem_IDs_and_exact_texts_disjoint=True,
            all_original_LongProc_IDs_excluded=len(s['excluded_all_LongProc_problem_ids']),
            frozen_preassignment_full_substring_and_hash_audit_passed=True,
            task='Supplementary SPoC-derived pseudocode-to-C++; not original LongProc benchmark'),
        repeated_trials='Three orders use the same reserved cohort with independent initial head and fresh Adam resets, not three independent datasets.',
        scope='Current artifact integrity and partition recheck against immutable PRE-assignment snapshots. Current trials are already using assigned data; this does not claim those data remain unused now or that the base pretrained model never saw them.',
        NN_dispatch=False)
    args.output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))


if __name__=='__main__':main()
