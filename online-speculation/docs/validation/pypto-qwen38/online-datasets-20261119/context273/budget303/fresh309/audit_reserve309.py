from pathlib import Path
import ast,json,subprocess,zlib,hashlib,concurrent.futures
import argparse
p=argparse.ArgumentParser(description='Read-only exposure audit; use a NEW output, never overwrite a pre-assignment proof.')
p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
D=args.validation_root;output=args.output
if output.exists():raise FileExistsError(output)
manifest=json.loads((D/'context273/data276/pseudo276-selection-manifest.json').read_text());reserved=[]
for name in ['pseudo276-confirm','pseudo276-holdout']:
 qs=json.loads((D/'context273/data276'/(name+'.json')).read_text());meta=manifest['sets'][name];assert len(qs)==meta['requests'];assert hashlib.sha256((D/'context273/data276'/(name+'.json')).read_bytes()).hexdigest()==meta['sha256']
 for q,row in zip(qs,meta['rows']):reserved.append(dict(problem_id=row['problem_id'],normalized=' '.join(q.split()),sha256=hashlib.sha256(q.encode()).hexdigest(),normalized_sha256=hashlib.sha256(' '.join(q.split()).encode()).hexdigest(),cohort=name))
source=(D/'context273/data276/audit_exposures_v2.py').read_text();tree=ast.parse(source);scan=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='scan'for t in n.targets));needle="pre=r.parent.parent/";i=scan.index(needle);extra="""hits=[]
for n in sorted(used_names):
 for text in json.loads((r/'assets'/n).read_text()):
  normal=' '.join(text.split())
  for row in P:
   if row['normalized'] in normal:hits.append(dict(asset=n,problem_id=row['problem_id'],cohort=row['cohort'],match='complete reserved question inside executed/assigned prompt'))
""";scan=scan[:i]+extra+scan[i:];needle="if pre.exists():strings(json.loads(pre.read_text()));used_hashes.update(parent)";assert needle in scan;scan=scan.replace(needle,needle+"\nreserved_hashes={row[key]for row in P for key in ['sha256','normalized_sha256']}\nintersections=sorted(used_hashes&reserved_hashes)\n",1);needle='sys.stdout.buffer.write(zlib.compress(json.dumps(out).encode()))';assert needle in scan;scan=scan.replace(needle,"out.update(reserved_question_substring_hits=hits,reserved_hash_intersections=intersections,completed=out['completed']and not hits and not intersections)\n"+needle,1)
ssh=['ssh','-o','ControlPath=none','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=10']
def call(item):
 host,sub=item;root='/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/oneserving-v109-b99e118/'+sub;code='P='+repr(reserved)+'\n'+scan.replace('ROOT',repr(root));x=subprocess.run([*ssh,'wangzili@'+host,'python3 -'],input=code.encode(),capture_output=True,timeout=90);assert x.returncode==0,x.stderr.decode()[-1600:];d=json.loads(zlib.decompress(x.stdout));print(json.dumps(dict(host=host,completed=d['completed'],tags=d['launched_tags'],unknown=d['unknown_assets'],hits=d['reserved_question_substring_hits'],intersections=d['reserved_hash_intersections'])),flush=True);return host,d
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:hosts=dict(pool.map(call,[('159.138.5.111','online-studies'),('189.1.227.167','online-background'),('159.138.54.61','online-tp2-test')]))
ids={x['problem_id']for x in reserved};explore_ids=set(manifest['sets']['pseudo276-explore']['problem_ids']);assert not ids&explore_ids and not ids&set(manifest['conservative_seen_problem_ids']);report=dict(completed=all(x['completed']for x in hosts.values()),expected_initial128_shards=['f0308fed571f54be1d75aec6ae33d9a58230ccc1bbcea1d3f4df0d1fab6de610','131649c093812f5d895e304ca0f7655b934bc456c9325a3917b1f9ccdcf0f7fd'],source_selection_manifest_sha256=hashlib.sha256((D/'context273/data276/pseudo276-selection-manifest.json').read_bytes()).hexdigest(),reserved_counts={'confirmation':24,'independent_holdout':8},problem_ids=sorted(ids),problem_ids_disjoint_from_all_registered_seen_and_explore=True,hosts=hosts,no_newNNrequests=True,performance_recipe_not_changed_by_audit=True,scope='Conservative launched/assigned-source+actualmodelreport textassets andparent128 hashes, normalizedhashes andfullquestion substrings against reserved24/8. Read-only CPU audit. Nothing copied to execution assets or dispatched by this script.')
spoc_path=D/'context273/spoc283/spoc283-selection-manifest.json'
spoc=json.loads(spoc_path.read_text());explore=spoc['sets']['spoc283-explore64']
spoc_ids={row['problem_id'] for row in explore['rows']}
report['spoc_explore64_problem_ids_disjoint']=not bool(ids&spoc_ids)
report['spoc_selection_manifest_sha256']=hashlib.sha256(spoc_path.read_bytes()).hexdigest()
report['completed']=report['completed']and report['spoc_explore64_problem_ids_disjoint']
with output.open('x')as f:f.write(json.dumps(report,indent=2)+'\n')
print('ALL_FRESH',report['completed'])
