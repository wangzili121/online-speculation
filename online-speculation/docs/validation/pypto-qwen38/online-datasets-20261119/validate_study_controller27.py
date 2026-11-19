from pathlib import Path
import tempfile,runpy,sys,json,copy,subprocess,time,hashlib
A=Path(__file__).resolve().parent;src=(A/'study_queue_wave27.py').read_text();receipts=[]
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)/'group/work';root.mkdir(parents=True);(root/'assets').mkdir();script=root/'study_queue_wave27.py';script.write_text(src);planfile=root/'plan.json'
 (root/'assets/reserved.json').write_text(json.dumps(['one','two','three','four']));(root/'assets/holdout.json').write_text(json.dumps(['five','six']))
 case={'tag':'recipe','launcher':'fake.py','args':['--learner-device','0','--temperature','1','--tokens','128','--learning-rate','1e-5'],'confirm':{'requests':'reserved.json','holdout':'holdout.json'}}
 def load(cases):
  planfile.write_text(json.dumps({'cases':[]}));argv=sys.argv;sys.argv=[str(script),'--plan',str(planfile)]
  try:ns=runpy.run_path(str(script))
  finally:sys.argv=argv
  ns['plan']['cases'][:]=cases
  return ns
 ns=load([case]);entry={'comparisons':{'32':{'net_increment':.06,'continuous':True}}};ns['confirm'](case,entry);assert len(ns['plan']['cases'])==4;assert len(json.loads(planfile.read_text())['cases'])==4
 saved={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'assets').glob('*confirmation-order*.json')};ns['confirm'](case,entry);assert len(ns['plan']['cases'])==4;receipts.append('persisted confirmations, idempotent re-entry')
 ns=load([copy.deepcopy(case)]);current=ns['plan']['cases'][0];ns['confirm'](current,{'comparisons':entry['comparisons']});assert len(ns['plan']['cases'])==4;assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'assets').glob('*confirmation-order*.json')}==saved;receipts.append('recovered queue after lock persisted but dispatcher plan lost')
 changed=copy.deepcopy(case);changed['args'][-1]='3e-5';ns=load([changed]);e={'comparisons':entry['comparisons']};ns['confirm'](changed,e);assert len(ns['plan']['cases'])==1 and 'already assigned' in e['confirmation_skipped'];receipts.append('rejected modified recipe on consumed reserved cohort')
 other=copy.deepcopy(case);other['tag']='other';ns=load([other]);e={'comparisons':entry['comparisons']};ns['confirm'](other,e);assert len(ns['plan']['cases'])==1 and e.get('confirmation_skipped');receipts.append('rejected a second tuning recipe on same reserved cohort')
 (root/'fake.py').write_text('raise SystemExit(2)\n');planfile.write_text(json.dumps({'cases':[{'tag':'failed','launcher':'fake.py','args':[]},{'tag':'blocked','launcher':'fake.py','args':[],'requires':[{'file':'failed.json'}]}]}));start=time.monotonic();x=subprocess.run([sys.executable,str(script),'--plan',str(planfile)],capture_output=True,text=True,timeout=20);assert x.returncode==0,(x.stdout,x.stderr);state=json.loads(planfile.with_suffix('.status.json').read_text());assert len(state['cases'])==2 and state['execution_complete'];assert state['cases'][0]['error'] and 'prerequisite failed' in state['cases'][1]['error'];assert json.loads((root/'failed.json').read_text())['terminal'];receipts.append('incomplete launcher creates terminal receipt and promptly blocks dependent tasks')
(A/'controller27-validation.json').write_text(json.dumps(dict(completed=True,checks=receipts,checks_count=len(receipts),terminal_failure_test_seconds=time.monotonic()-start,source_sha256=hashlib.sha256(src.encode()).hexdigest()),indent=2)+'\n');print(json.dumps(receipts))
