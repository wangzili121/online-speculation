"""比较真实词表下 TV/KL 的现有分块；训练数学不变。"""
from pathlib import Path
import importlib.util,json,time,statistics,random,sys,hashlib
import torch
from pypto.runtime import DistributedWorker,RunConfig,configure_log
from pypto.ir import DistributedConfig
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'oneserving-v109-b99e118/online-studies'))
from guarded_entry_v28 import Guard
configure_log('warn');torch.set_num_threads(4)
report=dict(completed=False,stage='compiling',physical_card=5,rows=32,vocabulary=248320,timings_include_host_dispatch=True,variants={},source_sha256={},validation_contract=dict(gradient_relative_l2=.01,loss_rtol=1e-4,loss_atol=1e-6,invalid_rows_exact_zero=True))
def save():
 p=root/'report.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2)+'\n');q.replace(p)
save();guard=Guard();worker=None
try:
 generator=torch.Generator().manual_seed(94719);shape=(32,248320)
 student=(torch.randn(shape,generator=generator)*2).bfloat16().share_memory_()
 teacher=(torch.randn(shape,generator=generator)*2).bfloat16().share_memory_()
 teacher[torch.arange(32),torch.randint(248320,(32,),generator=generator)]=14
 output=torch.empty(shape,dtype=torch.bfloat16).share_memory_();loss=torch.empty((32,8)).share_memory_()
 config=RunConfig(platform='a2a3',distributed_config=DistributedConfig(device_ids=[0],num_sub_workers=0),ring_heap=(512<<20,)*4)
 entries={}
 for tile in [512,1280,2560]:
  path=root/f'loss_{tile}.py';report['source_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
  spec=importlib.util.spec_from_file_location('loss_tile_'+str(tile),path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  for objective in ['tv','kl']:
   key=objective+':'+str(tile)
   try:
    fn=getattr(module,'make_'+objective)(32,248320);program=fn.compile(torch.empty(shape,dtype=torch.bfloat16,device='meta'),torch.empty(shape,dtype=torch.bfloat16,device='meta'),torch.empty(shape,dtype=torch.bfloat16,device='meta'),torch.empty((32,8),device='meta'),1,1,config=config);entries[key]=program;report['variants'][key]=dict(compiled=True)
   except Exception as e:report['variants'][key]=dict(compiled=False,error=repr(e))
   save()
 assert 'tv:512'in entries and 'kl:512'in entries
 report['stage']='running';save()
 worker=DistributedWorker(list(entries.values()),config=config,inherited_host_tensors=[student,teacher,output,loss])
 ds=worker.alloc_tensor(shape,torch.bfloat16,init=student);dt=worker.alloc_tensor(shape,torch.bfloat16,init=teacher);dg=worker.alloc_tensor(shape,torch.bfloat16);dl=worker.alloc_tensor((32,8),torch.float32)
 guard.release();guard.close()
 lp=teacher.float().log_softmax(-1);lq=student.float().log_softmax(-1);p,q=lp.exp(),lq.exp()
 gold={}
 for objective in ['tv','kl']:
  if objective=='kl':g=q-p;v=(p*(lp-lq)).sum(-1)
  else:
   sign=(q-p).sign()*.5;g=q*(sign-(q*sign).sum(-1,keepdim=True));v=(q-p).abs().sum(-1)*.5
  gold[objective]=(g,v)
 def run(key,valid):worker.run(entries[key],ds,dt,dg,dl,valid,valid,config=config)
 for valid in [1,3,7,31]:
  for key in list(entries):
   run(key,valid);worker.copy_from(output.data_ptr(),dg.data_ptr,output.nbytes);worker.copy_from(loss.data_ptr(),dl.data_ptr,loss.nbytes)
   obj=key.split(':')[0];gg,gl=gold[obj];want=(gg[:valid]/valid).bfloat16().float();got=output[:valid].float()
   relative=((got-want).norm()/want.norm().clamp_min(1e-20)).item();loss_error=(loss[:valid,0]-gl[:valid]/valid).abs().max().item()
   passed=bool(torch.isfinite(got).all()) and relative<.01 and torch.allclose(loss[:valid,0],gl[:valid]/valid,rtol=1e-4,atol=1e-6) and torch.count_nonzero(output[valid:]).item()==0 and torch.count_nonzero(loss[valid:]).item()==0
   report['variants'][key].setdefault('correctness',{})[str(valid)]=dict(passed=passed,relative_l2=relative,max_abs_gradient=(got-want).abs().max().item(),max_abs_loss=loss_error,golden_loss_max=(gl[:valid]/valid).abs().max().item(),max_relative_loss=((loss[:valid,0]-gl[:valid]/valid).abs()/(gl[:valid]/valid).abs().clamp_min(1e-12)).max().item())
   save()
   if key.endswith(':512'):assert passed,(key,valid,report['variants'][key])
   for _ in range(2):run(key,valid)
  samples={key:[] for key in entries}
  for repeat in range(10):
   keys=list(entries);random.Random(94719+repeat+valid).shuffle(keys)
   for key in keys:
    started=time.perf_counter_ns();run(key,valid);samples[key].append((time.perf_counter_ns()-started)/1e6)
  for key,values in samples.items():report['variants'][key].setdefault('timing_ms',{})[str(valid)]=dict(samples=values,median=statistics.median(values))
  save()
 report['accepted_variants']=[key for key,v in report['variants'].items() if v.get('compiled') and len(v.get('correctness',{}))==4 and all(x['passed'] for x in v['correctness'].values())]
 report['completed']=True;report['stage']='completed'
except BaseException as e:report['error']=repr(e);raise
finally:
 try:
  if worker is not None:worker.close()
 finally:guard.close();save()
