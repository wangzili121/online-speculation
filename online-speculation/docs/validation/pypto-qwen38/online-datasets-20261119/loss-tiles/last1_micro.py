"""真实单层权重与完整前反向，比较损失核的两种分块。"""
from pathlib import Path
import sys,os,json,importlib.util,time,statistics,random
import torch
from safetensors import safe_open
from pypto.runtime import DistributedWorker,RunConfig,configure_log
from pypto.ir import DistributedConfig
BASE=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');ROOT=BASE/'oneserving-v109-b99e118';here=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'online-studies/src-dense-v28'),str(ROOT/'pypto-serving-qwen3.8-27B'),str(ROOT/'online-studies')]
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from blockspec.pypto_qwen38.training_executor import TrainingPrograms,TrainingWorkspace
import blockspec.pypto_qwen38.training_pipeline as pipeline
from blockspec.pypto_qwen38.oneserving_tp2_online import read_raw_parameter
from serving.weight import TextWeights
from guarded_entry_v28 import Guard
configure_log('warn');torch.set_num_threads(4);torch.manual_seed(391)
report=dict(completed=False,stage='preparing',scope='real last1 weights and production shapes; synthetic hidden/history/teacher, complete forward/backward only; no optimizer or publication timing',physical_card=int(os.environ['LOSS_PHYSICAL_CARD']),results={},loss_factory_calls={})
def save():p=here/'report.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2)+'\n');q.replace(p)
save();guard=Guard();worker=None;work={}
try:
 config=RunConfig(platform='a2a3',distributed_config=DistributedConfig(device_ids=[0],num_sub_workers=0),ring_heap=(512<<20,)*4)
 head=ROOT/'step160-suffix-tp2';weights=TextWeights(Path('/data/disk/models/Qwen3.8-27B'));graph=SuffixGraph(1,objective='tv');raw={}
 with safe_open(head/'rank0.safetensors',framework='pt',device='cpu') as a,safe_open(head/'rank1.safetensors',framework='pt',device='cpu') as b:
  for p in graph.parameters:
   value=read_raw_parameter(p,(a,b));raw[p.name]=value.float().add(1) if p.offset_norm else value
 paths={'l63_input_gamma':'input_layernorm.weight','l63_post_gamma':'post_attention_layernorm.weight','l63_mlp_gate':'mlp.gate_proj.weight','l63_mlp_up':'mlp.up_proj.weight','l63_mlp_down':'mlp.down_proj.weight'}
 inputs={}
 for name,value in graph.inputs.items():
  if name in raw:x=raw[name]
  elif name in paths:
   x=weights.layer(63,paths[name]);x=(1+x.float()).reshape(value.shape) if name.endswith('gamma') else x
  elif name=='lm_head':x=weights.read('lm_head.weight')
  elif name=='final_gamma':x=(1+weights.read('model.language_model.norm.weight').float()).reshape(value.shape)
  elif name=='cos':x=torch.ones(value.shape)
  elif name=='sin':x=torch.zeros(value.shape)
  else:x=torch.randn(value.shape)*(.3 if name=='teacher' else .02)
  inputs[name]=x.to(torch.float32 if value.dtype=='fp32' else torch.bfloat16).contiguous().share_memory_()
 def loss_factory(factory,key):
  def build(*args,**kwargs):
   report['loss_factory_calls'][key]=report['loss_factory_calls'].get(key,0)+1
   return factory(*args,**kwargs)
  return build
 compiled={};graphs={};original_tv,original_kl=pipeline.make_tv,pipeline.make_kl
 for tile in [512,2560]:
  spec=importlib.util.spec_from_file_location('loss_training_'+str(tile),here/f'loss_{tile}.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);pipeline.make_tv,pipeline.make_kl=loss_factory(m.make_tv,'tv:'+str(tile)),loss_factory(m.make_kl,'kl:'+str(tile))
  for obj in ['tv','kl']:
   g=graphs[obj,tile]=SuffixGraph(1,objective=obj);compiled[obj,tile]=TrainingPrograms(g,config,combined=True,chunk_size=None);assert report['loss_factory_calls'][obj+':'+str(tile)]==1
 pipeline.make_tv,pipeline.make_kl=original_tv,original_kl
 report['stage']='running';save()
 worker=DistributedWorker([p for item in compiled.values() for p in item.programs.values()],config=config,inherited_host_tensors=list(inputs.values()))
 values={k:worker.alloc_tensor(tuple(v.shape),v.dtype,init=v) for k,v in inputs.items()};work={k:TrainingWorkspace(p,worker) for k,p in compiled.items()}
 guard.release();guard.close()
 def read(t):
  x=torch.empty(t.shape,dtype=t.dtype);worker.copy_from(x.data_ptr(),t.data_ptr,x.nbytes);return x
 def run(key):return work[key].run(values,past=64,width=32,valid=7)
 for obj in ['tv','kl']:
  before=run((obj,512));baseline={k:read(t) for k,t in before.items()};baseline['logits']=read(work[obj,512].values[graphs[obj,512].logits.name]);baseline['losses']=read(work[obj,512].values[graphs[obj,512].losses.name])
  after=run((obj,2560));diff={}
  for name,t in after.items():
   x,y=baseline[name].float(),read(t).float();rel=((x-y).norm()/x.norm().clamp_min(1e-20)).item();diff[name]=dict(relative_l2=rel,max_abs=(x-y).abs().max().item());assert rel<.01,(obj,name,diff[name])
  logits=read(work[obj,2560].values[graphs[obj,2560].logits.name]);assert torch.equal(logits,baseline['logits']),'loss tiling changed forward logits'
  losses=read(work[obj,2560].values[graphs[obj,2560].losses.name]);torch.testing.assert_close(losses,baseline['losses'],rtol=1e-4,atol=1e-6)
  report['results'][obj]=dict(gradient_differences=diff,forward_logits_bitwise_equal=True,losses_close=True);save();del baseline
 for key in compiled:
  for _ in range(2):run(key)
 samples={key:[] for key in compiled}
 for repeat in range(8):
  keys=list(compiled);random.Random(391+repeat).shuffle(keys)
  for key in keys:
   start=time.perf_counter_ns();run(key);read(work[key].values[graphs[key].losses.name]);samples[key].append((time.perf_counter_ns()-start)/1e6)
 for obj in ['tv','kl']:
  report['results'][obj]['timing_ms']={str(t):dict(samples=samples[obj,t],median=statistics.median(samples[obj,t])) for t in [512,2560]}
 report['completed']=True;report['stage']='completed'
except BaseException as e:report['error']=repr(e);raise
finally:
 try:
  for ws in work.values():ws.close()
  if worker is not None:worker.close()
 finally:guard.close();save()
