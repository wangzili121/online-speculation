"""真实 last4 参数，连续三次完整训练/Adam/发布，核对临时梯度空间复用。"""
from pathlib import Path
import sys,os,json,hashlib,time
import torch
from pypto.runtime import DistributedWorker,RunConfig,configure_log
from pypto.ir import DistributedConfig
B=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');S=B/'oneserving-v109-b99e118';Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q/'src'),str(S/'pypto-serving-qwen3.8-27B'),str(S/'online-tp2-test')]
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from blockspec.pypto_qwen38.training_executor import TrainingPrograms,TrainingWorkspace
from blockspec.pypto_qwen38.online_suffix_optimizer import SuffixOptimizerPrograms,SuffixOptimizer
from blockspec.pypto_qwen38.oneserving_tp2_online import master_from_shards,pack_execution
from blockspec.pypto_qwen38.oneserving_tp2 import SHAPES
from blockspec.pypto_qwen38.tp2_suffix_publication import SuffixPublicationPrograms
from serving.weight import TextWeights
import importlib.util
spec=importlib.util.spec_from_file_location('blockspec.pypto_qwen38._reference_optimizer_v74',Q/'reference_optimizer_v74.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);ReferenceOptimizer=mod.SuffixOptimizer
from guarded_entry import Guard
configure_log('warn');torch.set_num_threads(4);torch.manual_seed(391)
d=dict(completed=False,stage='preparing',physical_card=int(os.environ['PUB_PHYSICAL_CARD']),scope='singlecard compact optimizer/publication correctness micro: real last4 parameters, synthetic nonzero hidden/history/teacher; three consecutive full forward/backward/Adam/publications, private vs reused scratch; not online throughput',steps=[])
def save():p=Q/'report.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def digest(x):return hashlib.sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest()
def read(t):
 x=torch.empty(t.shape,dtype=t.dtype);worker.copy_from(x.data_ptr(),t.data_ptr,x.nbytes);return x
save();guard=Guard();worker=None;opts=[];workspace=None
try:
 config=RunConfig(platform='a2a3',distributed_config=DistributedConfig(device_ids=[0],num_sub_workers=0),ring_heap=(512<<20,)*4)
 graph=SuffixGraph(4,objective='tv');training=TrainingPrograms(graph,config,combined=True,chunk_size=24,external_gradients=True);optim=SuffixOptimizerPrograms(graph,config,clip_norm=1.);pub=SuffixPublicationPrograms(optim.layout,set(range(15)),config)
 head=S/'step160-suffix-tp2';master=master_from_shards(optim.layout,head).share_memory_();weights=TextWeights(B/'inference-tuning-20261116/model-view');names={p.name for p in graph.parameters};inputs={};paths={}
 for layer in range(60,64):
  for k,v in [('input_gamma','input_layernorm.weight'),('post_gamma','post_attention_layernorm.weight'),('mlp_gate','mlp.gate_proj.weight'),('mlp_up','mlp.up_proj.weight'),('mlp_down','mlp.down_proj.weight')]:paths[f'l{layer}_{k}']=(layer,v)
 for name,value in graph.inputs.items():
  if name in names:continue
  if name in paths:
   layer,path=paths[name];x=weights.layer(layer,path);x=(1+x.float()).reshape(value.shape)if name.endswith('gamma')else x
  elif name=='lm_head':x=weights.read('lm_head.weight')
  elif name=='final_gamma':x=(1+weights.read('model.language_model.norm.weight').float()).reshape(value.shape)
  elif name=='cos':x=torch.ones(value.shape)
  elif name=='sin':x=torch.zeros(value.shape)
  else:x=torch.randn(value.shape)*(.3 if name=='teacher'else .02)
  inputs[name]=x.to(torch.float32 if value.dtype=='fp32'else torch.bfloat16).contiguous().share_memory_()
 packed=[[torch.empty((2,*s),dtype=t).share_memory_()for s,t in SHAPES]for _ in range(2)]
 host=[torch.empty(optim.layout.shape,dtype=torch.bfloat16).share_memory_(),torch.empty(optim.layout.special_shape,dtype=torch.float32).share_memory_()]
 d.update(stage='starting_worker',scratch_bytes=4*optim.layout.elements,avoided_publication_allocation_bytes=pub.allocation_bytes);save()
 worker=DistributedWorker([*training.programs.values(),*optim.programs(),pub.program],config=config,inherited_host_tensors=[master,*inputs.values(),*host,*packed[0],*packed[1]])
 guard.release();guard.close()
 values={k:worker.alloc_tensor(tuple(x.shape),x.dtype,init=x)for k,x in inputs.items()}
 for i in range(2):
  opt=ReferenceOptimizer(optim,worker) if i==0 else SuffixOptimizer(optim,worker,compact=True);opts.append(opt);worker.copy_to(opt.state[0].data_ptr,master.data_ptr(),master.nbytes)
  for t in [*opt.state[1:],*opt.candidate]:opt.run(optim.zero,t)
  opt.render(0,opt.state[0])
 workspace=TrainingWorkspace(training,worker)
 destinations=[pub.allocate(worker),[pub.allocate(worker,scratch=x[0])for x in opts[1].execution]];d['optimizer_flat_bytes_saved']=4*optim.layout.elements
 class Publisher:
  def __init__(self,index):self.index=index
  def publish_flat(self,layout,buffers,*,expected_version):
   i=self.index
   dst=destinations[0]if i==0 else destinations[1][opts[1].index]
   pub.run(worker,layout,*buffers,dst);pub.readback(worker,layout,buffers,dst,packed[i],host)
 for step,valid in enumerate([7,11,3],1):
  measurements=[]
  for i,opt in enumerate(opts):
   opt.zero_grad();workspace.run({**values,**opt.tensors()},past=64,width=32,valid=valid,gradients=opt.gradients)
   losses=read(workspace.values[graph.losses.name]);opt.accumulate();r=opt.step(Publisher(i),3e-5)
   assert r['gradient_norm']>0 and r['version']==step
   measurements.append(dict(loss_sha256=digest(losses),loss=losses[:,0].sum().item(),**r,bank_sha256=[digest(x)for x in packed[i]],state_sha256=[digest(read(x))for x in opt.state]))
  assert measurements[0]==measurements[1],('consecutive update mismatch',step,measurements)
  # 每一步还与 CPU 排布逐位对比，不只比较两种设备输出。
  for x,t in zip(host,opts[1].execution[opts[1].index]):worker.copy_from(x.data_ptr(),t.data_ptr,x.nbytes)
  pack_execution(optim.layout,*host,packed[0]);assert [digest(x)for x in packed[0]]==measurements[1]['bank_sha256']
  d['steps'].append(dict(step=step,private=measurements[0],arena=measurements[1],all_15_banks_equal_cpu=True,master_and_adam_states_bitwise_equal=True));d['stage']='updating';save()
 d.update(completed=True,stage='completed',three_updates_exact=True);save()
except BaseException as err:d['error']=repr(err);raise
finally:
 try:
  if workspace is not None:workspace.close()
  for opt in opts:opt.close()
  if worker is not None:worker.close()
 finally:guard.close();save()
