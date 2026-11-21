"""拆分真实训练矩阵形状的32/16行延迟；不用于端到端吞吐结论。"""
from pathlib import Path
import json,statistics,random,time
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker,RunConfig,configure_log
from blockspec.pypto_qwen38.training_programs import make_linear
from serving.weight import TextWeights
q=Path(__file__).parent;report=dict(completed=False,stage='compiling',cases=[],scope='RealLM/MLP weights and physical trainingmatrix dimensions. Synthetic activations/upstream; independent component timing, fixed64element completion read. Not online throughput.');worker=None

def save():
 p=q/'report.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(p)
save();configure_log('warn');torch.set_num_threads(4);torch.manual_seed(261)
try:
 config=RunConfig(platform='a2a3',ring_heap=(512<<20,)*4,distributed_config=DistributedConfig(device_ids=[0],num_sub_workers=0))
 specs=dict(lm_forward=((5120),(248320,5120),False,True,False),lm_input_vjp=(248320,(248320,5120),False,False,False),mlp_forward=(5120,(17408,5120),False,True,False),mlp_input_vjp=(17408,(17408,5120),False,False,False),q_weight_vjp=(12288,None,True,False,True));programs={};shapes={}
 for name,(cols,b,ta,tb,grad)in specs.items():
  for rows in [32,16]:
   a=(rows,cols);bs=(rows,5120)if b is None else b;m,k=a[::-1]if ta else a;kb,n=bs[::-1]if tb else bs;out=(m,n);dt=torch.float32 if grad else torch.bfloat16;shapes[name,rows]=(a,bs,out,dt);programs[name,rows]=make_linear(a,bs,transpose_a=ta,transpose_b=tb,gradient_weight=grad).compile(torch.empty(a,dtype=torch.bfloat16,device='meta'),torch.empty(bs,dtype=torch.bfloat16,device='meta'),torch.empty(out,dtype=dt,device='meta'),config=config)
 weights=TextWeights(Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/inference-tuning-20261116/model-view'));lm=weights.read('lm_head.weight').share_memory_();mlp=weights.layer(63,'mlp.up_proj.weight').share_memory_();hosts={'lm':lm,'mlp':mlp};upstream={};extra={}
 for name,(cols,b,ta,tb,grad)in specs.items():
  x=(torch.randn(32,cols)*.02).bfloat16();x[16:].zero_();upstream[name]=x.share_memory_()
  if name=='q_weight_vjp':extra[name]=(torch.randn(32,5120)*.02).bfloat16().share_memory_()
 worker=DistributedWorker(list(programs.values()),config=config,inherited_host_tensors=[*hosts.values(),*upstream.values(),*extra.values()]);base={name:worker.alloc_tensor(t.shape,t.dtype,init=t)for name,t in hosts.items()};inputs={};outputs={}
 from pypto.runtime.task_interface import DataType
 for name in specs:
  x=worker.alloc_tensor(upstream[name].shape,torch.bfloat16,init=upstream[name]);b=base['lm'if name.startswith('lm_')else'mlp']if name!='q_weight_vjp'else worker.alloc_tensor(extra[name].shape,torch.bfloat16,init=extra[name])
  for rows in [32,16]:
   a,bs,shape,dt=shapes[name,rows];inputs[name,rows]=(x.buffer.tensor(a,DataType.BFLOAT16),b.buffer.tensor(bs,DataType.BFLOAT16));outputs[name,rows]=worker.alloc_tensor(shape,dt)
 report['stage']='running';save()
 def sync_run(name,rows):
  worker.run(programs[name,rows],*inputs[name,rows],outputs[name,rows],config=config);h=torch.empty(64,dtype=outputs[name,rows].dtype);worker.copy_from(h.data_ptr(),outputs[name,rows].data_ptr,h.nbytes)
 def read(name,rows):
  out=outputs[name,rows];h=torch.empty(out.shape,dtype=out.dtype);worker.copy_from(h.data_ptr(),out.data_ptr,h.nbytes);return h
 for name in specs:
  for rows in [32,16]:
   for _ in range(2):sync_run(name,rows)
  old=read(name,32);new=read(name,16);old=old if name=='q_weight_vjp'else old[:16];assert torch.equal(old,new),(name,'output differs');del old,new
  samples={32:[],16:[]}
  for repetition in range(8):
   order=[32,16];random.Random(261+repetition).shuffle(order)
   for rows in order:
    started=time.perf_counter_ns();sync_run(name,rows);samples[rows].append((time.perf_counter_ns()-started)/1e6)
  report['cases'].append(dict(name=name,active_output_bitwise=True,timing_ms={str(rows):dict(median=statistics.median(values),samples=values)for rows,values in samples.items()}));save()
 report.update(completed=True,stage='completed')
except BaseException as e:report['error']=repr(e);raise
finally:
 try:
  if worker is not None:worker.close()
 finally:save()
