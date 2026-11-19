"""真实 last-4 head：核对 TP2 排布逐字节一致，再比较排布加回读耗时。"""
from pathlib import Path
import hashlib,json,os,random,statistics,sys,time
import torch
from safetensors import safe_open
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker,RunConfig,configure_log
B=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');S=B/'oneserving-v109-b99e118';R=B/'pub-suffix-npu-v2'
sys.path[:0]=[str(R/'src'),str(S/'pypto-serving-qwen3.8-27B'),str(S/'online-studies')]
from blockspec.pypto_qwen38.training_flat import FlatLayout
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from blockspec.pypto_qwen38.oneserving_tp2 import SHAPES
from blockspec.pypto_qwen38.oneserving_tp2_online import pack_execution,read_raw_parameter
from blockspec.pypto_qwen38.tp2_suffix_publication import SuffixPublicationPrograms
from guarded_entry_v28 import Guard
configure_log('warn');torch.set_num_threads(4)
d=dict(completed=False,stage='preparing',physical_card=int(os.environ['PUB_PHYSICAL_CARD']),scope='real step160 last4 weights, scalar host padding retained; packing and readback only, not serving or whole-stream throughput')
def save():p=R/'report.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def digest(t):return hashlib.sha256(memoryview(t.view(torch.uint8).numpy())).hexdigest()
save();worker=None;guard=Guard()
try:
 layout=FlatLayout(SuffixGraph(4).parameters)
 ordinary=torch.zeros(layout.shape,dtype=torch.bfloat16).share_memory_();special=torch.zeros(layout.special_shape,dtype=torch.float32).share_memory_()
 with safe_open(S/'step160-suffix-tp2/rank0.safetensors',framework='pt',device='cpu') as a,safe_open(S/'step160-suffix-tp2/rank1.safetensors',framework='pt',device='cpu') as b:
  for e in layout.entries:
   raw=read_raw_parameter(e.parameter,(a,b)).reshape(-1);ordinary[0,e.offset:e.offset+raw.numel()].copy_(raw)
   if e.special is not None:
    value=raw.float().add(1) if e.parameter.offset_norm else raw.float();special[0,e.special:e.special+value.numel()].copy_(value)
 initial=[digest(x) for x in (ordinary,special)]
 reference=[torch.empty((2,*shape),dtype=dtype).share_memory_() for shape,dtype in SHAPES]
 packed=[torch.empty_like(x).share_memory_() for x in reference]
 pack_execution(layout,ordinary,special,reference)
 config=RunConfig(platform='a2a3',distributed_config=DistributedConfig(device_ids=[0],num_sub_workers=0),ring_heap=(512<<20,)*4)
 program=SuffixPublicationPrograms(layout,set(range(15)),config)
 worker=DistributedWorker([program.program],config=config,inherited_host_tensors=[ordinary,special,*packed])
 buffers=[worker.alloc_tensor(tuple(x.shape),x.dtype,init=x) for x in (ordinary,special)];outputs=program.allocate(worker)
 guard.release();guard.close();d['stage']='running';save()
 def cpu():
  for x,y in zip((ordinary,special),buffers):worker.copy_from(x.data_ptr(),y.data_ptr,x.nbytes)
  pack_execution(layout,ordinary,special,reference)
 def device():
  program.run(worker,layout,*buffers,outputs)
  program.readback(worker,layout,buffers,outputs,packed,(ordinary,special))
 device()
 for i,(a,b) in enumerate(zip(reference,packed)):
  assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),('bank mismatch',i,(a.float()-b.float()).abs().max().item())
 d['all_15_banks_bitwise_equal']=True;d['bank_sha256']=[digest(x) for x in packed];save()
 for _ in range(2):cpu();device()
 samples={'cpu':[],'device':[]}
 for rep in range(8):
  labels=['cpu','device'];random.Random(731+rep).shuffle(labels)
  for label in labels:
   start=time.perf_counter_ns();(cpu if label=='cpu' else device)();samples[label].append((time.perf_counter_ns()-start)/1e6)
 for i,(a,b) in enumerate(zip(reference,packed)):assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),('final bank mismatch',i)
 for x,y in zip((ordinary,special),buffers):worker.copy_from(x.data_ptr(),y.data_ptr,x.nbytes)
 assert [digest(x) for x in (ordinary,special)]==initial,'publication mutated source buffers'
 d.update(completed=True,stage='completed',source_buffers_unchanged=True,timing_ms={k:dict(samples=v,median=statistics.median(v)) for k,v in samples.items()})
except BaseException as e:d['error']=repr(e);raise
finally:
 try:
  if worker is not None:worker.close()
 finally:guard.close();save()
