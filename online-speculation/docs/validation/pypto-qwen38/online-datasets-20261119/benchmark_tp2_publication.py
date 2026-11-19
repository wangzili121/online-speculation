"""真实 head 的发布布局对照：原 CPU 拼接与设备布局/直接 DMA。"""
from pathlib import Path
import json,os,sys,time,statistics,subprocess
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker,RunConfig,configure_log
configure_log('warn')
BASE=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105');ROOT=BASE/'oneserving-v109-b99e118'
sys.path[:0]=[str(BASE/'publication-v1/src'),str(ROOT/'pypto-serving-qwen3.8-27B'),str(ROOT/'online-tp2-test')]
from blockspec.pypto_qwen38.tp2_publication import PublicationPrograms
from blockspec.pypto_qwen38.oneserving_tp2 import SHAPES
from blockspec.pypto_qwen38.oneserving_tp2_online import master_from_shards,pack_execution
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from blockspec.pypto_qwen38.training_flat import FlatLayout
from watch_glm53_validation import idle_devices,acquire_devices,release_devices

torch.set_num_threads(4)
config=RunConfig(platform='a2a3',distributed_config=DistributedConfig(device_ids=[0],num_sub_workers=0),ring_heap=(512<<20,)*4)
report=dict(completed=False,stage='compiling',scope='same real served head; publication layout and copies only, no training or generation',results={});output=BASE/'publication-v1/report.json'
def save():output.write_text(json.dumps(report,indent=2)+'\n')
save();compiled={};hosts={};layouts={};reference={};candidate={}
for scope,banks in [('full',set(range(9,15))),('norms',{12,13}),('output_norms',{12,13,14})]:
 layout=layouts[scope]=FlatLayout(SuffixGraph(1,parameter_scope=scope).parameters)
 master=master_from_shards(layout,ROOT/'step160-suffix-tp2');ordinary=master.bfloat16().share_memory_();special=torch.zeros(layout.special_shape).share_memory_()
 for e in layout.entries:
  if e.special is not None:
   n=__import__('math').prod(e.parameter.shape);v=ordinary[0,e.offset:e.offset+n].float();special[0,e.special:e.special+n]=v+1 if e.parameter.offset_norm else v
 compiled[scope]=PublicationPrograms(layout,banks,config);hosts[scope]=(ordinary,special)
 reference[scope]=[torch.empty((2,*shape),dtype=dtype).share_memory_() if i in banks else None for i,(shape,dtype) in enumerate(SHAPES)]
 candidate[scope]=[torch.empty_like(x).share_memory_() if x is not None else None for x in reference[scope]]
 del master
# 不在 CPU 编译期间占卡。编译结束后只选择无进程、低 HBM 的一张卡。
report['stage']='waiting_for_idle_device';save();deadline=time.monotonic()+14400;locks=None
while time.monotonic()<deadline:
 idle=idle_devices(subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True))
 for card in idle:
  if card==6:continue
  locks=acquire_devices(BASE.parent/'orthrus-glm53-flash-20261101',[card])
  if locks is not None:break
 else:time.sleep(1);continue
 break
if locks is None:raise TimeoutError('no idle device for publication audit')
os.environ['ASCEND_RT_VISIBLE_DEVICES']=str(card);report['physical_device']=card;save();worker=None
try:
 inherited=[t for pair in hosts.values() for t in pair]+[t for groups in (reference,candidate) for banks in groups.values() for t in banks if t is not None]
 report['stage']='validating';save()
 worker=DistributedWorker([p.program for p in compiled.values()],config=config,inherited_host_tensors=inherited)
 devices={scope:[worker.alloc_tensor(tuple(x.shape),x.dtype,init=x) for x in pair] for scope,pair in hosts.items()}
 targets={scope:p.allocate(worker) for scope,p in compiled.items()}
 def cpu(scope):
  for host,device in zip(hosts[scope],devices[scope]):worker.copy_from(host.data_ptr(),device.data_ptr,host.nbytes)
  pack_execution(layouts[scope],*hosts[scope],reference[scope])
 def npu(scope):
  p=compiled[scope];p.run(worker,layouts[scope],*devices[scope],targets[scope]);pointers=p.sources(layouts[scope],devices[scope][0],targets[scope])
  for bank,(ptr,offset) in pointers.items():
   host=candidate[scope][bank];worker.copy_from(host.data_ptr(),ptr,host.nbytes,src_offset=offset)
 for scope in compiled:
  cpu(scope);npu(scope);diff={}
  for bank,(x,y) in enumerate(zip(reference[scope],candidate[scope])):
   if x is not None:
    assert torch.equal(x,y),(scope,bank,(x.float()-y.float()).abs().max().item());diff[bank]=0
  samples={k:[] for k in ['cpu','device']}
  for f in (cpu,npu):
   for _ in range(2):f(scope)
  for repeat in range(8):
   for name,f in [('cpu',cpu),('device',npu)]:
    start=time.perf_counter();f(scope);samples[name].append(time.perf_counter()-start)
  median={k:statistics.median(v) for k,v in samples.items()};report['results'][scope]=dict(exact_banks=diff,seconds=samples,medians=median,speedup=median['cpu']/median['device']);save()
 report['completed']=True;report['stage']='completed'
except BaseException as error:
 report['error']=repr(error);raise
finally:
 try:
  if worker is not None:worker.close()
 finally:release_devices(locks);save()
