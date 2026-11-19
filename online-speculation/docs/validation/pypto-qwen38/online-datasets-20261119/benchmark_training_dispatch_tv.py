"""真实单层形状、真实权重：比较一个训练入口与 24 节点分段入口。输入为合成状态。"""
import json,os,sys,time,statistics
from pathlib import Path
import torch
from safetensors import safe_open
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker,RunConfig,configure_log
configure_log("warn")
BASE=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105')
ROOT=BASE/'oneserving-v109-b99e118';sys.path[:0]=[str(ROOT/'online-tp2-test/src-dense-v9'),str(ROOT/'pypto-serving-qwen3.8-27B')]
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from blockspec.pypto_qwen38.training_pipeline import source
from blockspec.pypto_qwen38.training_executor import TrainingPrograms,TrainingWorkspace
from blockspec.pypto_qwen38.oneserving_tp2_online import read_raw_parameter
from serving.weight import TextWeights
from guarded_entry import Guard

torch.set_num_threads(4);torch.manual_seed(391)
config=RunConfig(platform='a2a3',distributed_config=DistributedConfig(device_ids=[0],num_sub_workers=0))
head=ROOT/'step160-suffix-tp2';weights=TextWeights(BASE/'inference-tuning-20261116/model-view')
full=SuffixGraph(1,objective='tv');raw={}
with safe_open(head/'rank0.safetensors',framework='pt',device='cpu') as left,safe_open(head/'rank1.safetensors',framework='pt',device='cpu') as right:
 for p in full.parameters:raw[p.name]=read_raw_parameter(p,(left,right)).float().add(1) if p.offset_norm else read_raw_parameter(p,(left,right))
paths={'l63_input_gamma':'input_layernorm.weight','l63_post_gamma':'post_attention_layernorm.weight','l63_mlp_gate':'mlp.gate_proj.weight','l63_mlp_up':'mlp.up_proj.weight','l63_mlp_down':'mlp.down_proj.weight'}
inputs={}
for name,value in full.inputs.items():
 if name in raw:x=raw[name]
 elif name in paths:
  x=weights.layer(63,paths[name]);x=(1+x.float()).reshape(value.shape) if name.endswith('gamma') else x
 elif name=='lm_head':x=weights.read('lm_head.weight')
 elif name=='final_gamma':x=(1+weights.read('model.language_model.norm.weight').float()).reshape(value.shape)
 elif name=='cos':x=torch.ones(value.shape)
 elif name=='sin':x=torch.zeros(value.shape)
 else:x=torch.randn(value.shape)*(.3 if name=='teacher' else .02)
 inputs[name]=x.to(torch.float32 if value.dtype=='fp32' else torch.bfloat16).contiguous()
report=dict(completed=False,scope='real weights and production shapes; synthetic hidden/history/teacher; kernel dispatch comparison only',visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),results={})
output=Path(os.environ['DISPATCH_REPORT'])
def save():output.write_text(json.dumps(report,indent=2))
g=Guard();w=None;work=[]
try:
 compiled={};graphs={}
 for scope in ['full','norms','output_norms']:
  graph=graphs[scope]=SuffixGraph(1,objective='tv',parameter_scope=scope)
  text,args=source(graph);report['results'][scope]=dict(nodes=len(graph.nodes),arguments=len(args)+4)
  if len(args)+4>256:raise RuntimeError('monolithic argument limit')
  for label,chunk in [('chunk24',24),('single',None)]:compiled[scope,label]=TrainingPrograms(graph,config,combined=True,chunk_size=chunk)
 g.close();w=DistributedWorker([p for programs in compiled.values() for p in programs.programs.values()],config=config,inherited_host_tensors=list(inputs.values()))
 values={name:w.alloc_tensor(tuple(x.shape),x.dtype,init=x) for name,x in inputs.items()}
 work={key:TrainingWorkspace(p,w) for key,p in compiled.items()}
 def read(tensor):
  x=torch.empty(tensor.shape,dtype=tensor.dtype);w.copy_from(x.data_ptr(),tensor.data_ptr,x.nbytes);return x
 def run(key):return work[key].run(values,past=64,width=32,valid=7)
 for scope in ['full','norms','output_norms']:
  references={}
  for label in ['chunk24','single']:
   gradients=run((scope,label));refs={name:read(t) for name,t in gradients.items()};ws=work[scope,label];graph=graphs[scope]
   refs['logits']=read(ws.values[graph.logits.name]);refs['losses']=read(ws.values[graph.losses.name]);references[label]=refs
  teacher=inputs['teacher'][:7].float().softmax(-1)
  logits=references['chunk24']['logits'][:7].float()
  expected_loss=(teacher-logits.softmax(-1)).abs().sum(-1).mul(.5).mean()
  actual_loss=references['chunk24']['losses'][:,0].sum()
  torch.testing.assert_close(actual_loss,expected_loss,atol=2e-5,rtol=.001)
  report['results'][scope]['torch_loss_reference']=dict(expected=expected_loss.item(),actual=actual_loss.item())
  diffs={}
  for name,x in references['chunk24'].items():
   y=references['single'][name];torch.testing.assert_close(y,x,atol=2e-6,rtol=.001)
   diffs[name]=dict(max_abs=(x.float()-y.float()).abs().max().item(),relative_l2=((x.float()-y.float()).norm()/x.float().norm().clamp_min(1e-20)).item())
  del references
  for label in ['chunk24','single']:
   for _ in range(2):run((scope,label))
  timings={label:[] for label in ['chunk24','single']}
  for repeat in range(8):
   for label in ['chunk24','single']:
    start=time.perf_counter();run((scope,label));read(work[scope,label].values[graphs[scope].losses.name]);timings[label].append(time.perf_counter()-start)
  medians={name:statistics.median(v) for name,v in timings.items()}
  report['results'][scope].update(differences=diffs,seconds=timings,medians=medians,speedup=medians['chunk24']/medians['single']);save()
 report['completed']=True
except BaseException as error:
 report['error']=repr(error);raise
finally:
 if work:
  for ws in work.values():ws.close()
 if w is not None:w.close()
 g.close();save()
