from pathlib import Path
import json,time
import torch
from pypto.backend import BackendType
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig,ExecutionMode
from blockspec.pypto_qwen38.training_programs import make_tv
from blockspec.pypto_qwen38.training_executor import TrainingPrograms
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
r=Path(__file__).parent;report=dict(completed=False,scope='CPU compilation only; no NPU worker constructed',cases=[]);config=RunConfig(backend_type=BackendType.Ascend910B,execution_mode=ExecutionMode.ONBOARD,distributed_config=DistributedConfig(device_ids=[0],num_sub_workers=0));torch.set_num_threads(4)
try:
 for vocabulary in [128,1024,248320]:
  for weighted in [False,True]:
   tensors=[torch.empty((32,vocabulary),dtype=torch.bfloat16,device='meta')for _ in range(2)];tensors+=([torch.empty((32,8),device='meta')]if weighted else[]);tensors+=[torch.empty((32,vocabulary),dtype=torch.bfloat16,device='meta'),torch.empty((32,8),device='meta')];started=time.monotonic()
   program=make_tv(32,vocabulary,log_overlap=True,position_weighted=weighted).compile(*tensors,7,21,config=config);report['cases'].append(dict(vocabulary=vocabulary,weighted=weighted,output=str(program.output_dir),seconds=time.monotonic()-started))
 for layers in [1,4]:
  started=time.monotonic();graph=SuffixGraph(layers,objective='log_overlap');compiled=TrainingPrograms(graph,config,combined=True,chunk_size=24 if layers==4 else None,accumulate_gradients=True);report['cases'].append(dict(last_layers=layers,programs=[str(p.output_dir)for p in compiled.programs.values()],seconds=time.monotonic()-started))
 report['completed']=True
except BaseException as e:
 report['error']=repr(e);raise
finally:(r/'compile-report.json').write_text(json.dumps(report,indent=2)+'\n')
