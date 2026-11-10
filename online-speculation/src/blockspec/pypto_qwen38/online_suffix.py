"""独立 learner 更新末尾指定层；复用 last-1 的反馈和更新节奏。"""

from dataclasses import replace
import math

from .online import OnlineLearner,FeedbackBlock
from .online_feedback_programs import make_copy,make_gdn_history,make_history,make_loss_sum,make_teacher
from .online_suffix_optimizer import SuffixOptimizer,SuffixOptimizerPrograms
from .training_executor import TrainingPrograms,TrainingWorkspace
from .training_suffix_graph import SuffixGraph


class SuffixLearner(OnlineLearner):
    def __init__(self, head, options):
        super().__init__(head,options)
        self.graph = SuffixGraph(options.last_layers, objective=options.objective)

    def compile(self, samples, config):
        import torch
        if self.options.learner_device is None or self.options.learner_device in config.distributed_config.device_ids:
            raise ValueError("multi-layer updates require a separate learner device")
        self.config,self.dedicated = config,True
        self.training_config = replace(config,distributed_config=replace(config.distributed_config,
            device_ids=[self.options.learner_device]))
        self.training = TrainingPrograms(self.graph,self.training_config,combined=True,chunk_size=24,external_gradients=True)
        self.optimization = SuffixOptimizerPrograms(self.graph,self.training_config,clip_norm=self.options.clip_norm)
        meta = lambda shape,dtype=torch.bfloat16:torch.empty(shape,dtype=dtype,device="meta")
        self.history_programs = {layer:make_history(samples[31].shape[2],layer).compile(samples[31],samples[32],
            meta((32,),torch.int32),meta((2048,256)),meta((2048,256)),0,config=config)
            for layer in self.graph.histories if layer%4 == 3}
        self.gdn_history = {layer:make_gdn_history(layer).compile(samples[29],samples[30],
            meta((4,10240)),meta((6144,128),torch.float32),0,config=config)
            for layer in self.graph.histories if layer%4 != 3}
        self.teacher_program = make_teacher().compile(meta((8,16,248320)),meta((32,248320)),0,0,16,config=config)
        self.rope_program = make_copy(32,64).compile(meta((32,64)),meta((32,64)),config=config)
        self.loss_program = make_loss_sum().compile(meta((32,8),torch.float32),meta((1,8),torch.float32),config=self.training_config)
        self.training_programs = [*self.training.programs.values(),*self.optimization.programs(),self.loss_program]
        entries = sum(sum((path/"kernel_config.py").exists() for path in (program.output_dir/"next_levels").iterdir())
                      for program in self.training_programs)
        if entries > 64:
            raise RuntimeError(f"suffix learner exceeds pinned runtime capacity: {entries} > 64; reduce last-layers")
        from .online_bridge import TensorBridge
        self.bridge = TensorBridge()
        return [*self.history_programs.values(),*self.gdn_history.values(),self.teacher_program,self.rope_program]

    def start(self, native, draft):
        import torch
        from pypto.runtime import DistributedWorker
        from safetensors import safe_open
        from .head import HeadSource
        from .head_loading import raw_location
        from .online_bridge import SuffixPublisher
        from .online_checkpoint import identity
        from .online_suffix_checkpoint import restore
        self.native,self.draft = native,draft
        self.head_identity = identity(self.head)
        raw_norms = {}
        if self.options.resume_dir is None:
            raw = self.head if self.options.raw_head_dir is None else HeadSource(self.head.model_root,
                self.options.raw_head_dir,block_size=self.head.block_size,mask_token_id=self.head.mask_token_id)
            if raw.prepared_manifest is not None:
                raise ValueError("suffix initialization needs raw BF16 norms: pass --learner-head-dir")
            for parameter in self.graph.parameters:
                if not parameter.offset_norm:
                    continue
                name = parameter.name
                tensor = "q_norm.weight" if name.endswith("q_gamma") else "k_norm.weight"
                path,key = raw_location(raw,parameter.layer,tensor,0)
                with safe_open(path,framework="pt",device="cpu") as file:
                    raw_norms[name] = file.get_tensor(key).reshape(self.graph.inputs[name].shape)
        try:
            self.trainer = DistributedWorker(self.training_programs,config=self.training_config,
                inherited_host_tensors=[self.bridge.scratch])
            from .training_memory import require_suffix_memory
            self.memory_plan = require_suffix_memory(self.graph,self.optimization.layout,self.options.replay_blocks,
                native.worker.device_memory_info()[0],self.trainer.device_memory_info()[0])
            self.optimizer = SuffixOptimizer(self.optimization,self.trainer)
            self.workspace = TrainingWorkspace(self.training,self.trainer)
            self.loss_device = self.alloc((1,8),torch.float32,training=True)
            self.loss_host = torch.empty((1,8),dtype=torch.float32)
            self.serving_banks = [(self.alloc(self.optimizer.layout.shape,torch.bfloat16),
                self.alloc(self.optimizer.layout.special_shape,torch.float32)) for _ in range(2)]
            self.publisher = SuffixPublisher(self,draft.weights)
            if self.options.resume_dir is None:
                self.optimizer.initialize(self.head,raw_norms)
                # 初始末段也驻留在推理卡上，避免每轮重传待更新层。
                self.publisher.publish_flat(self.optimizer.layout,self.optimizer.execution[0],expected_version=0)
                draft.weights.version = 0
            else:
                restore(self.optimizer,self.publisher,self.options.resume_dir,self.head_identity)
                self.updates = self.optimizer.steps
                self.rounds = self.updates*self.options.stride
            for _ in range(self.options.replay_blocks):
                names = {"hidden","cos","sin","teacher"}|{value.name for values in self.graph.histories.values() for value in values.values()}
                inputs = {name:self.alloc(self.graph.inputs[name].shape,
                    torch.float32 if self.graph.inputs[name].dtype == "fp32" else torch.bfloat16) for name in names}
                block = FeedbackBlock(self,inputs)
                block.training_inputs = {name:self.alloc(value.shape,value.dtype,training=True) for name,value in inputs.items()}
                self.pool.append(block)
            self.shared = {}
            for name,(bank,offset) in self.graph.frozen.items():
                value = self.graph.inputs[name]
                target = self.alloc(value.shape,torch.float32 if value.dtype == "fp32" else torch.bfloat16,training=True)
                self.bridge.copy(native.worker,native.banks[bank],self.trainer,target,source_offset=offset)
                self.shared[name] = target
            print("online suffix learner ready",self.options.last_layers,
                  "free bytes",self.trainer.device_memory_info()[0],flush=True)
        except BaseException:
            self.close()
            raise

    def begin(self, draft, past, width, *, slot=None):
        if self.current is not None:
            raise RuntimeError("previous online feedback has not been committed or aborted")
        if self.options.stride-self.rounds%self.options.stride > self.options.replay_blocks:
            return None
        if slot is None:
            raise ValueError("GDN feedback requires the request state slot")
        used = {id(item) for item in self.replay}
        block = next((item for item in self.pool if id(item) not in used),None)
        if block is None:
            block = self.replay.popleft()
        block.past,block.width,block.rows,block.valid = past,width,0,0
        self.current = block
        for layer,program in self.history_programs.items():
            self.run(program,*self.native.resident[2:],draft.table,
                block.inputs[f"l{layer}_history_key"],block.inputs[f"l{layer}_history_value"],past)
        for layer,program in self.gdn_history.items():
            self.run(program,*self.native.resident[:2],block.inputs[f"l{layer}_conv_history"],
                     block.inputs[f"l{layer}_recurrent_history"],slot)
        self.run(self.rope_program,draft.cos,block.inputs["cos"])
        self.run(self.rope_program,draft.sin,block.inputs["sin"])
        return block

    def update(self):
        self.optimizer.zero_grad()
        positions = sum(item.valid for item in self.replay)
        loss = 0.
        for item in self.replay:
            item.transfer()
            inputs = {**self.shared,**item.training_inputs,**self.optimizer.tensors()}
            self.workspace.run(inputs,past=item.past,width=item.width,valid=item.valid,normalizer=positions,
                               gradients=self.optimizer.gradients)
            self.optimizer.accumulate()
            handle = self.workspace.values[self.graph.losses.name]
            self.trainer.run(self.loss_program,handle,self.loss_device,config=self.training_config)
            self.trainer.copy_from(self.loss_host.data_ptr(),self.loss_device.data_ptr,32)
            loss += self.loss_host[0,0].item()
        if not math.isfinite(loss):
            raise FloatingPointError("nonfinite online loss; head not published")
        result = self.optimizer.step(self.publisher,self.options.learning_rate)
        self.updates += 1
        self.last_result = dict(loss=loss,reached_positions=positions,**result)
        if self.options.checkpoint_dir is not None:
            from pathlib import Path
            from .online_suffix_checkpoint import save
            directory = Path(self.options.checkpoint_dir)/f"step-{self.optimizer.steps:08d}"
            save(self.optimizer,directory,self.head_identity)
            self.last_result["checkpoint"] = str(directory)
        return self.last_result
