"""训练诊断用 Torch 前向；梯度由 autograd 生成，不执行手写 VJP。"""


class AutogradReference:
    def __init__(self, graph, shared, masters, *, quantize=True):
        import torch
        self.graph, self.shared, self.masters = graph, shared, masters
        self.quantize = quantize
        self.parameters = ({p.name:p for p in graph.parameters} if hasattr(graph,"parameters") else {})
        # 只取 logits 的前向依赖，绝不把手写 backward 节点当作参考梯度。
        needed, selected = {graph.logits.name}, []
        for node in reversed(graph.nodes):
            if any(value.name in needed for value in node.outputs):
                selected.append(node)
                needed.update(value.name for value in node.inputs)
        self.nodes = list(reversed(selected))
        self.float_shared = {name:value.float() for name,value in shared.items() if value.ndim==2}
        self.torch = torch

    def cast(self, value):
        return value.bfloat16() if self.quantize else value

    def execution(self):
        result = {}
        for name, master in self.masters.items():
            parameter = self.parameters.get(name)
            offset = parameter.offset_norm if parameter is not None else "gamma" in name
            dtype = parameter.raw_dtype if parameter is not None else "bf16"
            if self.quantize and dtype == "bf16":
                # 执行值按 BF16 舍入，master 的 dW 仍以 FP32 累加。
                result[name] = master + (master.bfloat16().float()-master).detach() + (1 if offset else 0)
            else:
                result[name] = master + (1 if offset else 0)
        return result

    def forward(self, block, *, past, width):
        torch, s = self.torch, self.graph.shape
        F = torch.nn.functional
        values = {**self.shared, **block, **self.execution()}
        for name in ("hidden","cos","sin"):
            values[name] = values[name][:width]

        for node in self.nodes:
            args = [values[v.name] for v in node.inputs]
            cfg, kind = node.options, node.kind
            if kind == "linear":
                a,b = args
                batches,repeats = cfg["batches"],cfg["b_repeats"]
                a = a.float().reshape(batches,-1,a.shape[-1])
                b = self.float_shared.get(node.inputs[1].name,b).float()
                b = b.reshape(batches//repeats,-1,b.shape[-1])
                if repeats != 1:b = b.repeat_interleave(repeats,0)
                if cfg["transpose_a"]:a = a.transpose(-1,-2)
                if cfg["transpose_b"]:b = b.transpose(-1,-2)
                product = self.cast(a@b)
                out = (product.reshape(batches*product.shape[1],product.shape[2]),)
            elif kind == "norm":
                x,gamma = args
                inv = (x.float().square().mean(-1,keepdim=True)+1e-6).rsqrt()
                out = (self.cast(x.float()*inv*gamma),inv.expand(-1,8))
            elif kind == "element":
                a,b = (x.float() for x in args)
                op = cfg["kind"]
                if op == "add":value = a+b
                elif op == "mul":value = a*b
                elif op == "silu":value = F.silu(a)
                elif op == "sigmoid":value = a.sigmoid()
                elif op == "scale":value = a*cfg["scale"]
                else:raise ValueError("backward operation in forward golden: "+op)
                out = (self.cast(value),)
            elif kind == "layout":
                x, = args
                heads = cfg["heads"]
                if cfg["mode"] == "pack":
                    x = x.reshape(width,heads,-1)
                    begin = s.dimension if cfg["gate"] else 0
                    value = x[:,:,begin:begin+s.dimension].transpose(0,1).reshape(heads*width,s.dimension)
                else:
                    value = x.reshape(heads,width,s.dimension).transpose(0,1).reshape(width,heads*s.dimension)
                out = (value,)
            elif kind == "rope":
                x,cos,sin = args
                x = x.reshape(-1,width,s.dimension)
                a,b = x[...,:s.rotary//2].float(),x[...,s.rotary//2:s.rotary].float()
                c0,c1 = cos.float().chunk(2,-1);s0,s1 = sin.float().chunk(2,-1)
                out = (torch.cat((self.cast(a*c0-b*s0),self.cast(b*c1+a*s1),x[...,s.rotary:]),-1)
                       .reshape(-1,s.dimension),)
            elif kind == "join":
                x,history = args
                old = history.reshape(s.kv_heads,s.context,s.dimension)[:,:past]
                current = x.reshape(s.kv_heads,width,s.dimension)
                pad = x.new_zeros((s.kv_heads,s.context-past-width,s.dimension))
                out = (torch.cat((old,current,pad),1).reshape_as(history),)
            elif kind == "softmax":
                x, = args
                bias = x.new_zeros((1,s.context),dtype=torch.float32)
                bias[:,past+width:] = -10000.
                p = (x.float()+bias).softmax(-1)
                out = (p,self.cast(p))
            elif kind.startswith("gdn_"):
                from . import training_gdn_reference as ref
                if kind == "gdn_conv":out = ref.convolution_forward(*args,width)
                elif kind == "gdn_gates":
                    decay,beta = ref.gates_forward(*args,self.graph.gdn_shape.heads)
                    out = tuple(F.pad(x,(0,64-self.graph.gdn_shape.heads)) for x in (decay,beta))
                elif kind == "gdn_recurrence":
                    x,g,b,initial = args;gdn = self.graph.gdn_shape
                    y,states,_ = ref.recurrence_forward(x,g[:,:gdn.heads],b[:,:gdn.heads],
                        initial.reshape(gdn.heads,gdn.dimension,gdn.dimension),width,gdn)
                    out = (y[:width].reshape(width,-1),states.reshape(-1,gdn.dimension))
                elif kind == "gdn_tail":
                    x,z,weight = args;gdn = self.graph.gdn_shape
                    out = ref.tail_forward(x.reshape(width,gdn.heads,gdn.dimension),z,weight)
                else:raise ValueError("backward operation in forward golden: "+kind)
            else:
                raise ValueError("unsupported forward operation: "+kind)
            values.update((value.name,tensor) for value,tensor in zip(node.outputs,out,strict=True))
        return values[self.graph.logits.name]

    def loss(self, block, *, past, width, valid, normalizer):
        torch = self.torch
        logits = self.forward(block,past=past,width=width)
        teacher = block["teacher"][:valid].float().softmax(-1)
        loss = torch.nn.functional.kl_div(logits[:valid].float().log_softmax(-1),teacher,reduction="sum")/normalizer
        return loss,logits
