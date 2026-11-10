"""最后 2/3/4 层的显式梯度图，沿用 last-1 的 Full Attention 子图。"""

from dataclasses import dataclass

from .training_gdn_reference import GDNShape
from .training_graph import GraphBuilder, LastLayerGraph, Node, SuffixShape, Value


@dataclass(frozen=True)
class Parameter:
    name: str
    layer: int
    bank: int
    shape: tuple
    raw_dtype: str = "bf16"
    offset_norm: bool = False

    @property
    def execution_dtype(self):
        return "fp32" if self.offset_norm else self.raw_dtype


class SuffixGraph(GraphBuilder):
    def __init__(self, last_layers, shape=SuffixShape(), gdn_shape=GDNShape()):
        if type(last_layers) is not int or not 2 <= last_layers <= 4 or shape.block != gdn_shape.rows:
            raise ValueError("last-2/3/4 and a common K32 block required")
        self.last_layers, self.capture_layer = last_layers, 64-last_layers
        self.shape, self.gdn_shape = shape, gdn_shape
        self.inputs, self.nodes, self.gradients = {}, [], {}
        self.parameters, self.frozen, self.histories = [], {}, {}
        hidden = self.input("hidden", (shape.block, shape.hidden))
        layers = []
        for layer in range(self.capture_layer, 63):
            block = GDNLayer(self, layer, hidden)
            layers.append(block)
            hidden = block.output
        full = LastLayerGraph(shape, backprop_input=True)
        values = {}
        shared = {"teacher", "cos", "sin", "final_gamma", "lm_head"}
        for name, value in full.inputs.items():
            if name == "hidden":
                values[name] = hidden
            else:
                key = name if name in shared else "l63_"+name
                values[name] = self.input(key, value.shape, value.dtype)
        order = ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight")
        for bank,name in enumerate(order):
            value = full.inputs[name]
            self.parameters.append(Parameter("l63_"+name,63,bank,value.shape,offset_norm="gamma" in name))
        for name in ("input_gamma", "post_gamma", "mlp_gate", "mlp_up", "mlp_down"):
            self.bind_frozen("l63_"+name,name,63)
        self.bind_frozen("final_gamma","final_gamma",None)
        self.bind_frozen("lm_head","lm_head",None)
        self.histories[63] = {name:values[name] for name in ("history_key","history_value")}
        for node in full.nodes:
            mapped = tuple(Value("fa_"+v.name,v.shape,v.dtype) for v in node.outputs)
            self.nodes.append(Node(node.kind,tuple(values[v.name] for v in node.inputs),mapped,node.options))
            values.update({v.name:clone for v,clone in zip(node.outputs,mapped,strict=True)})
        self.logits, self.losses = values[full.logits.name], values[full.losses.name]
        self.gradients.update({"l63_"+name:values[v.name] for name,v in full.gradients.items()})
        gradient = values[full.input_gradient.name]
        for block in reversed(layers):
            gradient = block.backward(gradient, propagate=block.layer != self.capture_layer)
        self.input_gradient = None
        self.parameters.sort(key=lambda p:(p.layer,p.bank))
        assert set(self.gradients) == {p.name for p in self.parameters}

    def input(self, name, shape, dtype="bf16"):
        value = Value(name,tuple(shape),dtype)
        if name in self.inputs and self.inputs[name] != value:
            raise ValueError("conflicting suffix input "+name)
        self.inputs[name] = value
        return value

    def bind_frozen(self, key, kind, layer):
        H,I = self.shape.hidden,self.shape.intermediate
        bank, stride = {"input_gamma":(3,H*4),"post_gamma":(4,H*4),
                        "mlp_gate":(5,H*I*2),"mlp_up":(6,H*I*2),"mlp_down":(7,H*I*2),
                        "final_gamma":(1,0),"lm_head":(2,0)}[kind]
        self.frozen[key] = bank,0 if layer is None else layer*stride


class GDNLayer:
    def __init__(self, graph, layer, hidden):
        self.graph, self.layer, self.hidden = graph, layer, hidden
        g,s = graph,graph.gdn_shape
        K,H,I,D,C = s.rows,g.shape.hidden,g.shape.intermediate,s.dimension,s.channels
        self.prefix = f"l{layer}_"
        self.xnorm,self.postnorm = self.frozen("input_gamma",(1,H),"fp32"),self.frozen("post_gamma",(1,H),"fp32")
        self.wgate,self.wup,self.wdown = (self.frozen("mlp_gate",(I,H)),self.frozen("mlp_up",(I,H)),
                                        self.frozen("mlp_down",(H,I)))
        self.weights = {}
        definitions = (("qkv_weight",(C,H),"bf16"),("z_weight",(s.heads*D,H),"bf16"),
            ("a_weight",(128,H),"bf16"),("b_weight",(128,H),"bf16"),("conv_weight",(4,C),"bf16"),
            ("a_log",(1,64),"fp32"),("dt_bias",(1,64),"bf16"),("norm_weight",(1,D),"bf16"),
            ("out_weight",(H,s.heads*D),"bf16"))
        for bank,(name,shape,dtype) in enumerate(definitions):
            key = self.prefix+name
            self.weights[name] = g.input(key,shape,dtype)
            g.parameters.append(Parameter(key,layer,bank,shape,dtype))
        w = self.weights
        self.conv_history = g.input(self.prefix+"conv_history",(4,C))
        self.recurrent_history = g.input(self.prefix+"recurrent_history",(s.heads*D,D),"fp32")
        g.histories[layer] = dict(conv_history=self.conv_history,recurrent_history=self.recurrent_history)
        self.x,self.xinv = g.norm(hidden,self.xnorm)
        self.qkv,self.z,self.a,self.b = (g.mm(self.x,w[name]) for name in ("qkv_weight","z_weight","a_weight","b_weight"))
        self.conv,self.pre = g.op("gdn_conv",(self.qkv,w["conv_weight"],self.conv_history),
                                 (((K,C),"bf16"),((K,C),"bf16")))
        self.decay,self.beta = g.op("gdn_gates",(self.a,self.b,w["a_log"],w["dt_bias"]),
                                    (((K,64),"fp32"),((K,64),"fp32")))
        self.core,self.states = g.op("gdn_recurrence",(self.conv,self.decay,self.beta,self.recurrent_history),
            (((K,s.heads*D),"bf16"),(((K+1)*s.heads*D,D),"fp32")))
        self.gated,self.normed,self.weighted,self.inverse = g.op("gdn_tail",(self.core,self.z,w["norm_weight"]),
            (((K,s.heads*D),"bf16"),((K*s.heads,D),"bf16"),((K*s.heads,D),"bf16"),((K*s.heads,8),"fp32")))
        self.attention = g.mm(self.gated,w["out_weight"])
        self.skip = g.element(hidden,self.attention,"add")
        self.post,self.postinv = g.norm(self.skip,self.postnorm)
        self.pregate,self.up = g.mm(self.post,self.wgate),g.mm(self.post,self.wup)
        self.silu = g.element(self.pregate,self.pregate,"silu")
        self.middle = g.element(self.silu,self.up,"mul")
        self.output = g.element(self.skip,g.mm(self.middle,self.wdown),"add")

    def frozen(self, name, shape, dtype="bf16"):
        key = self.prefix+name
        value = self.graph.input(key,shape,dtype)
        self.graph.bind_frozen(key,name,self.layer)
        return value

    def backward(self, dy, *, propagate):
        g,s,w = self.graph,self.graph.gdn_shape,self.weights
        K,C,D,V = s.rows,s.channels,s.dimension,s.heads
        dm = g.mm(dy,self.wdown,transpose_b=False)
        du = g.element(dm,self.silu,"mul")
        dpre = g.element(self.pregate,g.element(dm,self.up,"mul"),"silu_grad")
        dp = g.element(g.mm(du,self.wup,transpose_b=False),g.mm(dpre,self.wgate,transpose_b=False),"add")
        ds = g.element(dy,g.norm_dx(self.skip,self.postnorm,self.postinv,dp),"add")
        gradients = {"out_weight":g.mm(ds,self.gated,transpose_a=True,transpose_b=False,gradient_weight=True)}
        dgated = g.mm(ds,w["out_weight"],transpose_b=False)
        dc,dz,dweighted = g.op("gdn_tail_dx",(self.core,self.z,w["norm_weight"],self.weighted,self.inverse,dgated),
            (((K,V*D),"bf16"),((K,V*D),"bf16"),((K*V,D),"bf16")))
        gradients["norm_weight"] = g.op("gdn_tail_dw",(self.normed,dweighted),(((1,D),"fp32"),))[0]
        partials = g.op("gdn_recurrence_dx",(self.conv,self.decay,self.beta,self.states,dc),
            (((K*V*(D//16),D),"fp32"),((K*V*(D//16),D),"fp32"),((K,V*D),"bf16"),
             ((K*V*(D//16),8),"fp32"),((K*V*(D//16),8),"fp32")))
        dconv,dg,dbeta = g.op("gdn_recurrence_reduce",(self.conv,*partials),
                            (((K,C),"bf16"),((K,64),"fp32"),((K,64),"fp32")))
        da,db,pa,pbias = g.op("gdn_gates_dx",(self.a,self.b,w["a_log"],w["dt_bias"],dg,dbeta),
            (((K,128),"bf16"),((K,128),"bf16"),((K,64),"fp32"),((K,64),"fp32")))
        gradients["a_log"],gradients["dt_bias"] = g.op("gdn_gates_dw",(pa,pbias),(((1,64),"fp32"),((1,64),"fp32")))
        dqkv,gradients["conv_weight"] = g.op("gdn_conv_dx",(self.qkv,w["conv_weight"],self.conv_history,self.pre,dconv),
                                           (((K,C),"bf16"),((4,C),"fp32")))
        projections = (("qkv_weight",dqkv),("z_weight",dz),("a_weight",da),("b_weight",db))
        for name,delta in projections:
            gradients[name] = g.mm(delta,self.x,transpose_a=True,transpose_b=False,gradient_weight=True)
        g.gradients.update({self.prefix+name:value for name,value in gradients.items()})
        if not propagate:
            return None
        dx = None
        for name,delta in projections:
            part = g.mm(delta,w[name],transpose_b=False)
            dx = part if dx is None else g.element(dx,part,"add")
        return g.element(ds,g.norm_dx(self.hidden,self.xnorm,self.xinv,dx),"add")
