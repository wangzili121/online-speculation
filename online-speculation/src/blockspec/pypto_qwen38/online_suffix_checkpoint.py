"""last-N 独立 checkpoint 契约，绑定基模、初始 head 和完整参数布局。"""

import json
from pathlib import Path

from .online_checkpoint import sha256, STATES

FORMAT = "orthrus-qwen38-pypto-lastn-adam-v2"


def contract(layout):
    return json.loads(json.dumps(layout.schema()))


def save(optimizer, directory, head_identity):
    import torch
    from safetensors.torch import save_file
    if optimizer.steps < 1:
        raise ValueError("checkpoint requires a committed suffix update")
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    tensors = {}
    for name,handle in zip(STATES,optimizer.state,strict=True):
        value = torch.empty(handle.shape,dtype=torch.float32)
        optimizer.worker.copy_from(value.data_ptr(),handle.data_ptr,value.numel()*4)
        tensors[name] = value
    path = directory/"optimizer.safetensors"
    save_file(tensors,str(path.with_suffix(".part")))
    path.with_suffix(".part").rename(path)
    manifest = dict(format=FORMAT,head=head_identity,step=optimizer.steps,layout=contract(optimizer.layout),
                    beta1=.9,beta2=.999,epsilon=1e-8,weight_decay=0.,payload_sha256=sha256(path))
    (directory/"manifest.json.part").write_text(json.dumps(manifest,indent=2)+"\n")
    (directory/"manifest.json.part").rename(directory/"manifest.json")
    return manifest


def read(directory, head_identity, layout):
    import torch
    from safetensors.torch import load_file
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    expected = dict(format=FORMAT,head=head_identity,step=manifest.get("step"),layout=contract(layout),
                    beta1=.9,beta2=.999,epsilon=1e-8,weight_decay=0.,payload_sha256=manifest.get("payload_sha256"))
    if manifest != expected or type(manifest["step"]) is not int or manifest["step"] < 1:
        raise ValueError("incompatible suffix optimizer checkpoint")
    path = directory/"optimizer.safetensors"
    if sha256(path) != manifest["payload_sha256"]:
        raise ValueError("suffix optimizer SHA256 mismatch")
    values = load_file(str(path),device="cpu")
    if set(values) != set(STATES) or any(tuple(value.shape) != layout.shape or value.dtype != torch.float32
                                       for value in values.values()):
        raise ValueError("suffix optimizer tensor schema changed")
    return manifest["step"],values


def restore(optimizer, publisher, directory, head_identity):
    if optimizer.steps or publisher.version:
        raise RuntimeError("restore is only allowed before serving starts")
    step,values = read(directory,head_identity,optimizer.layout)
    for name,handle in zip(STATES,optimizer.state,strict=True):
        value = values[name]
        optimizer.worker.copy_to(handle.data_ptr,value.data_ptr(),value.numel()*4)
        optimizer.statistics(handle)
    optimizer.render(0,optimizer.state[0])
    publisher.publish_flat(optimizer.layout,optimizer.execution[0],expected_version=0)
    optimizer.steps,optimizer.index,publisher.version = step,0,step
    optimizer.zero_grad()
