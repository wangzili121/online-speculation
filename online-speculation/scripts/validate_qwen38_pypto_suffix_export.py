#!/usr/bin/env python3
"""CPU 文件对照：导出 head 的更新层匹配 checkpoint，其他字节保持不变。"""

import argparse
import hashlib
import json
from math import prod
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model","head-dir","exported-head","checkpoint","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    import torch
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.head_loading import host_layout
    from blockspec.pypto_qwen38.online_checkpoint import identity
    from blockspec.pypto_qwen38.online_suffix_checkpoint import read
    from blockspec.pypto_qwen38.training_flat import FlatLayout
    from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
    head = HeadSource(args.model,args.head_dir)
    exported = HeadSource(args.model,args.exported_head)
    if head.prepared_manifest is None or exported.prepared_manifest is None:
        parser.error("source and exported directories must contain prepared payloads")
    for field in ("format","contract","runtime_tp","runtime_rank","producer_pypto","payload_bytes"):
        assert head.prepared_manifest[field] == exported.prepared_manifest[field],field
    manifest = json.loads((args.checkpoint/"manifest.json").read_text())
    count = len({p["layer"] for p in manifest["layout"]})
    layout = FlatLayout(SuffixGraph(count).parameters)
    step,values = read(args.checkpoint,identity(head),layout)
    master = values["master"]
    _,offsets,total = host_layout(head)
    changes = sorted(offsets[e.parameter.layer][e.parameter.bank] for e in layout.entries)
    source_hash,result_hash = hashlib.sha256(),hashlib.sha256()
    old_path,new_path = args.head_dir/"weights.bin",args.exported_head/"weights.bin"
    assert old_path.stat().st_size == new_path.stat().st_size == total
    with old_path.open("rb") as source,new_path.open("rb") as result:
        for begin in range(0,total,16<<20):
            old,new = source.read(16<<20),result.read(16<<20)
            assert len(old) == len(new)
            source_hash.update(old);result_hash.update(new)
            end,cursor = begin+len(old),begin
            for offset,size in changes:
                if offset+size <= begin or offset >= end:
                    continue
                assert old[cursor-begin:max(cursor,offset)-begin] == new[cursor-begin:max(cursor,offset)-begin]
                cursor = min(end,offset+size)
            assert old[cursor-begin:] == new[cursor-begin:]
    assert source_hash.hexdigest() == head.prepared_manifest["payload_sha256"]
    assert result_hash.hexdigest() == exported.prepared_manifest["payload_sha256"]
    checks = []
    with new_path.open("rb") as file:
        for entry in layout.entries:
            p = entry.parameter
            raw = master[0,entry.offset:entry.offset+prod(p.shape)].reshape(p.shape)
            expected = raw.bfloat16().float()+1 if p.offset_norm else raw if p.raw_dtype == "fp32" else raw.bfloat16()
            offset,size = offsets[p.layer][p.bank]
            file.seek(offset)
            actual = torch.frombuffer(bytearray(file.read(size)),dtype=expected.dtype).reshape(p.shape)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            checks.append(p.name)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True,step=step,last_layers=count,
        source_sha256=source_hash.hexdigest(),exported_sha256=result_hash.hexdigest(),
        frozen_regions_byte_exact=True,checkpoint_execution_byte_exact=True,parameters=checks),indent=2)+"\n")


if __name__ == "__main__":
    main()
