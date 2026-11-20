"""Use public weight files through a private view with preserved source metadata."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time

BASE=Path('/data/disk1/wangzili/orthrus-qwen38-pypto-20261105')
PUBLIC=Path('/data/disk/models/Qwen3.8-27B')
PARENT=BASE/'inference-tuning-20261116'
VIEW=PARENT/'model-view'
STAGE=PARENT/'model-view.verified-stage136'
REPORT=BASE/'oneserving-v109-b99e118/online-background/resume136-model-view.json'
METADATA = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0\n706cebd746c4b6f2b1d1f892630867acfdfd3df8\n1791780134.295195\n'
started=time.monotonic()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        while data:=f.read(8<<20):h.update(data)
    return h.hexdigest()

registry=BASE/'model-file-hashes.json'
expected=json.loads(registry.read_text())['files']
index=json.loads((PUBLIC/'model.safetensors.index.json').read_text())['weight_map']
shards=sorted(set(index.values()))
assert len(shards)==18
required=shards+['config.json','model.safetensors.index.json','tokenizer.json',
                 'tokenizer_config.json','vocab.json','merges.txt',
                 'generation_config.json','chat_template.jinja']
assert not VIEW.exists() and not STAGE.exists() and not REPORT.exists()
files={}
for name in required:
    path=PUBLIC/name
    before=path.stat()
    assert before.st_size==expected[name]['bytes'],name
    sha=digest(path)
    after=path.stat()
    assert (before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns),name
    assert sha==expected[name]['sha256'],name
    files[name]=dict(bytes=after.st_size,sha256=sha,inode=after.st_ino,mtime_ns=after.st_mtime_ns)
    print(json.dumps(dict(verified=len(files),total=len(required),file=name)),flush=True)
PARENT.mkdir(exist_ok=True)
STAGE.mkdir()
try:
    for name in required:(STAGE/name).symlink_to(PUBLIC/name)
    metadata=STAGE/'.cache/huggingface/download/config.json.metadata'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(METADATA)
    receipt=dict(completed=True,cpu_only=True,view=str(VIEW),public_model=str(PUBLIC),
                 registry=str(registry),registry_sha256=digest(registry),files=files,
                 metadata_source='Byte-for-byte copy of C existing model-view HF config receipt; contents match prior B full-file hash registry',
                 metadata_source_path='/data/disk1/wangzili/orthrus-qwen38-pypto-20261105/inference-tuning-20261116/model-view/.cache/huggingface/download/config.json.metadata on159.138.54.61',
                 metadata_sha256=hashlib.sha256(METADATA.encode()).hexdigest(),
                 source_script_sha256=digest(Path(__file__)),
                 weight_files_copied=0,public_directory_modified=False,
                 seconds=time.monotonic()-started)
    (STAGE/'identity-receipt136.json').write_text(json.dumps(receipt,indent=2)+'\n')
    STAGE.rename(VIEW)
    REPORT.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(completed=True,view=str(VIEW),verified_files=len(files),seconds=receipt['seconds'])),flush=True)
except BaseException:
    if STAGE.exists():shutil.rmtree(STAGE)
    raise
