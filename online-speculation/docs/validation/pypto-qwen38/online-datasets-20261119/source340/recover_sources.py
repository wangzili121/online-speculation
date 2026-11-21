"""恢复并逐文件核对冻结实验源码；不启动模型，也不安装运行环境。"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def recover(root, output=None):
    here = root / "source340"
    manifest = json.loads((here / "manifest.json").read_text())
    spoc = json.loads(gzip.decompress((root / "relocate318/source292_A_bundle.json.gz").read_bytes()))
    gsm = json.loads(gzip.decompress((root / "aux308/source233_bundle.json.gz").read_bytes()))
    extra = (here / "extra-guarded-sources.json.gz").read_bytes()
    assert sha(extra) == manifest["extra_archive_sha256"]
    texts = (list(spoc["files"].values()) + list(spoc["extras"].values())
             + list(gsm["files"].values()) + [gsm["benchmark"], (root / "aux308/run_dense_v308.py").read_text()])
    available = {sha(text.encode()): text.encode() for text in texts}
    for expected, text in json.loads(gzip.decompress(extra)).items():
        data = text.encode()
        assert sha(data) == expected
        available[expected] = data
    groups = {}
    for flavor, files in manifest["source_maps"].items():
        for name, expected in files.items():
            path = PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts and path.suffix == ".py"
            assert expected in available and sha(available[expected]) == expected, (flavor, name)
            ast.parse(available[expected], filename=name)
        groups[flavor] = dict(guarded_files=len(files), all_source_text_available=True,
                              all_hashes_match=True, syntax_parses=True)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        for flavor, files in manifest["source_maps"].items():
            base = output / flavor
            working = base / ("online-background" if flavor == "B_gsm" else "online-studies")
            for name, expected in files.items():
                path = (working if len(PurePosixPath(name).parts) == 1 else base) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(available[expected])
                assert sha(path.read_bytes()) == expected
            # 只补已有检查/资源选择辅助，不混入另一版本的模型入口。
            for name, text in spoc["extras"].items():
                if not (name.startswith("rows246-probe/") or name == "watch_glm53_validation.py"):
                    continue
                path = working / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            (base / "source-manifest.json").write_text(json.dumps(files, indent=2) + "\n")
            groups[flavor]["recovered_files_rehashed"] = True
    return dict(completed=True, groups=groups, unique_guarded_source_files=len(set(
        value for files in manifest["source_maps"].values() for value in files.values())),
                NN_started=False, environment_installed=False, weights_included=False,
                scope="Exact source-text coverage and byte recovery, not an offline runtime or a new NPU validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, help="新建源码目录；目标存在则拒绝写入")
    args = parser.parse_args()
    print(json.dumps(recover(args.validation_root, args.output), indent=2))
