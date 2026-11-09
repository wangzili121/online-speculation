"""加载区间、缓存契约与越界检查；数值转换另在真实 NPU 核对。"""

import json
from types import SimpleNamespace

import pytest
import torch

import test_pypto_qwen38_tp as fixture_definitions
from test_pypto_qwen38_tp import make_checkpoint, reference_shard
from blockspec.pypto_qwen38.head import HeadSource
from blockspec.pypto_qwen38.loading import DeviceLoader, parameter_pieces
from blockspec.pypto_qwen38.prepared_head import save
from blockspec.pypto_qwen38.tp import TPLayout


small_weights = fixture_definitions.small_weights


@pytest.mark.parametrize("source_tp", (1, 2, 4))
@pytest.mark.parametrize("target_tp", (1, 2, 4))
@pytest.mark.parametrize("layer", (0, 3))
def test_file_to_device_offsets_match_reference_resharding(small_weights, source_tp, target_tp, layer):
    for name, full in small_weights[layer].items():
        sources = [reference_shard(name, full, source_tp, rank) for rank in range(source_tp)]
        for rank in range(target_tp):
            axis, pieces = parameter_pieces(layer, name, source_tp, TPLayout(target_tp, rank))
            expected = reference_shard(name, full, target_tp, rank)
            if name.endswith("norm.weight"):
                assert all(torch.equal(sources[p.rank], expected) for p in pieces)
                continue
            actual = torch.empty_like(expected)
            for piece in pieces:
                src, dst = [slice(None)] * full.ndim, [slice(None)] * full.ndim
                src[axis] = slice(piece.source, piece.source + piece.width)
                dst[axis] = slice(piece.destination, piece.destination + piece.width)
                actual[tuple(dst)] = sources[piece.rank][tuple(src)]
            assert torch.equal(actual, expected)


def fixture_head(tmp_path, small_weights):
    model, directory = make_checkpoint(tmp_path, small_weights, 4)
    head = HeadSource(model, directory)
    head.prepare()
    # 仅构造缓存协议测试夹具；实机脚本单独验证 PyPTO 的计算结果。
    head.preparation_backend = "pypto"
    output = tmp_path / "prepared"
    save(head, output)
    return model, head, output


def test_prepared_head_roundtrip_uses_views_without_reformatting(tmp_path, small_weights, monkeypatch):
    model, source, directory = fixture_head(tmp_path, small_weights)
    actual = HeadSource(model, directory)
    monkeypatch.setattr(actual, "layer", lambda *a: pytest.fail("prepared weights must not reconstruct raw tensors"))
    actual.prepare()
    assert actual.preparation_backend == "pypto"
    assert torch.equal(actual.shared_weights, source.shared_weights)
    for layer in range(64):
        for expected, value in zip(source.banks(layer), actual.banks(layer), strict=True):
            assert torch.equal(value.view(torch.uint8), expected.view(torch.uint8))
    with pytest.raises(FileExistsError):
        save(source, directory)


@pytest.mark.parametrize("error", ("base", "runtime_tp", "runtime_rank", "producer", "size", "hash", "bytes"))
def test_prepared_head_rejects_corruption_and_contract_changes(tmp_path, small_weights, error):
    model, source, directory = fixture_head(tmp_path, small_weights)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if error == "base":
        manifest["contract"]["base_config_sha256"] = "0" * 64
    elif error in {"runtime_tp", "runtime_rank"}:
        manifest[error] = 2
    elif error == "producer":
        manifest["producer_pypto"] = "other"
    elif error == "size":
        manifest["payload_bytes"] += 1
    elif error == "hash":
        manifest["payload_sha256"] = "0" * 64
    elif error == "bytes":
        payload = directory / "weights.bin"
        data = bytearray(payload.read_bytes())
        data[0] ^= 1
        payload.write_bytes(data)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        candidate = HeadSource(model, directory)
        candidate.prepare()


def test_host_prepared_head_cannot_claim_pypto_provenance(tmp_path, small_weights):
    model, directory = make_checkpoint(tmp_path, small_weights, 1)
    head = HeadSource(model, directory)
    head.prepare()
    with pytest.raises(ValueError, match="PyPTO"):
        save(head, tmp_path / "bad")


def test_updated_head_export_preserves_frozen_layers_and_source(tmp_path, small_weights):
    import ctypes
    model, source, _ = fixture_head(tmp_path, small_weights)
    before = source.shared_weights.clone()
    updated = [(bank + .125).contiguous() for bank in source.banks(63)]
    devices = [SimpleNamespace(shape=tuple(bank.shape), dtype=bank.dtype, data_ptr=bank.data_ptr()) for bank in updated]
    class CopyWorker:
        @staticmethod
        def copy_from(target, pointer, size, *, src_offset=0):
            ctypes.memmove(target, pointer+src_offset, size)
    output = tmp_path/"updated"
    save(source, output, last_layer=devices, worker=CopyWorker())
    actual = HeadSource(model, output)
    actual.prepare()
    torch.testing.assert_close(source.shared_weights, before, atol=0, rtol=0)
    for layer in range(64):
        expected = updated if layer == 63 else source.banks(layer)
        for got, wanted in zip(actual.banks(layer), expected, strict=True):
            torch.testing.assert_close(got, wanted, atol=0, rtol=0)
    with pytest.raises(ValueError, match="six"):
        save(source, tmp_path/"invalid", last_layer=devices[:5], worker=CopyWorker())
    assert not (tmp_path/"invalid").exists()


@pytest.mark.parametrize("offset", ((-1, 0), (0, 128), (True, 0)))
def test_device_copy_bounds_rejected_before_dispatch(offset):
    loader = object.__new__(DeviceLoader)
    loader.registry = SimpleNamespace(copies={})
    key = ((1, 128), (1, 128), torch.bfloat16, torch.bfloat16, False, False, (1, 128))
    value = SimpleNamespace(shape=(1, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="layout"):
        loader.copy(key, value, value, source_offset=offset)
