from __future__ import annotations

import numpy as np
import pytest


gguf = pytest.importorskip("gguf")

from gguf_compress.gguf_io import CUSTOM_ARCH, compress_gguf, decompress_gguf


def _field_text(reader, key: str) -> str:
    field = reader.get_field(key)
    assert field is not None
    return str(field.contents())


def _write_tiny_gguf(path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    left = rng.standard_normal((64, 4), dtype=np.float32)
    right = rng.standard_normal((4, 64), dtype=np.float32)
    weight = np.ascontiguousarray(left @ right, dtype=np.float32)
    bias = np.arange(64, dtype=np.float32)

    writer = gguf.GGUFWriter(path, arch="test")
    writer.add_string("general.name", "tiny-roundtrip")
    writer.add_tensor("blk.0.weight", weight)
    writer.add_tensor("blk.0.bias", bias)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"blk.0.weight": weight, "blk.0.bias": bias}


def test_compress_decompress_roundtrip(tmp_path):
    source = tmp_path / "source.gguf"
    packed = tmp_path / "packed.gguf"
    restored = tmp_path / "restored.gguf"
    original = _write_tiny_gguf(source)

    stats = compress_gguf(source, packed, method="svd-lowrank", rank=4, qbits=8, min_elements=1, oversample=2, power_iters=1)
    assert stats.factorized_tensors == 1
    assert stats.passthrough_tensors == 1

    packed_reader = gguf.GGUFReader(packed, "r")
    assert _field_text(packed_reader, gguf.Keys.General.ARCHITECTURE) == CUSTOM_ARCH

    decompress_gguf(packed, restored)
    restored_reader = gguf.GGUFReader(restored, "r")
    assert _field_text(restored_reader, gguf.Keys.General.ARCHITECTURE) == "test"
    assert _field_text(restored_reader, "general.name") == "tiny-roundtrip"

    tensors = {t.name: gguf.dequantize(t.data, t.tensor_type).astype(np.float32) for t in restored_reader.tensors}
    assert set(tensors) == set(original)
    np.testing.assert_allclose(tensors["blk.0.bias"], original["blk.0.bias"], rtol=0, atol=0)
    rel = np.linalg.norm(tensors["blk.0.weight"] - original["blk.0.weight"]) / np.linalg.norm(original["blk.0.weight"])
    assert rel < 0.03
