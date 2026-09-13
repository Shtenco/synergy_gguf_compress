from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .codec import CompressedMatrix, compress_matrix, reconstruct_matrix
from .quant import QuantizedRows

PREFIX = "gguf_compress."
FORMAT_VERSION = 1
CUSTOM_ARCH = "gguf-seed-lowrank"


@dataclass
class CompressionStats:
    input_bytes: int
    output_bytes: int
    factorized_tensors: int
    passthrough_tensors: int
    mean_rel_rmse: float
    mean_cosine: float
    elapsed_seconds: float

    @property
    def file_ratio(self) -> float:
        return self.input_bytes / max(self.output_bytes, 1)


def _gguf():
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError("gguf-py is required. Install with: pip install 'gguf>=0.19.0'") from exc
    return gguf


def _copy_metadata(reader: Any, writer: Any, *, skip_custom: bool = True) -> None:
    gguf = _gguf()
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if skip_custom and field.name.startswith(PREFIX):
            continue
        if field.name == "general.file_type":
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def _tensor_float32(tensor: Any) -> np.ndarray:
    gguf = _gguf()
    return np.ascontiguousarray(gguf.dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)


def _q_to_manifest(q: QuantizedRows, data_name: str, scale_name: str) -> dict[str, Any]:
    return {"data": data_name, "scale": scale_name, "qbits": q.qbits, "cols": q.cols}


def _add_q(writer: Any, q: QuantizedRows, data_name: str, scale_name: str) -> None:
    writer.add_tensor(data_name, np.ascontiguousarray(q.data, dtype=np.int8))
    writer.add_tensor(scale_name, np.ascontiguousarray(q.scales, dtype=np.float16))


def compress_gguf(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    method: str = "svd-lowrank",
    rank: int = 8,
    qbits: int = 4,
    seed: str = "gguf-compress-v1",
    min_elements: int = 131072,
    max_rel_rmse: float | None = None,
    oversample: int = 8,
    power_iters: int = 1,
) -> CompressionStats:
    gguf = _gguf()
    input_path = str(input_path)
    output_path = str(output_path)
    reader = gguf.GGUFReader(input_path, "r")
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    original_arch = str(arch_field.contents()) if arch_field else "unknown"

    writer = gguf.GGUFWriter(output_path, arch=CUSTOM_ARCH, endianess=reader.endianess, use_temp_file=True)
    _copy_metadata(reader, writer, skip_custom=True)

    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "original_arch": original_arch,
        "method": method,
        "rank": rank,
        "qbits": qbits,
        "global_seed": seed,
        "tensors": [],
    }

    factor_errors: list[tuple[float, float]] = []
    factorized = 0
    passthrough = 0
    start = time.perf_counter()

    for idx, tensor in enumerate(reader.tensors):
        logical_shape = tuple(int(v) for v in reversed(tensor.shape.tolist()))
        eligible = len(logical_shape) == 2 and tensor.n_elements >= min_elements and min(logical_shape) > rank
        if eligible:
            try:
                w = _tensor_float32(tensor).reshape(logical_shape)
            except (NotImplementedError, ValueError):
                w = None
            if w is not None:
                cm = compress_matrix(
                    w,
                    method=method,  # type: ignore[arg-type]
                    rank=rank,
                    qbits=qbits,
                    seed=seed,
                    tensor_name=tensor.name,
                    oversample=oversample,
                    power_iters=power_iters,
                )
                accept = max_rel_rmse is None or cm.rel_rmse <= max_rel_rmse
                if accept:
                    base = f"__gguf_compress__.{idx}"
                    a_data, a_scale = f"{base}.a", f"{base}.a_scale"
                    _add_q(writer, cm.a, a_data, a_scale)
                    entry: dict[str, Any] = {
                        "name": tensor.name,
                        "kind": "factorized",
                        "shape": list(logical_shape),
                        "original_type": tensor.tensor_type.name,
                        "method": cm.method,
                        "rank": cm.rank,
                        "qbits": cm.qbits,
                        "seed": cm.seed,
                        "a": _q_to_manifest(cm.a, a_data, a_scale),
                        "rel_rmse": cm.rel_rmse,
                        "cosine": cm.cosine,
                    }
                    if cm.b is not None:
                        b_data, b_scale = f"{base}.b", f"{base}.b_scale"
                        _add_q(writer, cm.b, b_data, b_scale)
                        entry["b"] = _q_to_manifest(cm.b, b_data, b_scale)
                    manifest["tensors"].append(entry)
                    factor_errors.append((cm.rel_rmse, cm.cosine))
                    factorized += 1
                    continue

        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
        manifest["tensors"].append({
            "name": tensor.name,
            "kind": "raw",
            "shape": list(logical_shape),
            "original_type": tensor.tensor_type.name,
        })
        passthrough += 1

    writer.add_uint32(f"{PREFIX}format_version", FORMAT_VERSION)
    writer.add_string(f"{PREFIX}manifest", json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    elapsed = time.perf_counter() - start
    output_bytes = os.path.getsize(output_path)
    input_bytes = os.path.getsize(input_path)
    mean_rmse = float(np.mean([x[0] for x in factor_errors])) if factor_errors else 0.0
    mean_cos = float(np.mean([x[1] for x in factor_errors])) if factor_errors else 1.0
    return CompressionStats(input_bytes, output_bytes, factorized, passthrough, mean_rmse, mean_cos, elapsed)


def _read_manifest(reader: Any) -> dict[str, Any]:
    field = reader.get_field(f"{PREFIX}manifest")
    if field is None:
        raise ValueError("not a gguf-compress file: manifest metadata is missing")
    manifest = json.loads(field.contents())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported compressed format version: {manifest.get('format_version')}")
    return manifest


def _tensor_map(reader: Any) -> dict[str, Any]:
    return {t.name: t for t in reader.tensors}


def _load_q(tensors: dict[str, Any], meta: dict[str, Any]) -> QuantizedRows:
    data = np.array(tensors[meta["data"]].data, copy=True).astype(np.int8, copy=False)
    scales = np.array(tensors[meta["scale"]].data, copy=True).astype(np.float16, copy=False)
    return QuantizedRows(data, scales.reshape(-1), int(meta["qbits"]), int(meta["cols"]))


def decompress_gguf(input_path: str | os.PathLike[str], output_path: str | os.PathLike[str]) -> None:
    gguf = _gguf()
    reader = gguf.GGUFReader(str(input_path), "r")
    manifest = _read_manifest(reader)
    tensors = _tensor_map(reader)
    writer = gguf.GGUFWriter(str(output_path), arch=manifest["original_arch"], endianess=reader.endianess, use_temp_file=True)
    _copy_metadata(reader, writer, skip_custom=True)

    for entry in manifest["tensors"]:
        name = entry["name"]
        if entry["kind"] == "raw":
            t = tensors[name]
            writer.add_tensor(name, t.data, raw_dtype=t.tensor_type, tensor_endianess=reader.endianess)
            continue

        a = _load_q(tensors, entry["a"])
        b = _load_q(tensors, entry["b"]) if "b" in entry else None
        cm = CompressedMatrix(
            method=entry["method"],
            shape=tuple(entry["shape"]),
            rank=int(entry["rank"]),
            qbits=int(entry["qbits"]),
            a=a,
            b=b,
            seed=entry.get("seed"),
            fit_seconds=0.0,
            rel_rmse=float(entry.get("rel_rmse", 0.0)),
            cosine=float(entry.get("cosine", 0.0)),
        )
        w = reconstruct_matrix(cm).astype(np.float16)
        writer.add_tensor(name, w)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def inspect_compressed(path: str | os.PathLike[str]) -> dict[str, Any]:
    gguf = _gguf()
    reader = gguf.GGUFReader(str(path), "r")
    return _read_manifest(reader)
