from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .adaptive import AdaptivePolicy, adaptive_compress_matrix
from .gguf_io import _copy_metadata, _gguf, _tensor_float32
from .quant import QuantizedRows
from .residual import SparseRowResidual

V3_PREFIX = "gguf_compress.v3."
V3_FORMAT_VERSION = 3
V3_CUSTOM_ARCH = "gguf-adaptive-lowrank-v3"


@dataclass(frozen=True)
class V3CompressionStats:
    input_bytes: int
    output_bytes: int
    factorized_tensors: int
    passthrough_tensors: int
    mean_rel_rmse: float
    mean_corrected_rel_rmse: float
    mean_rank: float
    residual_nnz: int
    elapsed_seconds: float

    @property
    def file_ratio(self) -> float:
        return self.input_bytes / max(self.output_bytes, 1)


def _q_meta(q: QuantizedRows, data_name: str, scale_name: str) -> dict[str, Any]:
    return {"data": data_name, "scale": scale_name, "qbits": q.qbits, "cols": q.cols}


def _add_q(writer: Any, q: QuantizedRows, data_name: str, scale_name: str) -> None:
    writer.add_tensor(data_name, np.ascontiguousarray(q.data, dtype=np.int8))
    writer.add_tensor(scale_name, np.ascontiguousarray(q.scales, dtype=np.float16))


def _residual_meta(residual: SparseRowResidual, base: str) -> dict[str, Any]:
    return {
        "cols": f"{base}.cols",
        "values": f"{base}.values",
        "scales": f"{base}.scales",
        "in_features": residual.in_features,
        "k_per_row": residual.k_per_row,
        "nnz": residual.nnz,
        "col_dtype": str(residual.cols.dtype),
    }


def _add_residual(writer: Any, residual: SparseRowResidual, base: str) -> dict[str, Any]:
    meta = _residual_meta(residual, base)
    # GGUF tensor types do not include uint16/uint32 universally, so store
    # indices as signed I32 for portable reader support. Values remain I8.
    writer.add_tensor(meta["cols"], np.ascontiguousarray(residual.cols, dtype=np.int32))
    writer.add_tensor(meta["values"], np.ascontiguousarray(residual.values, dtype=np.int8))
    writer.add_tensor(meta["scales"], np.ascontiguousarray(residual.scales, dtype=np.float16))
    return meta


def compress_gguf_v3(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    candidates: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256),
    target_rel_rmse: float = 0.20,
    qbits: int = 4,
    residual_density: float = 0.01,
    min_elements: int = 131072,
    max_corrected_rel_rmse: float | None = None,
    seed: str = "gguf-compress-v3",
    oversample: int = 8,
    power_iters: int = 1,
) -> V3CompressionStats:
    gguf = _gguf()
    input_path = str(input_path)
    output_path = str(output_path)
    reader = gguf.GGUFReader(input_path, "r")
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    original_arch = str(arch_field.contents()) if arch_field else "unknown"

    writer = gguf.GGUFWriter(output_path, arch=V3_CUSTOM_ARCH, endianess=reader.endianess, use_temp_file=True)
    _copy_metadata(reader, writer, skip_custom=True)

    policy = AdaptivePolicy(
        candidates=candidates,
        target_rel_rmse=target_rel_rmse,
        qbits=qbits,
        oversample=oversample,
        power_iters=power_iters,
        residual_density=residual_density,
    )
    manifest: dict[str, Any] = {
        "format_version": V3_FORMAT_VERSION,
        "original_arch": original_arch,
        "qbits": qbits,
        "target_rel_rmse": target_rel_rmse,
        "residual_density": residual_density,
        "rank_candidates": list(candidates),
        "global_seed": seed,
        "tensors": [],
    }

    raw_count = 0
    factor_count = 0
    residual_nnz = 0
    rel_errors: list[float] = []
    corrected_errors: list[float] = []
    ranks: list[int] = []
    started = time.perf_counter()

    for idx, tensor in enumerate(reader.tensors):
        logical_shape = tuple(int(v) for v in reversed(tensor.shape.tolist()))
        eligible = len(logical_shape) == 2 and tensor.n_elements >= min_elements and min(logical_shape) > min(candidates)
        if eligible:
            try:
                w = _tensor_float32(tensor).reshape(logical_shape)
            except (NotImplementedError, ValueError):
                w = None
            if w is not None:
                result = adaptive_compress_matrix(w, tensor_name=tensor.name, policy=policy, seed=seed)
                accept = max_corrected_rel_rmse is None or result.corrected_rel_rmse <= max_corrected_rel_rmse
                if accept:
                    base = f"__gguf_compress_v3__.{idx}"
                    a_data, a_scale = f"{base}.a", f"{base}.a_scale"
                    b_data, b_scale = f"{base}.b", f"{base}.b_scale"
                    _add_q(writer, result.matrix.a, a_data, a_scale)
                    assert result.matrix.b is not None
                    _add_q(writer, result.matrix.b, b_data, b_scale)
                    entry: dict[str, Any] = {
                        "name": tensor.name,
                        "kind": "adaptive_factorized",
                        "shape": list(logical_shape),
                        "original_type": tensor.tensor_type.name,
                        "rank": result.matrix.rank,
                        "qbits": qbits,
                        "a": _q_meta(result.matrix.a, a_data, a_scale),
                        "b": _q_meta(result.matrix.b, b_data, b_scale),
                        "rel_rmse": result.matrix.rel_rmse,
                        "cosine": result.matrix.cosine,
                        "corrected_rel_rmse": result.corrected_rel_rmse,
                        "corrected_cosine": result.corrected_cosine,
                        "rank_search": result.candidates_tested,
                    }
                    if result.residual is not None:
                        rmeta = _add_residual(writer, result.residual, f"{base}.residual")
                        entry["residual"] = rmeta
                        residual_nnz += result.residual.nnz
                    manifest["tensors"].append(entry)
                    factor_count += 1
                    rel_errors.append(result.matrix.rel_rmse)
                    corrected_errors.append(result.corrected_rel_rmse)
                    ranks.append(result.matrix.rank)
                    continue

        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
        manifest["tensors"].append({
            "name": tensor.name,
            "kind": "raw",
            "shape": list(logical_shape),
            "original_type": tensor.tensor_type.name,
        })
        raw_count += 1

    writer.add_uint32(f"{V3_PREFIX}format_version", V3_FORMAT_VERSION)
    writer.add_string(f"{V3_PREFIX}manifest", json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return V3CompressionStats(
        input_bytes=os.path.getsize(input_path),
        output_bytes=os.path.getsize(output_path),
        factorized_tensors=factor_count,
        passthrough_tensors=raw_count,
        mean_rel_rmse=float(np.mean(rel_errors)) if rel_errors else 0.0,
        mean_corrected_rel_rmse=float(np.mean(corrected_errors)) if corrected_errors else 0.0,
        mean_rank=float(np.mean(ranks)) if ranks else 0.0,
        residual_nnz=residual_nnz,
        elapsed_seconds=time.perf_counter() - started,
    )


def inspect_v3(path: str | os.PathLike[str]) -> dict[str, Any]:
    gguf = _gguf()
    reader = gguf.GGUFReader(str(path), "r")
    field = reader.get_field(f"{V3_PREFIX}manifest")
    if field is None:
        raise ValueError("not a gguf-compress v3 file")
    manifest = json.loads(field.contents())
    if manifest.get("format_version") != V3_FORMAT_VERSION:
        raise ValueError(f"unsupported v3 format version: {manifest.get('format_version')}")
    return manifest
