from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .codec import CompressedMatrix, factors
from .gguf_io import _copy_metadata, _gguf, _load_q, _read_manifest, _tensor_map

NATIVE_PREFIX = "gguf_compress.native."
NATIVE_FORMAT_VERSION = 2
LR_A_SUFFIX = ".__gguf_lr_a"


@dataclass(frozen=True)
class NativeExportStats:
    input_bytes: int
    output_bytes: int
    factorized_tensors: int
    passthrough_tensors: int
    reconstructed_tensors: int

    @property
    def file_ratio(self) -> float:
        return self.input_bytes / max(self.output_bytes, 1)


_NATIVE_LINEAR_SUFFIXES = (
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_qkv.weight",
    ".attn_output.weight",
    ".ffn_up.weight",
    ".ffn_gate.weight",
    ".ffn_down.weight",
)


def is_native_linear_name(name: str) -> bool:
    """Conservative v2 allow-list for weights known to use the common linear path."""
    return name == "output.weight" or name.endswith(_NATIVE_LINEAR_SUFFIXES)


def native_tensor_names(original_name: str) -> tuple[str, str]:
    """Return (B_name, A_name). B deliberately keeps the original name.

    llama.cpp's model code asks for the original weight name. In native v2 the
    tensor under that name is B[in, rank] in GGML layout, while the companion
    A[rank, out] tensor is stored under a deterministic suffix. The llama.cpp
    integration patch recognizes the pair and replaces one dense matmul with
    two native ggml_mul_mat nodes.
    """
    return original_name, original_name + LR_A_SUFFIX


def factor_storage_bytes(shape: tuple[int, int], rank: int, itemsize: int = 2) -> int:
    out_features, in_features = shape
    return int((out_features * rank + rank * in_features) * itemsize)


def theoretical_native_ratio(shape: tuple[int, int], rank: int, dense_itemsize: int = 2, factor_itemsize: int = 2) -> float:
    out_features, in_features = shape
    dense = out_features * in_features * dense_itemsize
    return dense / max(factor_storage_bytes(shape, rank, factor_itemsize), 1)


def export_native_gguf(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    factor_dtype: str = "f16",
) -> NativeExportStats:
    """Convert the v1 compressed container into a llama.cpp-native factor GGUF.

    The output keeps the original model architecture and metadata. Raw tensors
    are copied byte-for-byte. A factorized W ~= A@B is written as:

      original tensor name  -> B  (numpy shape [rank, in])
      <name>.__gguf_lr_a    -> A  (numpy shape [out, rank])

    Both factors use standard F16/F32 GGUF tensor types so existing GGML CPU,
    CUDA, Metal, Vulkan, SYCL, etc. matrix-multiply kernels can execute them.
    No dense W is reconstructed or written for supported native linear tensors.
    """
    if factor_dtype not in {"f16", "f32"}:
        raise ValueError("factor_dtype must be 'f16' or 'f32'")
    np_dtype = np.float16 if factor_dtype == "f16" else np.float32

    gguf = _gguf()
    input_path = str(input_path)
    output_path = str(output_path)
    reader = gguf.GGUFReader(input_path, "r")
    manifest = _read_manifest(reader)
    tensors = _tensor_map(reader)

    writer = gguf.GGUFWriter(
        output_path,
        arch=manifest["original_arch"],
        endianess=reader.endianess,
        use_temp_file=True,
    )
    _copy_metadata(reader, writer, skip_custom=True)

    native_manifest: dict[str, Any] = {
        "format_version": NATIVE_FORMAT_VERSION,
        "source_format_version": manifest.get("format_version", 1),
        "original_arch": manifest["original_arch"],
        "factor_dtype": factor_dtype,
        "factorized": [],
    }

    factorized_count = 0
    passthrough_count = 0
    reconstructed_count = 0

    for entry in manifest["tensors"]:
        name = entry["name"]
        if entry["kind"] == "raw":
            t = tensors[name]
            writer.add_tensor(name, t.data, raw_dtype=t.tensor_type, tensor_endianess=reader.endianess)
            passthrough_count += 1
            continue

        a_q = _load_q(tensors, entry["a"])
        b_q = _load_q(tensors, entry["b"]) if "b" in entry else None
        cm = CompressedMatrix(
            method=entry["method"],
            shape=tuple(entry["shape"]),
            rank=int(entry["rank"]),
            qbits=int(entry["qbits"]),
            a=a_q,
            b=b_q,
            seed=entry.get("seed"),
            fit_seconds=0.0,
            rel_rmse=float(entry.get("rel_rmse", 0.0)),
            cosine=float(entry.get("cosine", 0.0)),
        )
        a, b = factors(cm)

        if not is_native_linear_name(name):
            # v2 must remain executable: tensors outside the central linear path
            # are reconstructed once during export, not during model inference.
            dense = np.ascontiguousarray(a @ b, dtype=np_dtype)
            writer.add_tensor(name, dense)
            native_manifest.setdefault("reconstructed", []).append({
                "name": name,
                "shape": list(cm.shape),
                "reason": "not-covered-by-native-linear-path",
            })
            reconstructed_count += 1
            continue

        b_name, a_name = native_tensor_names(name)

        # Numpy [rank, in] -> GGUF dims [in, rank], exactly what ggml_mul_mat(B, X) expects.
        writer.add_tensor(b_name, np.ascontiguousarray(b, dtype=np_dtype))
        # Numpy [out, rank] -> GGUF dims [rank, out], exactly what ggml_mul_mat(A, BX) expects.
        writer.add_tensor(a_name, np.ascontiguousarray(a, dtype=np_dtype))

        native_manifest["factorized"].append({
            "name": name,
            "a": a_name,
            "b": b_name,
            "shape": list(cm.shape),
            "rank": cm.rank,
            "method": cm.method,
            "rel_rmse": cm.rel_rmse,
            "cosine": cm.cosine,
        })
        factorized_count += 1

    writer.add_uint32(f"{NATIVE_PREFIX}format_version", NATIVE_FORMAT_VERSION)
    writer.add_string(
        f"{NATIVE_PREFIX}manifest",
        json.dumps(native_manifest, ensure_ascii=False, separators=(",", ":")),
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return NativeExportStats(
        input_bytes=os.path.getsize(input_path),
        output_bytes=os.path.getsize(output_path),
        factorized_tensors=factorized_count,
        passthrough_tensors=passthrough_count,
        reconstructed_tensors=reconstructed_count,
    )
