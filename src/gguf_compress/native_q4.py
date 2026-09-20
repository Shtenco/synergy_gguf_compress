from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .gguf_io import _copy_metadata, _gguf, _tensor_map
from .native import is_native_linear_name
from .quant import QuantizedRows, dequantize_rows
from .v3_io import V3_PREFIX, V3_FORMAT_VERSION

NATIVE_Q4_PREFIX = "gguf_compress.native_q4."
NATIVE_Q4_FORMAT_VERSION = 3

PROXY_SUFFIX = ".__gguf_lr_proxy"
A_DATA_SUFFIX = ".__gguf_lr_q4_a_data"
A_SCALE_SUFFIX = ".__gguf_lr_q4_a_scale"
B_DATA_SUFFIX = ".__gguf_lr_q4_b_data"
B_SCALE_SUFFIX = ".__gguf_lr_q4_b_scale"
R_COLS_SUFFIX = ".__gguf_lr_res_cols"
R_VALUES_SUFFIX = ".__gguf_lr_res_values"
R_SCALES_SUFFIX = ".__gguf_lr_res_scales"
GQ4_DATA_SUFFIX = ".__omega_gq4_data"
GQ4_SCALE_SUFFIX = ".__omega_gq4_scale"


@dataclass(frozen=True)
class NativeQ4Stats:
    input_bytes: int
    output_bytes: int
    q4_layers: int
    reconstructed_layers: int
    passthrough_tensors: int

    @property
    def file_ratio(self) -> float:
        return self.input_bytes / max(self.output_bytes, 1)


def _read_v3_manifest(reader: Any) -> dict[str, Any]:
    field = reader.get_field(f"{V3_PREFIX}manifest")
    if field is None:
        raise ValueError("not a gguf-compress v3 file")
    manifest = json.loads(field.contents())
    if manifest.get("format_version") != V3_FORMAT_VERSION:
        raise ValueError("unsupported v3 input")
    return manifest


def _load_q(tensors: dict[str, Any], meta: dict[str, Any]) -> QuantizedRows:
    data = np.array(tensors[meta["data"]].data, copy=True).astype(np.int8, copy=False)
    scales = np.array(tensors[meta["scale"]].data, copy=True).astype(np.float16, copy=False).reshape(-1)
    return QuantizedRows(data, scales, int(meta["qbits"]), int(meta["cols"]))


def _reconstruct_entry(tensors: dict[str, Any], entry: dict[str, Any]) -> np.ndarray:
    a = dequantize_rows(_load_q(tensors, entry["a"]))
    b = dequantize_rows(_load_q(tensors, entry["b"]))
    dense = a @ b
    rmeta = entry.get("residual")
    if rmeta:
        cols = np.array(tensors[rmeta["cols"]].data, copy=True).astype(np.int32, copy=False)
        vals = np.array(tensors[rmeta["values"]].data, copy=True).astype(np.int8, copy=False)
        scales = np.array(tensors[rmeta["scales"]].data, copy=True).astype(np.float16, copy=False).reshape(-1)
        deq = vals.astype(np.float32) * scales.astype(np.float32)[:, None]
        for row in range(dense.shape[0]):
            dense[row, cols[row].astype(np.int64)] += deq[row]
    return np.ascontiguousarray(dense, dtype=np.float16)


def export_native_q4_gguf(input_path: str | os.PathLike[str], output_path: str | os.PathLike[str]) -> NativeQ4Stats:
    """Export V3 to the packed format consumed by the V3 llama.cpp CPU kernel.

    Supported linear weights are represented by a tiny proxy tensor under the
    original name plus packed Q4 A/B payloads. The patched loader intercepts
    the proxy before dense shape validation and registers the packed factors.
    Unsupported factorized tensors are reconstructed once during export.
    """
    gguf = _gguf()
    input_path = str(input_path)
    output_path = str(output_path)
    reader = gguf.GGUFReader(input_path, "r")
    manifest = _read_v3_manifest(reader)
    tensors = _tensor_map(reader)

    writer = gguf.GGUFWriter(output_path, arch=manifest["original_arch"], endianess=reader.endianess, use_temp_file=True)
    _copy_metadata(reader, writer, skip_custom=True)

    out_manifest: dict[str, Any] = {
        "format_version": NATIVE_Q4_FORMAT_VERSION,
        "original_arch": manifest["original_arch"],
        "layers": [],
    }
    q4_layers = reconstructed = passthrough = 0

    for entry in manifest["tensors"]:
        name = entry["name"]
        if entry["kind"] == "raw":
            t = tensors[name]
            writer.add_tensor(name, t.data, raw_dtype=t.tensor_type, tensor_endianess=reader.endianess)
            passthrough += 1
            continue

        if entry["kind"] == "group_q4":
            if not is_native_linear_name(name):
                raise ValueError(f"GROUP_Q4 selected unsupported runtime tensor: {name}")
            data = np.array(tensors[entry["data"]].data, copy=True).astype(np.int8, copy=False)
            scales = np.array(tensors[entry["scales"]].data, copy=True).astype(np.float16, copy=False)
            writer.add_tensor(name, np.zeros((1, 1), dtype=np.float32))
            writer.add_tensor(name + GQ4_DATA_SUFFIX, np.ascontiguousarray(data, dtype=np.int8))
            writer.add_tensor(name + GQ4_SCALE_SUFFIX, np.ascontiguousarray(scales, dtype=np.float16))
            out_manifest["layers"].append({
                "name": name,
                "kind": "group_q4",
                "shape": entry["shape"],
                "group_size": int(entry["group_size"]),
                "rel_rmse": entry.get("rel_rmse"),
                "activation_rel_rmse": entry.get("activation_rel_rmse"),
            })
            q4_layers += 1
            continue

        if entry.get("qbits") != 4 or not is_native_linear_name(name):
            writer.add_tensor(name, _reconstruct_entry(tensors, entry))
            reconstructed += 1
            continue

        a = _load_q(tensors, entry["a"])
        b = _load_q(tensors, entry["b"])
        # The original-name proxy is never multiplied. It only lets the model
        # architecture ask for its ordinary tensor name while the patched
        # loader binds packed factors from companion tensors.
        writer.add_tensor(name, np.zeros((1, 1), dtype=np.float32))
        writer.add_tensor(name + A_DATA_SUFFIX, np.ascontiguousarray(a.data, dtype=np.int8))
        writer.add_tensor(name + A_SCALE_SUFFIX, np.ascontiguousarray(a.scales, dtype=np.float16))
        writer.add_tensor(name + B_DATA_SUFFIX, np.ascontiguousarray(b.data, dtype=np.int8))
        writer.add_tensor(name + B_SCALE_SUFFIX, np.ascontiguousarray(b.scales, dtype=np.float16))

        layer_meta: dict[str, Any] = {
            "name": name,
            "shape": entry["shape"],
            "rank": entry["rank"],
            "a_cols": a.cols,
            "b_cols": b.cols,
            "corrected_rel_rmse": entry.get("corrected_rel_rmse", entry.get("rel_rmse")),
        }
        rmeta = entry.get("residual")
        if rmeta:
            writer.add_tensor(name + R_COLS_SUFFIX, np.ascontiguousarray(tensors[rmeta["cols"]].data, dtype=np.int32))
            writer.add_tensor(name + R_VALUES_SUFFIX, np.ascontiguousarray(tensors[rmeta["values"]].data, dtype=np.int8))
            writer.add_tensor(name + R_SCALES_SUFFIX, np.ascontiguousarray(tensors[rmeta["scales"]].data, dtype=np.float16))
            layer_meta["residual_k"] = int(rmeta["k_per_row"])
        out_manifest["layers"].append(layer_meta)
        q4_layers += 1

    writer.add_uint32(f"{NATIVE_Q4_PREFIX}format_version", NATIVE_Q4_FORMAT_VERSION)
    writer.add_string(f"{NATIVE_Q4_PREFIX}manifest", json.dumps(out_manifest, ensure_ascii=False, separators=(",", ":")))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return NativeQ4Stats(
        input_bytes=os.path.getsize(input_path),
        output_bytes=os.path.getsize(output_path),
        q4_layers=q4_layers,
        reconstructed_layers=reconstructed,
        passthrough_tensors=passthrough,
    )