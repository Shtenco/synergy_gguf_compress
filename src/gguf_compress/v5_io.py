from __future__ import annotations

import json
import os
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .activation import ActivationCalibration, uniform_second_moment
from .gguf_io import _copy_metadata, _gguf, _tensor_float32
from .optimizer import CandidateScan, GlobalPlan, build_matrix_variant
from .group_q4 import quantize_group_q4
from .v3_io import (
    V3_CUSTOM_ARCH,
    V3_FORMAT_VERSION,
    V3_PREFIX,
    _add_q,
    _add_residual,
    _q_meta,
)


V5_PREFIX = "gguf_compress.v5."
V5_FORMAT_VERSION = 5


@dataclass(frozen=True)
class V5CompressionStats:
    input_bytes: int
    output_bytes: int
    factorized_tensors: int
    passthrough_tensors: int
    mean_rel_rmse: float
    mean_activation_rel_rmse: float
    mean_rank: float
    residual_nnz: int
    estimated_payload_bytes: int
    target_payload_bytes: int
    elapsed_seconds: float

    @property
    def file_ratio(self) -> float:
        return self.input_bytes / max(self.output_bytes, 1)


def compress_gguf_v5(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    plan: GlobalPlan,
    scan: CandidateScan,
    calibration: ActivationCalibration,
    ranks: tuple[int, ...],
    qbits: int = 4,
    seed: str = "gguf-compress-v5",
    oversample: int = 8,
    power_iters: int = 1,
) -> V5CompressionStats:
    """Materialize a globally selected V5 plan as a V3-compatible container.

    V5 writes the V3 compatibility manifest because the existing native-Q4
    exporter and llama.cpp runtime already understand that tensor layout. A
    second V5 manifest records the global budget decision and activation-aware
    evidence.
    """
    gguf = _gguf()
    input_path = str(input_path)
    output_path = str(output_path)
    reader = gguf.GGUFReader(input_path, "r")
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    original_arch = str(arch_field.contents()) if arch_field else "unknown"

    writer = gguf.GGUFWriter(
        output_path,
        arch=V3_CUSTOM_ARCH,
        endianess=reader.endianess,
        use_temp_file=True,
    )
    _copy_metadata(reader, writer, skip_custom=True)

    v3_manifest: dict[str, Any] = {
        "format_version": V3_FORMAT_VERSION,
        "original_arch": original_arch,
        "qbits": qbits,
        "target_rel_rmse": None,
        "residual_density": None,
        "rank_candidates": list(ranks),
        "global_seed": seed,
        "v5_global_budget": True,
        "tensors": [],
    }
    v5_manifest: dict[str, Any] = {
        "format_version": V5_FORMAT_VERSION,
        "original_arch": original_arch,
        "activation_calibration": str(calibration.path) if calibration.path else None,
        "target_ratio": plan.target_ratio,
        "target_payload_bytes": plan.target_bytes,
        "estimated_payload_bytes": plan.estimated_bytes,
        "estimated_activation_rel_rmse": plan.activation_rel_rmse,
        "feasible": plan.feasible,
        "fixed_payload_bytes": scan.fixed_payload_bytes,
        "original_payload_bytes": scan.original_payload_bytes,
        "choices": {},
        "tensors": [],
    }

    factor_count = raw_count = residual_nnz = 0
    rel_errors: list[float] = []
    activation_errors: list[float] = []
    ranks_used: list[int] = []
    started = time.perf_counter()

    for idx, tensor in enumerate(reader.tensors):
        shape = tuple(int(v) for v in reversed(tensor.shape.tolist()))
        choice = plan.choices.get(tensor.name)

        if choice is None or choice.kind == "raw":
            writer.add_tensor(
                tensor.name,
                tensor.data,
                raw_dtype=tensor.tensor_type,
                tensor_endianess=reader.endianess,
            )
            raw_entry = {
                "name": tensor.name,
                "kind": "raw",
                "shape": list(shape),
                "original_type": tensor.tensor_type.name,
            }
            v3_manifest["tensors"].append(raw_entry)
            v5_manifest["tensors"].append({
                **raw_entry,
                "reason": "fixed_or_global_raw",
            })
            if choice is not None:
                v5_manifest["choices"][tensor.name] = choice.to_dict()
            raw_count += 1
            continue

        if len(shape) != 2:
            raise RuntimeError(f"V5 plan selected non-matrix tensor: {tensor.name}")
        w = _tensor_float32(tensor).reshape(shape)
        second = calibration.get(tensor.name, shape[1])
        if second is None:
            if choice.calibration_source != "uniform_fallback":
                raise RuntimeError(f"missing activation calibration for selected tensor: {tensor.name}")
            second = uniform_second_moment(shape[1])

        if choice.kind == "group_q4":
            if choice.group_size is None:
                raise RuntimeError(f"GROUP_Q4 choice is missing group_size: {tensor.name}")
            q = quantize_group_q4(w, group_size=choice.group_size)
            base = f"__gguf_compress_v3__.{idx}"
            data_name = f"{base}.gq4_data"
            scale_name = f"{base}.gq4_scale"
            writer.add_tensor(data_name, np.ascontiguousarray(q.data, dtype=np.int8))
            writer.add_tensor(scale_name, np.ascontiguousarray(q.scales, dtype=np.float16))
            entry = {
                "name": tensor.name,
                "kind": "group_q4",
                "shape": list(shape),
                "original_type": tensor.tensor_type.name,
                "group_size": choice.group_size,
                "data": data_name,
                "scales": scale_name,
                "rel_rmse": choice.rel_rmse,
                "activation_rel_rmse": choice.activation_rel_rmse,
                "activation_distortion_energy": choice.distortion_energy,
                "activation_reference_energy": choice.reference_energy,
                "calibration_source": choice.calibration_source,
            }
            v3_manifest["tensors"].append(entry)
            v5_manifest["tensors"].append(entry)
            v5_manifest["choices"][tensor.name] = choice.to_dict()
            factor_count += 1
            rel_errors.append(choice.rel_rmse)
            activation_errors.append(choice.activation_rel_rmse)
            continue

        if choice.kind != "factorized":
            raise RuntimeError(f"unsupported V5 choice kind={choice.kind!r} for {tensor.name}")
        assert choice.rank is not None
        variant = build_matrix_variant(
            w,
            tensor_name=tensor.name,
            rank=choice.rank,
            residual_density=choice.residual_density,
            qbits=qbits,
            ranks_for_basis=ranks,
            second_moment=second,
            seed=seed,
            oversample=oversample,
            power_iters=power_iters,
        )

        base = f"__gguf_compress_v3__.{idx}"
        a_data, a_scale = f"{base}.a", f"{base}.a_scale"
        b_data, b_scale = f"{base}.b", f"{base}.b_scale"
        _add_q(writer, variant.a, a_data, a_scale)
        _add_q(writer, variant.b, b_data, b_scale)
        entry: dict[str, Any] = {
            "name": tensor.name,
            "kind": "adaptive_factorized",
            "shape": list(shape),
            "original_type": tensor.tensor_type.name,
            "rank": choice.rank,
            "qbits": qbits,
            "a": _q_meta(variant.a, a_data, a_scale),
            "b": _q_meta(variant.b, b_data, b_scale),
            "rel_rmse": variant.rel_rmse,
            "cosine": None,
            "corrected_rel_rmse": variant.rel_rmse,
            "corrected_cosine": None,
            "activation_rel_rmse": variant.activation_rel_rmse,
            "activation_distortion_energy": variant.distortion_energy,
            "activation_reference_energy": variant.reference_energy,
            "calibration_source": choice.calibration_source,
            "rank_search": [],
        }
        if variant.residual is not None:
            rmeta = _add_residual(writer, variant.residual, f"{base}.residual")
            entry["residual"] = rmeta
            residual_nnz += variant.residual.nnz

        v3_manifest["tensors"].append(entry)
        v5_manifest["tensors"].append(entry)
        v5_manifest["choices"][tensor.name] = choice.to_dict()
        factor_count += 1
        rel_errors.append(variant.rel_rmse)
        activation_errors.append(variant.activation_rel_rmse)
        ranks_used.append(choice.rank)

    writer.add_uint32(f"{V3_PREFIX}format_version", V3_FORMAT_VERSION)
    writer.add_string(
        f"{V3_PREFIX}manifest",
        json.dumps(v3_manifest, ensure_ascii=False, separators=(",", ":")),
    )
    writer.add_uint32(f"{V5_PREFIX}format_version", V5_FORMAT_VERSION)
    writer.add_string(
        f"{V5_PREFIX}manifest",
        json.dumps(v5_manifest, ensure_ascii=False, separators=(",", ":")),
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return V5CompressionStats(
        input_bytes=os.path.getsize(input_path),
        output_bytes=os.path.getsize(output_path),
        factorized_tensors=factor_count,
        passthrough_tensors=raw_count,
        mean_rel_rmse=float(np.mean(rel_errors)) if rel_errors else 0.0,
        mean_activation_rel_rmse=float(np.mean(activation_errors)) if activation_errors else 0.0,
        mean_rank=float(np.mean(ranks_used)) if ranks_used else 0.0,
        residual_nnz=residual_nnz,
        estimated_payload_bytes=plan.estimated_bytes,
        target_payload_bytes=plan.target_bytes,
        elapsed_seconds=time.perf_counter() - started,
    )


def inspect_v5(path: str | os.PathLike[str]) -> dict[str, Any]:
    reader = _gguf().GGUFReader(str(path), "r")
    field = reader.get_field(f"{V5_PREFIX}manifest")
    if field is None:
        raise ValueError("not a gguf-compress V5 file")
    manifest = json.loads(field.contents())
    if manifest.get("format_version") != V5_FORMAT_VERSION:
        raise ValueError(f"unsupported V5 format version: {manifest.get('format_version')}")
    return manifest