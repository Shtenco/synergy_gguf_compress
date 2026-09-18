from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any

import numpy as np

from .activation import ActivationCalibration, uniform_second_moment
from .adaptive import role_rank_cap
from .codec import _randomized_lowrank
from .native import is_native_linear_name
from .quant import QuantizedRows, dequantize_rows, quantize_rows
from .residual import SparseRowResidual, build_sparse_row_residual


@dataclass(frozen=True)
class TensorCandidate:
    tensor_name: str
    candidate_id: str
    kind: str
    rank: int | None
    residual_density: float
    estimated_bytes: int
    rel_rmse: float
    activation_rel_rmse: float
    distortion_energy: float
    reference_energy: float
    calibration_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateScan:
    candidates: dict[str, list[TensorCandidate]]
    fixed_payload_bytes: int
    original_payload_bytes: int
    optimizable_raw_bytes: int
    calibrated_tensors: int
    fallback_tensors: int
    skipped_tensors: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed_payload_bytes": self.fixed_payload_bytes,
            "original_payload_bytes": self.original_payload_bytes,
            "optimizable_raw_bytes": self.optimizable_raw_bytes,
            "calibrated_tensors": self.calibrated_tensors,
            "fallback_tensors": self.fallback_tensors,
            "skipped_tensors": self.skipped_tensors,
            "candidates": {
                name: [candidate.to_dict() for candidate in values]
                for name, values in self.candidates.items()
            },
        }


@dataclass
class GlobalPlan:
    target_ratio: float
    target_bytes: int
    estimated_bytes: int
    feasible: bool
    activation_rel_rmse: float
    distortion_energy: float
    reference_energy: float
    choices: dict[str, TensorCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_ratio": self.target_ratio,
            "target_bytes": self.target_bytes,
            "estimated_bytes": self.estimated_bytes,
            "feasible": self.feasible,
            "activation_rel_rmse": self.activation_rel_rmse,
            "distortion_energy": self.distortion_energy,
            "reference_energy": self.reference_energy,
            "choices": {name: candidate.to_dict() for name, candidate in self.choices.items()},
        }


@dataclass(frozen=True)
class MatrixVariant:
    a: QuantizedRows
    b: QuantizedRows
    residual: SparseRowResidual | None
    rel_rmse: float
    activation_rel_rmse: float
    distortion_energy: float
    reference_energy: float


def _seed_int(seed: str, tensor_name: str) -> int:
    digest = sha256(f"{seed}:{tensor_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _residual_storage_bytes(residual: SparseRowResidual | None) -> int:
    if residual is None:
        return 0
    return int(residual.values.size * 4 + residual.values.nbytes + residual.scales.nbytes)


def _metrics(
    w: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    residual: SparseRowResidual | None,
    second_moment: np.ndarray,
) -> tuple[float, float, float, float]:
    std_ref = std_diff = act_ref = act_diff = 0.0
    s = np.asarray(second_moment, dtype=np.float32).reshape(-1)
    if s.size != w.shape[1]:
        raise ValueError("activation second moment does not match input features")

    for row in range(w.shape[0]):
        rec = np.asarray(a[row] @ b, dtype=np.float32)
        if residual is not None:
            cols = residual.cols[row].astype(np.int64, copy=False)
            correction = residual.values[row].astype(np.float32) * np.float32(residual.scales[row])
            rec[cols] += correction
        ref = w[row]
        diff = ref - rec
        std_ref += float(np.sum(ref * ref, dtype=np.float64))
        std_diff += float(np.sum(diff * diff, dtype=np.float64))
        act_ref += float(np.sum((ref * ref) * s, dtype=np.float64))
        act_diff += float(np.sum((diff * diff) * s, dtype=np.float64))

    rel_rmse = math.sqrt(std_diff / max(std_ref, 1e-30))
    activation_rel_rmse = math.sqrt(act_diff / max(act_ref, 1e-30))
    return rel_rmse, activation_rel_rmse, act_diff, act_ref


def _rank_list(tensor_name: str, shape: tuple[int, int], ranks: tuple[int, ...]) -> list[int]:
    m, n = shape
    cap = role_rank_cap(tensor_name, max(ranks))
    max_rank = min(m, n, cap)
    values = sorted({min(int(r), max_rank) for r in ranks if int(r) > 0})
    return [r for r in values if r >= 1]


def build_matrix_variant(
    weight: np.ndarray,
    *,
    tensor_name: str,
    rank: int,
    residual_density: float,
    qbits: int,
    ranks_for_basis: tuple[int, ...],
    second_moment: np.ndarray,
    seed: str,
    oversample: int,
    power_iters: int,
) -> MatrixVariant:
    w = np.asarray(weight, dtype=np.float32)
    ranks = _rank_list(tensor_name, (w.shape[0], w.shape[1]), ranks_for_basis)
    if rank not in ranks:
        raise ValueError(f"rank {rank} is not valid for {tensor_name}; valid={ranks}")
    a_full, b_full = _randomized_lowrank(
        w,
        max(ranks),
        oversample,
        power_iters,
        _seed_int(seed, tensor_name),
    )
    qa = quantize_rows(a_full[:, :rank], qbits)
    qb = quantize_rows(b_full[:rank, :], qbits)
    a = dequantize_rows(qa)
    b = dequantize_rows(qb)
    residual = build_sparse_row_residual(w, a, b, residual_density)
    rel, act_rel, distortion, reference = _metrics(w, a, b, residual, second_moment)
    return MatrixVariant(qa, qb, residual, rel, act_rel, distortion, reference)


def build_tensor_candidates(
    weight: np.ndarray,
    *,
    tensor_name: str,
    raw_bytes: int,
    ranks: tuple[int, ...],
    residual_densities: tuple[float, ...],
    qbits: int,
    second_moment: np.ndarray,
    calibration_source: str,
    seed: str,
    oversample: int,
    power_iters: int,
    max_activation_rel_rmse: float | None = None,
) -> list[TensorCandidate]:
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("weight must be 2-D")
    valid_ranks = _rank_list(tensor_name, (w.shape[0], w.shape[1]), ranks)
    if not valid_ranks:
        return []

    s = np.asarray(second_moment, dtype=np.float32).reshape(-1)
    if s.size != w.shape[1]:
        raise ValueError("activation second moment does not match weight input features")
    reference_energy = float(np.sum((w * w) * s[None, :], dtype=np.float64))

    candidates: list[TensorCandidate] = [
        TensorCandidate(
            tensor_name=tensor_name,
            candidate_id="raw",
            kind="raw",
            rank=None,
            residual_density=0.0,
            estimated_bytes=int(raw_bytes),
            rel_rmse=0.0,
            activation_rel_rmse=0.0,
            distortion_energy=0.0,
            reference_energy=reference_energy,
            calibration_source=calibration_source,
        )
    ]

    a_full, b_full = _randomized_lowrank(
        w,
        max(valid_ranks),
        oversample,
        power_iters,
        _seed_int(seed, tensor_name),
    )
    densities = sorted({float(d) for d in residual_densities if 0.0 <= float(d) <= 1.0})
    if not densities:
        densities = [0.0]

    for rank in valid_ranks:
        qa = quantize_rows(a_full[:, :rank], qbits)
        qb = quantize_rows(b_full[:rank, :], qbits)
        a = dequantize_rows(qa)
        b = dequantize_rows(qb)
        for density in densities:
            residual = build_sparse_row_residual(w, a, b, density)
            rel, act_rel, distortion, reference = _metrics(w, a, b, residual, s)
            if max_activation_rel_rmse is not None and act_rel > max_activation_rel_rmse:
                continue
            estimated = int(qa.nbytes + qb.nbytes + _residual_storage_bytes(residual) + 4)
            candidates.append(
                TensorCandidate(
                    tensor_name=tensor_name,
                    candidate_id=f"r{rank}-d{density:.6g}",
                    kind="factorized",
                    rank=rank,
                    residual_density=density,
                    estimated_bytes=estimated,
                    rel_rmse=rel,
                    activation_rel_rmse=act_rel,
                    distortion_energy=distortion,
                    reference_energy=reference,
                    calibration_source=calibration_source,
                )
            )

    return _pareto_prune_candidates(candidates)


def _pareto_prune_candidates(candidates: list[TensorCandidate]) -> list[TensorCandidate]:
    ordered = sorted(candidates, key=lambda c: (c.estimated_bytes, c.distortion_energy, c.candidate_id))
    kept: list[TensorCandidate] = []
    best_distortion = math.inf
    for candidate in ordered:
        if candidate.distortion_energy < best_distortion - 1e-18:
            kept.append(candidate)
            best_distortion = candidate.distortion_energy
    raw = next((c for c in candidates if c.kind == "raw"), None)
    if raw is not None and all(c.candidate_id != "raw" for c in kept):
        kept.append(raw)
    return sorted(kept, key=lambda c: (c.estimated_bytes, c.distortion_energy))


def scan_gguf_candidates(
    input_path: str | Path,
    *,
    calibration: ActivationCalibration,
    ranks: tuple[int, ...],
    residual_densities: tuple[float, ...],
    qbits: int = 4,
    min_elements: int = 131072,
    seed: str = "gguf-compress-v5",
    oversample: int = 8,
    power_iters: int = 1,
    require_calibration: bool = True,
    max_activation_rel_rmse: float | None = 0.50,
) -> CandidateScan:
    from .gguf_io import _gguf, _tensor_float32

    reader = _gguf().GGUFReader(str(input_path), "r")
    candidate_map: dict[str, list[TensorCandidate]] = {}
    fixed = original = optimizable_raw = 0
    calibrated = fallback = skipped = 0

    for tensor in reader.tensors:
        raw_bytes = int(np.asarray(tensor.data).nbytes)
        original += raw_bytes
        shape = tuple(int(v) for v in reversed(tensor.shape.tolist()))
        eligible = (
            len(shape) == 2
            and tensor.n_elements >= min_elements
            and min(shape) > min(ranks)
            and is_native_linear_name(tensor.name)
        )
        if not eligible:
            fixed += raw_bytes
            skipped += 1
            continue

        second = calibration.get(tensor.name, shape[1])
        calibration_source = "activation"
        if second is None:
            if require_calibration:
                fixed += raw_bytes
                skipped += 1
                continue
            second = uniform_second_moment(shape[1])
            calibration_source = "uniform_fallback"
            fallback += 1
        else:
            calibrated += 1

        try:
            w = _tensor_float32(tensor).reshape(shape)
        except (NotImplementedError, ValueError):
            fixed += raw_bytes
            skipped += 1
            continue

        options = build_tensor_candidates(
            w,
            tensor_name=tensor.name,
            raw_bytes=raw_bytes,
            ranks=ranks,
            residual_densities=residual_densities,
            qbits=qbits,
            second_moment=second,
            calibration_source=calibration_source,
            seed=seed,
            oversample=oversample,
            power_iters=power_iters,
            max_activation_rel_rmse=max_activation_rel_rmse,
        )
        if len(options) <= 1:
            fixed += raw_bytes
            skipped += 1
            continue
        candidate_map[tensor.name] = options
        optimizable_raw += raw_bytes

    return CandidateScan(
        candidates=candidate_map,
        fixed_payload_bytes=fixed,
        original_payload_bytes=original,
        optimizable_raw_bytes=optimizable_raw,
        calibrated_tensors=calibrated,
        fallback_tensors=fallback,
        skipped_tensors=skipped,
    )


def _plan_key(choices: dict[str, TensorCandidate]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, candidate.candidate_id) for name, candidate in choices.items()))


def _evaluate_choices(
    choices: dict[str, TensorCandidate],
    fixed_payload_bytes: int,
) -> tuple[int, float, float]:
    total_bytes = int(fixed_payload_bytes + sum(c.estimated_bytes for c in choices.values()))
    distortion = float(sum(c.distortion_energy for c in choices.values()))
    reference = float(sum(c.reference_energy for c in choices.values()))
    return total_bytes, distortion, reference


def _choose_lambda(
    candidate_map: dict[str, list[TensorCandidate]],
    lam: float,
    total_reference: float,
    original_payload_bytes: int,
) -> dict[str, TensorCandidate]:
    ref = max(total_reference, 1e-30)
    raw = max(float(original_payload_bytes), 1.0)
    choices: dict[str, TensorCandidate] = {}
    for name, options in candidate_map.items():
        choices[name] = min(
            options,
            key=lambda c: (
                c.distortion_energy / ref + lam * (c.estimated_bytes / raw),
                c.estimated_bytes,
            ),
        )
    return choices


def _greedy_refine(
    choices: dict[str, TensorCandidate],
    candidate_map: dict[str, list[TensorCandidate]],
    fixed_payload_bytes: int,
    target_bytes: int,
) -> dict[str, TensorCandidate]:
    choices = dict(choices)
    used, _, _ = _evaluate_choices(choices, fixed_payload_bytes)
    remaining = target_bytes - used
    if remaining <= 0:
        return choices

    while True:
        best: tuple[float, str, TensorCandidate, int] | None = None
        for name, current in choices.items():
            for candidate in candidate_map[name]:
                extra = candidate.estimated_bytes - current.estimated_bytes
                benefit = current.distortion_energy - candidate.distortion_energy
                if extra <= 0 or extra > remaining or benefit <= 0:
                    continue
                score = benefit / extra
                if best is None or score > best[0]:
                    best = (score, name, candidate, extra)
        if best is None:
            break
        _, name, candidate, extra = best
        choices[name] = candidate
        remaining -= extra
    return choices


def select_global_plan(scan: CandidateScan, target_ratio: float) -> GlobalPlan:
    if not (0.0 < target_ratio <= 1.0):
        raise ValueError("target_ratio must be in (0, 1]")
    if not scan.candidates:
        raise ValueError("candidate scan contains no optimizable tensors")

    target_bytes = int(math.floor(scan.original_payload_bytes * target_ratio))
    total_reference = float(
        sum(options[0].reference_energy for options in scan.candidates.values())
    )

    cheapest = {
        name: min(options, key=lambda c: (c.estimated_bytes, c.distortion_energy))
        for name, options in scan.candidates.items()
    }
    cheapest_bytes, cheapest_dist, cheapest_ref = _evaluate_choices(
        cheapest, scan.fixed_payload_bytes
    )
    if cheapest_bytes > target_bytes:
        rel = math.sqrt(cheapest_dist / max(cheapest_ref, 1e-30))
        return GlobalPlan(
            target_ratio, target_bytes, cheapest_bytes, False, rel,
            cheapest_dist, cheapest_ref, cheapest
        )

    observed: dict[tuple[tuple[str, str], ...], dict[str, TensorCandidate]] = {}

    raw_choices = {
        name: next(c for c in options if c.kind == "raw")
        for name, options in scan.candidates.items()
    }
    observed[_plan_key(raw_choices)] = raw_choices
    observed[_plan_key(cheapest)] = cheapest

    hi = 1.0
    for _ in range(80):
        choices = _choose_lambda(scan.candidates, hi, total_reference, scan.original_payload_bytes)
        observed[_plan_key(choices)] = choices
        used, _, _ = _evaluate_choices(choices, scan.fixed_payload_bytes)
        if used <= target_bytes:
            break
        hi *= 2.0

    lo = 0.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        choices = _choose_lambda(scan.candidates, mid, total_reference, scan.original_payload_bytes)
        observed[_plan_key(choices)] = choices
        used, _, _ = _evaluate_choices(choices, scan.fixed_payload_bytes)
        if used <= target_bytes:
            hi = mid
        else:
            lo = mid

    feasible_choices: list[tuple[float, int, dict[str, TensorCandidate]]] = []
    for choices in observed.values():
        used, distortion, _ = _evaluate_choices(choices, scan.fixed_payload_bytes)
        if used <= target_bytes:
            feasible_choices.append((distortion, -used, choices))

    _, _, best = min(feasible_choices, key=lambda item: (item[0], item[1]))
    best = _greedy_refine(best, scan.candidates, scan.fixed_payload_bytes, target_bytes)
    used, distortion, reference = _evaluate_choices(best, scan.fixed_payload_bytes)
    rel = math.sqrt(distortion / max(reference, 1e-30))
    return GlobalPlan(
        target_ratio=target_ratio,
        target_bytes=target_bytes,
        estimated_bytes=used,
        feasible=True,
        activation_rel_rmse=rel,
        distortion_energy=distortion,
        reference_energy=reference,
        choices=best,
    )


def build_pareto_plans(scan: CandidateScan, target_ratios: tuple[float, ...]) -> list[GlobalPlan]:
    ratios = sorted({float(r) for r in target_ratios if 0.0 < float(r) <= 1.0}, reverse=True)
    if not ratios:
        raise ValueError("target_ratios must contain values in (0, 1]")
    return [select_global_plan(scan, ratio) for ratio in ratios]
