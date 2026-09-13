from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .codec import CompressedMatrix, _quality_from_compressed, _randomized_lowrank, factors
from .quant import quantize_rows
from .residual import SparseRowResidual, build_sparse_row_residual, residual_dense_rows


@dataclass(frozen=True)
class AdaptivePolicy:
    candidates: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
    target_rel_rmse: float = 0.20
    qbits: int = 4
    oversample: int = 8
    power_iters: int = 1
    residual_density: float = 0.0


@dataclass
class AdaptiveResult:
    matrix: CompressedMatrix
    residual: SparseRowResidual | None
    corrected_rel_rmse: float
    corrected_cosine: float
    candidates_tested: list[dict[str, float | int]]


def role_rank_cap(name: str, default_cap: int = 256) -> int:
    n = name.lower()
    if ".attn_k.weight" in n:
        return min(default_cap, 32)
    if any(s in n for s in (".attn_q.weight", ".attn_v.weight", ".attn_output.weight", ".attn_qkv.weight")):
        return min(default_cap, 64)
    if any(s in n for s in (".ffn_up.weight", ".ffn_gate.weight", ".ffn_down.weight")):
        return min(default_cap, 128)
    if n == "output.weight":
        return min(default_cap, 64)
    return default_cap


def _metrics_with_residual(
    w: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    residual: SparseRowResidual | None,
) -> tuple[float, float]:
    ref_sq = rec_sq = diff_sq = dot = 0.0
    r_dense = residual_dense_rows(residual) if residual is not None else None
    for row in range(w.shape[0]):
        rec = a[row] @ b
        if r_dense is not None:
            rec = rec + r_dense[row]
        ref = w[row]
        diff = ref - rec
        ref_sq += float(np.sum(ref * ref, dtype=np.float64))
        rec_sq += float(np.sum(rec * rec, dtype=np.float64))
        diff_sq += float(np.sum(diff * diff, dtype=np.float64))
        dot += float(np.sum(ref * rec, dtype=np.float64))
    rel_rmse = math.sqrt(diff_sq / max(ref_sq, 1e-30))
    cosine = dot / max(math.sqrt(ref_sq * rec_sq), 1e-30)
    return rel_rmse, cosine


def adaptive_compress_matrix(
    weight: np.ndarray,
    *,
    tensor_name: str,
    policy: AdaptivePolicy,
    seed: str = "gguf-compress-v3",
) -> AdaptiveResult:
    """Choose the smallest rank satisfying the requested reconstruction error.

    A single randomized-SVD basis is computed at the maximum candidate rank and
    truncated for each candidate. This makes rank search substantially cheaper
    than recomputing an SVD for every candidate.
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("only 2-D matrices can be factorized")

    m, n = w.shape
    cap = role_rank_cap(tensor_name, max(policy.candidates))
    max_rank = min(m, n, cap)
    candidates = sorted({min(int(r), max_rank) for r in policy.candidates if int(r) > 0})
    candidates = [r for r in candidates if r >= 1]
    if not candidates:
        candidates = [max_rank]

    seed_int = int.from_bytes(seed.encode("utf-8")[:8].ljust(8, b"\0"), "little", signed=False)
    start = time.perf_counter()
    a_full, b_full = _randomized_lowrank(
        w,
        max(candidates),
        policy.oversample,
        policy.power_iters,
        seed_int,
    )

    tested: list[dict[str, float | int]] = []
    chosen: CompressedMatrix | None = None
    for rank in candidates:
        qa = quantize_rows(a_full[:, :rank], policy.qbits)
        qb = quantize_rows(b_full[:rank, :], policy.qbits)
        cm = CompressedMatrix(
            "svd-lowrank",
            (m, n),
            rank,
            policy.qbits,
            qa,
            qb,
            None,
            time.perf_counter() - start,
            0.0,
            0.0,
        )
        rel_rmse, cosine = _quality_from_compressed(w, cm)
        cm.rel_rmse = rel_rmse
        cm.cosine = cosine
        tested.append({"rank": rank, "rel_rmse": rel_rmse, "cosine": cosine})
        chosen = cm
        if rel_rmse <= policy.target_rel_rmse:
            break

    assert chosen is not None
    a, b = factors(chosen)
    residual = build_sparse_row_residual(w, a, b, policy.residual_density)
    corrected_rel_rmse, corrected_cosine = _metrics_with_residual(w, a, b, residual)
    return AdaptiveResult(
        matrix=chosen,
        residual=residual,
        corrected_rel_rmse=corrected_rel_rmse,
        corrected_cosine=corrected_cosine,
        candidates_tested=tested,
    )
