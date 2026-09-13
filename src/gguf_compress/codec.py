from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .quant import QuantizedRows, dequantize_rows, quantize_rows
from .seed import seeded_orthonormal_basis, tensor_seed

Method = Literal["seed-projection", "svd-lowrank"]


@dataclass
class CompressedMatrix:
    method: Method
    shape: tuple[int, int]
    rank: int
    qbits: int
    a: QuantizedRows
    b: QuantizedRows | None
    seed: str | None
    fit_seconds: float
    rel_rmse: float
    cosine: float

    @property
    def payload_nbytes(self) -> int:
        return self.a.nbytes + (self.b.nbytes if self.b is not None else 0)

    def compression_ratio_vs(self, original_nbytes: int) -> float:
        return float(original_nbytes / max(self.payload_nbytes, 1))


def quality_metrics(reference: np.ndarray, approx: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=np.float32)
    rec = np.asarray(approx, dtype=np.float32)
    diff = ref - rec
    denom = float(np.sqrt(np.mean(ref * ref)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    rel_rmse = rmse / max(denom, 1e-12)
    dot = float(np.sum(ref * rec, dtype=np.float64))
    norms = math.sqrt(float(np.sum(ref * ref, dtype=np.float64)) * float(np.sum(rec * rec, dtype=np.float64)))
    cosine = dot / max(norms, 1e-30)
    return rel_rmse, cosine


def _randomized_lowrank(weight: np.ndarray, rank: int, oversample: int, n_iter: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomized SVD returning A[out,r], B[r,in] so W ~= A@B."""
    w = np.asarray(weight, dtype=np.float32)
    m, n = w.shape
    k = min(max(rank + oversample, rank), min(m, n))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, k), dtype=np.float32)
    y = w @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    for _ in range(max(0, n_iter)):
        z = w.T @ q
        q, _ = np.linalg.qr(w @ z, mode="reduced")
    small = q.T @ w
    uh, s, vt = np.linalg.svd(small, full_matrices=False)
    u = q @ uh[:, :rank]
    a = u * s[:rank][None, :]
    b = vt[:rank, :]
    return np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)


def _quality_from_compressed(reference: np.ndarray, matrix: CompressedMatrix, block_rows: int = 512) -> tuple[float, float]:
    """Quality metrics without materializing a second full-size weight matrix."""
    ref = np.asarray(reference, dtype=np.float32)
    a, b = factors(matrix)
    diff_sq = 0.0
    ref_sq = 0.0
    rec_sq = 0.0
    dot = 0.0
    for i in range(0, ref.shape[0], block_rows):
        j = min(i + block_rows, ref.shape[0])
        rr = ref[i:j]
        rec = a[i:j] @ b
        d = rr - rec
        diff_sq += float(np.sum(d * d, dtype=np.float64))
        ref_sq += float(np.sum(rr * rr, dtype=np.float64))
        rec_sq += float(np.sum(rec * rec, dtype=np.float64))
        dot += float(np.sum(rr * rec, dtype=np.float64))
    rel_rmse = math.sqrt(diff_sq / max(ref_sq, 1e-30))
    cosine = dot / max(math.sqrt(ref_sq * rec_sq), 1e-30)
    return rel_rmse, cosine


def compress_matrix(
    weight: np.ndarray,
    *,
    method: Method = "svd-lowrank",
    rank: int = 8,
    qbits: int = 4,
    seed: str = "gguf-compress-v1",
    tensor_name: str = "weight",
    oversample: int = 8,
    power_iters: int = 1,
) -> CompressedMatrix:
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("only 2-D matrices can be factorized")
    m, n = w.shape
    rank = min(int(rank), m, n)
    if rank < 1:
        raise ValueError("rank must be >= 1")

    start = time.perf_counter()
    used_seed: str | None = None
    if method == "seed-projection":
        used_seed = tensor_seed(seed, tensor_name)
        b = seeded_orthonormal_basis(n, rank, used_seed)
        a = w @ b.T
        qb = None
    elif method == "svd-lowrank":
        seed_int = int.from_bytes(seed.encode("utf-8")[:8].ljust(8, b"\0"), "little", signed=False)
        a, b = _randomized_lowrank(w, rank, oversample, power_iters, seed_int)
        qb = quantize_rows(b, qbits)
    else:
        raise ValueError(f"unknown method: {method}")

    qa = quantize_rows(a, qbits)
    fit_seconds = time.perf_counter() - start
    candidate = CompressedMatrix(method, (m, n), rank, qbits, qa, qb, used_seed, fit_seconds, 0.0, 0.0)
    rel_rmse, cosine = _quality_from_compressed(w, candidate)
    candidate.rel_rmse = rel_rmse
    candidate.cosine = cosine
    return candidate


def factors(matrix: CompressedMatrix) -> tuple[np.ndarray, np.ndarray]:
    a = dequantize_rows(matrix.a)
    if matrix.method == "seed-projection":
        if matrix.seed is None:
            raise ValueError("seed-projection payload is missing seed")
        b = seeded_orthonormal_basis(matrix.shape[1], matrix.rank, matrix.seed)
    else:
        if matrix.b is None:
            raise ValueError("svd-lowrank payload is missing B")
        b = dequantize_rows(matrix.b)
    return a, b


def reconstruct_matrix(matrix: CompressedMatrix) -> np.ndarray:
    a, b = factors(matrix)
    return np.ascontiguousarray(a @ b, dtype=np.float32)


def factorized_inference(x: np.ndarray, matrix: CompressedMatrix) -> np.ndarray:
    """Compute X @ W_hat.T without materializing W_hat."""
    a, b = factors(matrix)
    xx = np.asarray(x, dtype=np.float32)
    return (xx @ b.T) @ a.T


def theoretical_mac_speedup(shape: tuple[int, int], rank: int) -> float:
    out_features, in_features = shape
    dense = out_features * in_features
    lowrank = rank * (out_features + in_features)
    return float(dense / max(lowrank, 1))
