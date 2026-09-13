from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np

from .codec import compress_matrix, theoretical_mac_speedup


@dataclass
class BenchmarkResult:
    method: str
    shape: tuple[int, int]
    rank: int
    qbits: int
    original_bytes: int
    compressed_payload_bytes: int
    storage_ratio: float
    rel_rmse: float
    cosine: float
    fit_seconds: float
    dense_ms: float
    compressed_ms: float
    measured_speedup: float
    theoretical_mac_speedup: float
    output_rel_rmse: float

    def to_dict(self) -> dict:
        return asdict(self)


def _median_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return float(statistics.median(samples))


def synthetic_weight(size: int, *, kind: str = "lowrank", true_rank: int = 32, noise: float = 0.01, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "random":
        return rng.standard_normal((size, size), dtype=np.float32) / np.float32(np.sqrt(size))
    if kind != "lowrank":
        raise ValueError("kind must be lowrank or random")
    r = min(true_rank, size)
    a = rng.standard_normal((size, r), dtype=np.float32) / np.float32(np.sqrt(r))
    b = rng.standard_normal((r, size), dtype=np.float32) / np.float32(np.sqrt(size))
    w = a @ b
    if noise:
        w += rng.standard_normal(w.shape, dtype=np.float32) * np.float32(noise / np.sqrt(size))
    return np.ascontiguousarray(w)


def run_matrix_benchmark(
    *,
    size: int = 2048,
    batch: int = 32,
    rank: int = 8,
    qbits: int = 4,
    method: str = "svd-lowrank",
    matrix_kind: str = "lowrank",
    true_rank: int = 32,
    noise: float = 0.01,
    warmup: int = 5,
    iters: int = 20,
    seed: str = "gguf-compress-bench",
) -> BenchmarkResult:
    w = synthetic_weight(size, kind=matrix_kind, true_rank=true_rank, noise=noise)
    cm = compress_matrix(w, method=method, rank=rank, qbits=qbits, seed=seed)  # type: ignore[arg-type]
    rng = np.random.default_rng(77)
    x = rng.standard_normal((batch, size), dtype=np.float32)
    dense_out = x @ w.T

    from .codec import factors
    a_fac, b_fac = factors(cm)

    def compressed_forward():
        return (x @ b_fac.T) @ a_fac.T

    dense_ms = _median_ms(lambda: x @ w.T, warmup, iters)
    compressed_ms = _median_ms(compressed_forward, warmup, iters)
    comp_out = compressed_forward()
    denom = float(np.sqrt(np.mean(dense_out * dense_out)))
    out_err = float(np.sqrt(np.mean((dense_out - comp_out) ** 2)) / max(denom, 1e-12))

    return BenchmarkResult(
        method=method,
        shape=w.shape,
        rank=rank,
        qbits=qbits,
        original_bytes=w.nbytes,
        compressed_payload_bytes=cm.payload_nbytes,
        storage_ratio=w.nbytes / max(cm.payload_nbytes, 1),
        rel_rmse=cm.rel_rmse,
        cosine=cm.cosine,
        fit_seconds=cm.fit_seconds,
        dense_ms=dense_ms,
        compressed_ms=compressed_ms,
        measured_speedup=dense_ms / max(compressed_ms, 1e-12),
        theoretical_mac_speedup=theoretical_mac_speedup(w.shape, rank),
        output_rel_rmse=out_err,
    )
