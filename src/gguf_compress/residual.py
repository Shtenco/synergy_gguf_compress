from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SparseRowResidual:
    """Top-K sparse residual stored row-wise.

    For every output row we keep only the largest-magnitude correction terms.
    Values are int8 with one FP16 scale per output row. Column indices are
    uint16 when possible and uint32 otherwise.
    """

    cols: np.ndarray
    values: np.ndarray
    scales: np.ndarray
    in_features: int

    @property
    def nnz(self) -> int:
        return int(self.values.size)

    @property
    def k_per_row(self) -> int:
        return int(self.values.shape[1]) if self.values.ndim == 2 else 0

    @property
    def nbytes(self) -> int:
        return int(self.cols.nbytes + self.values.nbytes + self.scales.nbytes)


def build_sparse_row_residual(
    reference: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    density: float,
) -> SparseRowResidual | None:
    if density <= 0:
        return None
    if not (0 < density <= 1):
        raise ValueError("residual density must be in (0, 1]")

    w = np.asarray(reference, dtype=np.float32)
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if w.ndim != 2 or aa.ndim != 2 or bb.ndim != 2:
        raise ValueError("reference/A/B must be 2-D")
    if aa.shape[0] != w.shape[0] or bb.shape[1] != w.shape[1] or aa.shape[1] != bb.shape[0]:
        raise ValueError("A/B shapes do not match reference")

    out_features, in_features = w.shape
    k = min(in_features, max(1, int(math.ceil(in_features * density))))
    col_dtype = np.uint16 if in_features <= np.iinfo(np.uint16).max else np.uint32
    cols = np.empty((out_features, k), dtype=col_dtype)
    vals_f = np.empty((out_features, k), dtype=np.float32)

    # Row streaming avoids materializing the full dense residual matrix.
    for row in range(out_features):
        residual_row = w[row] - (aa[row] @ bb)
        if k == in_features:
            idx = np.arange(in_features)
        else:
            idx = np.argpartition(np.abs(residual_row), -k)[-k:]
        # Sorting by column improves locality in the runtime correction loop.
        idx = idx[np.argsort(idx)]
        cols[row] = idx.astype(col_dtype, copy=False)
        vals_f[row] = residual_row[idx]

    max_abs = np.max(np.abs(vals_f), axis=1)
    scales32 = np.where(max_abs > 0, max_abs / np.float32(127.0), np.float32(1.0))
    values = np.rint(vals_f / scales32[:, None]).clip(-127, 127).astype(np.int8)
    return SparseRowResidual(
        np.ascontiguousarray(cols),
        np.ascontiguousarray(values),
        scales32.astype(np.float16),
        in_features,
    )


def residual_dense_rows(residual: SparseRowResidual, out_features: int | None = None) -> np.ndarray:
    rows = residual.values.shape[0] if out_features is None else int(out_features)
    dense = np.zeros((rows, residual.in_features), dtype=np.float32)
    deq = residual.values.astype(np.float32) * residual.scales.astype(np.float32)[:, None]
    for row in range(rows):
        dense[row, residual.cols[row].astype(np.int64)] = deq[row]
    return dense


def apply_sparse_residual(x: np.ndarray, residual: SparseRowResidual) -> np.ndarray:
    """Compute X @ R.T without reconstructing the sparse residual matrix."""
    xx = np.asarray(x, dtype=np.float32)
    if xx.shape[-1] != residual.in_features:
        raise ValueError("input feature mismatch")

    flat = xx.reshape(-1, residual.in_features)
    out = np.empty((flat.shape[0], residual.values.shape[0]), dtype=np.float32)
    scales = residual.scales.astype(np.float32)
    values = residual.values.astype(np.float32)
    for row in range(residual.values.shape[0]):
        cols = residual.cols[row].astype(np.int64)
        weights = values[row] * scales[row]
        out[:, row] = (flat[:, cols] * weights[None, :]).sum(axis=1)
    return out.reshape(*xx.shape[:-1], residual.values.shape[0])
