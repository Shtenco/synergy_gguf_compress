from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizedRows:
    data: np.ndarray
    scales: np.ndarray
    qbits: int
    cols: int

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes + self.scales.nbytes)


def quantize_rows(matrix: np.ndarray, qbits: int = 4) -> QuantizedRows:
    """Symmetric per-row int8/int4 quantization."""
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if qbits not in (4, 8):
        raise ValueError("qbits must be 4 or 8")

    qmax = 7 if qbits == 4 else 127
    max_abs = np.max(np.abs(x), axis=1)
    scales32 = np.where(max_abs > 0, max_abs / np.float32(qmax), np.float32(1.0))
    q = np.rint(x / scales32[:, None]).clip(-qmax, qmax).astype(np.int8)
    scales = scales32.astype(np.float16)

    if qbits == 8:
        return QuantizedRows(np.ascontiguousarray(q), scales, qbits, x.shape[1])

    cols = x.shape[1]
    if cols & 1:
        q = np.pad(q, ((0, 0), (0, 1)), constant_values=0)
    u = (q.astype(np.int16) + 8).astype(np.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << np.uint8(4))).astype(np.uint8)
    return QuantizedRows(np.ascontiguousarray(packed.view(np.int8)), scales, qbits, cols)


def dequantize_rows(q: QuantizedRows) -> np.ndarray:
    scales = q.scales.astype(np.float32)
    if q.qbits == 8:
        vals = q.data.astype(np.int8, copy=False).astype(np.float32)
        return vals * scales[:, None]
    if q.qbits != 4:
        raise ValueError(f"unsupported qbits={q.qbits}")

    packed = q.data.view(np.uint8)
    lo = (packed & np.uint8(0x0F)).astype(np.int16) - 8
    hi = ((packed >> np.uint8(4)) & np.uint8(0x0F)).astype(np.int16) - 8
    vals = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    vals[:, 0::2] = lo
    vals[:, 1::2] = hi
    vals = vals[:, : q.cols]
    return vals * scales[:, None]
