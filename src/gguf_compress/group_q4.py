from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GroupQ4:
    """Direct group-wise Q4 representation consumed by the Omega-MIX runtime.

    Logical values are signed integers in [-7, 7]. Two values are packed per
    byte after adding 8, exactly matching q4_at() in lowrank_q4_kernel.h.
    Scales are FP16, one scale per output-row/group.
    """

    data: np.ndarray
    scales: np.ndarray
    shape: tuple[int, int]
    group_size: int

    @property
    def out_features(self) -> int:
        return int(self.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.shape[1])

    @property
    def groups_per_row(self) -> int:
        return int(math.ceil(self.in_features / self.group_size))

    @property
    def payload_nbytes(self) -> int:
        return int(self.data.nbytes + self.scales.nbytes)

    def ratio_vs(self, original_bytes: int) -> float:
        return self.payload_nbytes / max(int(original_bytes), 1)


@dataclass(frozen=True)
class GroupQ4Metrics:
    rel_rmse: float
    cosine: float
    payload_bytes: int
    payload_ratio: float


def quantize_group_q4(weight: np.ndarray, group_size: int = 64) -> GroupQ4:
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("GROUP_Q4 expects a 2-D matrix")
    if group_size < 2:
        raise ValueError("group_size must be >= 2")

    out_features, in_features = w.shape
    padded_cols = int(math.ceil(in_features / group_size) * group_size)
    padded = np.pad(w, ((0, 0), (0, padded_cols - in_features)))
    grouped = padded.reshape(out_features, -1, group_size)

    max_abs = np.max(np.abs(grouped), axis=2)
    scales32 = np.where(max_abs > 0, max_abs / np.float32(7.0), np.float32(1.0))
    q = np.rint(grouped / scales32[:, :, None]).clip(-7, 7).astype(np.int8)
    q = q.reshape(out_features, padded_cols)[:, :in_features]

    if in_features & 1:
        q = np.pad(q, ((0, 0), (0, 1)), constant_values=0)
    u = (q.astype(np.int16) + 8).astype(np.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << np.uint8(4))).astype(np.uint8)

    return GroupQ4(
        data=np.ascontiguousarray(packed.view(np.int8)),
        scales=np.ascontiguousarray(scales32.astype(np.float16)),
        shape=(out_features, in_features),
        group_size=int(group_size),
    )


def dequantize_group_q4(q: GroupQ4) -> np.ndarray:
    packed = np.asarray(q.data).view(np.uint8)
    lo = (packed & np.uint8(0x0F)).astype(np.int16) - 8
    hi = ((packed >> np.uint8(4)) & np.uint8(0x0F)).astype(np.int16) - 8
    vals = np.empty((q.out_features, packed.shape[1] * 2), dtype=np.float32)
    vals[:, 0::2] = lo
    vals[:, 1::2] = hi
    vals = vals[:, : q.in_features]

    groups = q.groups_per_row
    group_ids = np.minimum(np.arange(q.in_features) // q.group_size, groups - 1)
    scales = q.scales.astype(np.float32)
    return np.ascontiguousarray(vals * scales[:, group_ids], dtype=np.float32)


def group_q4_metrics(weight: np.ndarray, q: GroupQ4, original_bytes: int) -> GroupQ4Metrics:
    w = np.asarray(weight, dtype=np.float32)
    rec = dequantize_group_q4(q)
    if w.shape != rec.shape:
        raise ValueError("reference shape does not match GROUP_Q4 shape")
    diff = w - rec
    ref_sq = float(np.sum(w * w, dtype=np.float64))
    diff_sq = float(np.sum(diff * diff, dtype=np.float64))
    dot = float(np.sum(w * rec, dtype=np.float64))
    rec_sq = float(np.sum(rec * rec, dtype=np.float64))
    rel_rmse = math.sqrt(diff_sq / max(ref_sq, 1e-30))
    cosine = dot / max(math.sqrt(ref_sq * rec_sq), 1e-30)
    return GroupQ4Metrics(
        rel_rmse=rel_rmse,
        cosine=cosine,
        payload_bytes=q.payload_nbytes,
        payload_ratio=q.ratio_vs(original_bytes),
    )
