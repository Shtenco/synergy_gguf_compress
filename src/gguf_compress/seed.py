from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

_U32 = 0xFFFFFFFF


def _rotl32(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & _U32


def seed_words(seed: str | bytes | int) -> tuple[int, int, int, int]:
    """Map an arbitrary seed into four non-zero uint32 words."""
    if isinstance(seed, int):
        payload = str(seed).encode("utf-8")
    elif isinstance(seed, str):
        payload = seed.encode("utf-8")
    else:
        payload = seed
    digest = hashlib.blake2s(payload, digest_size=16, person=b"ggufseed").digest()
    words = tuple(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))
    if not any(words):
        return (0x9E3779B9, 0x243F6A88, 0xB7E15162, 1)
    return words  # type: ignore[return-value]


@dataclass
class SFC32:
    """Small Fast Counter PRNG with deterministic cross-platform uint32 output."""

    a: int
    b: int
    c: int
    d: int

    @classmethod
    def from_seed(cls, seed: str | bytes | int) -> "SFC32":
        g = cls(*seed_words(seed))
        for _ in range(16):
            g.next_u32()
        return g

    def next_u32(self) -> int:
        a, b, c, d = self.a, self.b, self.c, self.d
        t = (a + b + d) & _U32
        d = (d + 1) & _U32
        a = (b ^ (b >> 9)) & _U32
        b = (c + ((c << 3) & _U32)) & _U32
        c = (_rotl32(c, 21) + t) & _U32
        self.a, self.b, self.c, self.d = a, b, c, d
        return t

    def rademacher(self, count: int) -> np.ndarray:
        """Return count deterministic values in {-1,+1} as float32."""
        out = np.empty(count, dtype=np.float32)
        for i in range(count):
            out[i] = 1.0 if (self.next_u32() & 1) else -1.0
        return out


def tensor_seed(global_seed: str, tensor_name: str, trial: int = 0) -> str:
    return f"{global_seed}\x1f{tensor_name}\x1f{trial}"


def seeded_orthonormal_basis(
    in_features: int,
    rank: int,
    seed: str,
    *,
    include_dc: bool = True,
) -> np.ndarray:
    """Generate deterministic row-orthonormal B[rank, in_features]."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if rank > in_features:
        raise ValueError("rank cannot exceed in_features")

    rows = np.empty((rank, in_features), dtype=np.float32)
    start = 0
    if include_dc:
        rows[0] = np.float32(1.0 / math.sqrt(in_features))
        start = 1

    if start < rank:
        prng = SFC32.from_seed(seed)
        rows[start:] = prng.rademacher((rank - start) * in_features).reshape(rank - start, in_features)
        rows[start:] *= np.float32(1.0 / math.sqrt(in_features))

    q, _ = np.linalg.qr(rows.T, mode="reduced")
    return np.ascontiguousarray(q.T.astype(np.float32, copy=False))
