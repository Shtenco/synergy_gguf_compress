from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActivationCalibration:
    second_moment: dict[str, np.ndarray]
    path: Path | None = None

    def get(self, tensor_name: str, in_features: int) -> np.ndarray | None:
        value = self.second_moment.get(tensor_name)
        if value is None:
            return None
        x = np.asarray(value, dtype=np.float32).reshape(-1)
        if x.size != int(in_features):
            raise ValueError(
                f"activation calibration shape mismatch for {tensor_name}: "
                f"expected {in_features}, got {x.size}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError(f"activation calibration contains non-finite values for {tensor_name}")
        return np.maximum(x, np.float32(1e-12))


def load_activation_calibration(path: str | Path) -> ActivationCalibration:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as data:
        stats = {
            str(name): np.ascontiguousarray(np.asarray(data[name], dtype=np.float32).reshape(-1))
            for name in data.files
            if not str(name).startswith("__")
        }
    if not stats:
        raise ValueError(f"activation calibration contains no tensor statistics: {p}")
    return ActivationCalibration(stats, p.resolve())


def uniform_second_moment(in_features: int) -> np.ndarray:
    return np.ones(int(in_features), dtype=np.float32)


def activation_error_energy(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    second_moment: np.ndarray,
) -> tuple[float, float, float]:
    """Return (relative RMSE, squared error energy, reference energy).

    With a diagonal input covariance approximation, E[x_j^2] weights column j.
    The numerator approximates E||x(W-W_hat).T||^2.
    """
    w = np.asarray(reference, dtype=np.float32)
    rec = np.asarray(reconstructed, dtype=np.float32)
    s = np.asarray(second_moment, dtype=np.float32).reshape(-1)
    if w.shape != rec.shape or w.ndim != 2:
        raise ValueError("reference and reconstructed must be equal-shaped 2-D matrices")
    if s.size != w.shape[1]:
        raise ValueError("second_moment must match input features")
    diff = w - rec
    ref_energy = float(np.sum((w * w) * s[None, :], dtype=np.float64))
    diff_energy = float(np.sum((diff * diff) * s[None, :], dtype=np.float64))
    rel = float(np.sqrt(diff_energy / max(ref_energy, 1e-30)))
    return rel, diff_energy, ref_energy
