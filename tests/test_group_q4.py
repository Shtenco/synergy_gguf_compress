from __future__ import annotations

import numpy as np

from gguf_compress.functional_gate import evaluate_functional_gate
from gguf_compress.group_q4 import dequantize_group_q4, group_q4_metrics, quantize_group_q4


def test_group_q4_roundtrip_shape_and_payload() -> None:
    rng = np.random.default_rng(123)
    w = rng.normal(size=(17, 65)).astype(np.float32)
    q = quantize_group_q4(w, group_size=64)
    rec = dequantize_group_q4(q)
    assert rec.shape == w.shape
    assert q.data.shape == (17, 33)
    assert q.scales.shape == (17, 2)
    assert q.payload_nbytes < w.nbytes


def test_group_q4_grouping_improves_over_row_scale() -> None:
    rng = np.random.default_rng(7)
    w = rng.normal(size=(32, 128)).astype(np.float32)
    w[:, :64] *= 0.05
    row = quantize_group_q4(w, group_size=128)
    grouped = quantize_group_q4(w, group_size=64)
    row_m = group_q4_metrics(w, row, w.nbytes)
    grouped_m = group_q4_metrics(w, grouped, w.nbytes)
    assert grouped_m.rel_rmse < row_m.rel_rmse


def test_functional_gate_requires_retention_and_agreement() -> None:
    baseline = [
        {"id": "a", "answer": "A", "correct": True},
        {"id": "b", "answer": "B", "correct": False},
        {"id": "c", "answer": "C", "correct": True},
    ]
    candidate = [
        {"id": "a", "answer": "A", "correct": True},
        {"id": "b", "answer": "D", "correct": False},
        {"id": "c", "answer": "C", "correct": True},
    ]
    result = evaluate_functional_gate(
        baseline,
        candidate,
        min_retention=1.0,
        min_agreement=2 / 3,
        max_regressions=0,
    )
    assert result.passed
    assert result.baseline_correct_retention == 1.0
    assert result.regressions == 0
