from __future__ import annotations

import numpy as np

from gguf_compress.adaptive import AdaptivePolicy, adaptive_compress_matrix, role_rank_cap
from gguf_compress.residual import apply_sparse_residual, build_sparse_row_residual, residual_dense_rows


def test_sparse_residual_improves_reconstruction() -> None:
    rng = np.random.default_rng(1)
    w = rng.standard_normal((48, 64), dtype=np.float32)
    a = rng.standard_normal((48, 8), dtype=np.float32)
    b = rng.standard_normal((8, 64), dtype=np.float32) * 0.1
    residual = build_sparse_row_residual(w, a, b, 0.10)
    assert residual is not None
    base = a @ b
    corrected = base + residual_dense_rows(residual)
    assert np.linalg.norm(w - corrected) < np.linalg.norm(w - base)


def test_sparse_residual_inference_matches_dense() -> None:
    rng = np.random.default_rng(2)
    w = rng.standard_normal((16, 20), dtype=np.float32)
    a = np.zeros((16, 2), dtype=np.float32)
    b = np.zeros((2, 20), dtype=np.float32)
    residual = build_sparse_row_residual(w, a, b, 0.20)
    assert residual is not None
    x = rng.standard_normal((3, 20), dtype=np.float32)
    got = apply_sparse_residual(x, residual)
    expected = x @ residual_dense_rows(residual).T
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)


def test_adaptive_rank_finds_small_lowrank_solution() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((64, 4), dtype=np.float32)
    b = rng.standard_normal((4, 64), dtype=np.float32)
    w = a @ b
    result = adaptive_compress_matrix(
        w,
        tensor_name="blk.0.attn_q.weight",
        policy=AdaptivePolicy(candidates=(4, 8, 16), target_rel_rmse=0.12, qbits=8),
    )
    assert result.matrix.rank in (4, 8)
    assert result.matrix.rel_rmse < 0.12


def test_residual_reduces_adaptive_error() -> None:
    rng = np.random.default_rng(4)
    w = rng.standard_normal((64, 64), dtype=np.float32)
    no_res = adaptive_compress_matrix(
        w,
        tensor_name="blk.0.ffn_up.weight",
        policy=AdaptivePolicy(candidates=(8,), target_rel_rmse=0.0, qbits=4, residual_density=0.0),
    )
    with_res = adaptive_compress_matrix(
        w,
        tensor_name="blk.0.ffn_up.weight",
        policy=AdaptivePolicy(candidates=(8,), target_rel_rmse=0.0, qbits=4, residual_density=0.10),
    )
    assert with_res.corrected_rel_rmse < no_res.corrected_rel_rmse


def test_role_rank_caps() -> None:
    assert role_rank_cap("blk.0.attn_k.weight") == 32
    assert role_rank_cap("blk.0.attn_q.weight") == 64
    assert role_rank_cap("blk.0.ffn_up.weight") == 128
