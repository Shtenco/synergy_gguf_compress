import numpy as np

from gguf_compress.codec import compress_matrix, factorized_inference, reconstruct_matrix, theoretical_mac_speedup
from gguf_compress.quant import dequantize_rows, quantize_rows
from gguf_compress.seed import SFC32, seeded_orthonormal_basis


def make_lowrank(m=128, n=96, r=4):
    rng = np.random.default_rng(7)
    return (rng.standard_normal((m, r), dtype=np.float32) @ rng.standard_normal((r, n), dtype=np.float32)).astype(np.float32)


def test_sfc32_deterministic_and_basis_orthonormal():
    a = SFC32.from_seed("abc")
    b = SFC32.from_seed("abc")
    assert [a.next_u32() for _ in range(100)] == [b.next_u32() for _ in range(100)]
    basis = seeded_orthonormal_basis(128, 8, "test")
    np.testing.assert_allclose(basis @ basis.T, np.eye(8), atol=1e-5, rtol=1e-5)


def test_quant_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((17, 9), dtype=np.float32)
    y4 = dequantize_rows(quantize_rows(x, 4))
    y8 = dequantize_rows(quantize_rows(x, 8))
    assert y4.shape == x.shape
    assert np.mean((x - y8) ** 2) < np.mean((x - y4) ** 2)


def test_svd_lowrank_recovers_lowrank_matrix_and_factorized_runtime_matches():
    w = make_lowrank()
    c = compress_matrix(w, method="svd-lowrank", rank=4, qbits=8, power_iters=1)
    rec = reconstruct_matrix(c)
    assert rec.shape == w.shape
    assert c.rel_rmse < 0.03
    assert c.cosine > 0.999
    rng = np.random.default_rng(9)
    x = rng.standard_normal((5, w.shape[1]), dtype=np.float32)
    np.testing.assert_allclose(factorized_inference(x, c), x @ rec.T, atol=1e-4, rtol=1e-4)


def test_seed_projection_is_deterministic_and_small():
    w = make_lowrank(m=128, n=128, r=8)
    c1 = compress_matrix(w, method="seed-projection", rank=8, qbits=4, seed="same")
    c2 = compress_matrix(w, method="seed-projection", rank=8, qbits=4, seed="same")
    np.testing.assert_array_equal(c1.a.data, c2.a.data)
    assert c1.payload_nbytes < w.nbytes / 10


def test_4096_rank8_has_256x_theoretical_mac_reduction():
    assert theoretical_mac_speedup((4096, 4096), 8) == 256.0
