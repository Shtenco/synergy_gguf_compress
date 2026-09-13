import math

from gguf_compress.native import factor_storage_bytes, is_native_linear_name, native_tensor_names, theoretical_native_ratio


def test_native_names_keep_b_at_original_name():
    b, a = native_tensor_names("blk.0.attn_q.weight")
    assert b == "blk.0.attn_q.weight"
    assert a == "blk.0.attn_q.weight.__gguf_lr_a"


def test_native_ratio_square_4096_rank8_f16_is_256x():
    ratio = theoretical_native_ratio((4096, 4096), 8, dense_itemsize=2, factor_itemsize=2)
    assert math.isclose(ratio, 256.0)
    assert factor_storage_bytes((4096, 4096), 8, 2) == 131072


def test_native_ratio_rectangular_matches_formula():
    shape = (11008, 4096)
    rank = 16
    expected = (shape[0] * shape[1]) / (rank * (shape[0] + shape[1]))
    assert math.isclose(theoretical_native_ratio(shape, rank), expected)


def test_native_allowlist_is_conservative():
    assert is_native_linear_name("blk.0.attn_q.weight")
    assert is_native_linear_name("blk.12.ffn_down.weight")
    assert is_native_linear_name("output.weight")
    assert not is_native_linear_name("token_embd.weight")
    assert not is_native_linear_name("classifier.weight")
