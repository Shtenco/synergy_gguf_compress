from __future__ import annotations

import numpy as np

from gguf_compress.activation import activation_error_energy
from gguf_compress.hf_calibration import hf_module_to_gguf_weight
from gguf_compress.optimizer import CandidateScan, TensorCandidate, select_global_plan


def _candidate(
    name: str,
    cid: str,
    *,
    size: int,
    distortion: float,
    reference: float,
    rank: int | None,
) -> TensorCandidate:
    return TensorCandidate(
        tensor_name=name,
        candidate_id=cid,
        kind="raw" if rank is None else "factorized",
        rank=rank,
        residual_density=0.0,
        estimated_bytes=size,
        rel_rmse=0.0 if rank is None else (distortion / reference) ** 0.5,
        activation_rel_rmse=0.0 if rank is None else (distortion / reference) ** 0.5,
        distortion_energy=distortion,
        reference_energy=reference,
        calibration_source="activation",
    )


def test_activation_weighting_changes_error_priority() -> None:
    w = np.eye(2, dtype=np.float32)
    rec = w.copy()
    rec[0, 0] = 0.0
    rel_uniform, _, _ = activation_error_energy(w, rec, np.ones(2, dtype=np.float32))
    rel_weighted, _, _ = activation_error_energy(
        w, rec, np.array([100.0, 1.0], dtype=np.float32)
    )
    assert rel_weighted > rel_uniform


def test_global_budget_spends_bytes_on_high_value_layer() -> None:
    sensitive = [
        _candidate("blk.0.attn_q.weight", "cheap", size=20, distortion=100.0, reference=1000.0, rank=8),
        _candidate("blk.0.attn_q.weight", "better", size=50, distortion=1.0, reference=1000.0, rank=32),
        _candidate("blk.0.attn_q.weight", "raw", size=100, distortion=0.0, reference=1000.0, rank=None),
    ]
    insensitive = [
        _candidate("blk.0.ffn_up.weight", "cheap", size=20, distortion=10.0, reference=100.0, rank=8),
        _candidate("blk.0.ffn_up.weight", "better", size=50, distortion=5.0, reference=100.0, rank=32),
        _candidate("blk.0.ffn_up.weight", "raw", size=100, distortion=0.0, reference=100.0, rank=None),
    ]
    scan = CandidateScan(
        candidates={
            "blk.0.attn_q.weight": sensitive,
            "blk.0.ffn_up.weight": insensitive,
        },
        fixed_payload_bytes=0,
        original_payload_bytes=200,
        optimizable_raw_bytes=200,
        calibrated_tensors=2,
        fallback_tensors=0,
        skipped_tensors=0,
    )
    plan = select_global_plan(scan, 0.35)
    assert plan.feasible
    assert plan.estimated_bytes <= 70
    assert plan.choices["blk.0.attn_q.weight"].candidate_id == "better"
    assert plan.choices["blk.0.ffn_up.weight"].candidate_id == "cheap"


def test_hf_llama_qwen_name_mapping() -> None:
    assert hf_module_to_gguf_weight("model.layers.3.self_attn.q_proj") == "blk.3.attn_q.weight"
    assert hf_module_to_gguf_weight("model.layers.7.mlp.down_proj") == "blk.7.ffn_down.weight"
    assert hf_module_to_gguf_weight("lm_head") == "output.weight"
    assert hf_module_to_gguf_weight("model.embed_tokens") is None
