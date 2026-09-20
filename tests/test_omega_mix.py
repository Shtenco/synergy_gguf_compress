from __future__ import annotations

from gguf_compress.omega_mix import OmegaClass, OmegaConfig, apply_omega_policy, classify_tensor
from gguf_compress.optimizer import CandidateScan, TensorCandidate


def _c(name: str, cid: str, *, rank: int | None, rmse: float, density: float = 0.0, size: int = 10):
    return TensorCandidate(
        tensor_name=name,
        candidate_id=cid,
        kind="raw" if rank is None else "factorized",
        rank=rank,
        residual_density=density,
        estimated_bytes=size if rank is not None else 100,
        rel_rmse=rmse,
        activation_rel_rmse=rmse,
        distortion_energy=rmse * rmse * 100.0,
        reference_energy=100.0,
        calibration_source="activation",
    )


def test_omega_classifies_roles() -> None:
    assert classify_tensor("blk.3.attn_q.weight", max_layer=31) is OmegaClass.CRITICAL
    assert classify_tensor("blk.10.ffn_up.weight", max_layer=31) is OmegaClass.CHEAP
    assert classify_tensor("blk.10.ffn_gate.weight", max_layer=31) is OmegaClass.PROTECTED
    assert classify_tensor("output.weight", max_layer=31) is OmegaClass.AUTHORITY


def test_critical_tensor_rejects_damaging_cheap_candidate() -> None:
    name = "blk.10.attn_q.weight"
    scan = CandidateScan(
        candidates={
            name: [
                _c(name, "tiny-bad", rank=8, rmse=0.12, density=0.0, size=8),
                _c(name, "protected", rank=32, rmse=0.18, density=0.01, size=30),
                _c(name, "raw", rank=None, rmse=0.0),
            ]
        },
        fixed_payload_bytes=0,
        original_payload_bytes=100,
        optimizable_raw_bytes=100,
        calibrated_tensors=1,
        fallback_tensors=0,
        skipped_tensors=0,
    )
    filtered, classes = apply_omega_policy(scan, OmegaConfig())
    ids = {x.candidate_id for x in filtered.candidates[name]}
    assert classes[name] == "critical"
    assert "tiny-bad" not in ids
    assert "protected" in ids
    assert "raw" in ids

def test_group_q4_candidate_is_allowed_for_cheap_tensor() -> None:
    name = "blk.10.ffn_up.weight"
    group = TensorCandidate(
        tensor_name=name,
        candidate_id="gq4-g32",
        kind="group_q4",
        rank=None,
        residual_density=0.0,
        estimated_bytes=70,
        rel_rmse=0.10,
        activation_rel_rmse=0.10,
        distortion_energy=1.0,
        reference_energy=100.0,
        calibration_source="activation",
        group_size=32,
    )
    raw = TensorCandidate(
        tensor_name=name,
        candidate_id="raw",
        kind="raw",
        rank=None,
        residual_density=0.0,
        estimated_bytes=100,
        rel_rmse=0.0,
        activation_rel_rmse=0.0,
        distortion_energy=0.0,
        reference_energy=100.0,
        calibration_source="activation",
    )
    scan = CandidateScan(
        candidates={name: [group, raw]},
        fixed_payload_bytes=0,
        original_payload_bytes=100,
        optimizable_raw_bytes=100,
        calibrated_tensors=1,
        fallback_tensors=0,
        skipped_tensors=0,
    )
    filtered, _ = apply_omega_policy(scan, OmegaConfig())
    ids = {x.candidate_id for x in filtered.candidates[name]}
    assert "gq4-g32" in ids
