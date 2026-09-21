from __future__ import annotations

from gguf_compress.functional_safety import FunctionalSafetyDecision, apply_functional_safety
from gguf_compress.optimizer import CandidateScan, TensorCandidate


def _candidate(name: str, cid: str, kind: str, size: int) -> TensorCandidate:
    return TensorCandidate(
        tensor_name=name,
        candidate_id=cid,
        kind=kind,
        rank=None,
        residual_density=0.0,
        estimated_bytes=size,
        rel_rmse=0.1 if kind != "raw" else 0.0,
        activation_rel_rmse=0.1 if kind != "raw" else 0.0,
        distortion_energy=1.0 if kind != "raw" else 0.0,
        reference_energy=100.0,
        calibration_source="activation",
        group_size=8 if kind == "group_q4" else None,
    )


def test_functional_safety_removes_known_fail_and_keeps_pass() -> None:
    name = "blk.23.ffn_up.weight"
    scan = CandidateScan(
        candidates={
            name: [
                _candidate(name, "gq4-g8", "group_q4", 60),
                _candidate(name, "gq4-g4", "group_q4", 80),
                _candidate(name, "raw", "raw", 100),
            ]
        },
        fixed_payload_bytes=0,
        original_payload_bytes=100,
        optimizable_raw_bytes=100,
        calibrated_tensors=1,
        fallback_tensors=0,
        skipped_tensors=0,
    )
    decisions = {
        (name, "gq4-g8"): FunctionalSafetyDecision(name, "gq4-g8", False),
        (name, "gq4-g4"): FunctionalSafetyDecision(name, "gq4-g4", True),
    }
    filtered, stats = apply_functional_safety(scan, decisions)
    ids = {candidate.candidate_id for candidate in filtered.candidates[name]}
    assert ids == {"gq4-g4", "raw"}
    assert stats["known_pass"] == 1
    assert stats["known_fail_removed"] == 1


def test_functional_safety_known_only_falls_back_to_raw() -> None:
    name = "blk.10.ffn_gate.weight"
    scan = CandidateScan(
        candidates={
            name: [
                _candidate(name, "gq4-g8", "group_q4", 60),
                _candidate(name, "raw", "raw", 100),
            ]
        },
        fixed_payload_bytes=0,
        original_payload_bytes=100,
        optimizable_raw_bytes=100,
        calibrated_tensors=1,
        fallback_tensors=0,
        skipped_tensors=0,
    )
    filtered, stats = apply_functional_safety(scan, {}, require_known=True)
    assert [candidate.candidate_id for candidate in filtered.candidates[name]] == ["raw"]
    assert stats["unknown_removed"] == 1
