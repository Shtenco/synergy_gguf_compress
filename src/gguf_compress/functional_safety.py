from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .optimizer import CandidateScan, TensorCandidate


@dataclass(frozen=True)
class FunctionalSafetyDecision:
    tensor_name: str
    candidate_id: str
    passed: bool
    source: str | None = None
    reason: str | None = None
    baseline_correct_retention: float | None = None
    regressions: int | None = None
    answer_agreement: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FunctionalSafetyDecision":
        return cls(
            tensor_name=str(payload["tensor_name"]),
            candidate_id=str(payload["candidate_id"]),
            passed=bool(payload["pass"]),
            source=payload.get("source"),
            reason=payload.get("reason"),
            baseline_correct_retention=payload.get("baseline_correct_retention"),
            regressions=payload.get("regressions"),
            answer_agreement=payload.get("answer_agreement"),
        )


def load_functional_safety(
    path: str | Path,
) -> tuple[dict[tuple[str, str], FunctionalSafetyDecision], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "gguf_compress.functional_safety/v1":
        raise ValueError(f"unsupported functional safety format: {payload.get('format')!r}")

    decisions: dict[tuple[str, str], FunctionalSafetyDecision] = {}
    for row in payload.get("records", []):
        decision = FunctionalSafetyDecision.from_dict(row)
        decisions[(decision.tensor_name, decision.candidate_id)] = decision
    return decisions, payload


def apply_functional_safety(
    scan: CandidateScan,
    decisions: dict[tuple[str, str], FunctionalSafetyDecision],
    *,
    require_known: bool = False,
) -> tuple[CandidateScan, dict[str, int]]:
    filtered: dict[str, list[TensorCandidate]] = {}
    stats = {
        "known_pass": 0,
        "known_fail_removed": 0,
        "unknown_kept": 0,
        "unknown_removed": 0,
        "raw_kept": 0,
    }

    for name, options in scan.candidates.items():
        kept: list[TensorCandidate] = []
        raw = next((candidate for candidate in options if candidate.kind == "raw"), None)

        for candidate in options:
            if candidate.kind == "raw":
                kept.append(candidate)
                stats["raw_kept"] += 1
                continue

            decision = decisions.get((name, candidate.candidate_id))
            if decision is None:
                if require_known:
                    stats["unknown_removed"] += 1
                else:
                    kept.append(candidate)
                    stats["unknown_kept"] += 1
                continue

            if decision.passed:
                kept.append(candidate)
                stats["known_pass"] += 1
            else:
                stats["known_fail_removed"] += 1

        if not kept and raw is not None:
            kept = [raw]
        elif not kept:
            kept = list(options)

        filtered[name] = sorted(
            kept,
            key=lambda candidate: (candidate.estimated_bytes, candidate.distortion_energy),
        )

    safe_scan = CandidateScan(
        candidates=filtered,
        fixed_payload_bytes=scan.fixed_payload_bytes,
        original_payload_bytes=scan.original_payload_bytes,
        optimizable_raw_bytes=scan.optimizable_raw_bytes,
        calibrated_tensors=scan.calibrated_tensors,
        fallback_tensors=scan.fallback_tensors,
        skipped_tensors=scan.skipped_tensors,
    )
    return safe_scan, stats
