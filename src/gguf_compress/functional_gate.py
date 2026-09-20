from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FunctionalGateResult:
    total: int
    baseline_correct: int
    candidate_correct: int
    answer_agreement: float
    baseline_correct_retention: float
    regressions: int
    improvements: int
    passed: bool


def evaluate_functional_gate(
    baseline_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
    *,
    min_retention: float = 1.0,
    min_agreement: float = 0.95,
    max_regressions: int = 0,
) -> FunctionalGateResult:
    baseline = list(baseline_rows)
    candidate = list(candidate_rows)
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must contain the same number of rows")

    by_id = {str(row["id"]): row for row in candidate}
    ordered_candidate = []
    for row in baseline:
        key = str(row["id"])
        if key not in by_id:
            raise ValueError(f"candidate is missing row {key}")
        ordered_candidate.append(by_id[key])

    total = len(baseline)
    baseline_correct = candidate_correct = agreement = regressions = improvements = retained = 0
    for b, c in zip(baseline, ordered_candidate, strict=True):
        b_ok = bool(b.get("correct"))
        c_ok = bool(c.get("correct"))
        baseline_correct += int(b_ok)
        candidate_correct += int(c_ok)
        agreement += int(b.get("answer") == c.get("answer"))
        retained += int(b_ok and c_ok)
        regressions += int(b_ok and not c_ok)
        improvements += int((not b_ok) and c_ok)

    answer_agreement = agreement / max(total, 1)
    retention = retained / max(baseline_correct, 1)
    passed = (
        retention >= float(min_retention)
        and answer_agreement >= float(min_agreement)
        and regressions <= int(max_regressions)
    )
    return FunctionalGateResult(
        total=total,
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
        answer_agreement=answer_agreement,
        baseline_correct_retention=retention,
        regressions=regressions,
        improvements=improvements,
        passed=passed,
    )
