from __future__ import annotations

from gguf_compress.omega_v7_search import RetentionPolicy, compare_rows, passes_policy, select_smallest_passing


def _row(qid: str, *, answer: str | None, gold: str, correct: bool):
    return {
        "id": qid,
        "category": "test",
        "gold": gold,
        "answer": answer,
        "correct": correct,
        "raw": answer or "",
        "seconds": 0.01,
    }


def test_compare_rows_tracks_regressions_and_improvements() -> None:
    baseline = [
        _row("q1", answer="A", gold="A", correct=True),
        _row("q2", answer="B", gold="C", correct=False),
    ]
    candidate = [
        _row("q1", answer="D", gold="A", correct=False),
        _row("q2", answer="C", gold="C", correct=True),
    ]
    result = compare_rows(baseline, candidate)
    cmp = result["comparison"]
    assert cmp["baseline_correct_retention"] == 0.0
    assert cmp["regressions"] == 1
    assert cmp["improvements"] == 1
    assert cmp["accuracy_delta"] == 0.0


def test_strict_policy_rejects_any_forgetting() -> None:
    result = {
        "candidate": {"unparsed": 0},
        "comparison": {
            "baseline_correct_retention": 0.99,
            "accuracy_delta": 0.10,
            "regressions": 1,
        },
    }
    assert not passes_policy(result, RetentionPolicy())


def test_select_smallest_passing_uses_real_native_bytes() -> None:
    rows = [
        {
            "strict_pass": True,
            "native_q4_bytes": 900,
            "activation_rel_rmse": 0.12,
            "full": {"candidate": {"accuracy": 0.50}},
        },
        {
            "strict_pass": True,
            "native_q4_bytes": 700,
            "activation_rel_rmse": 0.20,
            "full": {"candidate": {"accuracy": 0.45}},
        },
        {
            "strict_pass": False,
            "native_q4_bytes": 500,
            "activation_rel_rmse": 0.10,
            "full": {"candidate": {"accuracy": 0.55}},
        },
    ]
    selected = select_smallest_passing(rows)
    assert selected is rows[1]
