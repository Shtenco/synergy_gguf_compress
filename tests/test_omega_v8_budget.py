from __future__ import annotations

from gguf_compress.omega_v8_budget import (
    AdaptiveSearchConfig,
    coarse_ratios,
    find_pass_fail_bracket,
    next_refinement_ratio,
)


def test_coarse_ratios_descend_and_include_edges() -> None:
    cfg = AdaptiveSearchConfig(min_ratio=0.50, max_ratio=0.98, coarse_step=0.08)
    values = coarse_ratios(cfg)
    assert values[0] == 0.98
    assert values[-1] == 0.50
    assert all(a > b for a, b in zip(values, values[1:]))


def test_find_tightest_pass_fail_bracket() -> None:
    rows = [
        {"target_ratio": 0.98, "strict_pass": True},
        {"target_ratio": 0.90, "strict_pass": True},
        {"target_ratio": 0.82, "strict_pass": False},
        {"target_ratio": 0.86, "strict_pass": False},
    ]
    assert find_pass_fail_bracket(rows) == (0.86, 0.90)


def test_refinement_uses_midpoint_until_tolerance() -> None:
    cfg = AdaptiveSearchConfig(
        min_ratio=0.50,
        max_ratio=0.98,
        coarse_step=0.08,
        tolerance=0.01,
        max_refinements=8,
    )
    rows = [
        {"target_ratio": 0.90, "strict_pass": True},
        {"target_ratio": 0.82, "strict_pass": False},
    ]
    assert next_refinement_ratio(rows, cfg) == 0.86

    rows.append({"target_ratio": 0.86, "strict_pass": True})
    assert next_refinement_ratio(rows, cfg) == 0.84

    rows.extend([
        {"target_ratio": 0.84, "strict_pass": False},
        {"target_ratio": 0.85, "strict_pass": True},
    ])
    assert next_refinement_ratio(rows, cfg) is None
