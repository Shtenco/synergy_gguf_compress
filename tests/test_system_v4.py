from __future__ import annotations

from gguf_compress.system import (
    RuntimeQualityGates,
    StaticQualityGates,
    evaluate_runtime_gates,
    evaluate_static_gates,
)


def test_static_gates_pass_and_fail() -> None:
    gates = StaticQualityGates(
        min_file_compression_ratio=2.0,
        min_factorized_tensors=2,
        max_mean_corrected_rel_rmse=0.20,
        max_tensor_corrected_rel_rmse=0.30,
    )
    good = evaluate_static_gates(
        file_compression_ratio=3.5,
        quality_summary={
            "factorized_tensors": 12,
            "mean_corrected_rel_rmse": 0.11,
            "max_corrected_rel_rmse": 0.25,
        },
        gates=gates,
    )
    assert all(r.passed is True for r in good)

    bad = evaluate_static_gates(
        file_compression_ratio=1.5,
        quality_summary={
            "factorized_tensors": 1,
            "mean_corrected_rel_rmse": 0.21,
            "max_corrected_rel_rmse": 0.31,
        },
        gates=gates,
    )
    assert all(r.passed is False for r in bad)


def test_runtime_gates_do_not_fake_missing_metrics() -> None:
    gates = RuntimeQualityGates(
        max_perplexity_ratio=1.10,
        max_kl_divergence=0.05,
        min_same_top_p=0.90,
        min_prompt_processing_speedup=1.0,
        min_generation_speedup=1.0,
    )
    report = {
        "performance": {
            "comparison": {
                "prompt_processing_speedup": 1.3,
                "generation_speedup": 1.1,
            }
        },
        "quality": {
            "comparison": {
                "perplexity_ratio": 1.05,
                "kl_divergence": None,
                "same_top_p": 0.95,
            }
        },
    }
    results = {r.name: r for r in evaluate_runtime_gates(report, gates)}
    assert results["perplexity_ratio"].passed is True
    assert results["prompt_processing_speedup"].passed is True
    assert results["generation_speedup"].passed is True
    assert results["same_top_p"].passed is True
    assert results["kl_divergence"].passed is False
    assert results["kl_divergence"].reason == "metric was not parsed"


def test_report_only_runtime_gate_is_not_failure() -> None:
    gates = RuntimeQualityGates(max_perplexity_ratio=None)
    report = {
        "performance": {"comparison": {}},
        "quality": {"comparison": {"perplexity_ratio": None}},
    }
    result = evaluate_runtime_gates(report, gates)[0]
    assert result.passed is None
    assert result.reason == "report-only threshold"
