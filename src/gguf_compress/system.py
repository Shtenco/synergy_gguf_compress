from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


@dataclass(frozen=True)
class StaticQualityGates:
    """Static gates that can be evaluated immediately after compression/export."""

    min_file_compression_ratio: float = 1.0
    min_factorized_tensors: int = 1
    max_mean_corrected_rel_rmse: float = 0.30
    max_tensor_corrected_rel_rmse: float = 0.40


@dataclass(frozen=True)
class RuntimeQualityGates:
    """Optional end-to-end gates. A None threshold means report-only."""

    max_perplexity_ratio: float | None = 1.10
    max_kl_divergence: float | None = None
    min_same_top_p: float | None = None
    min_prompt_processing_speedup: float | None = None
    min_generation_speedup: float | None = None


@dataclass(frozen=True)
class DistilledConfig:
    ranks: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
    target_rel_rmse: float = 0.20
    qbits: int = 4
    residual_density: float = 0.01
    max_corrected_rel_rmse: float = 0.40
    min_elements: int = 131072
    seed: str = "gguf-compress-v4"
    oversample: int = 8
    power_iters: int = 1
    static_gates: StaticQualityGates = field(default_factory=StaticQualityGates)
    runtime_gates: RuntimeQualityGates = field(default_factory=RuntimeQualityGates)


@dataclass(frozen=True)
class RuntimeInputs:
    llama_bench: Path
    llama_perplexity: Path
    corpus: Path
    prompt_tokens: int = 128
    gen_tokens: int = 64
    repetitions: int = 3
    threads: int | None = None
    n_gpu_layers: int = 0
    context_size: int = 512
    timeout: int | None = 1800


@dataclass
class GateResult:
    name: str
    passed: bool | None
    observed: float | int | None
    threshold: float | int | None
    rule: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _tensor_quality_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    factorized = [t for t in manifest.get("tensors", []) if t.get("kind") == "adaptive_factorized"]
    corrected = [float(t["corrected_rel_rmse"]) for t in factorized if t.get("corrected_rel_rmse") is not None]
    ranks = [int(t["rank"]) for t in factorized if t.get("rank") is not None]
    return {
        "factorized_tensors": len(factorized),
        "raw_tensors": sum(1 for t in manifest.get("tensors", []) if t.get("kind") == "raw"),
        "max_corrected_rel_rmse": max(corrected) if corrected else 0.0,
        "mean_corrected_rel_rmse": (sum(corrected) / len(corrected)) if corrected else 0.0,
        "mean_rank": (sum(ranks) / len(ranks)) if ranks else 0.0,
        "max_rank": max(ranks) if ranks else 0,
    }


def evaluate_static_gates(
    *,
    file_compression_ratio: float,
    quality_summary: dict[str, Any],
    gates: StaticQualityGates,
) -> list[GateResult]:
    observed_factorized = int(quality_summary.get("factorized_tensors", 0))
    observed_mean = float(quality_summary.get("mean_corrected_rel_rmse", 0.0))
    observed_max = float(quality_summary.get("max_corrected_rel_rmse", 0.0))
    return [
        GateResult(
            "file_compression_ratio",
            file_compression_ratio >= gates.min_file_compression_ratio,
            file_compression_ratio,
            gates.min_file_compression_ratio,
            ">=",
        ),
        GateResult(
            "factorized_tensors",
            observed_factorized >= gates.min_factorized_tensors,
            observed_factorized,
            gates.min_factorized_tensors,
            ">=",
        ),
        GateResult(
            "mean_corrected_rel_rmse",
            observed_mean <= gates.max_mean_corrected_rel_rmse,
            observed_mean,
            gates.max_mean_corrected_rel_rmse,
            "<=",
        ),
        GateResult(
            "max_tensor_corrected_rel_rmse",
            observed_max <= gates.max_tensor_corrected_rel_rmse,
            observed_max,
            gates.max_tensor_corrected_rel_rmse,
            "<=",
        ),
    ]


def _optional_gate(name: str, observed: float | None, threshold: float | None, rule: str) -> GateResult:
    if threshold is None:
        return GateResult(name, None, observed, None, rule, "report-only threshold")
    if observed is None:
        return GateResult(name, False, None, threshold, rule, "metric was not parsed")
    if rule == "<=":
        passed = observed <= threshold
    elif rule == ">=":
        passed = observed >= threshold
    else:
        raise ValueError(f"unsupported rule: {rule}")
    return GateResult(name, passed, observed, threshold, rule)


def evaluate_runtime_gates(report: dict[str, Any], gates: RuntimeQualityGates) -> list[GateResult]:
    perf = report.get("performance", {}).get("comparison", {})
    quality = report.get("quality", {}).get("comparison", {})
    return [
        _optional_gate("perplexity_ratio", quality.get("perplexity_ratio"), gates.max_perplexity_ratio, "<="),
        _optional_gate("kl_divergence", quality.get("kl_divergence"), gates.max_kl_divergence, "<="),
        _optional_gate("same_top_p", quality.get("same_top_p"), gates.min_same_top_p, ">="),
        _optional_gate(
            "prompt_processing_speedup",
            perf.get("prompt_processing_speedup"),
            gates.min_prompt_processing_speedup,
            ">=",
        ),
        _optional_gate(
            "generation_speedup",
            perf.get("generation_speedup"),
            gates.min_generation_speedup,
            ">=",
        ),
    ]


def _gate_status(results: list[GateResult]) -> str:
    evaluated = [r for r in results if r.passed is not None]
    if not evaluated:
        return "NOT_EVALUATED"
    return "PASS" if all(r.passed for r in evaluated) else "FAIL"


def _validate_runtime_inputs(runtime: RuntimeInputs | None) -> None:
    if runtime is None:
        return
    for path in (runtime.llama_bench, runtime.llama_perplexity, runtime.corpus):
        if not Path(path).exists():
            raise FileNotFoundError(path)


def run_distilled_system(
    input_model: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    config: DistilledConfig | None = None,
    runtime: RuntimeInputs | None = None,
) -> dict[str, Any]:
    """Run the canonical distilled GGUF compression pipeline.

    The pipeline intentionally keeps storage compression, operator execution and
    quality evidence as separate stages. It never reports a runtime PASS when
    the llama.cpp benchmark was not actually executed.
    """

    cfg = config or DistilledConfig()
    source = Path(input_model).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".gguf":
        raise ValueError(f"expected a .gguf model, got: {source}")
    if cfg.qbits != 4:
        raise ValueError("the canonical native runtime currently requires qbits=4")
    if not cfg.ranks or any(r <= 0 for r in cfg.ranks):
        raise ValueError("ranks must contain positive integers")
    if not (0.0 <= cfg.residual_density <= 1.0):
        raise ValueError("residual_density must be in [0, 1]")
    _validate_runtime_inputs(runtime)

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = source.name[:-5] if source.name.lower().endswith(".gguf") else source.stem
    compressed = out / f"{stem}.v3.gguf"
    native = out / f"{stem}.native-q4.gguf"
    report_path = out / f"{stem}.distilled-report.json"
    kld_path = out / f"{stem}.baseline.kld"

    from .v3_io import compress_gguf_v3, inspect_v3
    from .native_q4 import export_native_q4_gguf

    compression = compress_gguf_v3(
        source,
        compressed,
        candidates=tuple(sorted(set(cfg.ranks))),
        target_rel_rmse=cfg.target_rel_rmse,
        qbits=cfg.qbits,
        residual_density=cfg.residual_density,
        min_elements=cfg.min_elements,
        max_corrected_rel_rmse=cfg.max_corrected_rel_rmse,
        seed=cfg.seed,
        oversample=cfg.oversample,
        power_iters=cfg.power_iters,
    )
    manifest = inspect_v3(compressed)
    quality_summary = _tensor_quality_summary(manifest)
    native_stats = export_native_q4_gguf(compressed, native)

    source_bytes = source.stat().st_size
    native_bytes = native.stat().st_size
    final_ratio = source_bytes / max(native_bytes, 1)
    static_results = evaluate_static_gates(
        file_compression_ratio=final_ratio,
        quality_summary=quality_summary,
        gates=cfg.static_gates,
    )

    e2e_report: dict[str, Any] | None = None
    runtime_results: list[GateResult] = []
    if runtime is not None:
        from .e2e_quality import run_end_to_end_benchmark

        e2e_report = run_end_to_end_benchmark(
            llama_bench=runtime.llama_bench,
            llama_perplexity=runtime.llama_perplexity,
            dense_model=source,
            compressed_model=native,
            corpus=runtime.corpus,
            kld_file=kld_path,
            prompt_tokens=runtime.prompt_tokens,
            gen_tokens=runtime.gen_tokens,
            repetitions=runtime.repetitions,
            threads=runtime.threads,
            n_gpu_layers=runtime.n_gpu_layers,
            context_size=runtime.context_size,
            timeout=runtime.timeout,
        )
        runtime_results = evaluate_runtime_gates(e2e_report, cfg.runtime_gates)

    static_status = _gate_status(static_results)
    runtime_status = _gate_status(runtime_results) if runtime is not None else "NOT_RUN"
    overall = "PASS" if static_status == "PASS" and runtime_status in ("PASS", "NOT_RUN") else "FAIL"

    report: dict[str, Any] = {
        "format": "gguf_compress.distilled_system/v4",
        "overall_status": overall,
        "static_status": static_status,
        "runtime_status": runtime_status,
        "warning": (
            "Runtime quality/performance was not executed; static PASS is not an end-to-end model-quality claim."
            if runtime is None
            else None
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "config": {
            **asdict(cfg),
            "ranks": list(cfg.ranks),
        },
        "artifacts": {
            "source": str(source),
            "compressed_v3": str(compressed),
            "native_q4": str(native),
            "report": str(report_path),
            "source_bytes": source_bytes,
            "compressed_v3_bytes": compressed.stat().st_size,
            "native_q4_bytes": native_bytes,
            "source_sha256": _sha256_file(source),
            "compressed_v3_sha256": _sha256_file(compressed),
            "native_q4_sha256": _sha256_file(native),
            "final_file_compression_ratio": final_ratio,
        },
        "compression": {
            "input_bytes": compression.input_bytes,
            "output_bytes": compression.output_bytes,
            "file_compression_ratio": compression.file_ratio,
            "factorized_tensors": compression.factorized_tensors,
            "passthrough_tensors": compression.passthrough_tensors,
            "mean_rel_rmse_before_residual": compression.mean_rel_rmse,
            "mean_rel_rmse_after_residual": compression.mean_corrected_rel_rmse,
            "mean_rank": compression.mean_rank,
            "residual_nnz": compression.residual_nnz,
            "elapsed_seconds": compression.elapsed_seconds,
        },
        "quality_summary": quality_summary,
        "native_export": {
            "input_bytes": native_stats.input_bytes,
            "output_bytes": native_stats.output_bytes,
            "file_compression_ratio": native_stats.file_ratio,
            "native_q4_layers": native_stats.q4_layers,
            "reconstructed_layers": native_stats.reconstructed_layers,
            "passthrough_tensors": native_stats.passthrough_tensors,
        },
        "static_gates": [r.to_dict() for r in static_results],
        "runtime_gates": [r.to_dict() for r in runtime_results],
        "e2e": e2e_report,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
