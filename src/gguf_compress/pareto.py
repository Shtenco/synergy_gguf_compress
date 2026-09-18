from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
from typing import Any

from .activation import load_activation_calibration
from .native_q4 import export_native_q4_gguf
from .optimizer import build_pareto_plans, scan_gguf_candidates
from .system import RuntimeInputs
from .v5_io import compress_gguf_v5


@dataclass(frozen=True)
class V5Config:
    ranks: tuple[int, ...] = (8, 16, 32, 64, 128)
    residual_densities: tuple[float, ...] = (0.0, 0.0025, 0.01)
    target_ratios: tuple[float, ...] = (0.90, 0.75, 0.60, 0.45)
    qbits: int = 4
    min_elements: int = 131072
    seed: str = "gguf-compress-v5"
    oversample: int = 8
    power_iters: int = 1
    require_calibration: bool = True
    max_activation_rel_rmse: float | None = 0.50


def _safe_budget_tag(ratio: float) -> str:
    return f"b{int(round(ratio * 1000)):04d}"


def _quality_value(point: dict[str, Any], runtime_enabled: bool) -> float | None:
    if runtime_enabled:
        value = point.get("perplexity_ratio")
        return float(value) if value is not None else None
    value = point.get("estimated_activation_rel_rmse")
    return float(value) if value is not None else None


def _mark_pareto(points: list[dict[str, Any]], runtime_enabled: bool) -> None:
    valid = [
        p for p in points
        if p.get("status") == "OK"
        and p.get("native_q4_bytes") is not None
        and _quality_value(p, runtime_enabled) is not None
    ]
    for p in points:
        p["pareto"] = False
    for p in valid:
        px = int(p["native_q4_bytes"])
        py = float(_quality_value(p, runtime_enabled))
        dominated = False
        for q in valid:
            if q is p:
                continue
            qx = int(q["native_q4_bytes"])
            qy = float(_quality_value(q, runtime_enabled))
            if qx <= px and qy <= py and (qx < px or qy < py):
                dominated = True
                break
        p["pareto"] = not dominated


def _write_csv(points: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "target_ratio",
        "status",
        "feasible",
        "estimated_payload_bytes",
        "native_q4_bytes",
        "file_compression_ratio",
        "estimated_activation_rel_rmse",
        "mean_activation_rel_rmse",
        "perplexity_ratio",
        "perplexity_delta",
        "kl_divergence",
        "same_top_p",
        "prompt_processing_speedup",
        "generation_speedup",
        "pareto",
        "native_q4_path",
        "runtime_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({key: point.get(key) for key in fields})


def _write_svg(points: list[dict[str, Any]], path: Path, runtime_enabled: bool) -> None:
    values = []
    for point in points:
        if point.get("status") != "OK":
            continue
        x = point.get("native_q4_bytes")
        y = _quality_value(point, runtime_enabled)
        if x is None or y is None:
            continue
        values.append((float(x) / (1024 * 1024), float(y), bool(point.get("pareto")), point))
    if not values:
        return

    width, height = 900, 560
    ml, mr, mt, mb = 90, 30, 40, 80
    xs = [v[0] for v in values]
    ys = [v[1] for v in values]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1e-6

    def sx(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * (width - ml - mr)

    def sy(y: float) -> float:
        return mt + (ymax - y) / (ymax - ymin) * (height - mt - mb)

    y_label = "Perplexity ratio (lower is better)" if runtime_enabled else "Activation weighted relative RMSE"
    frontier = sorted((v for v in values if v[2]), key=lambda v: v[0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="black"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="sans-serif" font-size="16">Native Q4 size, MiB (lower is better)</text>',
        f'<text x="24" y="{height/2}" transform="rotate(-90 24 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="16">{html.escape(y_label)}</text>',
        '<text x="450" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">GGUF Compress V5 Pareto frontier</text>',
    ]
    if len(frontier) >= 2:
        points_attr = " ".join(f"{sx(v[0]):.1f},{sy(v[1]):.1f}" for v in frontier)
        lines.append(f'<polyline points="{points_attr}" fill="none" stroke="black" stroke-width="2"/>')

    for x, y, pareto, point in values:
        radius = 7 if pareto else 5
        fill = "black" if pareto else "white"
        lines.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius}" fill="{fill}" stroke="black" stroke-width="2"/>'
        )
        label = f'{point["target_ratio"]:.2f}'
        lines.append(
            f'<text x="{sx(x)+9:.1f}" y="{sy(y)-9:.1f}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )

    for tick in range(6):
        tx = xmin + (xmax - xmin) * tick / 5
        xcoord = sx(tx)
        lines.append(f'<line x1="{xcoord:.1f}" y1="{height-mb}" x2="{xcoord:.1f}" y2="{height-mb+6}" stroke="black"/>')
        lines.append(f'<text x="{xcoord:.1f}" y="{height-mb+24}" text-anchor="middle" font-family="sans-serif" font-size="11">{tx:.1f}</text>')
        ty = ymin + (ymax - ymin) * tick / 5
        ycoord = sy(ty)
        lines.append(f'<line x1="{ml-6}" y1="{ycoord:.1f}" x2="{ml}" y2="{ycoord:.1f}" stroke="black"/>')
        lines.append(f'<text x="{ml-10}" y="{ycoord+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{ty:.4g}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_v5_pareto(
    input_model: str | Path,
    output_dir: str | Path,
    activation_stats: str | Path,
    *,
    config: V5Config | None = None,
    runtime: RuntimeInputs | None = None,
) -> dict[str, Any]:
    cfg = config or V5Config()
    source = Path(input_model).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    calibration = load_activation_calibration(activation_stats)

    scan = scan_gguf_candidates(
        source,
        calibration=calibration,
        ranks=cfg.ranks,
        residual_densities=cfg.residual_densities,
        qbits=cfg.qbits,
        min_elements=cfg.min_elements,
        seed=cfg.seed,
        oversample=cfg.oversample,
        power_iters=cfg.power_iters,
        require_calibration=cfg.require_calibration,
        max_activation_rel_rmse=cfg.max_activation_rel_rmse,
    )
    scan_path = out / "v5-candidate-scan.json"
    scan_path.write_text(json.dumps(scan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    plans = build_pareto_plans(scan, cfg.target_ratios)

    points: list[dict[str, Any]] = []
    for plan in plans:
        point: dict[str, Any] = {
            "target_ratio": plan.target_ratio,
            "target_payload_bytes": plan.target_bytes,
            "estimated_payload_bytes": plan.estimated_bytes,
            "estimated_activation_rel_rmse": plan.activation_rel_rmse,
            "feasible": plan.feasible,
            "status": "INFEASIBLE" if not plan.feasible else "BUILDING",
        }
        if not plan.feasible:
            points.append(point)
            continue

        tag = _safe_budget_tag(plan.target_ratio)
        v5_path = out / f"{source.stem}.{tag}.v5.gguf"
        native_path = out / f"{source.stem}.{tag}.native-q4.gguf"
        plan_path = out / f"{source.stem}.{tag}.plan.json"
        plan_path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        stats = compress_gguf_v5(
            source,
            v5_path,
            plan=plan,
            scan=scan,
            calibration=calibration,
            ranks=cfg.ranks,
            qbits=cfg.qbits,
            seed=cfg.seed,
            oversample=cfg.oversample,
            power_iters=cfg.power_iters,
        )
        native_stats = export_native_q4_gguf(v5_path, native_path)
        native_bytes = native_path.stat().st_size
        point.update({
            "status": "OK",
            "v5_path": str(v5_path),
            "native_q4_path": str(native_path),
            "native_q4_bytes": native_bytes,
            "file_compression_ratio": source.stat().st_size / max(native_bytes, 1),
            "mean_rel_rmse": stats.mean_rel_rmse,
            "mean_activation_rel_rmse": stats.mean_activation_rel_rmse,
            "mean_rank": stats.mean_rank,
            "factorized_tensors": stats.factorized_tensors,
            "passthrough_tensors": stats.passthrough_tensors,
            "native_q4_layers": native_stats.q4_layers,
            "reconstructed_layers": native_stats.reconstructed_layers,
            "runtime_status": "NOT_RUN" if runtime is None else "PENDING",
        })

        if runtime is not None:
            from .e2e_quality import run_end_to_end_benchmark
            try:
                e2e = run_end_to_end_benchmark(
                    llama_bench=runtime.llama_bench,
                    llama_perplexity=runtime.llama_perplexity,
                    dense_model=source,
                    compressed_model=native_path,
                    corpus=runtime.corpus,
                    kld_file=out / f"{source.stem}.{tag}.baseline.kld",
                    prompt_tokens=runtime.prompt_tokens,
                    gen_tokens=runtime.gen_tokens,
                    repetitions=runtime.repetitions,
                    threads=runtime.threads,
                    n_gpu_layers=runtime.n_gpu_layers,
                    context_size=runtime.context_size,
                    timeout=runtime.timeout,
                )
                (out / f"{source.stem}.{tag}.e2e.json").write_text(
                    json.dumps(e2e, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                perf = e2e.get("performance", {}).get("comparison", {})
                quality = e2e.get("quality", {}).get("comparison", {})
                point.update({
                    "runtime_status": "OK",
                    "perplexity_ratio": quality.get("perplexity_ratio"),
                    "perplexity_delta": quality.get("perplexity_delta"),
                    "kl_divergence": quality.get("kl_divergence"),
                    "same_top_p": quality.get("same_top_p"),
                    "rms_probability_change": quality.get("rms_probability_change"),
                    "prompt_processing_speedup": perf.get("prompt_processing_speedup"),
                    "generation_speedup": perf.get("generation_speedup"),
                    "rss_ratio_dense_over_compressed": perf.get("rss_ratio_dense_over_compressed"),
                })
            except Exception as exc:
                point["runtime_status"] = "ERROR"
                point["runtime_error"] = f"{type(exc).__name__}: {exc}"

        points.append(point)

    runtime_enabled = runtime is not None
    _mark_pareto(points, runtime_enabled)
    report = {
        "format": "gguf_compress.pareto/v5",
        "source_model": str(source),
        "activation_stats": str(Path(activation_stats).resolve()),
        "runtime_enabled": runtime_enabled,
        "config": {
            **asdict(cfg),
            "ranks": list(cfg.ranks),
            "residual_densities": list(cfg.residual_densities),
            "target_ratios": list(cfg.target_ratios),
        },
        "scan_summary": {
            "fixed_payload_bytes": scan.fixed_payload_bytes,
            "original_payload_bytes": scan.original_payload_bytes,
            "optimizable_raw_bytes": scan.optimizable_raw_bytes,
            "calibrated_tensors": scan.calibrated_tensors,
            "fallback_tensors": scan.fallback_tensors,
            "skipped_tensors": scan.skipped_tensors,
        },
        "points": points,
    }
    json_path = out / "pareto-v5.json"
    csv_path = out / "pareto-v5.csv"
    svg_path = out / "pareto-v5.svg"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(points, csv_path)
    _write_svg(points, svg_path, runtime_enabled)
    report["artifacts"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "svg": str(svg_path) if svg_path.exists() else None,
        "candidate_scan": str(scan_path),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
