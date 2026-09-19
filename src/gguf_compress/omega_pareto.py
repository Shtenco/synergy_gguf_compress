from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .activation import load_activation_calibration
from .native_q4 import export_native_q4_gguf
from .omega_mix import OmegaConfig, apply_omega_policy, plan_breakdown
from .optimizer import build_pareto_plans, scan_gguf_candidates
from .v5_io import compress_gguf_v5


def _tag(ratio: float) -> str:
    return f"b{int(round(ratio * 1000)):04d}"


def run_omega_pareto(
    input_model: str | Path,
    activation_stats: str | Path,
    output_dir: str | Path,
    *,
    target_ratios: tuple[float, ...] = (0.90, 0.75, 0.60),
    ranks: tuple[int, ...] = (8, 16, 32, 64, 128),
    residual_densities: tuple[float, ...] = (0.0, 0.0025, 0.01, 0.03),
    qbits: int = 4,
    min_elements: int = 131072,
    max_activation_rel_rmse: float = 0.60,
    omega_config: OmegaConfig | None = None,
) -> dict[str, Any]:
    source = Path(input_model).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    calibration = load_activation_calibration(activation_stats)
    cfg = omega_config or OmegaConfig()

    base_scan = scan_gguf_candidates(
        source,
        calibration=calibration,
        ranks=ranks,
        residual_densities=residual_densities,
        qbits=qbits,
        min_elements=min_elements,
        seed="nexus-omega-mix-v6",
        oversample=8,
        power_iters=1,
        require_calibration=True,
        max_activation_rel_rmse=max_activation_rel_rmse,
    )
    scan, classes = apply_omega_policy(base_scan, cfg)
    plans = build_pareto_plans(scan, target_ratios)

    points: list[dict[str, Any]] = []
    for plan in plans:
        point: dict[str, Any] = {
            **plan_breakdown(plan, classes),
            "status": "INFEASIBLE" if not plan.feasible else "BUILDING",
        }
        if not plan.feasible:
            points.append(point)
            continue

        tag = _tag(plan.target_ratio)
        container = out / f"{source.stem}.{tag}.omega-v6.gguf"
        native = out / f"{source.stem}.{tag}.omega-v6.native-q4.gguf"
        stats = compress_gguf_v5(
            source,
            container,
            plan=plan,
            scan=scan,
            calibration=calibration,
            ranks=ranks,
            qbits=qbits,
            seed="nexus-omega-mix-v6",
            oversample=8,
            power_iters=1,
        )
        native_stats = export_native_q4_gguf(container, native)
        point.update({
            "status": "OK",
            "container_path": str(container),
            "native_q4_path": str(native),
            "native_q4_bytes": native.stat().st_size,
            "source_bytes": source.stat().st_size,
            "file_compression_ratio": source.stat().st_size / max(native.stat().st_size, 1),
            "mean_rel_rmse": stats.mean_rel_rmse,
            "mean_activation_rel_rmse": stats.mean_activation_rel_rmse,
            "mean_rank": stats.mean_rank,
            "residual_nnz": stats.residual_nnz,
            "factorized_tensors": stats.factorized_tensors,
            "passthrough_tensors": stats.passthrough_tensors,
            "native_q4_layers": native_stats.q4_layers,
            "reconstructed_layers": native_stats.reconstructed_layers,
        })
        points.append(point)

    valid = [p for p in points if p.get("status") == "OK"]
    safe = [p for p in valid if float(p.get("activation_rel_rmse", 1.0)) <= 0.35]
    pool = safe or valid
    recommended = None
    if pool:
        # Prefer the smallest model that stays inside the activation error guard.
        recommended = min(pool, key=lambda p: (int(p["native_q4_bytes"]), float(p["activation_rel_rmse"])))

    report = {
        "format": "gguf_compress.omega_mix/v6",
        "source_model": str(source),
        "activation_stats": str(Path(activation_stats).resolve()),
        "omega_config": cfg.to_dict(),
        "ranks": list(ranks),
        "residual_densities": list(residual_densities),
        "target_ratios": list(target_ratios),
        "scan": {
            "fixed_payload_bytes": scan.fixed_payload_bytes,
            "original_payload_bytes": scan.original_payload_bytes,
            "optimizable_raw_bytes": scan.optimizable_raw_bytes,
            "calibrated_tensors": scan.calibrated_tensors,
            "fallback_tensors": scan.fallback_tensors,
            "skipped_tensors": scan.skipped_tensors,
        },
        "classes": classes,
        "points": points,
        "recommended": recommended,
    }
    path = out / "omega-v6-report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="NEXUS Omega-MIX V6 protected activation-aware GGUF optimizer")
    ap.add_argument("input_model")
    ap.add_argument("activation_stats")
    ap.add_argument("output_dir")
    ap.add_argument("--target-ratios", default="0.90,0.75,0.60")
    ap.add_argument("--ranks", default="8,16,32,64,128")
    ap.add_argument("--residual-densities", default="0,0.0025,0.01,0.03")
    ap.add_argument("--min-elements", type=int, default=131072)
    args = ap.parse_args()

    report = run_omega_pareto(
        args.input_model,
        args.activation_stats,
        args.output_dir,
        target_ratios=_floats(args.target_ratios),
        ranks=_ints(args.ranks),
        residual_densities=_floats(args.residual_densities),
        min_elements=args.min_elements,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
