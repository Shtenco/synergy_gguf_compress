from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any

from .activation import load_activation_calibration
from .intelligence import _summary, llama_server, run_model
from .native_q4 import export_native_q4_gguf
from .omega_mix import OmegaConfig, apply_omega_policy, plan_breakdown
from .omega_v7_search import (
    RetentionPolicy,
    _candidate_gate,
    compare_rows,
    load_questions,
    passes_policy,
    select_smallest_passing,
)
from .optimizer import scan_gguf_candidates, select_global_plan
from .v5_io import compress_gguf_v5


@dataclass(frozen=True)
class AdaptiveSearchConfig:
    min_ratio: float = 0.50
    max_ratio: float = 0.98
    coarse_step: float = 0.08
    tolerance: float = 0.01
    max_refinements: int = 8

    def validate(self) -> None:
        if not (0.0 < self.min_ratio < self.max_ratio <= 1.0):
            raise ValueError("require 0 < min_ratio < max_ratio <= 1")
        if not (0.0 < self.coarse_step < 1.0):
            raise ValueError("coarse_step must be in (0, 1)")
        if not (0.0 < self.tolerance < 1.0):
            raise ValueError("tolerance must be in (0, 1)")
        if self.max_refinements < 0:
            raise ValueError("max_refinements must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coarse_ratios(cfg: AdaptiveSearchConfig) -> list[float]:
    """Generate descending budget ratios including both endpoints."""
    cfg.validate()
    values = [cfg.max_ratio]
    x = cfg.max_ratio
    while x - cfg.coarse_step > cfg.min_ratio + 1e-12:
        x = x - cfg.coarse_step
        values.append(round(x, 10))
    if abs(values[-1] - cfg.min_ratio) > 1e-12:
        values.append(cfg.min_ratio)
    return values


def _ratio_tag(ratio: float) -> str:
    return f"r{int(round(ratio * 10000)):05d}"


def _ratio_key(ratio: float) -> float:
    return round(float(ratio), 10)


def find_pass_fail_bracket(
    evaluations: list[dict[str, Any]],
) -> tuple[float, float] | None:
    """Return (fail_ratio, pass_ratio) for the tightest observed boundary.

    Smaller ratio means stronger compression. We therefore need a failed point
    below a passing point. Among all observed pairs, choose the smallest width.
    """
    passes = sorted(
        float(row["target_ratio"])
        for row in evaluations
        if row.get("strict_pass") is True
    )
    fails = sorted(
        float(row["target_ratio"])
        for row in evaluations
        if row.get("strict_pass") is False
    )
    best: tuple[float, float] | None = None
    best_width = math.inf
    for fail in fails:
        for passed in passes:
            if fail >= passed:
                continue
            width = passed - fail
            if width < best_width:
                best = (fail, passed)
                best_width = width
    return best


def next_refinement_ratio(
    evaluations: list[dict[str, Any]],
    cfg: AdaptiveSearchConfig,
) -> float | None:
    bracket = find_pass_fail_bracket(evaluations)
    if bracket is None:
        return None
    fail_ratio, pass_ratio = bracket
    if pass_ratio - fail_ratio <= cfg.tolerance:
        return None
    mid = _ratio_key((fail_ratio + pass_ratio) / 2.0)
    seen = {_ratio_key(float(row["target_ratio"])) for row in evaluations}
    return None if mid in seen else mid


def _build_candidate(
    *,
    source: Path,
    out: Path,
    ratio: float,
    scan: Any,
    classes: dict[str, str],
    calibration: Any,
    ranks: tuple[int, ...],
    qbits: int,
) -> dict[str, Any]:
    plan = select_global_plan(scan, ratio)
    point: dict[str, Any] = {
        **plan_breakdown(plan, classes),
        "status": "INFEASIBLE" if not plan.feasible else "BUILDING",
    }
    if not plan.feasible:
        return point

    tag = _ratio_tag(ratio)
    container = out / f"{source.stem}.{tag}.omega-v8.gguf"
    native = out / f"{source.stem}.{tag}.omega-v8.native-q4.gguf"
    stats = compress_gguf_v5(
        source,
        container,
        plan=plan,
        scan=scan,
        calibration=calibration,
        ranks=ranks,
        qbits=qbits,
        seed="nexus-omega-mix-v8",
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
    return point


def run_v8_behavioral_budget(
    *,
    server_binary: str | Path,
    baseline_model: str | Path,
    activation_stats: str | Path,
    questions_path: str | Path,
    output_dir: str | Path,
    ranks: tuple[int, ...] = (8, 16, 32, 64, 128),
    residual_densities: tuple[float, ...] = (0.0, 0.0025, 0.01, 0.03),
    qbits: int = 4,
    min_elements: int = 131072,
    max_activation_rel_rmse: float = 0.60,
    threads: int = 4,
    policy: RetentionPolicy | None = None,
    search: AdaptiveSearchConfig | None = None,
    omega_config: OmegaConfig | None = None,
) -> dict[str, Any]:
    policy_cfg = policy or RetentionPolicy()
    search_cfg = search or AdaptiveSearchConfig()
    search_cfg.validate()

    source = Path(baseline_model).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    calibration = load_activation_calibration(activation_stats)
    omega_cfg = omega_config or OmegaConfig()

    questions = load_questions(questions_path)
    question_by_id = {q["id"]: q for q in questions}

    # Baseline runs exactly once for the whole optimization.
    with llama_server(server_binary, source, threads=threads) as url:
        baseline_rows = run_model(url, questions)
    baseline_summary = _summary(baseline_rows)
    baseline_correct_ids = {
        str(row["id"]) for row in baseline_rows if bool(row.get("correct"))
    }
    gate_questions = [
        question_by_id[row["id"]]
        for row in baseline_rows
        if bool(row.get("correct"))
    ]

    # Candidate geometry/distortion scan is also executed exactly once.
    base_scan = scan_gguf_candidates(
        source,
        calibration=calibration,
        ranks=ranks,
        residual_densities=residual_densities,
        qbits=qbits,
        min_elements=min_elements,
        seed="nexus-omega-mix-v8",
        oversample=8,
        power_iters=1,
        require_calibration=True,
        max_activation_rel_rmse=max_activation_rel_rmse,
    )
    scan, classes = apply_omega_policy(base_scan, omega_cfg)

    evaluations: list[dict[str, Any]] = []
    cache: dict[float, dict[str, Any]] = {}

    def evaluate_ratio(ratio: float, stage: str) -> dict[str, Any]:
        ratio = _ratio_key(ratio)
        if ratio in cache:
            return cache[ratio]

        point = _build_candidate(
            source=source,
            out=out,
            ratio=ratio,
            scan=scan,
            classes=classes,
            calibration=calibration,
            ranks=ranks,
            qbits=qbits,
        )
        record: dict[str, Any] = {
            "stage": stage,
            "target_ratio": ratio,
            **point,
        }

        if point.get("status") != "OK":
            record["strict_pass"] = False
            record["behavior_status"] = "BUILD_INFEASIBLE"
            evaluations.append(record)
            cache[ratio] = record
            return record

        candidate_path = Path(point["native_q4_path"])
        with llama_server(server_binary, candidate_path, threads=threads) as url:
            gate_rows = run_model(url, gate_questions)
        gate = _candidate_gate(gate_rows, baseline_correct_ids, policy_cfg)
        record["retention_gate"] = gate

        if not gate["pass"]:
            record["strict_pass"] = False
            record["behavior_status"] = "RETENTION_FAIL"
            evaluations.append(record)
            cache[ratio] = record
            return record

        with llama_server(server_binary, candidate_path, threads=threads) as url:
            candidate_rows = run_model(url, questions)
        full = compare_rows(baseline_rows, candidate_rows)
        strict_pass = passes_policy(full, policy_cfg)
        record["full"] = full
        record["strict_pass"] = strict_pass
        record["behavior_status"] = "PASS" if strict_pass else "FULL_FAIL"
        evaluations.append(record)
        cache[ratio] = record
        return record

    # Stage 1: coarse descending sweep. Stop after the first observed PASS->FAIL
    # boundary, because refinement will spend the remaining model executions
    # where they are most informative.
    last_pass: float | None = None
    bracket_found = False
    for ratio in coarse_ratios(search_cfg):
        row = evaluate_ratio(ratio, "coarse")
        if row.get("strict_pass") is True:
            last_pass = ratio
            continue
        if last_pass is not None and ratio < last_pass:
            bracket_found = True
            break

    # If no failure was observed after a pass, make sure the minimum ratio was
    # evaluated; the coarse generator normally already does this.
    if not bracket_found and _ratio_key(search_cfg.min_ratio) not in cache:
        evaluate_ratio(search_cfg.min_ratio, "coarse-edge")

    # Stage 2: adaptive bisection of the behavioral boundary.
    refinements = 0
    while refinements < search_cfg.max_refinements:
        ratio = next_refinement_ratio(evaluations, search_cfg)
        if ratio is None:
            break
        evaluate_ratio(ratio, "refine")
        refinements += 1

    # Stage 3: select by real file bytes, not target ratio.
    selected = select_smallest_passing(evaluations)
    bracket = find_pass_fail_bracket(evaluations)

    report = {
        "format": "gguf_compress.omega_mix/v8-behavioral-budget",
        "goal": "maximum compression subject to measured zero behavioral regression",
        "baseline_model": str(source),
        "activation_stats": str(Path(activation_stats).resolve()),
        "questions": str(Path(questions_path).resolve()),
        "policy": policy_cfg.to_dict(),
        "search": search_cfg.to_dict(),
        "omega_config": omega_cfg.to_dict(),
        "grid": {
            "ranks": list(ranks),
            "residual_densities": list(residual_densities),
            "qbits": qbits,
        },
        "baseline": baseline_summary,
        "baseline_correct_ids": sorted(baseline_correct_ids),
        "scan": {
            "fixed_payload_bytes": scan.fixed_payload_bytes,
            "original_payload_bytes": scan.original_payload_bytes,
            "optimizable_raw_bytes": scan.optimizable_raw_bytes,
            "calibrated_tensors": scan.calibrated_tensors,
            "fallback_tensors": scan.fallback_tensors,
            "skipped_tensors": scan.skipped_tensors,
        },
        "classes": classes,
        "evaluations": sorted(evaluations, key=lambda x: float(x["target_ratio"]), reverse=True),
        "observed_pass_fail_bracket": (
            {"fail_ratio": bracket[0], "pass_ratio": bracket[1], "width": bracket[1] - bracket[0]}
            if bracket is not None else None
        ),
        "refinements": refinements,
        "selected": selected,
        "verdict": "PASS" if selected is not None else "NO_STRICT_CANDIDATE",
    }
    (out / "omega-v8-behavioral-budget.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="NEXUS Omega-MIX V8 behavioral budget optimizer")
    ap.add_argument("--server", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--activation-stats", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ranks", default="8,16,32,64,128")
    ap.add_argument("--residual-densities", default="0,0.0025,0.01,0.03")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-ratio", type=float, default=0.50)
    ap.add_argument("--max-ratio", type=float, default=0.98)
    ap.add_argument("--coarse-step", type=float, default=0.08)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--max-refinements", type=int, default=8)
    ap.add_argument("--min-retention", type=float, default=1.0)
    ap.add_argument("--min-accuracy-delta", type=float, default=0.0)
    ap.add_argument("--max-regressions", type=int, default=0)
    ap.add_argument("--allow-unparsed", action="store_true")
    args = ap.parse_args()

    report = run_v8_behavioral_budget(
        server_binary=args.server,
        baseline_model=args.baseline,
        activation_stats=args.activation_stats,
        questions_path=args.questions,
        output_dir=args.output_dir,
        ranks=_ints(args.ranks),
        residual_densities=_floats(args.residual_densities),
        threads=args.threads,
        policy=RetentionPolicy(
            min_baseline_correct_retention=args.min_retention,
            min_accuracy_delta=args.min_accuracy_delta,
            max_regressions=args.max_regressions,
            require_no_unparsed=not args.allow_unparsed,
        ),
        search=AdaptiveSearchConfig(
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            coarse_step=args.coarse_step,
            tolerance=args.tolerance,
            max_refinements=args.max_refinements,
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
