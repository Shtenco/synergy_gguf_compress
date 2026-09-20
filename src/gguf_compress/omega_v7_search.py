from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .intelligence import _summary, llama_server, run_model
from .omega_pareto import run_omega_pareto


@dataclass(frozen=True)
class RetentionPolicy:
    """Hard behavioral gate for compression search.

    The baseline defines the set of questions the original model already knows.
    A candidate is not allowed to forget those answers unless the caller
    explicitly relaxes min_baseline_correct_retention/max_regressions.
    """

    min_baseline_correct_retention: float = 1.0
    min_accuracy_delta: float = 0.0
    max_regressions: int = 0
    require_no_unparsed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compare_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    candidate_by_id = {row["id"]: row for row in candidate_rows}
    if baseline_by_id.keys() != candidate_by_id.keys():
        missing = sorted(baseline_by_id.keys() - candidate_by_id.keys())
        extra = sorted(candidate_by_id.keys() - baseline_by_id.keys())
        raise ValueError(f"row id mismatch: missing={missing}, extra={extra}")

    ids = [row["id"] for row in baseline_rows]
    baseline_correct = 0
    retained = 0
    regressions = 0
    improvements = 0
    agreements = 0
    paired: list[dict[str, Any]] = []

    for qid in ids:
        b = baseline_by_id[qid]
        c = candidate_by_id[qid]
        if b.get("answer") == c.get("answer"):
            agreements += 1
        if bool(b.get("correct")):
            baseline_correct += 1
            if bool(c.get("correct")):
                retained += 1
            else:
                regressions += 1
        elif bool(c.get("correct")):
            improvements += 1
        paired.append({
            "id": qid,
            "gold": b.get("gold"),
            "baseline": b.get("answer"),
            "candidate": c.get("answer"),
            "baseline_correct": bool(b.get("correct")),
            "candidate_correct": bool(c.get("correct")),
        })

    baseline_summary = _summary(baseline_rows)
    candidate_summary = _summary(candidate_rows)
    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "comparison": {
            "accuracy_delta": float(candidate_summary["accuracy"]) - float(baseline_summary["accuracy"]),
            "answer_agreement": agreements / max(len(ids), 1),
            "baseline_correct_retention": retained / max(baseline_correct, 1),
            "baseline_correct": baseline_correct,
            "retained_baseline_correct": retained,
            "regressions": regressions,
            "improvements": improvements,
        },
        "paired": paired,
    }


def passes_policy(comparison: dict[str, Any], policy: RetentionPolicy) -> bool:
    cmp = comparison["comparison"]
    candidate = comparison["candidate"]
    if float(cmp["baseline_correct_retention"]) + 1e-12 < policy.min_baseline_correct_retention:
        return False
    if float(cmp["accuracy_delta"]) + 1e-12 < policy.min_accuracy_delta:
        return False
    if int(cmp["regressions"]) > policy.max_regressions:
        return False
    if policy.require_no_unparsed and int(candidate.get("unparsed", 0)) != 0:
        return False
    return True


def _candidate_gate(
    gate_rows: list[dict[str, Any]],
    baseline_correct_ids: set[str],
    policy: RetentionPolicy,
) -> dict[str, Any]:
    by_id = {row["id"]: row for row in gate_rows}
    missing = sorted(baseline_correct_ids - by_id.keys())
    retained = sum(
        1 for qid in baseline_correct_ids
        if qid in by_id and bool(by_id[qid].get("correct"))
    )
    regressions = len(baseline_correct_ids) - retained
    unparsed = sum(
        1 for qid in baseline_correct_ids
        if qid in by_id and by_id[qid].get("answer") is None
    )
    retention = retained / max(len(baseline_correct_ids), 1)
    passed = (
        not missing
        and retention + 1e-12 >= policy.min_baseline_correct_retention
        and regressions <= policy.max_regressions
        and (not policy.require_no_unparsed or unparsed == 0)
    )
    return {
        "total": len(baseline_correct_ids),
        "retained": retained,
        "retention": retention,
        "regressions": regressions,
        "unparsed": unparsed,
        "missing_ids": missing,
        "pass": passed,
    }


def select_smallest_passing(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in evaluations if row.get("strict_pass") is True]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            int(row["native_q4_bytes"]),
            float(row.get("activation_rel_rmse", 1e9)),
            -float(row["full"]["candidate"]["accuracy"]),
        ),
    )


def run_v7_retention_search(
    *,
    server_binary: str | Path,
    baseline_model: str | Path,
    activation_stats: str | Path,
    questions_path: str | Path,
    output_dir: str | Path,
    target_ratios: tuple[float, ...] = (0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60),
    ranks: tuple[int, ...] = (8, 16, 32, 64, 128),
    residual_densities: tuple[float, ...] = (0.0, 0.0025, 0.01, 0.03),
    threads: int = 4,
    policy: RetentionPolicy | None = None,
    max_activation_rel_rmse: float = 0.60,
) -> dict[str, Any]:
    """Build Omega-MIX candidates and find the smallest one that preserves baseline skill.

    The baseline is executed once. Each candidate first runs only the questions
    the baseline answered correctly. Candidates that forget any protected
    baseline answer are rejected before the full MCQ-40 pass.
    """

    cfg = policy or RetentionPolicy()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    questions = load_questions(questions_path)
    question_by_id = {q["id"]: q for q in questions}

    with llama_server(server_binary, baseline_model, threads=threads) as url:
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

    omega = run_omega_pareto(
        baseline_model,
        activation_stats,
        out / "candidates",
        target_ratios=target_ratios,
        ranks=ranks,
        residual_densities=residual_densities,
        max_activation_rel_rmse=max_activation_rel_rmse,
    )

    points = [
        dict(point)
        for point in omega.get("points", [])
        if point.get("status") == "OK" and point.get("native_q4_path")
    ]
    # Evaluate every built candidate. We intentionally do not assume quality is
    # monotonic in file size or target ratio.
    points.sort(key=lambda p: int(p["native_q4_bytes"]))

    evaluations: list[dict[str, Any]] = []
    for point in points:
        candidate_path = Path(point["native_q4_path"])
        record: dict[str, Any] = {
            "target_ratio": point.get("target_ratio"),
            "estimated_bytes": point.get("estimated_bytes"),
            "native_q4_path": str(candidate_path),
            "native_q4_bytes": int(point["native_q4_bytes"]),
            "file_compression_ratio": point.get("file_compression_ratio"),
            "activation_rel_rmse": point.get("activation_rel_rmse"),
            "mean_activation_rel_rmse": point.get("mean_activation_rel_rmse"),
            "mean_rel_rmse": point.get("mean_rel_rmse"),
            "mean_rank": point.get("mean_rank"),
            "residual_nnz": point.get("residual_nnz"),
        }

        with llama_server(server_binary, candidate_path, threads=threads) as url:
            gate_rows = run_model(url, gate_questions)
        gate = _candidate_gate(gate_rows, baseline_correct_ids, cfg)
        record["retention_gate"] = gate

        if not gate["pass"]:
            record["status"] = "RETENTION_FAIL"
            record["strict_pass"] = False
            evaluations.append(record)
            continue

        with llama_server(server_binary, candidate_path, threads=threads) as url:
            candidate_rows = run_model(url, questions)
        full = compare_rows(baseline_rows, candidate_rows)
        strict_pass = passes_policy(full, cfg)
        record["full"] = full
        record["strict_pass"] = strict_pass
        record["status"] = "PASS" if strict_pass else "FULL_FAIL"
        evaluations.append(record)

    selected = select_smallest_passing(evaluations)
    report = {
        "format": "gguf_compress.omega_mix/v7-retention-search",
        "goal": "smallest real native-Q4 Omega-MIX model that preserves baseline-correct behavior",
        "baseline_model": str(Path(baseline_model).resolve()),
        "activation_stats": str(Path(activation_stats).resolve()),
        "questions": str(Path(questions_path).resolve()),
        "policy": cfg.to_dict(),
        "grid": {
            "target_ratios": list(target_ratios),
            "ranks": list(ranks),
            "residual_densities": list(residual_densities),
        },
        "baseline": baseline_summary,
        "baseline_correct_ids": sorted(baseline_correct_ids),
        "omega_v6_build_report": omega,
        "evaluations": evaluations,
        "selected": selected,
        "verdict": "PASS" if selected is not None else "NO_STRICT_CANDIDATE",
    }
    report_path = out / "omega-v7-retention-search.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NEXUS Omega-MIX V7 baseline-retention search"
    )
    ap.add_argument("--server", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--activation-stats", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-ratios", default="0.98,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60")
    ap.add_argument("--ranks", default="8,16,32,64,128")
    ap.add_argument("--residual-densities", default="0,0.0025,0.01,0.03")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-retention", type=float, default=1.0)
    ap.add_argument("--min-accuracy-delta", type=float, default=0.0)
    ap.add_argument("--max-regressions", type=int, default=0)
    ap.add_argument("--allow-unparsed", action="store_true")
    ap.add_argument("--max-activation-rel-rmse", type=float, default=0.60)
    args = ap.parse_args()

    policy = RetentionPolicy(
        min_baseline_correct_retention=args.min_retention,
        min_accuracy_delta=args.min_accuracy_delta,
        max_regressions=args.max_regressions,
        require_no_unparsed=not args.allow_unparsed,
    )
    report = run_v7_retention_search(
        server_binary=args.server,
        baseline_model=args.baseline,
        activation_stats=args.activation_stats,
        questions_path=args.questions,
        output_dir=args.output_dir,
        target_ratios=_floats(args.target_ratios),
        ranks=_ints(args.ranks),
        residual_densities=_floats(args.residual_densities),
        threads=args.threads,
        policy=policy,
        max_activation_rel_rmse=args.max_activation_rel_rmse,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
