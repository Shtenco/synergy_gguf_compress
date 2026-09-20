from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any
from urllib import request


LETTER_RE = re.compile(r"(?<![A-ZА-Я])([ABCD])(?![A-ZА-Я])", re.IGNORECASE)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@contextmanager
def llama_server(binary: str | Path, model: str | Path, *, threads: int = 1, context: int = 1024):
    port = _free_port()
    log_path = Path(str(model) + ".server.log")
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(binary),
            "-m", str(model),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-t", str(threads),
            "-c", str(context),
            "-np", "1",
            "--no-cache-prompt",
            "--no-cache-idle-slots",
            "--slot-prompt-similarity", "0",
            "--no-webui",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 120
        last_error: Exception | None = None
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {proc.returncode}; see {log_path}")
            try:
                health = _get_json(f"http://127.0.0.1:{port}/health")
                if health.get("status") in {"ok", "no slot available"}:
                    break
            except Exception as exc:
                last_error = exc
            time.sleep(0.5)
        else:
            raise TimeoutError(f"llama-server did not become healthy: {last_error}; see {log_path}")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log.close()


def _question_prompt(item: dict[str, Any]) -> str:
    choices = item["choices"]
    return (
        f'{item["question"]}\n\n'
        f'A. {choices["A"]}\n'
        f'B. {choices["B"]}\n'
        f'C. {choices["C"]}\n'
        f'D. {choices["D"]}\n\n'
        "Выбери единственный правильный вариант. Ответь только одной латинской буквой A, B, C или D."
    )


def _extract_letter(text: str) -> str | None:
    m = LETTER_RE.search(text.strip().upper())
    return m.group(1).upper() if m else None


def run_model(
    base_url: str,
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in questions:
        started = time.perf_counter()
        payload = {
            "model": "local",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты проходишь объективный тест. Не объясняй решение. "
                        "После внутреннего решения верни только A, B, C или D."
                    ),
                },
                {"role": "user", "content": _question_prompt(item)},
            ],
            "temperature": 0,
            "max_tokens": 4,
            "seed": 12345,
            "cache_prompt": False,
        }
        response = _post_json(f"{base_url}/v1/chat/completions", payload)
        content = response["choices"][0]["message"]["content"]
        answer = _extract_letter(content)
        rows.append({
            "id": item["id"],
            "category": item.get("category", "other"),
            "gold": item["answer"],
            "answer": answer,
            "correct": answer == item["answer"],
            "raw": content,
            "seconds": time.perf_counter() - started,
        })
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    by_category: dict[str, dict[str, int | float]] = {}
    for row in rows:
        cat = row["category"]
        bucket = by_category.setdefault(cat, {"total": 0, "correct": 0})
        bucket["total"] = int(bucket["total"]) + 1
        bucket["correct"] = int(bucket["correct"]) + int(bool(row["correct"]))
    for bucket in by_category.values():
        bucket["accuracy"] = int(bucket["correct"]) / max(int(bucket["total"]), 1)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "unparsed": sum(row["answer"] is None for row in rows),
        "mean_seconds": sum(float(row["seconds"]) for row in rows) / max(total, 1),
        "by_category": by_category,
    }


def _balanced_antibias_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    letters = "ABCD"
    out: list[dict[str, Any]] = []
    for index, item in enumerate(questions):
        target = letters[index % 4]
        original_gold = item["answer"]
        original_index = letters.index(original_gold)
        target_index = letters.index(target)
        shift = (original_index - target_index) % 4
        rotated = letters[shift:] + letters[:shift]
        clone = dict(item)
        clone["choices"] = {
            letters[j]: item["choices"][rotated[j]]
            for j in range(4)
        }
        clone["answer"] = letters[rotated.index(original_gold)]
        clone["anti_bias_shift"] = shift
        out.append(clone)
    return out


def _compare_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    paired = []
    regressions = improvements = agreements = 0
    baseline_correct = retained = 0
    for b, c in zip(baseline_rows, candidate_rows, strict=True):
        agreements += int(b["answer"] == c["answer"])
        if b["correct"]:
            baseline_correct += 1
            retained += int(c["correct"])
        regressions += int(b["correct"] and not c["correct"])
        improvements += int((not b["correct"]) and c["correct"])
        paired.append({
            "id": b["id"],
            "gold": b["gold"],
            "baseline": b["answer"],
            "candidate": c["answer"],
            "baseline_correct": b["correct"],
            "candidate_correct": c["correct"],
        })

    baseline = _summary(baseline_rows)
    candidate = _summary(candidate_rows)
    accuracy_delta = candidate["accuracy"] - baseline["accuracy"]
    retention = retained / max(baseline_correct, 1)
    strict_pass = (
        retention == 1.0
        and regressions == 0
        and accuracy_delta >= 0.0
        and baseline["unparsed"] == 0
        and candidate["unparsed"] == 0
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "accuracy_delta": accuracy_delta,
            "answer_agreement": agreements / max(len(paired), 1),
            "baseline_correct_retention": retention,
            "regressions": regressions,
            "improvements": improvements,
        },
        "strict_gate": {
            "baseline_correct_retention_required": 1.0,
            "regressions_required": 0,
            "accuracy_delta_min": 0.0,
            "unparsed_required": 0,
            "pass": strict_pass,
        },
        "paired": paired,
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
    }


def run_intelligence_benchmark(
    *,
    server_binary: str | Path,
    baseline_model: str | Path,
    candidate_model: str | Path,
    questions_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    threads: int = 1,
    anti_bias: bool = True,
) -> dict[str, Any]:
    questions = [
        json.loads(line)
        for line in Path(questions_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        questions = questions[:limit]

    with llama_server(server_binary, baseline_model, threads=threads) as url:
        baseline_rows = run_model(url, questions)
    with llama_server(server_binary, candidate_model, threads=threads) as url:
        candidate_rows = run_model(url, questions)
    primary = _compare_rows(baseline_rows, candidate_rows)

    anti_bias_result: dict[str, Any] | None = None
    if anti_bias:
        anti_questions = _balanced_antibias_questions(questions)
        with llama_server(server_binary, baseline_model, threads=threads) as url:
            anti_baseline_rows = run_model(url, anti_questions)
        with llama_server(server_binary, candidate_model, threads=threads) as url:
            anti_candidate_rows = run_model(url, anti_questions)
        anti_bias_result = _compare_rows(anti_baseline_rows, anti_candidate_rows)

    strict_pass = bool(primary["strict_gate"]["pass"]) and (
        anti_bias_result is None or bool(anti_bias_result["strict_gate"]["pass"])
    )
    agreements = [float(primary["comparison"]["answer_agreement"])]
    if anti_bias_result is not None:
        agreements.append(float(anti_bias_result["comparison"]["answer_agreement"]))

    report = {
        "format": "gguf_compress.intelligence/v3",
        "benchmark": "NEXUS Core MCQ-40",
        "note": (
            "Paired retention gate with deterministic no-cache inference and "
            "balanced anti-bias permutation. Not a replacement for MMLU/ARC/GSM8K."
        ),
        "baseline_model": str(Path(baseline_model).resolve()),
        "candidate_model": str(Path(candidate_model).resolve()),
        "runtime_protocol": {
            "threads": threads,
            "parallel_slots": 1,
            "prompt_cache": False,
            "slot_prompt_similarity": 0,
            "temperature": 0,
            "seed": 12345,
        },
        "primary": primary,
        "balanced_anti_bias": anti_bias_result,
        "strict_gate": {
            "pass": strict_pass,
            "requires_primary_pass": True,
            "requires_balanced_anti_bias_pass": anti_bias,
        },
        "behavior_identity": {
            "minimum_answer_agreement": min(agreements),
            "pass_95pct": min(agreements) >= 0.95,
            "note": "Diagnostic only; not part of the legacy strict retention gate.",
        },
    }
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-anti-bias", action="store_true")
    args = ap.parse_args()
    report = run_intelligence_benchmark(
        server_binary=args.server,
        baseline_model=args.baseline,
        candidate_model=args.candidate,
        questions_path=args.questions,
        output_path=args.output,
        limit=args.limit,
        threads=args.threads,
        anti_bias=not args.no_anti_bias,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()