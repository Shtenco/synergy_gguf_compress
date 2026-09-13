from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from .real_bench import run_real_benchmark


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _last_float(text: str, patterns: tuple[str, ...]) -> float | None:
    value: float | None = None
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                pass
    return value


def parse_perplexity_output(text: str) -> dict[str, float | None]:
    """Best-effort parser for llama-perplexity output across llama.cpp versions.

    Raw stdout/stderr is always kept by the caller, so a future upstream wording
    change cannot hide information even if one of these optional fields is null.
    """
    return {
        "perplexity": _last_float(text, (
            rf"\bPPL\s*(?:=|:)\s*({_FLOAT})",
            rf"\bperplexity\s*(?:=|:)\s*({_FLOAT})",
        )),
        "kl_divergence": _last_float(text, (
            rf"\bKL(?:[- ]?divergence|D)?\s*(?:=|:)\s*({_FLOAT})",
            rf"\bKullback(?:[- ]Leibler)?(?:\s+divergence)?\s*(?:=|:)\s*({_FLOAT})",
        )),
        "pearson": _last_float(text, (
            rf"\bPearson(?:\s+(?:correlation|r))?\s*(?:=|:)\s*({_FLOAT})",
        )),
        "same_top_p": _last_float(text, (
            rf"\bsame[- ]top[- ]p\s*(?:=|:)\s*({_FLOAT})",
            rf"\bsame top p\s*(?:=|:)\s*({_FLOAT})",
        )),
        "rms_probability_change": _last_float(text, (
            rf"\bRMS(?:\s+probability)?(?:\s+change)?\s*(?:=|:)\s*({_FLOAT})",
        )),
    }


def _run(cmd: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    combined = (p.stdout or "") + ("\n" if p.stdout and p.stderr else "") + (p.stderr or "")
    if p.returncode != 0:
        tail = combined[-8000:]
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{tail}")
    return {
        "command": cmd,
        "elapsed_seconds": elapsed,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "combined": combined,
    }


def run_quality_benchmark(
    *,
    llama_perplexity: str | os.PathLike[str],
    dense_model: str | os.PathLike[str],
    compressed_model: str | os.PathLike[str],
    corpus: str | os.PathLike[str],
    kld_file: str | os.PathLike[str],
    threads: int | None = None,
    n_gpu_layers: int = 0,
    context_size: int | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Measure PPL and logit-distribution drift using llama.cpp's KLD mode."""
    exe = Path(llama_perplexity)
    dense = Path(dense_model)
    compressed = Path(compressed_model)
    corpus_path = Path(corpus)
    kld_path = Path(kld_file)
    for path in (exe, dense, compressed, corpus_path):
        if not path.exists():
            raise FileNotFoundError(path)

    common = ["-f", str(corpus_path), "-ngl", str(n_gpu_layers)]
    if threads is not None:
        common += ["-t", str(threads)]
    if context_size is not None:
        common += ["-c", str(context_size)]

    if kld_path.exists():
        kld_path.unlink()

    base = _run([
        str(exe), "-m", str(dense), *common,
        "--kl-divergence-base", str(kld_path),
    ], timeout=timeout)
    if not kld_path.exists() or kld_path.stat().st_size == 0:
        raise RuntimeError("llama-perplexity did not create the KLD baseline file")

    candidate = _run([
        str(exe), "-m", str(compressed), *common,
        "--kl-divergence-base", str(kld_path),
        "--kl-divergence",
    ], timeout=timeout)

    base_metrics = parse_perplexity_output(base["combined"])
    candidate_metrics = parse_perplexity_output(candidate["combined"])
    base_ppl = base_metrics["perplexity"]
    candidate_ppl = candidate_metrics["perplexity"]

    return {
        "settings": {
            "corpus": str(corpus_path),
            "threads": threads,
            "n_gpu_layers": n_gpu_layers,
            "context_size": context_size,
        },
        "dense": {
            "model": str(dense),
            "file_bytes": dense.stat().st_size,
            "metrics": base_metrics,
            "elapsed_seconds": base["elapsed_seconds"],
            "stdout": base["stdout"],
            "stderr": base["stderr"],
        },
        "compressed": {
            "model": str(compressed),
            "file_bytes": compressed.stat().st_size,
            "metrics": candidate_metrics,
            "elapsed_seconds": candidate["elapsed_seconds"],
            "stdout": candidate["stdout"],
            "stderr": candidate["stderr"],
        },
        "comparison": {
            "perplexity_ratio": (
                candidate_ppl / base_ppl
                if candidate_ppl is not None and base_ppl not in (None, 0.0)
                else None
            ),
            "perplexity_delta": (
                candidate_ppl - base_ppl
                if candidate_ppl is not None and base_ppl is not None
                else None
            ),
            "kl_divergence": candidate_metrics["kl_divergence"],
            "pearson": candidate_metrics["pearson"],
            "same_top_p": candidate_metrics["same_top_p"],
            "rms_probability_change": candidate_metrics["rms_probability_change"],
            "kld_reference_bytes": kld_path.stat().st_size,
        },
        "notes": [
            "Lower perplexity is better; compare the ratio/delta against the baseline model.",
            "KL divergence is computed from llama.cpp's baseline-logit file; zero means identical distributions.",
            "Raw llama-perplexity stdout/stderr is preserved because upstream metric wording can change.",
        ],
    }


def run_end_to_end_benchmark(
    *,
    llama_bench: str | os.PathLike[str],
    llama_perplexity: str | os.PathLike[str],
    dense_model: str | os.PathLike[str],
    compressed_model: str | os.PathLike[str],
    corpus: str | os.PathLike[str],
    kld_file: str | os.PathLike[str],
    prompt_tokens: int = 128,
    gen_tokens: int = 64,
    repetitions: int = 3,
    threads: int | None = None,
    n_gpu_layers: int = 0,
    context_size: int | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """One report combining file/RAM/speed and PPL/logit-distribution quality."""
    performance = run_real_benchmark(
        llama_bench=llama_bench,
        dense_model=dense_model,
        compressed_model=compressed_model,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        repetitions=repetitions,
        threads=threads,
        n_gpu_layers=n_gpu_layers,
    )
    quality = run_quality_benchmark(
        llama_perplexity=llama_perplexity,
        dense_model=dense_model,
        compressed_model=compressed_model,
        corpus=corpus,
        kld_file=kld_file,
        threads=threads,
        n_gpu_layers=n_gpu_layers,
        context_size=context_size,
        timeout=timeout,
    )
    return {
        "performance": performance,
        "quality": quality,
    }


def write_json_report(report: dict[str, Any], output: str | os.PathLike[str]) -> None:
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
