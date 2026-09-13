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
DEFAULT_MAX_KLD_CORPUS_BYTES = 4096


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

    Raw stdout/stderr is always retained, so an upstream wording change cannot
    hide the evidence even when an optional parsed metric becomes null.
    """
    return {
        "perplexity": _last_float(text, (
            rf"\bPPL\s*(?:=|:)\s*({_FLOAT})",
            rf"\bperplexity\s*(?:=|:)\s*({_FLOAT})",
        )),
        "kl_divergence": _last_float(text, (
            rf"\bKLD\s*(?:=|:)\s*({_FLOAT})",
            rf"\bKL(?:[- ]?divergence|D)?\s*(?:=|:)\s*({_FLOAT})",
            rf"\bKullback(?:[- ]Leibler)?(?:\s+divergence)?\s*(?:=|:)\s*({_FLOAT})",
        )),
        "pearson": _last_float(text, (
            rf"\bPearson(?:\s+(?:correlation|r))?\s*(?:=|:)\s*({_FLOAT})",
            rf"\bcorr(?:elation)?\s*(?:=|:)\s*({_FLOAT})",
        )),
        "same_top_p": _last_float(text, (
            rf"\bsame[- ]top[- ]p\s*(?:=|:)\s*({_FLOAT})",
            rf"\bsame top p\s*(?:=|:)\s*({_FLOAT})",
        )),
        "rms_probability_change": _last_float(text, (
            rf"\bRMS(?:\s+probability)?(?:\s+change|\s+delta[- ]?p|\s+Δp)?\s*(?:=|:)\s*({_FLOAT})",
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


def _bounded_corpus(source: Path, kld_path: Path, max_bytes: int | None) -> tuple[Path, int, bool]:
    """Create a bounded UTF-8 corpus next to the KLD file when needed.

    llama-perplexity's KLD reference stores vocabulary-wide logit information
    per evaluated token. On modern large-vocabulary models an innocent full
    corpus can therefore create multi-gigabyte files. The first real E2E gate
    deliberately uses a small deterministic prefix; users can opt out by
    passing max_corpus_bytes=None.
    """
    original_size = source.stat().st_size
    if max_bytes is None or original_size <= max_bytes:
        return source, original_size, False
    if max_bytes < 512:
        raise ValueError("max_corpus_bytes must be >= 512 or None")

    data = source.read_bytes()[:max_bytes]
    # Avoid ending in the middle of a UTF-8 code point.
    text = data.decode("utf-8", errors="ignore")
    bounded = kld_path.with_suffix(kld_path.suffix + ".corpus.txt")
    bounded.parent.mkdir(parents=True, exist_ok=True)
    bounded.write_text(text, encoding="utf-8")
    return bounded, bounded.stat().st_size, True


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
    max_corpus_bytes: int | None = DEFAULT_MAX_KLD_CORPUS_BYTES,
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

    eval_corpus, eval_corpus_bytes, corpus_was_bounded = _bounded_corpus(
        corpus_path, kld_path, max_corpus_bytes
    )
    common = ["-f", str(eval_corpus), "-ngl", str(n_gpu_layers)]
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
            "source_corpus": str(corpus_path),
            "evaluation_corpus": str(eval_corpus),
            "source_corpus_bytes": corpus_path.stat().st_size,
            "evaluation_corpus_bytes": eval_corpus_bytes,
            "corpus_was_bounded": corpus_was_bounded,
            "max_corpus_bytes": max_corpus_bytes,
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
            "Lower perplexity is better; compare the ratio/delta against the exact baseline model.",
            "KL divergence is computed from llama.cpp's baseline-logit file; zero means identical distributions.",
            "The default KLD corpus is intentionally bounded to avoid accidental multi-gigabyte logit files.",
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
    max_corpus_bytes: int | None = DEFAULT_MAX_KLD_CORPUS_BYTES,
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
        max_corpus_bytes=max_corpus_bytes,
    )
    return {
        "performance": performance,
        "quality": quality,
    }


def write_json_report(report: dict[str, Any], output: str | os.PathLike[str]) -> None:
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
