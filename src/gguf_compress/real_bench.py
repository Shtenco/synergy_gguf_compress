from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


def _rss_kib(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return 0
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def _gpu_used_mib() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        vals = [int(v.strip()) for v in out.splitlines() if v.strip().isdigit()]
        return sum(vals)
    except Exception:
        return None


def _run_one(
    llama_bench: Path,
    model: Path,
    *,
    prompt_tokens: int,
    gen_tokens: int,
    repetitions: int,
    threads: int | None,
    n_gpu_layers: int,
) -> dict[str, Any]:
    cmd = [
        str(llama_bench),
        "-m", str(model),
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-r", str(repetitions),
        "-ngl", str(n_gpu_layers),
        "-o", "json",
    ]
    if threads is not None:
        cmd += ["-t", str(threads)]

    baseline_gpu = _gpu_used_mib()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_rss_kib = 0
    peak_gpu_mib = baseline_gpu
    while p.poll() is None:
        peak_rss_kib = max(peak_rss_kib, _rss_kib(p.pid))
        gpu = _gpu_used_mib()
        if gpu is not None:
            peak_gpu_mib = gpu if peak_gpu_mib is None else max(peak_gpu_mib, gpu)
        time.sleep(0.05)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"llama-bench failed for {model}:\n{stderr[-4000:]}")

    rows = json.loads(stdout)
    if not isinstance(rows, list):
        raise RuntimeError("unexpected llama-bench JSON output")
    pp = next((r for r in rows if int(r.get("n_prompt", 0)) > 0 and int(r.get("n_gen", 0)) == 0), None)
    tg = next((r for r in rows if int(r.get("n_gen", 0)) > 0 and int(r.get("n_prompt", 0)) == 0), None)
    return {
        "model": str(model),
        "file_bytes": os.path.getsize(model),
        "prompt_tokens_per_second": float(pp["avg_ts"]) if pp else None,
        "generation_tokens_per_second": float(tg["avg_ts"]) if tg else None,
        "peak_rss_mib": peak_rss_kib / 1024.0,
        "peak_gpu_used_mib": peak_gpu_mib,
        "gpu_baseline_mib": baseline_gpu,
        "gpu_peak_delta_mib": (peak_gpu_mib - baseline_gpu) if peak_gpu_mib is not None and baseline_gpu is not None else None,
        "raw_rows": rows,
    }


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def run_real_benchmark(
    *,
    llama_bench: str | os.PathLike[str],
    dense_model: str | os.PathLike[str],
    compressed_model: str | os.PathLike[str],
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    repetitions: int = 5,
    threads: int | None = None,
    n_gpu_layers: int = 0,
) -> dict[str, Any]:
    """Run exactly the same llama-bench settings on dense and compressed GGUFs."""
    bench = Path(llama_bench)
    dense = Path(dense_model)
    compressed = Path(compressed_model)
    for path in (bench, dense, compressed):
        if not path.exists():
            raise FileNotFoundError(path)

    dense_r = _run_one(
        bench, dense,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        repetitions=repetitions,
        threads=threads,
        n_gpu_layers=n_gpu_layers,
    )
    compressed_r = _run_one(
        bench, compressed,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        repetitions=repetitions,
        threads=threads,
        n_gpu_layers=n_gpu_layers,
    )

    return {
        "settings": {
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "repetitions": repetitions,
            "threads": threads,
            "n_gpu_layers": n_gpu_layers,
        },
        "dense": dense_r,
        "compressed": compressed_r,
        "comparison": {
            "file_compression_ratio": _ratio(float(dense_r["file_bytes"]), float(compressed_r["file_bytes"])),
            "prompt_processing_speedup": _ratio(compressed_r["prompt_tokens_per_second"], dense_r["prompt_tokens_per_second"]),
            "generation_speedup": _ratio(compressed_r["generation_tokens_per_second"], dense_r["generation_tokens_per_second"]),
            "rss_ratio_dense_over_compressed": _ratio(dense_r["peak_rss_mib"], compressed_r["peak_rss_mib"]),
            "gpu_delta_ratio_dense_over_compressed": _ratio(dense_r["gpu_peak_delta_mib"], compressed_r["gpu_peak_delta_mib"]),
        },
        "notes": [
            "llama-bench excludes tokenization and sampling time.",
            "GPU memory is a best-effort nvidia-smi process-memory sample and can be null on non-NVIDIA systems.",
            "Use identical build/backend/settings for both models; this function enforces identical llama-bench arguments.",
        ],
    }
