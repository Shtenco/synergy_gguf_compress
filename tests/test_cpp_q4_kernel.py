from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_cpp_q4_kernel_compiles_and_runs(tmp_path: Path) -> None:
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler available")

    repo = Path(__file__).resolve().parents[1]
    exe = tmp_path / "bench_q4"
    subprocess.run(
        [
            cxx,
            "-std=c++17",
            "-O2",
            "-I", str(repo / "cpp"),
            str(repo / "cpp" / "bench_lowrank_q4.cpp"),
            "-o", str(exe),
        ],
        check=True,
    )
    out = subprocess.check_output([str(exe), "128", "8", "2", "2"], text=True)
    result = json.loads(out)
    assert result["speedup"] > 1.0
    assert result["max_abs_error"] < 0.05
