from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TrainingBenchmark:
    size: int
    batch: int
    rank: int
    dense_trainable: int
    factor_trainable: int
    dense_ms_per_step: float
    factor_ms_per_step: float

    @property
    def parameter_reduction(self) -> float:
        return self.dense_trainable / max(self.factor_trainable, 1)

    @property
    def measured_speedup(self) -> float:
        return self.dense_ms_per_step / max(self.factor_ms_per_step, 1e-12)


def run_training_benchmark(size: int = 2048, batch: int = 32, rank: int = 8, steps: int = 20, threads: int | None = None) -> TrainingBenchmark:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("training benchmark requires PyTorch: pip install 'gguf-compress[train]'") from exc

    if threads is not None:
        torch.set_num_threads(threads)
    torch.manual_seed(123)
    x = torch.randn(batch, size)
    target = torch.randn(batch, size)

    dense = torch.nn.Linear(size, size, bias=False)
    a = torch.nn.Parameter(torch.randn(size, rank) * 0.01)
    b = torch.nn.Parameter(torch.randn(rank, size) * 0.01)

    opt_dense = torch.optim.AdamW(dense.parameters(), lr=1e-3)
    opt_factor = torch.optim.AdamW([a, b], lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    def dense_step() -> None:
        opt_dense.zero_grad(set_to_none=True)
        loss = loss_fn(dense(x), target)
        loss.backward()
        opt_dense.step()

    def factor_step() -> None:
        opt_factor.zero_grad(set_to_none=True)
        y = (x @ b.t()) @ a.t()
        loss = loss_fn(y, target)
        loss.backward()
        opt_factor.step()

    for _ in range(3):
        dense_step(); factor_step()

    t0 = time.perf_counter()
    for _ in range(steps):
        dense_step()
    dense_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(steps):
        factor_step()
    factor_time = time.perf_counter() - t0

    return TrainingBenchmark(
        size=size,
        batch=batch,
        rank=rank,
        dense_trainable=size * size,
        factor_trainable=rank * (size + size),
        dense_ms_per_step=dense_time * 1000 / steps,
        factor_ms_per_step=factor_time * 1000 / steps,
    )
