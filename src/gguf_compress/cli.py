from __future__ import annotations

import argparse
import json
from pathlib import Path


def _preset(name: str) -> dict:
    presets = {
        "extreme": {"rank": 8, "qbits": 4, "max_rel_rmse": None},
        "medium": {"rank": 32, "qbits": 4, "max_rel_rmse": None},
        "guarded": {"rank": 64, "qbits": 8, "max_rel_rmse": 0.35},
    }
    return presets[name]


def cmd_compress(args: argparse.Namespace) -> None:
    from .gguf_io import compress_gguf
    p = _preset(args.preset)
    rank = args.rank if args.rank is not None else p["rank"]
    qbits = args.qbits if args.qbits is not None else p["qbits"]
    max_err = args.max_rel_rmse if args.max_rel_rmse is not None else p["max_rel_rmse"]
    stats = compress_gguf(
        args.input, args.output, method=args.method, rank=rank, qbits=qbits,
        seed=args.seed, min_elements=args.min_elements, max_rel_rmse=max_err,
        oversample=args.oversample, power_iters=args.power_iters,
    )
    print(json.dumps({
        "input_bytes": stats.input_bytes,
        "output_bytes": stats.output_bytes,
        "file_compression_ratio": stats.file_ratio,
        "factorized_tensors": stats.factorized_tensors,
        "passthrough_tensors": stats.passthrough_tensors,
        "mean_rel_rmse": stats.mean_rel_rmse,
        "mean_cosine": stats.mean_cosine,
        "elapsed_seconds": stats.elapsed_seconds,
    }, indent=2))


def cmd_decompress(args: argparse.Namespace) -> None:
    from .gguf_io import decompress_gguf
    decompress_gguf(args.input, args.output)
    print(f"reconstructed GGUF: {args.output}")


def cmd_inspect(args: argparse.Namespace) -> None:
    from .gguf_io import inspect_compressed
    print(json.dumps(inspect_compressed(args.input), indent=2, ensure_ascii=False))


def cmd_bench(args: argparse.Namespace) -> None:
    from .benchmark import run_matrix_benchmark
    r = run_matrix_benchmark(
        size=args.size, batch=args.batch, rank=args.rank, qbits=args.qbits,
        method=args.method, matrix_kind=args.matrix_kind, true_rank=args.true_rank,
        noise=args.noise, warmup=args.warmup, iters=args.iters,
    )
    print(json.dumps(r.to_dict(), indent=2))


def cmd_train_bench(args: argparse.Namespace) -> None:
    from .training import run_training_benchmark
    r = run_training_benchmark(args.size, args.batch, args.rank, args.steps, args.threads)
    print(json.dumps({
        "size": r.size,
        "batch": r.batch,
        "rank": r.rank,
        "dense_trainable": r.dense_trainable,
        "factor_trainable": r.factor_trainable,
        "parameter_reduction": r.parameter_reduction,
        "dense_ms_per_step": r.dense_ms_per_step,
        "factor_ms_per_step": r.factor_ms_per_step,
        "measured_speedup": r.measured_speedup,
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gguf-compress", description="Experimental measurable GGUF factorization compressor")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="compress a GGUF into the custom factorized GGUF container")
    c.add_argument("input", type=Path)
    c.add_argument("output", type=Path)
    c.add_argument("--method", choices=["svd-lowrank", "seed-projection"], default="svd-lowrank")
    c.add_argument("--preset", choices=["extreme", "medium", "guarded"], default="extreme")
    c.add_argument("--rank", type=int)
    c.add_argument("--qbits", type=int, choices=[4, 8])
    c.add_argument("--seed", default="gguf-compress-v1")
    c.add_argument("--min-elements", type=int, default=131072)
    c.add_argument("--max-rel-rmse", type=float)
    c.add_argument("--oversample", type=int, default=8)
    c.add_argument("--power-iters", type=int, default=1)
    c.set_defaults(func=cmd_compress)

    d = sub.add_parser("decompress", help="reconstruct a standard F16 GGUF")
    d.add_argument("input", type=Path)
    d.add_argument("output", type=Path)
    d.set_defaults(func=cmd_decompress)

    i = sub.add_parser("inspect", help="show compressed manifest and per-tensor error")
    i.add_argument("input", type=Path)
    i.set_defaults(func=cmd_inspect)

    b = sub.add_parser("benchmark", help="honest dense-vs-factorized CPU benchmark")
    b.add_argument("--size", type=int, default=2048)
    b.add_argument("--batch", type=int, default=32)
    b.add_argument("--rank", type=int, default=8)
    b.add_argument("--qbits", type=int, choices=[4, 8], default=4)
    b.add_argument("--method", choices=["svd-lowrank", "seed-projection"], default="svd-lowrank")
    b.add_argument("--matrix-kind", choices=["lowrank", "random"], default="lowrank")
    b.add_argument("--true-rank", type=int, default=32)
    b.add_argument("--noise", type=float, default=0.01)
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--iters", type=int, default=20)
    b.set_defaults(func=cmd_bench)

    t = sub.add_parser("train-benchmark", help="dense vs low-rank training benchmark (PyTorch)")
    t.add_argument("--size", type=int, default=2048)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--steps", type=int, default=20)
    t.add_argument("--threads", type=int)
    t.set_defaults(func=cmd_train_bench)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
