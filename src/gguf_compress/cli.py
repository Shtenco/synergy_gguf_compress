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


def _parse_ranks(value: str) -> tuple[int, ...]:
    ranks = tuple(sorted({int(v.strip()) for v in value.split(",") if v.strip()}))
    if not ranks or any(r < 1 for r in ranks):
        raise argparse.ArgumentTypeError("ranks must be a comma-separated list of positive integers")
    return ranks


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(sorted({float(v.strip()) for v in value.split(",") if v.strip()}))
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")
    return values


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


def cmd_compress_v3(args: argparse.Namespace) -> None:
    from .v3_io import compress_gguf_v3
    stats = compress_gguf_v3(
        args.input,
        args.output,
        candidates=args.ranks,
        target_rel_rmse=args.target_rel_rmse,
        qbits=args.qbits,
        residual_density=args.residual_density,
        min_elements=args.min_elements,
        max_corrected_rel_rmse=args.max_corrected_rel_rmse,
        seed=args.seed,
        oversample=args.oversample,
        power_iters=args.power_iters,
    )
    print(json.dumps({
        "input_bytes": stats.input_bytes,
        "output_bytes": stats.output_bytes,
        "file_compression_ratio": stats.file_ratio,
        "factorized_tensors": stats.factorized_tensors,
        "passthrough_tensors": stats.passthrough_tensors,
        "mean_rel_rmse_before_residual": stats.mean_rel_rmse,
        "mean_rel_rmse_after_residual": stats.mean_corrected_rel_rmse,
        "mean_rank": stats.mean_rank,
        "residual_nnz": stats.residual_nnz,
        "elapsed_seconds": stats.elapsed_seconds,
    }, indent=2))


def cmd_decompress(args: argparse.Namespace) -> None:
    from .gguf_io import decompress_gguf
    decompress_gguf(args.input, args.output)
    print(f"reconstructed GGUF: {args.output}")


def cmd_inspect(args: argparse.Namespace) -> None:
    from .gguf_io import inspect_compressed
    print(json.dumps(inspect_compressed(args.input), indent=2, ensure_ascii=False))


def cmd_inspect_v3(args: argparse.Namespace) -> None:
    from .v3_io import inspect_v3
    print(json.dumps(inspect_v3(args.input), indent=2, ensure_ascii=False))


def cmd_export_native(args: argparse.Namespace) -> None:
    from .native import export_native_gguf
    stats = export_native_gguf(args.input, args.output, factor_dtype=args.factor_dtype)
    print(json.dumps({
        "input_bytes": stats.input_bytes,
        "output_bytes": stats.output_bytes,
        "file_compression_ratio": stats.file_ratio,
        "factorized_tensors": stats.factorized_tensors,
        "passthrough_tensors": stats.passthrough_tensors,
        "reconstructed_tensors": stats.reconstructed_tensors,
    }, indent=2))


def cmd_export_native_q4(args: argparse.Namespace) -> None:
    from .native_q4 import export_native_q4_gguf
    stats = export_native_q4_gguf(args.input, args.output)
    print(json.dumps({
        "input_bytes": stats.input_bytes,
        "output_bytes": stats.output_bytes,
        "file_compression_ratio": stats.file_ratio,
        "native_q4_layers": stats.q4_layers,
        "reconstructed_layers": stats.reconstructed_layers,
        "passthrough_tensors": stats.passthrough_tensors,
    }, indent=2))


def cmd_real_bench(args: argparse.Namespace) -> None:
    from .real_bench import run_real_benchmark
    report = run_real_benchmark(
        llama_bench=args.llama_bench,
        dense_model=args.dense_model,
        compressed_model=args.compressed_model,
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        repetitions=args.repetitions,
        threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_e2e_bench(args: argparse.Namespace) -> None:
    from .e2e_quality import run_end_to_end_benchmark, write_json_report
    report = run_end_to_end_benchmark(
        llama_bench=args.llama_bench,
        llama_perplexity=args.llama_perplexity,
        dense_model=args.dense_model,
        compressed_model=args.compressed_model,
        corpus=args.corpus,
        kld_file=args.kld_file,
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        repetitions=args.repetitions,
        threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
        context_size=args.context_size,
        timeout=args.timeout,
    )
    if args.output:
        write_json_report(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))



def cmd_distill(args: argparse.Namespace) -> None:
    from .system import (
        DistilledConfig,
        RuntimeInputs,
        RuntimeQualityGates,
        StaticQualityGates,
        run_distilled_system,
    )

    static_gates = StaticQualityGates(
        min_file_compression_ratio=args.min_file_compression_ratio,
        min_factorized_tensors=args.min_factorized_tensors,
        max_mean_corrected_rel_rmse=args.max_mean_corrected_rel_rmse,
        max_tensor_corrected_rel_rmse=args.max_tensor_corrected_rel_rmse,
    )
    runtime_gates = RuntimeQualityGates(
        max_perplexity_ratio=args.max_perplexity_ratio,
        max_kl_divergence=args.max_kl_divergence,
        min_same_top_p=args.min_same_top_p,
        min_prompt_processing_speedup=args.min_prompt_processing_speedup,
        min_generation_speedup=args.min_generation_speedup,
    )
    config = DistilledConfig(
        ranks=args.ranks,
        target_rel_rmse=args.target_rel_rmse,
        qbits=4,
        residual_density=args.residual_density,
        max_corrected_rel_rmse=args.max_corrected_rel_rmse,
        min_elements=args.min_elements,
        seed=args.seed,
        oversample=args.oversample,
        power_iters=args.power_iters,
        static_gates=static_gates,
        runtime_gates=runtime_gates,
    )

    runtime = None
    runtime_fields = (args.llama_bench, args.llama_perplexity, args.corpus)
    if any(v is not None for v in runtime_fields):
        if not all(v is not None for v in runtime_fields):
            raise SystemExit("--llama-bench, --llama-perplexity and --corpus must be supplied together")
        runtime = RuntimeInputs(
            llama_bench=args.llama_bench,
            llama_perplexity=args.llama_perplexity,
            corpus=args.corpus,
            prompt_tokens=args.prompt_tokens,
            gen_tokens=args.gen_tokens,
            repetitions=args.repetitions,
            threads=args.threads,
            n_gpu_layers=args.n_gpu_layers,
            context_size=args.context_size,
            timeout=args.timeout,
        )

    report = run_distilled_system(args.input, args.output_dir, config=config, runtime=runtime)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_gate and report["overall_status"] != "PASS":
        raise SystemExit(2)




def cmd_calibrate_hf(args: argparse.Namespace) -> None:
    from .hf_calibration import capture_hf_activation_stats
    report = capture_hf_activation_stats(
        args.model_id,
        args.corpus,
        args.output,
        max_samples=args.max_samples,
        max_length=args.max_length,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_pareto_v5(args: argparse.Namespace) -> None:
    from .pareto import V5Config, run_v5_pareto
    from .system import RuntimeInputs

    config = V5Config(
        ranks=args.ranks,
        residual_densities=args.residual_densities,
        target_ratios=args.target_ratios,
        qbits=4,
        min_elements=args.min_elements,
        seed=args.seed,
        oversample=args.oversample,
        power_iters=args.power_iters,
        require_calibration=not args.allow_uniform_fallback,
        max_activation_rel_rmse=args.max_activation_rel_rmse,
    )

    runtime = None
    runtime_fields = (args.llama_bench, args.llama_perplexity, args.corpus)
    if any(v is not None for v in runtime_fields):
        if not all(v is not None for v in runtime_fields):
            raise SystemExit("--llama-bench, --llama-perplexity and --corpus must be supplied together")
        runtime = RuntimeInputs(
            llama_bench=args.llama_bench,
            llama_perplexity=args.llama_perplexity,
            corpus=args.corpus,
            prompt_tokens=args.prompt_tokens,
            gen_tokens=args.gen_tokens,
            repetitions=args.repetitions,
            threads=args.threads,
            n_gpu_layers=args.n_gpu_layers,
            context_size=args.context_size,
            timeout=args.timeout,
        )

    report = run_v5_pareto(
        args.input,
        args.output_dir,
        args.activation_stats,
        config=config,
        runtime=runtime,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))



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

    c = sub.add_parser("compress", help="compress a GGUF into the v1 custom factorized container")
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

    v3 = sub.add_parser("compress-v3", help="adaptive-rank Q4/Q8 compression with sparse residual correction")
    v3.add_argument("input", type=Path)
    v3.add_argument("output", type=Path)
    v3.add_argument("--ranks", type=_parse_ranks, default=(4, 8, 16, 32, 64, 128, 256))
    v3.add_argument("--target-rel-rmse", type=float, default=0.20)
    v3.add_argument("--qbits", type=int, choices=[4, 8], default=4)
    v3.add_argument("--residual-density", type=float, default=0.01)
    v3.add_argument("--max-corrected-rel-rmse", type=float)
    v3.add_argument("--seed", default="gguf-compress-v3")
    v3.add_argument("--min-elements", type=int, default=131072)
    v3.add_argument("--oversample", type=int, default=8)
    v3.add_argument("--power-iters", type=int, default=1)
    v3.set_defaults(func=cmd_compress_v3)

    d = sub.add_parser("decompress", help="reconstruct a standard F16 GGUF from v1")
    d.add_argument("input", type=Path)
    d.add_argument("output", type=Path)
    d.set_defaults(func=cmd_decompress)

    i = sub.add_parser("inspect", help="show v1 compressed manifest and per-tensor error")
    i.add_argument("input", type=Path)
    i.set_defaults(func=cmd_inspect)

    i3 = sub.add_parser("inspect-v3", help="show adaptive rank search and residual quality for V3")
    i3.add_argument("input", type=Path)
    i3.set_defaults(func=cmd_inspect_v3)

    n = sub.add_parser("export-native", help="export v1 factors as standard F16/F32 tensors for native llama.cpp/GGML execution")
    n.add_argument("input", type=Path)
    n.add_argument("output", type=Path)
    n.add_argument("--factor-dtype", choices=["f16", "f32"], default="f16")
    n.set_defaults(func=cmd_export_native)

    nq = sub.add_parser("export-native-q4", help="export V3 as packed Q4 factors + sparse residual for the V3 llama.cpp CPU kernel")
    nq.add_argument("input", type=Path)
    nq.add_argument("output", type=Path)
    nq.set_defaults(func=cmd_export_native_q4)

    rb = sub.add_parser("real-benchmark", help="compare dense and native models with llama-bench JSON output")
    rb.add_argument("llama_bench", type=Path)
    rb.add_argument("dense_model", type=Path)
    rb.add_argument("compressed_model", type=Path)
    rb.add_argument("--prompt-tokens", type=int, default=512)
    rb.add_argument("--gen-tokens", type=int, default=128)
    rb.add_argument("--repetitions", type=int, default=5)
    rb.add_argument("--threads", type=int)
    rb.add_argument("--n-gpu-layers", type=int, default=0)
    rb.set_defaults(func=cmd_real_bench)

    e2e = sub.add_parser("e2e-benchmark", help="combined llama-bench + perplexity/KL logit-drift comparison")
    e2e.add_argument("llama_bench", type=Path)
    e2e.add_argument("llama_perplexity", type=Path)
    e2e.add_argument("dense_model", type=Path)
    e2e.add_argument("compressed_model", type=Path)
    e2e.add_argument("corpus", type=Path)
    e2e.add_argument("--kld-file", type=Path, default=Path("baseline.kld"))
    e2e.add_argument("--output", type=Path)
    e2e.add_argument("--prompt-tokens", type=int, default=128)
    e2e.add_argument("--gen-tokens", type=int, default=64)
    e2e.add_argument("--repetitions", type=int, default=3)
    e2e.add_argument("--threads", type=int)
    e2e.add_argument("--n-gpu-layers", type=int, default=0)
    e2e.add_argument("--context-size", type=int, default=512)
    e2e.add_argument("--timeout", type=int)
    e2e.set_defaults(func=cmd_e2e_bench)


    ds = sub.add_parser(
        "distill",
        help="canonical V4 pipeline: adaptive compression -> native Q4 -> quality gates -> optional llama.cpp E2E",
    )
    ds.add_argument("input", type=Path)
    ds.add_argument("output_dir", type=Path)
    ds.add_argument("--ranks", type=_parse_ranks, default=(4, 8, 16, 32, 64, 128, 256))
    ds.add_argument("--target-rel-rmse", type=float, default=0.20)
    ds.add_argument("--residual-density", type=float, default=0.01)
    ds.add_argument("--max-corrected-rel-rmse", type=float, default=0.40)
    ds.add_argument("--min-elements", type=int, default=131072)
    ds.add_argument("--seed", default="gguf-compress-v4")
    ds.add_argument("--oversample", type=int, default=8)
    ds.add_argument("--power-iters", type=int, default=1)
    ds.add_argument("--min-file-compression-ratio", type=float, default=1.0)
    ds.add_argument("--min-factorized-tensors", type=int, default=1)
    ds.add_argument("--max-mean-corrected-rel-rmse", type=float, default=0.30)
    ds.add_argument("--max-tensor-corrected-rel-rmse", type=float, default=0.40)
    ds.add_argument("--llama-bench", type=Path)
    ds.add_argument("--llama-perplexity", type=Path)
    ds.add_argument("--corpus", type=Path)
    ds.add_argument("--prompt-tokens", type=int, default=128)
    ds.add_argument("--gen-tokens", type=int, default=64)
    ds.add_argument("--repetitions", type=int, default=3)
    ds.add_argument("--threads", type=int)
    ds.add_argument("--n-gpu-layers", type=int, default=0)
    ds.add_argument("--context-size", type=int, default=512)
    ds.add_argument("--timeout", type=int, default=1800)
    ds.add_argument("--max-perplexity-ratio", type=float, default=1.10)
    ds.add_argument("--max-kl-divergence", type=float)
    ds.add_argument("--min-same-top-p", type=float)
    ds.add_argument("--min-prompt-processing-speedup", type=float)
    ds.add_argument("--min-generation-speedup", type=float)
    ds.add_argument("--fail-on-gate", action="store_true")
    ds.set_defaults(func=cmd_distill)


    cal = sub.add_parser(
        "calibrate-hf",
        help="capture activation second moments from a Hugging Face Llama/Qwen model",
    )
    cal.add_argument("model_id")
    cal.add_argument("corpus", type=Path)
    cal.add_argument("output", type=Path)
    cal.add_argument("--max-samples", type=int, default=8)
    cal.add_argument("--max-length", type=int, default=256)
    cal.add_argument("--device", default="cpu")
    cal.add_argument("--trust-remote-code", action="store_true")
    cal.set_defaults(func=cmd_calibrate_hf)

    p5 = sub.add_parser(
        "pareto-v5",
        help="global activation-aware rank/residual allocation and Pareto frontier",
    )
    p5.add_argument("input", type=Path)
    p5.add_argument("activation_stats", type=Path)
    p5.add_argument("output_dir", type=Path)
    p5.add_argument("--ranks", type=_parse_ranks, default=(8, 16, 32, 64, 128))
    p5.add_argument("--residual-densities", type=_parse_floats, default=(0.0, 0.0025, 0.01))
    p5.add_argument("--target-ratios", type=_parse_floats, default=(0.90, 0.75, 0.60, 0.45))
    p5.add_argument("--min-elements", type=int, default=131072)
    p5.add_argument("--seed", default="gguf-compress-v5")
    p5.add_argument("--oversample", type=int, default=8)
    p5.add_argument("--power-iters", type=int, default=1)
    p5.add_argument("--max-activation-rel-rmse", type=float, default=0.50)
    p5.add_argument("--allow-uniform-fallback", action="store_true")
    p5.add_argument("--llama-bench", type=Path)
    p5.add_argument("--llama-perplexity", type=Path)
    p5.add_argument("--corpus", type=Path)
    p5.add_argument("--prompt-tokens", type=int, default=128)
    p5.add_argument("--gen-tokens", type=int, default=64)
    p5.add_argument("--repetitions", type=int, default=3)
    p5.add_argument("--threads", type=int)
    p5.add_argument("--n-gpu-layers", type=int, default=0)
    p5.add_argument("--context-size", type=int, default=512)
    p5.add_argument("--timeout", type=int, default=1800)
    p5.set_defaults(func=cmd_pareto_v5)

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
