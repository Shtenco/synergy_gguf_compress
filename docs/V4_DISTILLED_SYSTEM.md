# V4 Distilled GGUF System

## Purpose

V4 turns the repository from a sequence of compression experiments into one canonical, measurable system.

The central rule is simple: disk compression, operator execution, matrix reconstruction quality and end-to-end model quality are different claims and must be measured separately.

V4 therefore produces artifacts and evidence for every stage instead of collapsing everything into one compression number.

## Canonical pipeline

```text
input GGUF
  |
  v
adaptive rank search
  |
  v
Q4 A/B factors
  |
  +--> sparse row residual
  |
  v
V3 compressed container
  |
  v
native packed-Q4 GGUF
  |
  +--> static quality gates
  |
  +--> optional patched llama.cpp runtime
          |
          +--> llama-bench
          +--> llama-perplexity
          +--> KLD / top-p / probability drift
  |
  v
distilled-report.json
```

## What V4 reuses

V4 intentionally does not duplicate the V3 codec. It reuses:

- adaptive per-tensor rank selection;
- role-specific rank caps;
- Q4 factor quantization;
- row-sparse residual correction;
- native packed-Q4 exporter;
- llama.cpp CPU-first packed-Q4 runtime patch;
- real benchmark and perplexity/KLD harnesses.

The new contribution is orchestration, reproducibility and rejection logic.

## One command

Static pipeline:

```bash
gguf-compress distill model.gguf out/
```

Full runtime validation:

```bash
gguf-compress distill model.gguf out/ \
  --llama-bench /path/to/llama-bench \
  --llama-perplexity /path/to/llama-perplexity \
  --corpus corpus.txt \
  --fail-on-gate
```

## Artifacts

For an input named `model.gguf`, V4 writes:

```text
out/model.v3.gguf
out/model.native-q4.gguf
out/model.distilled-report.json
out/model.baseline.kld            # only when runtime validation is enabled
```

The report records SHA-256 hashes and byte sizes for the source, V3 container and native-Q4 artifact.

## Static gates

Static gates run even without llama.cpp.

Default gates:

- final file compression ratio >= 1.0;
- at least one tensor factorized;
- mean corrected relative RMSE <= 0.30;
- worst factorized tensor corrected relative RMSE <= 0.40.

Static gates are deliberately conservative and configurable.

A static PASS means only:

> the compressed representation satisfies the configured storage and tensor-reconstruction constraints.

It does not mean that the language model retained acceptable perplexity or generation quality.

## Runtime gates

Runtime gates are evaluated only when real llama.cpp executables and a corpus are supplied.

Default hard runtime gate:

- perplexity ratio <= 1.10.

Other metrics are report-only unless thresholds are explicitly supplied:

- KL divergence;
- same-top-p agreement;
- prompt-processing speedup;
- generation speedup.

If a metric has a configured threshold but cannot be parsed, the gate fails instead of silently passing.

## Overall status

`overall_status=PASS` requires all evaluated static gates to pass and, if runtime validation was requested, all evaluated runtime gates to pass.

If runtime validation was not requested:

- `static_status` can be PASS;
- `runtime_status` is `NOT_RUN`;
- the report contains an explicit warning that this is not an end-to-end model-quality claim.

This prevents an unexecuted benchmark from being presented as evidence.

## Why this is the distilled architecture

The earlier generations solved different parts of the problem:

V1 proved the basic factorized-storage and factorized-compute path.

V2 moved supported weights into a native llama.cpp/GGML execution path without reconstructing dense weights.

V3 added adaptive rank, sparse correction, packed Q4 factors and a direct runtime kernel.

V4 defines the system boundary around all of them:

```text
codec + runtime + evidence + gates + reproducibility
```

That is the minimum useful unit for a real compressor product.

## Non-claims

V4 does not claim:

- that arbitrary trained matrices are reconstructible from a short seed;
- that a theoretical MAC reduction equals wall-clock speedup;
- that tensor RMSE guarantees language-model quality;
- that all transformer layers are safely compressible to the same rank;
- that the current scalar CPU kernel is faster than every optimized GGML backend;
- that a GitHub Actions job which never reached a runner is a successful validation.

## Next research track

The next mathematical generation should focus on allocation rather than another fixed preset:

1. calibration-activation-weighted sensitivity;
2. global byte budget across all layers;
3. joint rank and residual-density optimization;
4. Pareto frontier: bytes vs perplexity vs latency;
5. AVX2/AVX-512 kernel;
6. CUDA/Metal/Vulkan native kernels;
7. architecture-aware handling of MoE and grouped expert matrices.

The key next objective is:

```text
minimize total model bytes and latency
subject to an end-to-end quality budget
```

rather than minimizing matrix reconstruction error independently layer by layer.
