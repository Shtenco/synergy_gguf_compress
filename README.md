# gguf_compress

Experimental **working** GGUF compression lab for very aggressive model shrinking, fast factorized inference and low-rank training.

The repository deliberately separates two claims that are often mixed together:

1. A short PRNG seed can reproduce a deterministic basis, but it **cannot losslessly encode arbitrary trained weights**.
2. Huge compression and compute reduction are possible when a weight matrix is replaced by a low-rank approximation and inference runs directly on the factors.

For a dense weight `W[out, in]`, this project uses:

```text
standard:        Y = X @ W.T
svd-lowrank:     W ~= A @ B       -> Y = (X @ B.T) @ A.T
seed-projection: W ~= A @ B(seed) -> Y = (X @ B(seed).T) @ A.T
V3:              W ~= A @ B + R_sparse
```

With a square 4096×4096 matrix and rank 8, the factorized path reduces the theoretical multiply-accumulate count by **256×**. That is a complexity ratio, not a promise that wall-clock latency will be 256× on every CPU/GPU. The CLI measures the real number.

## Generations

### V1

- randomized-SVD or seed-projection factors;
- Q4/Q8 row quantization;
- quality guard and F16 reconstruction;
- synthetic inference/training benchmarks.

### V2

- native factor GGUF for patched llama.cpp;
- dense `W` is not reconstructed for supported linear layers;
- execution expands into two ordinary `ggml_mul_mat` operations, so existing GGML backends can execute it.

### V3

- **adaptive rank** chosen per tensor from candidate ranks;
- role-specific rank caps for Q/K/V/O and FFN projections;
- **sparse residual correction** `W ~= A@B + R_sparse`;
- native packed-Q4 A/B storage with FP16 row scales;
- direct C++ packed-Q4 CPU kernel that does not materialize dense A/B/W;
- CPU-first llama.cpp `ggml_custom_4d` patcher;
- `llama-bench` comparison harness reporting prompt/generation t/s, file ratio, peak RSS and best-effort NVIDIA memory delta.

See [`docs/V3.md`](docs/V3.md) for the V3 format and validation details.

## Install

```bash
python -m pip install -e .
```

Current code targets `gguf-py >= 0.19.0`.

## V1 quick start

```bash
gguf-compress benchmark --size 2048 --rank 8 --qbits 4 --method svd-lowrank --matrix-kind lowrank

gguf-compress compress model.gguf model.extreme.gguf \
  --method svd-lowrank --preset extreme

gguf-compress inspect model.extreme.gguf

gguf-compress decompress model.extreme.gguf model.reconstructed-f16.gguf
```

Quality-guarded V1 mode:

```bash
gguf-compress compress model.gguf model.guarded.gguf \
  --method svd-lowrank --rank 64 --qbits 8 --max-rel-rmse 0.20
```

Any tensor whose measured relative RMSE is above the threshold is left raw rather than silently damaged.

## V2 native factor execution

```bash
gguf-compress export-native model.extreme.gguf model.native-lr.gguf --factor-dtype f16
python llama_cpp_v2/apply_patch.py /path/to/llama.cpp
```

V2 stores ordinary F16/F32 factors and executes two standard GGML matmuls. It is the more portable native runtime path.

## V3 adaptive + residual + packed Q4

Compress:

```bash
gguf-compress compress-v3 model.gguf model.v3.gguf \
  --ranks 4,8,16,32,64,128 \
  --target-rel-rmse 0.20 \
  --qbits 4 \
  --residual-density 0.01
```

Inspect rank decisions and corrected errors:

```bash
gguf-compress inspect-v3 model.v3.gguf
```

Export packed-Q4 native format:

```bash
gguf-compress export-native-q4 model.v3.gguf model.native-q4.gguf
```

Patch pinned llama.cpp for the CPU-first packed-Q4 custom op:

```bash
python llama_cpp_v3/apply_patch.py /path/to/llama.cpp
```

Compare a normal model and the native-Q4 model using exactly the same llama-bench arguments:

```bash
gguf-compress real-benchmark \
  /path/to/llama-bench \
  model.gguf \
  model.native-q4.gguf \
  --prompt-tokens 512 \
  --gen-tokens 128 \
  --repetitions 5 \
  --n-gpu-layers 0
```

Build the standalone packed-Q4 kernel benchmark:

```bash
g++ -std=c++17 -O3 -Icpp cpp/bench_lowrank_q4.cpp -o bench_lowrank_q4
./bench_lowrank_q4 1024 16 4 5
```

On the development scalar reference run, `1024×1024`, rank 16, batch 4 measured about **26.5×** speedup over the same scalar dense reference with max absolute numerical difference around **0.0053**. This is a kernel-path sanity benchmark, **not** a claim of 26.5× over optimized BLAS/GGML.

## Training benchmark

```bash
python -m pip install -e '.[train]'
gguf-compress train-benchmark --size 2048 --rank 8 --batch 32 --steps 20
```

The factorized model trains `rank * (in + out)` parameters instead of `in * out` for the dense matrix.

## Why the old `W_seed + LoRA` idea does not automatically accelerate inference

A model of the form:

```text
W = W_seed + A @ B
```

still has to evaluate `X @ W_seed.T`. If `W_seed` is a full dense 4096×4096 matrix, the dominant dense multiply remains. A seed may reduce **disk storage**, but it does not erase the arithmetic.

To reduce inference compute, the runtime itself must use the low-rank/operator representation instead of reconstructing the original matrix.

## Compression math

Dense MACs for one matrix: `out * in` per input vector.

Factorized MACs: `rank * (in + out)`.

For a square matrix of width `N`:

```text
compute reduction ~= N / (2 * rank)
```

So for `N=4096, rank=8`: `4096 / 16 = 256×` theoretical MAC reduction before residual cost.

Sparse residual cost is approximately `out * residual_k` MACs per input vector. V3 therefore exposes the quality/speed tradeoff directly through rank and residual density instead of hiding it behind one fixed compression number.

## Quality warning

Low-rank compression is **lossy**. Transformer matrices are not guaranteed to have rank 8, rank 32, or even rank 256 at the accuracy you need. Always validate the model with real task metrics/perplexity and token/logit comparisons, not only matrix RMSE.

## Current validation status

Verified locally in the development environment:

- adaptive-rank and residual unit tests;
- sparse residual inference equivalence;
- direct packed-Q4 C++ kernel compilation and numerical test;
- scalar packed-Q4 microbenchmark;
- V2/V3 llama.cpp patcher anchor tests.

A complete patched `llama.cpp` link/build and real Llama/Qwen end-to-end run still requires an executing GitHub Actions runner or another machine with the pinned upstream source and model files. The repository's Actions jobs have previously been created with `runner_id=0` and no steps, so this limitation is kept explicit rather than reporting an unexecuted CI job as a successful build.

## Roadmap

1. Calibration/activation-weighted adaptive rank rather than weight-only RMSE.
2. Per-layer residual density allocation under a global byte/quality budget.
3. End-to-end Llama/Qwen perplexity, token/logit parity and benchmark suite.
4. SIMD/AVX2/AVX-512 optimization of the V3 CPU Q4 kernel.
5. CUDA/Metal/Vulkan fused packed-Q4 kernels so V3 no longer needs the CPU-only custom-op path.
