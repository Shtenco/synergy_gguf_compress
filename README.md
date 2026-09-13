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
```

With a square 4096×4096 matrix and rank 8, the factorized path reduces the theoretical multiply-accumulate count by **256×**. That is a complexity ratio, not a promise that wall-clock latency will be 256× on every CPU/GPU. The CLI measures the real number.

## What is implemented

- `svd-lowrank`: randomized-SVD factorization for substantially better approximation quality.
- `seed-projection`: SFC32-generated deterministic orthonormal basis; only coefficient matrix `A` is stored.
- per-row **Q4 or Q8** factor quantization.
- custom GGUF container that preserves original model metadata and raw-pass-through tensors.
- decompression back to a standard **F16 GGUF**.
- per-tensor relative RMSE and cosine similarity in the manifest.
- honest dense-vs-factorized inference benchmark with no `sleep()` or invented baseline.
- PyTorch dense-vs-low-rank training benchmark.
- `--max-rel-rmse` guard that falls back to byte-identical raw storage for tensors whose approximation is too poor.

> The compressed custom GGUF is **not directly loadable by stock llama.cpp**. Use `gguf-compress decompress` to reconstruct a normal F16 GGUF, or add the factorized runtime op described in the roadmap below. Direct factorized execution is implemented in the benchmark/runtime Python path.

## Install

```bash
python -m pip install -e .
```

Current code targets `gguf-py >= 0.19.0`.

## 1. Benchmark before touching a real model

Low-rank synthetic matrix:

```bash
gguf-compress benchmark --size 2048 --rank 8 --qbits 4 --method svd-lowrank --matrix-kind lowrank
```

Adversarial/full-rank random matrix:

```bash
gguf-compress benchmark --size 2048 --rank 8 --qbits 4 --method svd-lowrank --matrix-kind random
```

The second command is important: it shows why extreme compression can destroy an arbitrary high-rank tensor.

## 2. Compress a GGUF

Maximum shrink, lossy:

```bash
gguf-compress compress model.gguf model.extreme.gguf \
  --method svd-lowrank --preset extreme
```

Seed-core mode (smallest factor payload, normally worse reconstruction at equal rank):

```bash
gguf-compress compress model.gguf model.seed.gguf \
  --method seed-projection --rank 8 --qbits 4 --seed "my-secret-model-seed"
```

Quality-guarded mode:

```bash
gguf-compress compress model.gguf model.guarded.gguf \
  --method svd-lowrank --rank 64 --qbits 8 --max-rel-rmse 0.20
```

Any tensor whose measured relative RMSE is above `0.20` is left raw rather than silently damaged.

## 3. Inspect actual errors

```bash
gguf-compress inspect model.extreme.gguf
```

The manifest records, per factorized tensor:

- original shape/type;
- rank and quantization bits;
- factor tensor names;
- seed when `seed-projection` is used;
- relative RMSE;
- cosine similarity.

## 4. Reconstruct a standard GGUF

```bash
gguf-compress decompress model.extreme.gguf model.reconstructed-f16.gguf
```

The output restores original tensor names and model/tokenizer metadata; factorized tensors are reconstructed as F16.

## 5. Training benchmark

```bash
python -m pip install -e '.[train]'
gguf-compress train-benchmark --size 2048 --rank 8 --batch 32 --steps 20
```

The factorized model trains `rank * (in + out)` parameters instead of `in * out` for the dense matrix. Real wall-clock speedup depends on BLAS/kernel efficiency and batch size, so the command prints the measured value rather than a fixed claim.

## Why the old `W_seed + LoRA` idea does not automatically accelerate inference

A model of the form:

```text
W = W_seed + A @ B
```

still has to evaluate `X @ W_seed.T`. If `W_seed` is a full dense 4096×4096 matrix, the dominant dense multiply remains. A seed may reduce **disk storage**, but it does not erase the arithmetic.

To reduce inference compute by 100×–256×, the runtime itself must use a low-rank/operator representation such as:

```text
Y = (X @ B.T) @ A.T
```

This repository implements that operator in Python and stores the corresponding factors in GGUF. A future C++/GGML runtime patch can execute the same two GEMMs directly in llama.cpp without reconstructing `W`.

## Compression math

Dense MACs for one matrix: `out * in` per input vector.

Factorized MACs: `rank * (in + out)`.

For a square matrix of width `N`:

```text
compute reduction ~= N / (2 * rank)
```

So for `N=4096, rank=8`: `4096 / 16 = 256×` theoretical MAC reduction.

Storage is even smaller in `seed-projection` because `B` is generated from SFC32 and is not stored. Actual file ratio also includes tokenizer/metadata and tensors that were passed through.

## Quality warning

Low-rank compression is **lossy**. Transformer matrices are not guaranteed to have rank 8, rank 32, or even rank 256 at the accuracy you need. Always validate the reconstructed model with task metrics/perplexity, not only matrix RMSE. Use `--max-rel-rmse` if you need an automatic guardrail.

## Roadmap

1. Add calibration-aware rank allocation: spend rank where activation-weighted error matters most.
2. Add sparse/outlier residuals on top of the low-rank core.
3. Add importance-matrix weighted factorization for LLMs.
4. Implement a GGML `LOWRANK_MATMUL` custom op / llama.cpp loader so compressed files run directly without decompression.
5. Add GPU kernels and layer-wise end-to-end perplexity benchmark.
