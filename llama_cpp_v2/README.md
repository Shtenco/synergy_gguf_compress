# llama.cpp / GGML native low-rank v2

Pinned upstream llama.cpp commit:

`5f436dddb440a288ee5611d7d1eca564a6aca9f4`

## Native execution

For `W ~= A @ B`, the native GGUF stores:

- `<original weight name>` = `B` with GGML dims `[in, rank]`
- `<original weight name>.__gguf_lr_a` = `A` with GGML dims `[rank, out]`

The patched common linear path executes:

```text
hidden = ggml_mul_mat(B, X)
Y      = ggml_mul_mat(A, hidden)
```

The dense W is never materialized. v2 deliberately composes two existing `GGML_OP_MUL_MAT` nodes so existing CPU/CUDA/Metal/Vulkan/SYCL backends can be reused before a backend-specific fused kernel is introduced.

For F16 `4096 x 4096`, rank 8, ideal factor storage and MAC count are both 256x smaller than dense F16. This is a complexity/storage ratio, not a promised 256x wall-clock speedup.

## Export

```bash
gguf-compress export-native model.extreme.gguf model.native-lr.gguf --factor-dtype f16
```

The exporter is conservative. Only known linear weights routed through the common llama.cpp linear path remain factorized. Unsupported 2D tensors (for example token embeddings/classifier-like weights) are reconstructed once during export so the native model is not silently corrupted.

## Patch and build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 5f436dddb440a288ee5611d7d1eca564a6aca9f4
python llama_cpp_v2/apply_patch.py llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build llama.cpp/build --target llama -j
```

GitHub Actions performs the same pinned patch-and-build check.

## Current scope

Covered first: common dense linear projections such as standard Q/K/V/QKV, attention output, FFN up/gate/down and LM output where tensor names match the conservative native allow-list.

Not yet covered: MoE `MUL_MAT_ID`, embeddings, convolutions, arbitrary direct `ggml_mul_mat` call sites, fused A/B backend kernels, and native Q4/Q8 factor kernels. Those are v3 targets after end-to-end correctness and real-model latency/perplexity measurements.
