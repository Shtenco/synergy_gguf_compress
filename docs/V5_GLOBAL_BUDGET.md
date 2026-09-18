# V5 Global Budget Optimizer + Activation-Aware Calibration

## Goal

V5 changes the optimization target from:

```text
compress each tensor until its local RMSE is small enough
```

to:

```text
allocate one global byte budget across the model
while minimizing activation-weighted distortion
```

The unit of optimization is no longer one tensor. It is the whole model.

## Activation-aware distortion

For a linear layer:

```text
Y = X W^T
```

V5 approximates the expected output distortion using a diagonal activation covariance:

```text
E ||X(W-W_hat)^T||^2
  ~= sum_j E[x_j^2] * ||W[:,j] - W_hat[:,j]||^2
```

The calibration step records `E[x_j^2]` for each input channel from real model activations.

This is stronger than plain weight RMSE because errors on frequently active channels cost more than errors on nearly unused channels.

## Calibration

For Hugging Face Llama/Qwen-family models:

```bash
python -m pip install -e '.[calibrate]'

gguf-compress calibrate-hf \
  Qwen/Qwen2.5-0.5B-Instruct \
  corpus.txt \
  qwen-activation-stats.npz \
  --max-samples 8 \
  --max-length 256
```

The NPZ uses GGUF tensor names such as:

```text
blk.0.attn_q.weight
blk.0.attn_k.weight
blk.0.attn_v.weight
blk.0.attn_output.weight
blk.0.ffn_gate.weight
blk.0.ffn_up.weight
blk.0.ffn_down.weight
```

Strict mode is the default. If a compressible tensor has no matching calibration statistics, V5 leaves it raw instead of inventing a sensitivity score.

## Candidate generation

For each calibrated native linear tensor V5 evaluates a grid of:

```text
(rank, residual_density)
```

Example:

```text
ranks              = 8,16,32,64,128
residual densities = 0,0.0025,0.01
```

Each candidate records:

- estimated bytes;
- ordinary reconstruction RMSE;
- activation-weighted relative RMSE;
- activation distortion energy;
- calibration source.

Dominated candidates are removed before global optimization.

## Global allocation

For a target model payload ratio `B`, V5 chooses one candidate per tensor.

It minimizes total activation distortion subject to:

```text
fixed_bytes + sum(selected_tensor_bytes) <= target_bytes
```

A Lagrangian search finds useful global allocations, then a greedy refinement spends any remaining budget on the upgrades with the highest distortion reduction per extra byte.

This means two equally shaped layers can receive different ranks and residual densities when their calibrated sensitivity differs.

## Pareto frontier

One command evaluates several global budgets:

```bash
gguf-compress pareto-v5 \
  model.gguf \
  qwen-activation-stats.npz \
  out/ \
  --target-ratios 0.90,0.75,0.60,0.45
```

Outputs include:

```text
v5-candidate-scan.json
*.plan.json
*.v5.gguf
*.native-q4.gguf
pareto-v5.json
pareto-v5.csv
pareto-v5.svg
```

Without llama.cpp runtime validation, the Pareto Y-axis is activation-weighted relative RMSE.

With runtime validation enabled, Pareto membership is recalculated using actual perplexity ratio versus native-Q4 file size.

## Real end-to-end mode

```bash
gguf-compress pareto-v5 \
  model.gguf \
  activation-stats.npz \
  out/ \
  --target-ratios 0.90,0.75,0.60,0.45 \
  --llama-bench /path/to/llama-bench \
  --llama-perplexity /path/to/llama-perplexity \
  --corpus corpus.txt
```

For each feasible budget point V5 records:

- actual native-Q4 bytes;
- actual file compression ratio;
- activation-aware estimated error;
- perplexity ratio and delta;
- KL divergence;
- same-top-p;
- RMS probability change;
- prompt-processing speedup;
- generation speedup;
- RSS ratio.

## First reference experiment

The repository contains:

```text
.github/workflows/v5-qwen-pareto.yml
```

It is configured for the official:

```text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-0.5B-Instruct-GGUF Q4_K_M
```

The workflow:

1. installs V5 and calibration dependencies;
2. runs unit tests;
3. builds the pinned patched llama.cpp;
4. downloads the official Qwen GGUF;
5. captures real Qwen activation statistics;
6. generates four global budget points;
7. runs llama-bench and llama-perplexity for each feasible point;
8. uploads JSON/CSV/SVG evidence.

## Current infrastructure limitation

At the time V5 was added, GitHub created the V5 workflow job but did not start a runner. The job record had no steps and no log payload.

Therefore the repository does not claim a real Qwen Pareto result yet.

The experiment is wired and reproducible, but the first end-to-end numbers require a functioning GitHub-hosted runner or another machine with sufficient RAM and network access.

## What is already validated

The allocator logic itself has been exercised independently with a synthetic budget case.

Given two layers and enough budget to upgrade only one, V5 spent the extra bytes on the layer with the larger activation-distortion reduction per byte and kept the lower-value layer at the cheaper rank.

That is the intended global behavior.

## Next research improvements

V5 uses a diagonal activation covariance approximation. Stronger future versions can add:

1. block-diagonal or low-rank activation covariance;
2. Hessian/Fisher-aware sensitivity;
3. measured latency cost in the optimizer rather than byte cost alone;
4. architecture-specific constraints for MoE;
5. automatic calibration-corpus selection;
6. joint quantization-bit allocation;
7. direct search against perplexity surrogate models.

The long-term objective is:

```text
minimize bytes + latency + RAM
subject to a measured end-to-end quality budget
```

rather than minimizing local tensor error.
