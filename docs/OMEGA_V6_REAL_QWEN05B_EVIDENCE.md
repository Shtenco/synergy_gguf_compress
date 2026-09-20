# NEXUS Ω-MIX V6 — real Qwen2.5-0.5B evidence

Date: 2026-09-20

## Baseline

- Model: `Qwen2.5-0.5B-Instruct-Q4_K_M`
- File size: `491,400,032` bytes
- SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- Runtime: pinned `llama.cpp 5f436dddb440a288ee5611d7d1eca564a6aca9f4`
- Balanced MCQ-40 baseline: `11/40 = 27.5%`

The MCQ answer positions were rebalanced to 10 A / 10 B / 10 C / 10 D before using the benchmark as a functional-retention gate. The original smoke set was skewed and could reward answer-letter collapse.

## Low-rank result

Packed low-rank Q4 is executable, but on an already-Q4_K_M source it is not the preferred representation.

Examples:

- `blk.0.ffn_up`, rank 512 + 3% residual: tensor ratio ~0.718, but only 45.5% baseline-correct retention.
- `blk.23.ffn_up`, rank 512 + 3% residual: retention 72.7%.
- `blk.23.ffn_up`, rank 896: retention 90.9%, answer agreement 85%.

Dense reconstruction and packed runtime controls showed that the main loss came from the approximation itself, not from the packed low-rank kernel.

## GROUP_Q4

Direct group-wise Q4 with FP16 scales and group size 64 was added as a separate Ω-MIX representation.

For `blk.23.ffn_up.weight`:

- source tensor payload: `2,996,224` bytes
- GROUP_Q4 payload: `2,315,264` bytes
- tensor payload ratio: `0.772727`
- tensor reduction: `22.7273%`
- reconstruction relative RMSE: ~`0.102883`

### One-layer executable proof

A real compact GGUF using the GROUP_Q4 runtime was executed locally.

- baseline: `11/40`
- compact: `11/40`
- answer agreement: `100%`
- baseline-correct retention: `100%`
- regressions: `0`
- improvements: `0`
- whole GGUF: `491,400,032 -> 490,719,232` bytes

This is the first real executable Ω-MIX V6 PASS on an actual LLM in this branch.

## Coverage search

All coverage experiments below use only `ffn_up` and GROUP_Q4 group=64.

| Coverage | Output bytes | Saved bytes | Whole-file reduction | Strict baseline-correct gate |
|---|---:|---:|---:|---|
| blk.20–23 (last4) | 488,676,928 | 2,723,104 | 0.5542% | 11/11 PASS |
| blk.16–23 (last8) | 485,953,824 | 5,446,208 | 1.1083% | 11/11 PASS |
| blk.13–23 (last11) | 483,911,520 | 7,488,512 | 1.5239% | 11/11 PASS |
| blk.12–23 (last12) | 483,230,720 | 8,169,312 | 1.6625% | FAIL: q11 regression |

The observed safe frontier for this exact test is therefore between blocks 12 and 13.

### Full balanced-40 for last4

- baseline: `11/40`
- candidate: `12/40`
- accuracy delta: +2.5 pp (not treated as evidence of intelligence gain)
- answer agreement: `95%`
- baseline-correct retention: `100%`
- regressions: `0`
- improvements: `1`

The previous scalar GROUP_Q4 kernel was ~6.30x slower on this four-layer case. Runtime optimization is required before scaling this representation broadly.

## V6 gate rule

Weight RMSE is not sufficient. A representation is allowed only after a paired functional check.

For the current smoke protocol:

- baseline-correct retention = 100%
- regressions = 0
- answer agreement >= 95% on the full balanced smoke benchmark when run
- candidate file/payload must be smaller than source

Larger 1.5B / 13B experiments should use a larger benchmark and confidence intervals rather than treating MCQ-40 as a final intelligence measurement.

## Architectural conclusion

For already quantized Q4_K_M source tensors, direct GROUP_Q4 can dominate low-rank Q4 in the size/quality trade-off. Ω-MIX should choose representations per tensor/source type rather than applying low-rank compression universally.
