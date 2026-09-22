# NEXUS Ω-NanoQuant V9 — Design Specification

Date: 2026-09-23  
Branch: `feat/nexus-omega-nanoquant-v9`  
Base: `feat/omega-mix-v8-behavioral-budget`

## 1. Objective

Build a new compression layer on top of the existing V8 behavioral governor without changing V8 semantics.

V9 must expose **one Python entrypoint** that can execute the complete experiment:

```text
teacher / source model
    -> activation calibration
    -> NanoQuant-style binary low-rank candidate generation
    -> optional learned/structured rotation
    -> optional sparse/lattice residual correction
    -> global byte-budget allocation
    -> artifact build
    -> behavioral retention gate
    -> full benchmark / perplexity / KL checks when available
    -> adaptive budget refinement
    -> select smallest passing candidate
    -> write final machine-readable report
```

The success criterion is not minimum BPW in isolation. It is:

```text
minimum realized BPW
subject to:
  baseline-correct retention = 100%
  regressions = 0
  candidate accuracy >= baseline accuracy
  unparsed answers = 0
  optional PPL/KL thresholds pass when configured
```

Extreme targets such as 0.10–0.25 BPW are research goals, not pre-declared achievements.

## 2. Non-goals

V9 does not:
- rewrite or weaken V8 behavioral gates;
- claim lossless reconstruction;
- claim that a small implicit network can reproduce arbitrary 70B weights;
- require a KAN/hypernetwork for the first working version;
- replace every tensor with the same codec;
- report target BPW as achieved BPW unless serialized artifact size supports it.

## 3. Compatibility rule

V8 remains a stable governor. V9 may import and reuse:
- `RetentionPolicy`
- baseline/candidate comparison functions
- pass/fail semantics
- adaptive bracket/refinement logic where appropriate

V9 must not silently change V8 defaults.

Existing V1–V8 paths must continue to work.

## 4. New modules

### 4.1 `nanoquant_backend.py`

Purpose: implement a self-contained NanoQuant-inspired binary low-rank backend that can run from Python.

Core representation for a weight matrix:

```text
W ~= D_out * (B1 @ B2.T) * D_in + R
```

where:
- `B1`, `B2` are binary or signed-binary factors;
- `D_out`, `D_in` are learned/scalar row/column scales;
- `R` is optional residual handled outside or through the residual codec.

First implementation requirements:
- deterministic initialization from seed;
- low-rank factorization;
- binary projection with scale fitting;
- alternating refinement resembling ADMM/block-coordinate optimization;
- activation-weighted reconstruction loss when calibration is supplied;
- candidate statistics: rank, factor bytes, scale bytes, reconstruction RMSE, activation-weighted RMSE, realized BPW estimate;
- no dependency on Samsung NanoQuant internals at runtime.

The code may use PyTorch if available, but must have a NumPy fallback for unit tests and small matrices.

### 4.2 `rotation.py`

Purpose: transform difficult weight matrices into a more binary-friendly basis.

Supported V9 modes:
- `none`
- deterministic Hadamard/sign rotation where shape permits;
- learned orthogonal rotation for small/calibration runs.

Requirements:
- numerically invertible transform metadata;
- deterministic seed;
- report rotation metadata bytes;
- include transform cost in storage accounting;
- allow the optimizer to reject a rotation when quality gains do not justify size/runtime cost.

### 4.3 `residual_codec.py`

Purpose: encode the correction term after binary factorization.

V9 residual modes:
- `none`
- sparse top-k residual;
- block sparse residual;
- E8-inspired 8D lattice/vector quantization as an experimental codec.

The E8 path is explicitly experimental. It must be compared against a simpler scalar/vector baseline and may only be selected if it improves the real distortion-vs-bytes frontier.

All residual metadata and indices count toward realized storage.

### 4.4 `omega_v9_candidates.py`

Purpose: generate heterogeneous per-tensor candidate sets.

Candidate dimensions:
- codec type;
- rank;
- binary factor mode;
- rotation mode;
- residual mode;
- residual density;
- protected precision class.

The module consumes V5/V8 calibration and Ω tensor classes:
- authority;
- critical;
- protected;
- normal;
- cheap.

Policy:
- authority tensors default to passthrough/high precision;
- critical tensors require stricter activation error and minimum rank/residual protection;
- protected tensors allow moderate compression;
- normal/cheap tensors may enter aggressive sub-1-bit regimes.

Each candidate records:
- estimated bytes;
- realized serialized bytes when built;
- effective BPW;
- static reconstruction metrics;
- activation-weighted distortion;
- runtime metadata cost;
- codec configuration.

Dominated candidates are removed before global budget optimization.

### 4.5 `omega_v9_behavioral_budget.py`

Purpose: orchestrate V9 while reusing V8 behavior policy.

Responsibilities:
1. run baseline once;
2. load or create activation calibration;
3. generate V9 candidate space once;
4. apply Ω policy;
5. solve global model budget;
6. materialize candidate model/artifact;
7. run baseline-correct retention gate;
8. run full question set only after retention passes;
9. optionally run perplexity/KL/logit checks;
10. adaptively refine around pass/fail boundary;
11. choose smallest **realized** passing artifact;
12. emit JSON/CSV summary.

No candidate that fails the strict behavior policy may be selected even if its BPW is lower.

## 5. Single Python entrypoint

Add a top-level executable Python entrypoint:

```text
python -m gguf_compress.omega_v9_run ...
```

or equivalent console command:

```text
gguf-compress omega-v9-run ...
```

This one invocation must be sufficient for the complete experiment.

Minimum CLI:

```text
--baseline
--questions
--output-dir
--server
--activation-stats optional
--calibration-corpus optional
--ranks
--target-bpw
--min-bpw
--max-bpw
--coarse-step-bpw
--tolerance-bpw
--rotation-modes
--residual-modes
--residual-densities
--threads
--seed
--min-retention
--min-accuracy-delta
--max-regressions
--allow-unparsed
```

Optional:
- `--perplexity-bin`
- `--perplexity-corpus`
- `--max-ppl-delta`
- `--max-kl`
- `--max-logit-rmse`

If activation stats are not supplied but a calibration corpus/model path is supplied, the entrypoint should invoke the existing calibration path automatically.

## 6. One-run execution contract

A successful invocation creates:

```text
output/
  omega-v9-config.json
  omega-v9-candidate-scan.json
  omega-v9-evaluations.json
  omega-v9-frontier.csv
  omega-v9-report.json
  selected/
      model-or-runtime-artifact
      manifest.json
```

The final report must contain:
- source hash;
- source bytes;
- parameter count if known;
- baseline metrics;
- all evaluated BPW points;
- static distortion metrics;
- retention/full behavior metrics;
- optional PPL/KL/logit metrics;
- realized artifact bytes;
- realized BPW;
- selected candidate;
- explicit verdict.

Verdicts:
- `PASS`
- `NO_STRICT_CANDIDATE`
- `BUILD_FAILED`
- `BASELINE_FAILED`

## 7. Storage accounting

For every candidate:

```text
total_bytes =
  binary_factor_bytes
+ scale_bytes
+ rotation_metadata_bytes
+ residual_values_bytes
+ residual_index_bytes
+ tensor_metadata_bytes
+ alignment_overhead
+ passthrough_tensor_bytes
```

Effective BPW:

```text
effective_bpw = 8 * total_bytes / total_parameter_count
```

Both estimated and realized BPW must be reported. Selection uses realized bytes when an artifact exists.

## 8. Optimization objective

Primary optimization:

```text
minimize realized_bytes
```

subject to V8 behavior constraints.

Secondary tie-breakers:
1. lower activation-weighted reconstruction error;
2. lower PPL delta;
3. lower KL;
4. higher runtime throughput;
5. lower peak RSS.

Do not optimize for compression ratio alone.

## 9. Search schedule

Default research ladder:

```text
1.00
0.80
0.70
0.60
0.55
0.45
0.35
0.25
0.20
0.15
0.10 BPW
```

The actual implementation may use a coarse-to-fine adaptive search rather than evaluate every point.

If behavior is non-monotonic, retain the V8 principle: evaluate enough coarse points to avoid assuming monotonicity, then refine the tightest observed pass/fail region.

## 10. Testing strategy

### Unit tests

New tests:
- binary factorization shape and determinism;
- scale fitting improves or preserves reconstruction loss;
- activation-weighted objective changes candidate ordering on a controlled fixture;
- rotation round-trip;
- residual encode/decode;
- storage accounting;
- candidate dominance pruning;
- authority tensor protection;
- critical tensor minimum protection;
- BPW search bracket/refinement;
- one-entrypoint config parsing.

### Integration tests

Small synthetic matrices:
- dense -> binary factors -> residual -> reconstruction;
- compare no rotation vs rotation;
- compare sparse residual vs lattice experimental residual;
- verify lower byte candidate can be rejected on weighted distortion.

Tiny model fixture:
- baseline answers generated once;
- deliberately damaged candidate must fail retention;
- safe candidate must pass;
- selector must never choose failed lower-BPW candidate.

### End-to-end acceptance

For a real small Llama/Qwen model:
- one Python command completes the full pipeline;
- candidate search executes;
- behavior gate executes;
- selected artifact is loadable by the configured runtime;
- report is reproducible with same seed.

After the small-model gate is green, run the 70B experiment. Do not begin with 70B.

## 11. Scientific claim discipline

V9 reports measured facts only.

Allowed:
- “artifact is X bytes”;
- “effective BPW is Y”;
- “baseline-correct retention was Z%”;
- “PPL delta was D”;
- “this candidate passed/failed the declared gate.”

Not allowed without evidence:
- “70B fits under 1 GB”;
- “0.1 BPW preserves intelligence”;
- “KAN can regenerate the full model from 10 MB”;
- “E8 guarantees 30–40% better rate-distortion on this model.”

## 12. First implementation milestone

Milestone V9.1 is intentionally narrower than the full MAQ-IR vision:

```text
existing V8
+ NanoQuant-inspired binary low-rank backend
+ deterministic rotation
+ sparse residual
+ heterogeneous Ω candidates
+ single Python orchestration entrypoint
+ strict behavioral selection
```

E8/lattice and implicit residual generator are experimental follow-ons and must not block V9.1.

## 13. Definition of done for V9.1

V9.1 is complete only when:
- one Python command runs the whole pipeline;
- all new unit/integration tests pass;
- existing V8 tests still pass unchanged;
- a real small Qwen/Llama run produces a valid report;
- at least two distinct compression candidates are evaluated;
- the smallest strict passing candidate is selected by realized bytes;
- failures remain visible and are not rewritten as successful compression;
- no claim about 70B is made until the 70B run itself is executed.
