# NEXUS Ω-NanoQuant V9.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single Python-run NEXUS Ω-NanoQuant V9.1 pipeline that creates bit-packed binary low-rank GGUF artifacts, searches the realized BPW boundary, and selects the smallest candidate that preserves V8 baseline-correct behavior.

**Architecture:** V9.1 keeps V8 behavior policy untouched and reuses the existing V5 global allocator. A new NanoQuant-inspired backend produces signed-binary low-rank factors with optional deterministic block-Hadamard rotation and sparse residual correction; V9 serializes those factors into a compact custom GGUF, then reconstructs a temporary standard GGUF only for behavioral/perplexity evaluation with the existing llama-server. This proves codec size/quality in one Python invocation without falsely claiming a native sub-1-bit llama.cpp kernel; direct binary GEMV is a later milestone.

**Tech Stack:** Python 3.10+, NumPy 2.x, gguf-py >= 0.19, pytest 8+, existing optional PyTorch/Transformers calibration path, existing llama.cpp server/benchmark binaries.

**Spec:** `docs/superpowers/specs/2026-09-23-nexus-omega-nanoquant-v9-design.md`

## Global Constraints

- Do not modify V8 behavioral defaults: baseline-correct retention `1.0`, minimum accuracy delta `0.0`, maximum regressions `0`, unparsed answers forbidden by default.
- V9 selection is by **realized serialized bytes/BPW**, not target BPW or estimated payload alone.
- `0.10–0.25 BPW` are research targets, never pre-declared achievements.
- Authority tensors remain raw/high precision by default.
- V9.1 must run end-to-end from one Python entrypoint.
- V9.1 may reconstruct a temporary standard GGUF for evaluation; reports must state `evaluation_mode="reconstructed_standard_gguf"` and `native_binary_runtime=false`.
- No runtime dependency on Samsung NanoQuant source code. The backend is NanoQuant-inspired and self-contained.
- Existing V1–V8 commands/tests must keep working.
- Every compressed byte class, GGUF container overhead, metadata, residual indices, passthrough tensors, and alignment count toward realized file BPW.
- A failed behavioral candidate is never selectable even when smaller.
- V9.1 uses sparse residual only. E8/lattice and implicit/KAN residual generation stay out of this milestone.

## Review Focus

1. **A target BPW below the fixed/passthrough payload floor** must be reported as infeasible, not rounded into a fake candidate. Task 5 adds `test_bpw_below_fixed_floor_is_infeasible`.
2. **A model whose baseline answers zero benchmark questions correctly** must produce `BASELINE_FAILED`, not vacuous 100% retention. Task 6 adds `test_zero_correct_baseline_is_baseline_failed`.
3. **Missing/mismatched activation statistics** must either leave the tensor raw under strict calibration or fail explicitly at CLI validation; they must never silently use a wrong-length vector. Tasks 4 and 7 pin both paths.
4. **Behavior can be non-monotonic with BPW.** The search must evaluate the full coarse ladder before refinement and must not stop on the first PASS→FAIL transition. Task 6 adds a non-monotonic fixture.
5. **A compact V9 artifact that cannot be reconstructed/load-tested** must never be selected. Task 6 converts reconstruction/server startup failures to `BUILD_FAILED` for that point and excludes it from selection.

---

## File Structure

New files:

- `src/gguf_compress/nanoquant_backend.py` — signed-binary low-rank fitting, bit packing, reconstruction, byte accounting.
- `src/gguf_compress/rotation.py` — deterministic block-Hadamard transform/inverse and transformed calibration approximation.
- `src/gguf_compress/residual_codec.py` — sparse top-k residual quantization, reconstruction, storage accounting.
- `src/gguf_compress/omega_v9_candidates.py` — heterogeneous Ω-aware candidate generation, Pareto pruning, conversion to existing `CandidateScan`.
- `src/gguf_compress/omega_v9_behavioral_budget.py` — BPW search, materialization, temporary GGUF reconstruction, V8 behavior gate, selection/report.
- `src/gguf_compress/omega_v9_run.py` — one-shot Python entrypoint and automatic activation calibration routing.
- `tests/test_nanoquant_backend.py`
- `tests/test_rotation_v9.py`
- `tests/test_residual_codec_v9.py`
- `tests/test_omega_v9_candidates.py`
- `tests/test_omega_v9_budget.py`
- `tests/test_omega_v9_run.py`
- `.github/workflows/omega-v9-qwen-e2e.yml` — reproducible real small-Qwen invocation of the same Python entrypoint.

Modified files:

- `src/gguf_compress/cli.py` — add `omega-v9-run` command that delegates to `omega_v9_run.main_from_namespace`.
- `README.md` — document the honest V9.1 execution/claim boundary.

No V8 source file is modified.

---

### Task 1: Signed-Binary Low-Rank Backend

**Files:**
- Create: `src/gguf_compress/nanoquant_backend.py`
- Create: `tests/test_nanoquant_backend.py`

**Interfaces:**
- Consumes: `np.ndarray weight`, rank, optional activation second moment, deterministic seed.
- Produces:
  - `BinaryLowRankFactors`
  - `fit_binary_lowrank(weight, rank, second_moment, seed, iterations=3) -> BinaryLowRankFactors`
  - `reconstruct_binary_lowrank(factors) -> np.ndarray`
  - `pack_sign_matrix(signs) -> tuple[np.ndarray, int]`
  - `unpack_sign_matrix(packed, count, shape) -> np.ndarray`
  - `binary_factor_storage_bytes(factors) -> int`

- [ ] **Step 1: Write deterministic bit-pack round-trip tests**

```python
import numpy as np

from gguf_compress.nanoquant_backend import pack_sign_matrix, unpack_sign_matrix


def test_sign_pack_round_trip() -> None:
    signs = np.array([[1, -1, 1, 1], [-1, -1, 1, -1]], dtype=np.int8)
    packed, count = pack_sign_matrix(signs)
    restored = unpack_sign_matrix(packed, count, signs.shape)
    np.testing.assert_array_equal(restored, signs)
    assert packed.dtype == np.uint8
    assert packed.nbytes == 1
```

- [ ] **Step 2: Run the test and verify it fails because the module does not exist**

Run:

```bash
pytest tests/test_nanoquant_backend.py::test_sign_pack_round_trip -v
```

Expected: FAIL with import/module error.

- [ ] **Step 3: Implement exact pack/unpack primitives**

```python
def pack_sign_matrix(signs: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(signs, dtype=np.int8)
    if x.ndim != 2 or not np.all((x == 1) | (x == -1)):
        raise ValueError("sign matrix must be 2-D and contain only -1/+1")
    bits = (x.reshape(-1) > 0).astype(np.uint8)
    return np.packbits(bits, bitorder="little"), int(bits.size)


def unpack_sign_matrix(packed: np.ndarray, count: int, shape: tuple[int, int]) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")[:count]
    if bits.size != int(np.prod(shape)):
        raise ValueError("packed sign count does not match shape")
    return np.where(bits.reshape(shape) != 0, 1, -1).astype(np.int8)
```

- [ ] **Step 4: Add binary factorization quality/determinism tests**

```python
from gguf_compress.nanoquant_backend import fit_binary_lowrank, reconstruct_binary_lowrank


def test_binary_lowrank_is_deterministic_and_beats_zero() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(size=(24, 4)).astype(np.float32)
    right = rng.normal(size=(4, 20)).astype(np.float32)
    weight = left @ right
    second = np.linspace(0.5, 2.0, weight.shape[1], dtype=np.float32)

    a = fit_binary_lowrank(weight, rank=8, second_moment=second, seed="v9", iterations=2)
    b = fit_binary_lowrank(weight, rank=8, second_moment=second, seed="v9", iterations=2)

    np.testing.assert_array_equal(a.left_sign, b.left_sign)
    np.testing.assert_array_equal(a.right_sign, b.right_sign)
    np.testing.assert_allclose(a.component_scale, b.component_scale)

    rec = reconstruct_binary_lowrank(a)
    zero_error = np.sum((weight * weight) * second[None, :])
    rec_error = np.sum(((weight - rec) ** 2) * second[None, :])
    assert rec_error < zero_error
```

- [ ] **Step 5: Implement `BinaryLowRankFactors` and weighted coefficient fitting**

The implementation must store signed-binary matrices and one FP16/FP32 component scale per rank component. Initialize from the repository's deterministic randomized low-rank routine, then solve the weighted least-squares component amplitudes without materializing a rank×M×N tensor:

```python
@dataclass(frozen=True)
class BinaryLowRankFactors:
    left_sign: np.ndarray      # [out, rank], int8 {-1,+1}
    right_sign: np.ndarray     # [rank, in], int8 {-1,+1}
    component_scale: np.ndarray  # [rank], float32
    rank: int


def _solve_component_scales(
    weight: np.ndarray,
    left_sign: np.ndarray,
    right_sign: np.ndarray,
    second_moment: np.ndarray,
) -> np.ndarray:
    a = left_sign.astype(np.float64)
    b = right_sign.astype(np.float64)
    s = second_moment.astype(np.float64)
    gram = (a.T @ a) * ((b * s[None, :]) @ b.T)
    rhs = np.einsum("mk,mn,kn,n->k", a, weight.astype(np.float64), b, s, optimize=True)
    ridge = max(float(np.trace(gram)), 1.0) * 1e-10
    return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs).astype(np.float32)
```

Use deterministic coordinate sign refinement for `iterations`: reconstruct the other components, update each left sign from the weighted residual projection, update each right sign from the residual projection, then refit component scales.

- [ ] **Step 6: Add storage-accounting test**

```python
def test_binary_storage_counts_bits_and_scales() -> None:
    factors = fit_binary_lowrank(
        np.eye(16, dtype=np.float32),
        rank=4,
        second_moment=np.ones(16, dtype=np.float32),
        seed="bytes",
        iterations=0,
    )
    expected_bits = 16 * 4 + 4 * 16
    expected_packed = (expected_bits + 7) // 8
    assert binary_factor_storage_bytes(factors) == expected_packed + 4 * 4
```

- [ ] **Step 7: Run backend tests**

Run:

```bash
pytest tests/test_nanoquant_backend.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/gguf_compress/nanoquant_backend.py tests/test_nanoquant_backend.py
git commit -m "feat: add V9 binary low-rank backend"
```

---

### Task 2: Deterministic Block-Hadamard Rotation

**Files:**
- Create: `src/gguf_compress/rotation.py`
- Create: `tests/test_rotation_v9.py`

**Interfaces:**
- Consumes: weight matrix, second moment, block size, seed.
- Produces:
  - `RotationSpec(mode: str, row_block: int, col_block: int, seed: str)`
  - `rotate_weight(weight, spec) -> np.ndarray`
  - `inverse_rotate_weight(weight, spec) -> np.ndarray`
  - `rotate_second_moment(second_moment, spec) -> np.ndarray`
  - `rotation_metadata_bytes(spec) -> int`

- [ ] **Step 1: Write round-trip and energy tests**

```python
def test_hadamard_rotation_round_trip() -> None:
    rng = np.random.default_rng(1)
    w = rng.normal(size=(32, 64)).astype(np.float32)
    spec = RotationSpec("hadamard", row_block=16, col_block=16, seed="rot")
    rotated = rotate_weight(w, spec)
    restored = inverse_rotate_weight(rotated, spec)
    np.testing.assert_allclose(restored, w, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.sum(rotated * rotated), np.sum(w * w), rtol=2e-5)
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest tests/test_rotation_v9.py::test_hadamard_rotation_round_trip -v
```

Expected: FAIL before implementation.

- [ ] **Step 3: Implement normalized block FWHT with deterministic sign masks**

Implement an in-place-style NumPy FWHT on contiguous blocks whose block size is a power of two. Apply a seed-derived ±1 diagonal sign mask before each transform; the inverse applies the transforms/sign masks in reverse order. Reject non-power-of-two block sizes.

- [ ] **Step 4: Add calibration approximation test**

Within each transformed input block, the diagonal activation second moment is the block mean because normalized Hadamard mixing redistributes the diagonal energy:

```python
def test_rotated_second_moment_preserves_sum_per_block() -> None:
    s = np.arange(1, 33, dtype=np.float32)
    spec = RotationSpec("hadamard", row_block=8, col_block=8, seed="rot")
    rotated = rotate_second_moment(s, spec)
    for start in range(0, 32, 8):
        np.testing.assert_allclose(rotated[start:start+8], np.mean(s[start:start+8]))
        np.testing.assert_allclose(rotated[start:start+8].sum(), s[start:start+8].sum())
```

- [ ] **Step 5: Run all rotation tests**

Run:

```bash
pytest tests/test_rotation_v9.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gguf_compress/rotation.py tests/test_rotation_v9.py
git commit -m "feat: add deterministic V9 rotations"
```

---

### Task 3: Sparse Residual Codec

**Files:**
- Create: `src/gguf_compress/residual_codec.py`
- Create: `tests/test_residual_codec_v9.py`

**Interfaces:**
- Consumes: original/reconstructed matrices and residual density.
- Produces:
  - `V9SparseResidual(cols, values, scales, shape)`
  - `encode_sparse_residual(reference, approximation, density) -> V9SparseResidual | None`
  - `apply_sparse_residual(approximation, residual) -> np.ndarray`
  - `residual_storage_bytes(residual) -> int`

- [ ] **Step 1: Write correction-improves-error test**

```python
def test_sparse_residual_reduces_error() -> None:
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(16, 32)).astype(np.float32)
    approximation = reference.copy()
    approximation[:, ::4] = 0.0
    residual = encode_sparse_residual(reference, approximation, density=0.25)
    corrected = apply_sparse_residual(approximation, residual)
    assert np.linalg.norm(reference - corrected) < np.linalg.norm(reference - approximation)
```

- [ ] **Step 2: Verify it fails before implementation**

Run:

```bash
pytest tests/test_residual_codec_v9.py::test_sparse_residual_reduces_error -v
```

Expected: FAIL.

- [ ] **Step 3: Implement per-row top-k INT8 residual**

For each row choose `k=max(1, round(cols*density))` largest absolute residuals, sort selected column indices, set row scale to `max_abs/127`, quantize to signed int8, and store column indices using the smallest safe unsigned dtype selected from input width (`uint16` when width <= 65535, else `uint32`).

- [ ] **Step 4: Add exact byte-accounting tests**

```python
def test_residual_storage_counts_values_indices_scales() -> None:
    ref = np.eye(8, dtype=np.float32)
    approx = np.zeros_like(ref)
    residual = encode_sparse_residual(ref, approx, density=0.25)
    assert residual is not None
    expected = residual.cols.nbytes + residual.values.nbytes + residual.scales.astype(np.float16).nbytes
    assert residual_storage_bytes(residual) == expected
```

- [ ] **Step 5: Run residual tests**

Run:

```bash
pytest tests/test_residual_codec_v9.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gguf_compress/residual_codec.py tests/test_residual_codec_v9.py
git commit -m "feat: add V9 sparse residual codec"
```

---

### Task 4: Ω-Aware V9 Candidate Generation

**Files:**
- Create: `src/gguf_compress/omega_v9_candidates.py`
- Create: `tests/test_omega_v9_candidates.py`

**Interfaces:**
- Consumes: GGUF tensors, `ActivationCalibration`, `OmegaConfig`, ranks, rotation modes, residual densities.
- Produces:
  - `V9CandidateDetail`
  - `V9CandidateScan(optimizer_scan: CandidateScan, details: dict[str, V9CandidateDetail], total_parameters: int)`
  - `build_v9_tensor_candidates(...)`
  - `scan_v9_gguf_candidates(...)`
  - `candidate_detail(scan, tensor_name, candidate_id)`
- Reuses unchanged: `TensorCandidate`, `CandidateScan`, `select_global_plan`, `classify_tensor`.

- [ ] **Step 1: Write heterogeneous protection test**

```python
def test_authority_is_raw_and_cheap_tensor_gets_binary_candidates() -> None:
    authority = build_v9_tensor_candidates(
        np.eye(32, dtype=np.float32),
        tensor_name="output.weight",
        raw_bytes=4096,
        second_moment=np.ones(32, dtype=np.float32),
        ranks=(4, 8),
        rotations=("none",),
        residual_densities=(0.0,),
        seed="v9",
        omega_class=OmegaClass.AUTHORITY,
    )
    cheap = build_v9_tensor_candidates(
        np.eye(32, dtype=np.float32),
        tensor_name="blk.3.ffn_up.weight",
        raw_bytes=4096,
        second_moment=np.ones(32, dtype=np.float32),
        ranks=(4, 8),
        rotations=("none",),
        residual_densities=(0.0,),
        seed="v9",
        omega_class=OmegaClass.CHEAP,
    )
    assert [c.kind for c in authority.optimizer_candidates] == ["raw"]
    assert any(c.kind == "v9_binary" for c in cheap.optimizer_candidates)
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest tests/test_omega_v9_candidates.py::test_authority_is_raw_and_cheap_tensor_gets_binary_candidates -v
```

Expected: FAIL.

- [ ] **Step 3: Implement V9 candidate detail + optimizer adapter**

Use ordinary `TensorCandidate` for global allocation, with `candidate_id` encoding the V9 configuration:

```text
v9-r{rank}-rot{mode}-d{density}
```

Keep full binary factors/residual/rotation only in `V9CandidateDetail`, keyed by `(tensor_name, candidate_id)`. This lets the unmodified V5 allocator select V9 candidates using only fields it already understands.

- [ ] **Step 4: Compute final metrics in original basis**

For a rotated candidate:
1. rotate weight;
2. fit binary factors;
3. reconstruct in rotated basis;
4. inverse-rotate reconstruction;
5. build residual in original basis;
6. compute `activation_error_energy(reference, corrected, original_second_moment)`.

Do not score a rotated candidate using the transformed-basis approximation alone.

- [ ] **Step 5: Add strict-calibration missing-stat test**

Create a fake two-tensor GGUF fixture or monkeypatch reader helpers so one eligible tensor lacks calibration. Verify `scan_v9_gguf_candidates(..., require_calibration=True)` leaves that tensor in fixed payload and does not emit candidates.

- [ ] **Step 6: Add Pareto dominance test**

```python
def test_dominated_candidate_is_removed() -> None:
    # same/higher bytes and same/higher distortion must not survive
    pruned = pareto_prune_v9_candidates([...])
    ids = {c.candidate_id for c in pruned.optimizer_candidates}
    assert "dominated" not in ids
    assert "raw" in ids
```

- [ ] **Step 7: Run candidate tests**

Run:

```bash
pytest tests/test_omega_v9_candidates.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/gguf_compress/omega_v9_candidates.py tests/test_omega_v9_candidates.py
git commit -m "feat: add Omega-aware V9 candidate scan"
```

---

### Task 5: Compact V9 GGUF Serialization and Temporary Reconstruction

**Files:**
- Extend: `src/gguf_compress/omega_v9_candidates.py`
- Extend: `tests/test_omega_v9_candidates.py`

**Interfaces:**
- Produces:
  - `materialize_v9_plan(source, output, plan, scan, seed) -> V9ArtifactStats`
  - `reconstruct_v9_artifact(compact_path, standard_path) -> Path`
  - `inspect_v9_artifact(path) -> dict[str, Any]`
- Artifact prefix: `gguf_compress.omega_v9.`
- Format version: `9`.

- [ ] **Step 1: Write compact-artifact round-trip fixture**

Use a tiny synthetic GGUF containing at least one eligible 2-D F32 tensor and one passthrough tensor. Materialize a selected binary plan, reconstruct it to standard GGUF, dequantize both source/reconstructed tensor, and compare reconstructed tensor to the candidate detail's in-memory reconstruction.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest tests/test_omega_v9_candidates.py::test_v9_artifact_reconstructs_selected_candidate -v
```

Expected: FAIL.

- [ ] **Step 3: Implement custom GGUF payload**

For each selected V9 binary tensor write:
- packed left bits as `uint8`;
- packed right bits as `uint8`;
- component scales as `float16`;
- residual indices/values/scales when present;
- no full reconstructed matrix.

The manifest records logical shape, rank, exact bit counts, rotation spec, residual metadata, original tensor type, candidate id, and static error metrics.

Raw tensors are copied unchanged.

- [ ] **Step 4: Implement reconstruction to standard GGUF**

The reconstructed file uses the original architecture metadata and tensor names. V9 binary tensors are rebuilt into contiguous F16 tensors; raw tensors retain their original representation. This reconstructed file is an **evaluation artifact only**, not the selected compressed artifact.

- [ ] **Step 5: Add realized-BPW accounting**

```python
@dataclass(frozen=True)
class V9ArtifactStats:
    source_bytes: int
    artifact_bytes: int
    total_parameters: int
    estimated_payload_bytes: int

    @property
    def realized_bpw(self) -> float:
        return 8.0 * self.artifact_bytes / max(self.total_parameters, 1)
```

The final BPW uses `artifact_bytes = os.path.getsize(output)`, thereby including all GGUF metadata/alignment overhead.

- [ ] **Step 6: Add fixed-floor infeasibility test**

Build a `CandidateScan` whose `fixed_payload_bytes` alone exceeds the bytes implied by a target BPW. Verify the BPW-to-ratio helper returns an infeasible plan and no artifact is materialized.

- [ ] **Step 7: Run candidate/serialization tests**

Run:

```bash
pytest tests/test_omega_v9_candidates.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/gguf_compress/omega_v9_candidates.py tests/test_omega_v9_candidates.py
git commit -m "feat: serialize and reconstruct V9 artifacts"
```

---

### Task 6: Behavioral BPW Search Using Unchanged V8 Policy

**Files:**
- Create: `src/gguf_compress/omega_v9_behavioral_budget.py`
- Create: `tests/test_omega_v9_budget.py`

**Interfaces:**
- Consumes unchanged:
  - `RetentionPolicy`
  - `load_questions`
  - `_candidate_gate`
  - `compare_rows`
  - `passes_policy`
  - `llama_server`
  - `run_model`
- Produces:
  - `V9SearchConfig`
  - `bpw_ladder(config) -> list[float]`
  - `find_bpw_pass_fail_bracket(evaluations) -> tuple[float, float] | None`
  - `next_bpw_refinement(evaluations, config) -> float | None`
  - `run_v9_behavioral_budget(...) -> dict[str, Any]`

- [ ] **Step 1: Write BPW ladder/refinement tests**

```python
def test_v9_bpw_search_refines_observed_boundary() -> None:
    cfg = V9SearchConfig(
        coarse_bpw=(1.0, 0.8, 0.6, 0.4, 0.25),
        tolerance_bpw=0.025,
        max_refinements=8,
    )
    rows = [
        {"target_bpw": 0.8, "strict_pass": True},
        {"target_bpw": 0.6, "strict_pass": True},
        {"target_bpw": 0.4, "strict_pass": False},
    ]
    assert next_bpw_refinement(rows, cfg) == 0.5
```

- [ ] **Step 2: Add non-monotonic coarse-search test**

Mock `evaluate_bpw` so outcomes are `1.0 PASS, 0.8 FAIL, 0.6 PASS, 0.4 FAIL`. Assert all coarse points are evaluated before refinement. This preserves V8's rule that behavior is not assumed monotonic.

- [ ] **Step 3: Implement baseline-once logic**

Run baseline once, derive `baseline_correct_ids`, and fail early:

```python
if not baseline_correct_ids:
    return {
        "format": "gguf_compress.omega_v9/v9.1",
        "verdict": "BASELINE_FAILED",
        "reason": "baseline answered zero benchmark questions correctly",
        ...
    }
```

- [ ] **Step 4: Implement point evaluation**

For each target BPW:
1. convert BPW to target ratio using total parameters and original payload bytes;
2. call unchanged `select_global_plan`;
3. materialize compact V9 artifact;
4. measure realized bytes/BPW;
5. reconstruct to a temporary standard GGUF;
6. start `llama_server` with reconstruction;
7. run baseline-correct gate;
8. if gate passes, run full question set;
9. call unchanged `passes_policy`;
10. delete the temporary reconstructed GGUF after evaluation, preserving compact artifact and logs.

Catch materialization/reconstruction/server startup exceptions per point and record:

```python
{
    "behavior_status": "BUILD_FAILED",
    "strict_pass": False,
    "error_type": type(exc).__name__,
    "error": str(exc),
}
```

Do not swallow baseline startup failure; that yields top-level `BASELINE_FAILED`.

- [ ] **Step 5: Select by realized compact bytes**

Do not reuse V7's `select_smallest_passing` because it assumes `native_q4_bytes`. Add:

```python
def select_smallest_v9_passing(rows):
    passing = [r for r in rows if r.get("strict_pass") is True]
    if not passing:
        return None
    return min(
        passing,
        key=lambda r: (
            int(r["artifact_bytes"]),
            float(r.get("activation_rel_rmse", 1e9)),
            -float(r["full"]["candidate"]["accuracy"]),
        ),
    )
```

- [ ] **Step 6: Add zero-correct-baseline test**

Mock baseline rows all incorrect and assert `verdict == "BASELINE_FAILED"` and no candidate evaluator was called.

- [ ] **Step 7: Add failed-reconstruction exclusion test**

Make the smallest artifact raise during reconstruction and a larger candidate pass behavior. Assert the larger candidate is selected and the smaller one remains visible with `BUILD_FAILED`.

- [ ] **Step 8: Add report-output test**

Verify report contains:
- `evaluation_mode == "reconstructed_standard_gguf"`;
- `native_binary_runtime is False`;
- source SHA-256;
- baseline;
- policy;
- every evaluated target/realized BPW;
- selected candidate;
- explicit verdict.

- [ ] **Step 9: Run budget tests plus unchanged V8 tests**

Run:

```bash
pytest tests/test_omega_v9_budget.py tests/test_omega_v8_budget.py tests/test_omega_v7_search.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/gguf_compress/omega_v9_behavioral_budget.py tests/test_omega_v9_budget.py
git commit -m "feat: add V9 behavioral BPW search"
```

---

### Task 7: One Python Entrypoint with Automatic Calibration

**Files:**
- Create: `src/gguf_compress/omega_v9_run.py`
- Create: `tests/test_omega_v9_run.py`
- Modify: `src/gguf_compress/cli.py`

**Interfaces:**
- Produces:
  - `build_parser() -> argparse.ArgumentParser` in `omega_v9_run.py`
  - `run_from_args(args: argparse.Namespace) -> dict[str, Any]`
  - `main() -> None`
  - `main_from_namespace(args: argparse.Namespace) -> None`
- Shell entrypoints:
  - `python -m gguf_compress.omega_v9_run ...`
  - `gguf-compress omega-v9-run ...`

- [ ] **Step 1: Write parser contract test**

```python
def test_v9_parser_accepts_complete_single_run_contract(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--baseline", "model.gguf",
        "--server", "llama-server",
        "--questions", "benchmarks/intelligence_core_40.jsonl",
        "--output-dir", str(tmp_path),
        "--activation-stats", "stats.npz",
        "--ranks", "8,16,32",
        "--rotation-modes", "none,hadamard",
        "--residual-densities", "0,0.0025,0.01",
    ])
    assert args.baseline == Path("model.gguf")
    assert args.ranks == (8, 16, 32)
```

- [ ] **Step 2: Verify parser test fails**

Run:

```bash
pytest tests/test_omega_v9_run.py::test_v9_parser_accepts_complete_single_run_contract -v
```

Expected: FAIL.

- [ ] **Step 3: Implement CLI validation**

Required:
- `--baseline`
- `--server`
- `--questions`
- `--output-dir`

Calibration must be exactly one of:
1. `--activation-stats existing.npz`, or
2. `--calibration-model-id MODEL --calibration-corpus corpus.txt`.

If neither/partial/mixed configuration is given, raise `SystemExit` with a precise message.

- [ ] **Step 4: Implement automatic HF calibration path**

If activation stats are absent:
1. create `output/activation-stats.npz`;
2. call existing `capture_hf_activation_stats`;
3. pass that path into `run_v9_behavioral_budget`.

Do not duplicate calibration code.

- [ ] **Step 5: Implement complete single invocation options**

Defaults:

```text
--ranks 8,16,32,64
--coarse-bpw 1.0,0.8,0.7,0.6,0.55,0.45,0.35,0.25,0.20,0.15,0.10
--tolerance-bpw 0.025
--max-refinements 8
--rotation-modes none,hadamard
--rotation-block 32
--residual-densities 0,0.0025,0.01
--threads 4
--seed nexus-omega-nanoquant-v9
--min-retention 1.0
--min-accuracy-delta 0.0
--max-regressions 0
```

Optional runtime-quality arguments remain opt-in:
- `--perplexity-bin`
- `--perplexity-corpus`
- `--llama-bench`
- `--max-ppl-delta`
- `--max-kl`
- `--max-logit-rmse`.

V9.1 should call existing end-to-end quality helpers where their input contract is satisfied rather than reimplementing perplexity parsing.

- [ ] **Step 6: Write output topology test**

Mock the budget runner and assert a successful one-run invocation creates:
- `omega-v9-config.json`;
- `omega-v9-candidate-scan.json`;
- `omega-v9-evaluations.json`;
- `omega-v9-frontier.csv`;
- `omega-v9-report.json`;
- `selected/manifest.json`;
- selected compact artifact copied under `selected/`.

- [ ] **Step 7: Add activation-shape failure test**

Feed an NPZ whose tensor statistic length mismatches a selected matrix input width. Assert the run fails explicitly with the existing `ActivationCalibration.get` shape-mismatch error and does not fall back silently.

- [ ] **Step 8: Wire `gguf-compress omega-v9-run`**

Add a CLI subparser that mirrors the module parser and delegates to `main_from_namespace`. Keep `python -m gguf_compress.omega_v9_run` as the canonical implementation; the package CLI is only a forwarding surface.

- [ ] **Step 9: Run entrypoint + CLI regression tests**

Run:

```bash
pytest tests/test_omega_v9_run.py tests/test_core.py tests/test_omega_v8_budget.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/gguf_compress/omega_v9_run.py src/gguf_compress/cli.py tests/test_omega_v9_run.py
git commit -m "feat: add one-shot Omega NanoQuant V9 runner"
```

---

### Task 8: Optional PPL/KL Gate Integration

**Files:**
- Modify: `src/gguf_compress/omega_v9_behavioral_budget.py`
- Modify: `src/gguf_compress/omega_v9_run.py`
- Modify: `tests/test_omega_v9_budget.py`

**Interfaces:**
- New `V9RuntimeQualityPolicy(max_ppl_delta, max_kl, max_logit_rmse)`.
- Candidate behavior PASS becomes final strict PASS only after configured runtime-quality checks pass.

- [ ] **Step 1: Write optional-gate semantics test**

Mock a candidate that passes behavior but exceeds configured PPL delta. Assert:
- `behavior_status == "PASS"`;
- `quality_status == "PPL_FAIL"`;
- `strict_pass is False`.

- [ ] **Step 2: Implement opt-in quality evaluation**

Run quality checks only after behavioral PASS, against the temporary reconstructed standard GGUF. Absence of optional binaries/corpus means `quality_status="NOT_REQUESTED"`, not failure.

- [ ] **Step 3: Add CLI grouping validation**

If any of `--perplexity-bin`, `--perplexity-corpus`, or `--llama-bench` is provided for the existing combined helper, require the complete set needed by that helper. Emit a precise CLI error for partial configuration.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_omega_v9_budget.py tests/test_omega_v9_run.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gguf_compress/omega_v9_behavioral_budget.py src/gguf_compress/omega_v9_run.py tests/test_omega_v9_budget.py tests/test_omega_v9_run.py
git commit -m "feat: gate V9 candidates on optional runtime quality"
```

---

### Task 9: Real Small-Qwen End-to-End Proof

**Files:**
- Create: `.github/workflows/omega-v9-qwen-e2e.yml`
- Modify: `README.md`

**Interfaces:**
- Workflow runs the exact public Python interface from Task 7; it contains no alternate compression logic.
- Evidence artifact contains `omega-v9-report.json`, frontier CSV, selected manifest, and logs.

- [ ] **Step 1: Add workflow smoke test for command shape**

In `tests/test_omega_v9_run.py`, parse the exact arguments the workflow will use so CI command drift is caught by unit tests.

- [ ] **Step 2: Add GitHub Actions workflow**

Workflow stages:
1. checkout `feat/nexus-omega-nanoquant-v9`;
2. install package with `.[dev,calibrate]`;
3. obtain a verified small Qwen Instruct GGUF baseline and the matching HF model id for calibration;
4. obtain/build a pinned llama.cpp `llama-server`;
5. invoke exactly one V9 command;
6. upload the complete output directory as evidence.

The workflow must not declare success from static/unit tests alone; the real Qwen command has to return a report.

- [ ] **Step 3: Document claim boundary in README**

Add:

```text
V9.1 compact artifacts are bit-packed on disk.
Current behavioral validation reconstructs candidate weights to a temporary
standard GGUF and runs ordinary llama-server. Therefore V9.1 proves
storage rate/distortion/behavior, not native sub-1-bit inference speed or RAM.
```

Include the one-command example and explain `realized_bpw` vs `target_bpw`.

- [ ] **Step 4: Run the full local unit/regression suite**

Run:

```bash
pytest -q
```

Expected: all existing and V9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/omega-v9-qwen-e2e.yml README.md tests/test_omega_v9_run.py
git commit -m "test: add real Qwen V9 end-to-end proof"
```

- [ ] **Step 6: Execute the real workflow and inspect evidence**

The implementation is not considered end-to-end green until the workflow actually starts a model and emits a report. Record from the report:
- baseline model/file hash;
- source bytes/parameters;
- every target BPW;
- every realized BPW;
- behavior pass/fail;
- selected compact artifact bytes/BPW;
- whether optional PPL/KL gates ran.

If the workflow itself cannot start due runner/infrastructure problems, report `RUNNER_BLOCKED` separately; do not relabel that as algorithm PASS.

---

### Task 10: Final Verification and Branch Review

**Files:**
- No new production file required.
- Verify all files from Tasks 1–9.

**Interfaces:**
- Final output is the feature branch and evidence, not a new API.

- [ ] **Step 1: Run focused V9 suite**

```bash
pytest tests/test_nanoquant_backend.py        tests/test_rotation_v9.py        tests/test_residual_codec_v9.py        tests/test_omega_v9_candidates.py        tests/test_omega_v9_budget.py        tests/test_omega_v9_run.py -v
```

Expected: PASS.

- [ ] **Step 2: Run unchanged V5–V8 regression suite**

```bash
pytest tests/test_v5_optimizer.py        tests/test_omega_mix.py        tests/test_omega_v7_search.py        tests/test_omega_v8_budget.py -v
```

Expected: PASS.

- [ ] **Step 3: Run complete suite**

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 4: Verify one-command help surface**

```bash
python -m gguf_compress.omega_v9_run --help
gguf-compress omega-v9-run --help
```

Expected: both expose the same essential V9 options and exit 0.

- [ ] **Step 5: Inspect scientific evidence**

Reject completion if:
- selected row has `strict_pass != true`;
- selected artifact size is estimated rather than measured;
- report omits source hash;
- report claims native binary runtime;
- baseline has zero correct items;
- reconstructed evaluation artifact is confused with compact selected artifact.

- [ ] **Step 6: Commit any verification-only documentation updates**

```bash
git add README.md
git commit -m "docs: record V9 verification evidence"
```

Only commit if README evidence changed; otherwise no empty commit.

