from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import re
from typing import Any

from .optimizer import CandidateScan, GlobalPlan, TensorCandidate, select_global_plan


_BLOCK_RE = re.compile(r"(?:^|\.)blk\.(\d+)\.")


class OmegaClass(str, Enum):
    AUTHORITY = "authority"
    CRITICAL = "critical"
    PROTECTED = "protected"
    NORMAL = "normal"
    CHEAP = "cheap"


@dataclass(frozen=True)
class OmegaConfig:
    """Policy layer over the V5 activation-aware candidate scan.

    V5 already supplies the expensive part: real per-tensor candidates measured
    with activation-weighted distortion. Omega-MIX adds information governance:
    critical tensors are not allowed to buy size savings with arbitrary damage.
    """

    authority_patterns: tuple[str, ...] = (
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
    )
    critical_roles: tuple[str, ...] = ("attn_q", "attn_k")
    protected_roles: tuple[str, ...] = ("attn_output", "ffn_gate")
    critical_max_activation_rmse: float = 0.20
    protected_max_activation_rmse: float = 0.30
    normal_max_activation_rmse: float = 0.45
    cheap_max_activation_rmse: float = 0.55
    critical_min_rank: int = 32
    protected_min_rank: int = 16
    critical_min_residual_density: float = 0.0025
    protect_edge_fraction: float = 0.12

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _layer_index(name: str) -> int | None:
    m = _BLOCK_RE.search(name)
    return int(m.group(1)) if m else None


def _max_layer(candidate_names: list[str]) -> int | None:
    values = [idx for idx in (_layer_index(x) for x in candidate_names) if idx is not None]
    return max(values) if values else None


def classify_tensor(name: str, *, max_layer: int | None = None) -> OmegaClass:
    lower = name.lower()
    if any(pattern in lower for pattern in (
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
    )):
        return OmegaClass.AUTHORITY

    idx = _layer_index(lower)
    edge = False
    if idx is not None and max_layer is not None and max_layer >= 1:
        width = max(1, int(math.ceil((max_layer + 1) * 0.12)))
        edge = idx < width or idx > max_layer - width

    if any(f".{role}." in lower for role in ("attn_q", "attn_k")):
        return OmegaClass.CRITICAL
    if edge and any(f".{role}." in lower for role in ("attn_output", "ffn_gate", "ffn_down")):
        return OmegaClass.CRITICAL
    if any(f".{role}." in lower for role in ("attn_output", "ffn_gate")):
        return OmegaClass.PROTECTED
    if any(f".{role}." in lower for role in ("attn_v", "ffn_down")):
        return OmegaClass.NORMAL
    if ".ffn_up." in lower:
        return OmegaClass.CHEAP
    return OmegaClass.NORMAL


def _allowed(candidate: TensorCandidate, cls: OmegaClass, cfg: OmegaConfig) -> bool:
    if candidate.kind == "raw":
        return True
    rank = int(candidate.rank or 0)

    if cls is OmegaClass.AUTHORITY:
        return False
    if cls is OmegaClass.CRITICAL:
        return (
            rank >= cfg.critical_min_rank
            and candidate.activation_rel_rmse <= cfg.critical_max_activation_rmse
            and candidate.residual_density >= cfg.critical_min_residual_density
        )
    if cls is OmegaClass.PROTECTED:
        return (
            rank >= cfg.protected_min_rank
            and candidate.activation_rel_rmse <= cfg.protected_max_activation_rmse
        )
    if cls is OmegaClass.NORMAL:
        return candidate.activation_rel_rmse <= cfg.normal_max_activation_rmse
    return candidate.activation_rel_rmse <= cfg.cheap_max_activation_rmse


def apply_omega_policy(scan: CandidateScan, config: OmegaConfig | None = None) -> tuple[CandidateScan, dict[str, str]]:
    cfg = config or OmegaConfig()
    names = list(scan.candidates)
    max_layer = _max_layer(names)
    filtered: dict[str, list[TensorCandidate]] = {}
    classes: dict[str, str] = {}

    for name, options in scan.candidates.items():
        cls = classify_tensor(name, max_layer=max_layer)
        classes[name] = cls.value
        kept = [candidate for candidate in options if _allowed(candidate, cls, cfg)]
        raw = next((candidate for candidate in options if candidate.kind == "raw"), None)
        if raw is not None and all(candidate.candidate_id != raw.candidate_id for candidate in kept):
            kept.append(raw)
        if not kept:
            # A V5 scan should always have raw, but never silently drop a tensor.
            kept = list(options)
        filtered[name] = sorted(
            kept,
            key=lambda candidate: (candidate.estimated_bytes, candidate.distortion_energy),
        )

    policy_scan = CandidateScan(
        candidates=filtered,
        fixed_payload_bytes=scan.fixed_payload_bytes,
        original_payload_bytes=scan.original_payload_bytes,
        optimizable_raw_bytes=scan.optimizable_raw_bytes,
        calibrated_tensors=scan.calibrated_tensors,
        fallback_tensors=scan.fallback_tensors,
        skipped_tensors=scan.skipped_tensors,
    )
    return policy_scan, classes


def select_omega_plan(
    scan: CandidateScan,
    target_ratio: float,
    *,
    config: OmegaConfig | None = None,
) -> tuple[GlobalPlan, dict[str, str]]:
    policy_scan, classes = apply_omega_policy(scan, config)
    return select_global_plan(policy_scan, target_ratio), classes


def plan_breakdown(
    plan: GlobalPlan,
    classes: dict[str, str],
) -> dict[str, Any]:
    by_class: dict[str, dict[str, float | int]] = {}
    for name, choice in plan.choices.items():
        cls = classes.get(name, OmegaClass.NORMAL.value)
        row = by_class.setdefault(
            cls,
            {
                "tensors": 0,
                "bytes": 0,
                "raw": 0,
                "factorized": 0,
                "activation_distortion": 0.0,
                "activation_reference": 0.0,
            },
        )
        row["tensors"] = int(row["tensors"]) + 1
        row["bytes"] = int(row["bytes"]) + int(choice.estimated_bytes)
        row["raw" if choice.kind == "raw" else "factorized"] = (
            int(row["raw" if choice.kind == "raw" else "factorized"]) + 1
        )
        row["activation_distortion"] = float(row["activation_distortion"]) + float(choice.distortion_energy)
        row["activation_reference"] = float(row["activation_reference"]) + float(choice.reference_energy)

    for row in by_class.values():
        row["activation_rel_rmse"] = math.sqrt(
            float(row["activation_distortion"]) / max(float(row["activation_reference"]), 1e-30)
        )

    return {
        "target_ratio": plan.target_ratio,
        "target_bytes": plan.target_bytes,
        "estimated_bytes": plan.estimated_bytes,
        "feasible": plan.feasible,
        "activation_rel_rmse": plan.activation_rel_rmse,
        "classes": by_class,
    }
