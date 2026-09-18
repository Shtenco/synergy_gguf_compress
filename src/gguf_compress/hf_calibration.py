from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np


def hf_module_to_gguf_weight(module_name: str) -> str | None:
    """Map common Llama/Qwen HF linear modules to llama.cpp GGUF tensor names."""
    if module_name == "lm_head":
        return "output.weight"

    prefixes = ("model.layers.", "transformer.layers.")
    prefix = next((p for p in prefixes if module_name.startswith(p)), None)
    if prefix is None:
        return None
    rest = module_name[len(prefix):]
    parts = rest.split(".", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    block = int(parts[0])
    tail = parts[1]
    mapping = {
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }
    role = mapping.get(tail)
    if role is None:
        return None
    return f"blk.{block}.{role}.weight"


def _sha256_path(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_hf_activation_stats(
    model_id: str,
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    max_samples: int = 8,
    max_length: int = 256,
    device: str = "cpu",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Capture per-input-channel E[x^2] for common Llama/Qwen linear layers.

    Activations are reduced online, so full activation tensors are never kept.
    The NPZ keys are GGUF weight names, allowing direct lookup during V5 scan.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "HF calibration requires the optional 'calibrate' dependencies: "
            "python -m pip install -e '.[calibrate]'"
        ) from exc

    corpus = Path(corpus_path)
    output = Path(output_path)
    if not corpus.exists():
        raise FileNotFoundError(corpus)
    if max_samples < 1 or max_length < 8:
        raise ValueError("max_samples must be >= 1 and max_length must be >= 8")

    text = corpus.read_text(encoding="utf-8", errors="ignore")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    token_ids = [int(v) for v in encoded]
    if not token_ids:
        raise ValueError("calibration corpus produced no tokens")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.to(device)

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(gguf_name: str):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                return
            x = args[0]
            if not hasattr(x, "shape") or x.shape[-1] <= 0:
                return
            detached = x.detach().to(dtype=torch.float32)
            reduce_dims = tuple(range(detached.ndim - 1))
            ss = detached.square().sum(dim=reduce_dims).cpu().numpy().astype(np.float64, copy=False)
            vectors = int(detached.numel() // detached.shape[-1])
            if gguf_name not in sums:
                sums[gguf_name] = np.array(ss, dtype=np.float64, copy=True)
                counts[gguf_name] = vectors
            else:
                sums[gguf_name] += ss
                counts[gguf_name] += vectors
        return hook

    mapped_modules: list[tuple[str, str]] = []
    for module_name, module in model.named_modules():
        gguf_name = hf_module_to_gguf_weight(module_name)
        if gguf_name is None:
            continue
        handles.append(module.register_forward_pre_hook(make_hook(gguf_name)))
        mapped_modules.append((module_name, gguf_name))

    if not handles:
        raise RuntimeError("no supported Llama/Qwen linear modules were found in the HF model")

    samples_run = 0
    try:
        with torch.no_grad():
            for start in range(0, len(token_ids), max_length):
                if samples_run >= max_samples:
                    break
                chunk = token_ids[start:start + max_length]
                if len(chunk) < 2:
                    continue
                input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
                model(input_ids=input_ids, use_cache=False)
                samples_run += 1
    finally:
        for handle in handles:
            handle.remove()

    if samples_run == 0 or not sums:
        raise RuntimeError("activation calibration captured no statistics")

    stats: dict[str, np.ndarray] = {}
    for name, value in sums.items():
        denom = max(counts[name], 1)
        second = np.maximum(value / denom, 1e-12).astype(np.float32)
        stats[name] = np.ascontiguousarray(second)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **stats)

    metadata = {
        "format": "gguf_compress.activation_calibration/v1",
        "model_id": model_id,
        "corpus": str(corpus.resolve()),
        "corpus_sha256": _sha256_path(corpus),
        "max_samples": max_samples,
        "max_length": max_length,
        "samples_run": samples_run,
        "device": device,
        "mapped_modules": mapped_modules,
        "captured_tensors": sorted(stats),
        "vector_counts": counts,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata
