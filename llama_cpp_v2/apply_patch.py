#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PINNED_LLAMA_COMMIT = "5f436dddb440a288ee5611d7d1eca564a6aca9f4"
LR_A_SUFFIX = ".__gguf_lr_a"


class PatchError(RuntimeError):
    pass


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected anchor exactly once, found {count}")
    return text.replace(old, new, 1)


def _replace_all_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise PatchError(f"{label}: expected {expected} anchors, found {count}")
    return text.replace(old, new)


def patch_llama_graph_h(text: str) -> str:
    text = _replace_once(text, "#include <map>\n", "#include <map>\n#include <unordered_map>\n", "llama-graph.h include")
    text = _replace_once(
        text,
        "struct ggml_cgraph;\nstruct ggml_context;\nstruct ggml_tensor;\n",
        "struct ggml_cgraph;\nstruct ggml_context;\nstruct ggml_tensor;\n\n"
        "// gguf_compress v2: model-owned registry mapping the factor stored under\n"
        "// the original weight name (B) to its companion A factor.\n"
        "struct llama_lowrank_registry {\n"
        "    std::unordered_map<const ggml_tensor *, ggml_tensor *> a_by_b;\n"
        "};\n",
        "llama-graph.h registry",
    )
    text = _replace_all_exact(
        text,
        "    const llama_adapter_loras    * loras;\n",
        "    const llama_adapter_loras    * loras;\n"
        "    const llama_lowrank_registry * lowrank = nullptr;\n",
        2,
        "llama-graph.h lowrank pointers",
    )
    text = _replace_once(
        text,
        "            loras == other.loras &&\n            cross == other.cross;",
        "            loras   == other.loras   &&\n"
        "            lowrank == other.lowrank &&\n"
        "            cross   == other.cross;",
        "llama-graph.h cache equality",
    )
    return text


def patch_llama_graph_cpp(text: str) -> str:
    text = _replace_once(
        text,
        "    cvec             (params.cvec),\n"
        "    loras            (params.loras),\n"
        "    mctx             (params.mctx),",
        "    cvec             (params.cvec),\n"
        "    loras            (params.loras),\n"
        "    lowrank          (params.lowrank),\n"
        "    mctx             (params.mctx),",
        "llama-graph.cpp constructor",
    )
    old = """ggml_tensor * llm_graph_context::build_lora_mm(\n          ggml_tensor * w,\n          ggml_tensor * cur,\n          ggml_tensor * w_s) const {\n    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);\n"""
    new = """ggml_tensor * llm_graph_context::build_lora_mm(\n          ggml_tensor * w,\n          ggml_tensor * cur,\n          ggml_tensor * w_s) const {\n    ggml_tensor * res = nullptr;\n\n    // gguf_compress v2 native low-rank path. The tensor under the original\n    // weight name is B[in, rank] and the companion tensor is A[rank, out]\n    // in GGML dimension order. Execute B*x followed by A*(B*x); never\n    // materialize the dense W ~= A@B. Both nodes are ordinary MUL_MAT ops,\n    // so every backend that already supports ggml_mul_mat can execute them.\n    if (lowrank) {\n        const auto it = lowrank->a_by_b.find(w);\n        if (it != lowrank->a_by_b.end()) {\n            ggml_tensor * hidden = ggml_mul_mat(ctx0, w, cur);\n            res = ggml_mul_mat(ctx0, it->second, hidden);\n        }\n    }\n\n    if (!res) {\n        res = ggml_mul_mat(ctx0, w, cur);\n    }\n"""
    return _replace_once(text, old, new, "llama-graph.cpp build_lora_mm")


def patch_llama_model_h(text: str) -> str:
    return _replace_once(
        text,
        "struct llama_model {\n    llm_type type = LLM_TYPE_UNKNOWN;\n    llm_arch arch = LLM_ARCH_UNKNOWN;\n",
        "struct llama_model {\n    llm_type type = LLM_TYPE_UNKNOWN;\n    llm_arch arch = LLM_ARCH_UNKNOWN;\n\n"
        "    // gguf_compress v2: pairs populated while architecture tensors are loaded.\n"
        "    llama_lowrank_registry lowrank;\n",
        "llama-model.h registry member",
    )


def patch_llama_model_cpp(text: str) -> str:
    old_create = """ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {\n    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;\n    return ml.create_tensor(\n        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,\n        tn, ne, flags);\n}\n"""
    new_create = f"""ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {{\n    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;\n\n    // gguf_compress v2 convention:\n    //   <original-name>              -> B, GGML dims [in, rank]\n    //   <original-name>{LR_A_SUFFIX} -> A, GGML dims [rank, out]\n    // Detect the companion before normal dense-shape validation because B is\n    // intentionally much smaller than the original dense [in, out] tensor.\n    const std::string original_name = tn.str();\n    const std::string a_name = original_name + \"{LR_A_SUFFIX}\";\n    const ggml_tensor * b_meta = ml.get_tensor_meta(original_name.c_str());\n    const ggml_tensor * a_meta = ml.get_tensor_meta(a_name.c_str());\n\n    if (b_meta && a_meta) {{\n        if (ne.size() != 2 || ggml_n_dims(b_meta) != 2 || ggml_n_dims(a_meta) != 2) {{\n            throw std::runtime_error(\"gguf_compress: low-rank factors must replace a 2D tensor\");\n        }}\n\n        const auto it_ne = ne.begin();\n        const int64_t expected_in  = it_ne[0];\n        const int64_t expected_out = it_ne[1];\n        const int64_t rank_b = b_meta->ne[1];\n        const int64_t rank_a = a_meta->ne[0];\n\n        if (b_meta->ne[0] != expected_in || a_meta->ne[1] != expected_out || rank_a != rank_b) {{\n            throw std::runtime_error(format(\n                \"gguf_compress: invalid low-rank pair for '%s': B=[%lld,%lld], A=[%lld,%lld], expected dense=[%lld,%lld]\",\n                original_name.c_str(),\n                (long long) b_meta->ne[0], (long long) b_meta->ne[1],\n                (long long) a_meta->ne[0], (long long) a_meta->ne[1],\n                (long long) expected_in, (long long) expected_out));\n        }}\n        if (!tn.suffix) {{\n            throw std::runtime_error(\"gguf_compress: native low-rank tensor requires a suffix-bearing tensor name\");\n        }}\n\n        ggml_tensor * b = ml.create_tensor(\n            hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,\n            tn, {{b_meta->ne[0], b_meta->ne[1]}}, flags);\n\n        const std::string a_suffix = std::string(tn.suffix) + \"{LR_A_SUFFIX}\";\n        const LLM_TN_IMPL tn_a(tn.arch, tn.tensor, a_suffix.c_str(), tn.bid, tn.xid);\n        ggml_tensor * a = ml.create_tensor(\n            hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,\n            tn_a, {{a_meta->ne[0], a_meta->ne[1]}}, flags);\n\n        if (!a || !b) {{\n            throw std::runtime_error(\"gguf_compress: failed to create native low-rank tensor pair\");\n        }}\n        lowrank.a_by_b[b] = a;\n        LLAMA_LOG_INFO(\"%s: native low-rank %s rank=%lld\\n\", __func__, original_name.c_str(), (long long) rank_b);\n        return b;\n    }}\n\n    return ml.create_tensor(\n        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,\n        tn, ne, flags);\n}}\n"""
    text = _replace_once(text, old_create, new_create, "llama-model.cpp create_tensor")
    return _replace_once(
        text,
        "ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {\n"
        "    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);",
        "ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {\n"
        "    llm_graph_params params_lowrank = params;\n"
        "    params_lowrank.lowrank = &lowrank;\n"
        "    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params_lowrank);",
        "llama-model.cpp build_graph",
    )


def patch_tree(root: Path) -> list[Path]:
    files = {
        Path("src/llama-graph.h"): patch_llama_graph_h,
        Path("src/llama-graph.cpp"): patch_llama_graph_cpp,
        Path("src/llama-model.h"): patch_llama_model_h,
        Path("src/llama-model.cpp"): patch_llama_model_cpp,
    }
    changed: list[Path] = []
    for rel, fn in files.items():
        path = root / rel
        if not path.exists():
            raise PatchError(f"missing upstream file: {path}")
        original = path.read_text(encoding="utf-8")
        patched = fn(original)
        path.write_text(patched, encoding="utf-8")
        changed.append(path)
    return changed


def git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Patch a pinned llama.cpp tree for gguf_compress v2 native low-rank GGML execution")
    p.add_argument("llama_cpp", type=Path)
    p.add_argument("--force", action="store_true", help="allow a different llama.cpp commit; anchors still must match")
    args = p.parse_args()

    root = args.llama_cpp.resolve()
    head = git_head(root)
    if head and head != PINNED_LLAMA_COMMIT and not args.force:
        raise SystemExit(f"refusing to patch llama.cpp {head}; expected {PINNED_LLAMA_COMMIT}. Use --force to try anchor-based patching.")

    changed = patch_tree(root)
    print(f"patched llama.cpp for native low-rank execution ({len(changed)} files)")
    for path in changed:
        print(path.relative_to(root))


if __name__ == "__main__":
    main()
