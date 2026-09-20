#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

PINNED_LLAMA_COMMIT = "5f436dddb440a288ee5611d7d1eca564a6aca9f4"

A_DATA_SUFFIX = ".__gguf_lr_q4_a_data"
A_SCALE_SUFFIX = ".__gguf_lr_q4_a_scale"
B_DATA_SUFFIX = ".__gguf_lr_q4_b_data"
B_SCALE_SUFFIX = ".__gguf_lr_q4_b_scale"
R_COLS_SUFFIX = ".__gguf_lr_res_cols"
R_VALUES_SUFFIX = ".__gguf_lr_res_values"
R_SCALES_SUFFIX = ".__gguf_lr_res_scales"
GQ4_DATA_SUFFIX = ".__omega_gq4_data"
GQ4_SCALE_SUFFIX = ".__omega_gq4_scale"

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
    text = _replace_once(text, "#include <map>\n", "#include <map>\n#include <unordered_map>\n", "graph include")
    text = _replace_once(
        text,
        "struct ggml_cgraph;\nstruct ggml_context;\nstruct ggml_tensor;\n",
        "struct ggml_cgraph;\nstruct ggml_context;\nstruct ggml_tensor;\n\n"
        "// gguf_compress V3 packed-Q4 registry.\n"
        "struct llama_lowrank_q4_entry {\n"
        "    ggml_tensor * a_data = nullptr;\n"
        "    ggml_tensor * a_scale = nullptr;\n"
        "    ggml_tensor * b_data = nullptr;\n"
        "    ggml_tensor * b_scale = nullptr;\n"
        "    ggml_tensor * r_cols = nullptr;\n"
        "    ggml_tensor * r_values = nullptr;\n"
        "    ggml_tensor * r_scales = nullptr;\n"
        "    ggml_tensor * direct_data = nullptr;\n"
        "    ggml_tensor * direct_scales = nullptr;\n"
        "    int64_t in_features = 0;\n"
        "    int64_t out_features = 0;\n"
        "    int64_t rank = 0;\n"
        "    int64_t residual_k = 0;\n"
        "    int64_t direct_group_size = 0;\n"
        "};\n"
        "struct llama_lowrank_q4_registry {\n"
        "    std::unordered_map<const ggml_tensor *, llama_lowrank_q4_entry> by_proxy;\n"
        "};\n",
        "graph registry",
    )
    text = _replace_all_exact(
        text,
        "    const llama_adapter_loras    * loras;\n",
        "    const llama_adapter_loras    * loras;\n"
        "    const llama_lowrank_q4_registry * lowrank_q4 = nullptr;\n",
        2,
        "graph registry pointers",
    )
    text = _replace_once(
        text,
        "            loras == other.loras &&\n            cross == other.cross;",
        "            loras      == other.loras      &&\n"
        "            lowrank_q4 == other.lowrank_q4 &&\n"
        "            cross      == other.cross;",
        "graph cache equality",
    )
    return text

def patch_llama_model_h(text: str) -> str:
    return _replace_once(
        text,
        "struct llama_model {\n    llm_type type = LLM_TYPE_UNKNOWN;\n    llm_arch arch = LLM_ARCH_UNKNOWN;\n",
        "struct llama_model {\n    llm_type type = LLM_TYPE_UNKNOWN;\n    llm_arch arch = LLM_ARCH_UNKNOWN;\n\n"
        "    llama_lowrank_q4_registry lowrank_q4;\n",
        "model registry member",
    )

def patch_llama_model_cpp(text: str) -> str:
    old_create = '''ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}
'''
    new_create = f'''ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {{
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;

    const std::string original_name = tn.str();
    const ggml_tensor * direct_data_meta  = ml.get_tensor_meta((original_name + "{GQ4_DATA_SUFFIX}").c_str());
    const ggml_tensor * direct_scale_meta = ml.get_tensor_meta((original_name + "{GQ4_SCALE_SUFFIX}").c_str());

    if (direct_data_meta && direct_scale_meta) {{
        if (ne.size() != 2 || !tn.suffix) {{
            throw std::runtime_error("gguf_compress V6: direct group-Q4 can replace only a suffix-bearing 2D weight");
        }}
        const auto it_ne_direct = ne.begin();
        const int64_t expected_in_direct = it_ne_direct[0];
        const int64_t expected_out_direct = it_ne_direct[1];
        const int64_t groups = direct_scale_meta->ne[0];
        if (groups <= 0 || direct_scale_meta->ne[1] != expected_out_direct ||
            direct_data_meta->ne[1] != expected_out_direct || direct_data_meta->ne[0] != (expected_in_direct + 1)/2) {{
            throw std::runtime_error(format("gguf_compress V6: invalid direct group-Q4 shapes for '%s'", original_name.c_str()));
        }}
        const ggml_tensor * proxy_meta = ml.get_tensor_meta(original_name.c_str());
        if (!proxy_meta) throw std::runtime_error("gguf_compress V6: direct group-Q4 proxy tensor missing");
        ggml_tensor * proxy = ml.create_tensor(
            hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list,
            tn, {{proxy_meta->ne[0], proxy_meta->ne[1]}}, flags | TENSOR_ALLOW_RESHAPE);

        const std::string dd_suffix = std::string(tn.suffix) + "{GQ4_DATA_SUFFIX}";
        const std::string ds_suffix = std::string(tn.suffix) + "{GQ4_SCALE_SUFFIX}";
        const LLM_TN_IMPL tn_dd(tn.arch, tn.tensor, dd_suffix.c_str(), tn.bid, tn.xid);
        const LLM_TN_IMPL tn_ds(tn.arch, tn.tensor, ds_suffix.c_str(), tn.bid, tn.xid);
        ggml_tensor * direct_data = ml.create_tensor(
            hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list,
            tn_dd, {{direct_data_meta->ne[0], direct_data_meta->ne[1]}}, flags);
        ggml_tensor * direct_scales = ml.create_tensor(
            hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list,
            tn_ds, {{direct_scale_meta->ne[0], direct_scale_meta->ne[1]}}, flags);

        llama_lowrank_q4_entry entry;
        entry.direct_data = direct_data;
        entry.direct_scales = direct_scales;
        entry.in_features = expected_in_direct;
        entry.out_features = expected_out_direct;
        entry.direct_group_size = (expected_in_direct + groups - 1)/groups;
        lowrank_q4.by_proxy[proxy] = entry;
        LLAMA_LOG_INFO("%s: direct group-Q4 %s group=%lld\\n", __func__, original_name.c_str(), (long long) entry.direct_group_size);
        return proxy;
    }}

    const ggml_tensor * a_data_meta  = ml.get_tensor_meta((original_name + "{A_DATA_SUFFIX}").c_str());
    const ggml_tensor * a_scale_meta = ml.get_tensor_meta((original_name + "{A_SCALE_SUFFIX}").c_str());
    const ggml_tensor * b_data_meta  = ml.get_tensor_meta((original_name + "{B_DATA_SUFFIX}").c_str());
    const ggml_tensor * b_scale_meta = ml.get_tensor_meta((original_name + "{B_SCALE_SUFFIX}").c_str());

    if (a_data_meta && a_scale_meta && b_data_meta && b_scale_meta) {{
        if (ne.size() != 2 || !tn.suffix) {{
            throw std::runtime_error("gguf_compress V3: packed Q4 can replace only a suffix-bearing 2D weight");
        }}
        const auto it_ne = ne.begin();
        const int64_t expected_in = it_ne[0];
        const int64_t expected_out = it_ne[1];
        const int64_t rank = b_scale_meta->ne[0];
        if (a_scale_meta->ne[0] != expected_out || a_data_meta->ne[1] != expected_out ||
            b_data_meta->ne[1] != rank || b_data_meta->ne[0] != (expected_in + 1)/2 ||
            a_data_meta->ne[0] != (rank + 1)/2) {{
            throw std::runtime_error(format("gguf_compress V3: invalid packed Q4 shapes for '%s'", original_name.c_str()));
        }}

        const ggml_tensor * proxy_meta = ml.get_tensor_meta(original_name.c_str());
        if (!proxy_meta) throw std::runtime_error("gguf_compress V3: proxy tensor missing");
        ggml_tensor * proxy = ml.create_tensor(
            hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list,
            tn, {{proxy_meta->ne[0], proxy_meta->ne[1]}}, flags | TENSOR_ALLOW_RESHAPE);

        const std::string a_data_suffix = std::string(tn.suffix) + "{A_DATA_SUFFIX}";
        const std::string a_scale_suffix = std::string(tn.suffix) + "{A_SCALE_SUFFIX}";
        const std::string b_data_suffix = std::string(tn.suffix) + "{B_DATA_SUFFIX}";
        const std::string b_scale_suffix = std::string(tn.suffix) + "{B_SCALE_SUFFIX}";
        const LLM_TN_IMPL tn_a_data(tn.arch, tn.tensor, a_data_suffix.c_str(), tn.bid, tn.xid);
        const LLM_TN_IMPL tn_a_scale(tn.arch, tn.tensor, a_scale_suffix.c_str(), tn.bid, tn.xid);
        const LLM_TN_IMPL tn_b_data(tn.arch, tn.tensor, b_data_suffix.c_str(), tn.bid, tn.xid);
        const LLM_TN_IMPL tn_b_scale(tn.arch, tn.tensor, b_scale_suffix.c_str(), tn.bid, tn.xid);
        ggml_tensor * a_data = ml.create_tensor(hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list, tn_a_data, {{a_data_meta->ne[0], a_data_meta->ne[1]}}, flags);
        ggml_tensor * a_scale = ml.create_tensor(hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list, tn_a_scale, {{a_scale_meta->ne[0]}}, flags);
        ggml_tensor * b_data = ml.create_tensor(hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list, tn_b_data, {{b_data_meta->ne[0], b_data_meta->ne[1]}}, flags);
        ggml_tensor * b_scale = ml.create_tensor(hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list, tn_b_scale, {{b_scale_meta->ne[0]}}, flags);

        llama_lowrank_q4_entry entry;
        entry.a_data = a_data; entry.a_scale = a_scale; entry.b_data = b_data; entry.b_scale = b_scale;
        entry.in_features = expected_in; entry.out_features = expected_out; entry.rank = rank;

        const ggml_tensor * r_cols_meta = ml.get_tensor_meta((original_name + "{R_COLS_SUFFIX}").c_str());
        const ggml_tensor * r_values_meta = ml.get_tensor_meta((original_name + "{R_VALUES_SUFFIX}").c_str());
        const ggml_tensor * r_scales_meta = ml.get_tensor_meta((original_name + "{R_SCALES_SUFFIX}").c_str());
        if (r_cols_meta && r_values_meta && r_scales_meta) {{
            const std::string rc_suffix = std::string(tn.suffix) + "{R_COLS_SUFFIX}";
            const std::string rv_suffix = std::string(tn.suffix) + "{R_VALUES_SUFFIX}";
            const std::string rs_suffix = std::string(tn.suffix) + "{R_SCALES_SUFFIX}";
            const LLM_TN_IMPL tn_rc(tn.arch, tn.tensor, rc_suffix.c_str(), tn.bid, tn.xid);
            const LLM_TN_IMPL tn_rv(tn.arch, tn.tensor, rv_suffix.c_str(), tn.bid, tn.xid);
            const LLM_TN_IMPL tn_rs(tn.arch, tn.tensor, rs_suffix.c_str(), tn.bid, tn.xid);
            entry.r_cols = ml.create_tensor(hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list, tn_rc, {{r_cols_meta->ne[0], r_cols_meta->ne[1]}}, flags);
            entry.r_values = ml.create_tensor(hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list, tn_rv, {{r_values_meta->ne[0], r_values_meta->ne[1]}}, flags);
            entry.r_scales = ml.create_tensor(hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, &pimpl->cpu_buft_list, tn_rs, {{r_scales_meta->ne[0]}}, flags);
            entry.residual_k = r_cols_meta->ne[0];
        }}
        lowrank_q4.by_proxy[proxy] = entry;
        LLAMA_LOG_INFO("%s: packed-Q4 low-rank %s rank=%lld residual_k=%lld\\n", __func__, original_name.c_str(), (long long) rank, (long long) entry.residual_k);
        return proxy;
    }}

    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}}
'''
    text = _replace_once(text, old_create, new_create, "model create_tensor")
    return _replace_once(
        text,
        "ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {\n    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);",
        "ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {\n"
        "    llm_graph_params params_lr = params;\n"
        "    params_lr.lowrank_q4 = &lowrank_q4;\n"
        "    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params_lr);",
        "model build_graph",
    )

def patch_llama_context_cpp(text: str) -> str:
    return _replace_once(
        text,
        "        /*.loras       =*/ loras.get(),\n        /*.mctx        =*/ mctx,\n",
        "        /*.loras       =*/ loras.get(),\n        /*.lowrank_q4  =*/ &model.lowrank_q4,\n        /*.mctx        =*/ mctx,\n",
        "context graph params",
    )


def patch_llama_graph_cpp(text: str) -> str:
    text = _replace_once(text, '#include "llama-graph.h"\n', '#include "llama-graph.h"\n#include "gguf-compress-lowrank-q4.h"\n', "graph kernel include")
    text = _replace_once(
        text,
        "    cvec             (params.cvec),\n    loras            (params.loras),\n    mctx             (params.mctx),",
        "    cvec             (params.cvec),\n    loras            (params.loras),\n    lowrank_q4       (params.lowrank_q4),\n    mctx             (params.mctx),",
        "graph constructor",
    )
    helper = r'''
static void gguf_compress_direct_q4_forward(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) return;
    auto * entry = static_cast<llama_lowrank_q4_entry *>(userdata);
    const ggml_tensor * cur = dst->src[0];
    GGML_ASSERT(cur && ggml_is_contiguous(cur) && cur->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    const int64_t batch64 = ggml_nelements(cur) / entry->in_features;
    GGML_ASSERT(batch64 <= INT32_MAX);
    gguf_compress_v3::direct_group_q4(
        reinterpret_cast<const float *>(cur->data),
        int(batch64), int(entry->in_features), int(entry->out_features), int(entry->direct_group_size),
        reinterpret_cast<const int8_t *>(entry->direct_data->data),
        reinterpret_cast<const uint16_t *>(entry->direct_scales->data),
        reinterpret_cast<float *>(dst->data));
}

static void gguf_compress_lowrank_q4_forward(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) return;
    auto * entry = static_cast<llama_lowrank_q4_entry *>(userdata);
    const ggml_tensor * cur = dst->src[0];
    GGML_ASSERT(cur && ggml_is_contiguous(cur) && cur->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    gguf_compress_v3::SparseResidualView residual;
    gguf_compress_v3::SparseResidualView * residual_ptr = nullptr;
    if (entry->residual_k > 0) {
        residual.cols = reinterpret_cast<const int32_t *>(entry->r_cols->data);
        residual.values = reinterpret_cast<const int8_t *>(entry->r_values->data);
        residual.scales_f16 = reinterpret_cast<const uint16_t *>(entry->r_scales->data);
        residual.out_features = int(entry->out_features);
        residual.k_per_row = int(entry->residual_k);
        residual_ptr = &residual;
    }
    const int64_t batch64 = ggml_nelements(cur) / entry->in_features;
    GGML_ASSERT(batch64 <= INT32_MAX);
    gguf_compress_v3::lowrank_q4_fused(
        reinterpret_cast<const float *>(cur->data),
        int(batch64), int(entry->in_features), int(entry->out_features), int(entry->rank),
        reinterpret_cast<const int8_t *>(entry->a_data->data),
        reinterpret_cast<const uint16_t *>(entry->a_scale->data),
        reinterpret_cast<const int8_t *>(entry->b_data->data),
        reinterpret_cast<const uint16_t *>(entry->b_scale->data),
        reinterpret_cast<float *>(dst->data), residual_ptr);
}
'''
    marker = "ggml_tensor * llm_graph_context::build_lora_mm(\n"
    text = _replace_once(text, marker, helper + "\n" + marker, "graph q4 helper")
    old = '''ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
'''
    new = '''ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = nullptr;
    if (lowrank_q4) {
        const auto it = lowrank_q4->by_proxy.find(w);
        if (it != lowrank_q4->by_proxy.end()) {
            auto * entry = const_cast<llama_lowrank_q4_entry *>(&it->second);
            ggml_tensor * args[1] = { cur };
            auto fn = entry->direct_data ? gguf_compress_direct_q4_forward : gguf_compress_lowrank_q4_forward;
            res = ggml_custom_4d(ctx0, GGML_TYPE_F32,
                    entry->out_features, cur->ne[1], cur->ne[2], cur->ne[3],
                    args, 1, fn, 1, entry);
        }
    }
    if (!res) {
        res = ggml_mul_mat(ctx0, w, cur);
    }
'''
    return _replace_once(text, old, new, "graph build_lora_mm")

def patch_tree(root: Path, repo_root: Path) -> list[Path]:
    files = {
        Path("src/llama-graph.h"): patch_llama_graph_h,
        Path("src/llama-model.h"): patch_llama_model_h,
        Path("src/llama-model.cpp"): patch_llama_model_cpp,
        Path("src/llama-context.cpp"): patch_llama_context_cpp,
        Path("src/llama-graph.cpp"): patch_llama_graph_cpp,
    }
    changed=[]
    for rel, fn in files.items():
        path=root/rel
        original=path.read_text(encoding="utf-8")
        path.write_text(fn(original),encoding="utf-8")
        changed.append(path)
    shutil.copyfile(repo_root/"cpp"/"lowrank_q4_kernel.h", root/"src"/"gguf-compress-lowrank-q4.h")
    return changed

def git_head(root: Path) -> str|None:
    try:
        return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    except Exception:
        return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("llama_cpp",type=Path)
    p.add_argument("--force",action="store_true")
    args=p.parse_args()
    root=args.llama_cpp.resolve()
    repo_root=Path(__file__).resolve().parents[1]
    head=git_head(root)
    if head and head != PINNED_LLAMA_COMMIT and not args.force:
        raise SystemExit(f"refusing to patch llama.cpp {head}; expected {PINNED_LLAMA_COMMIT}")
    changed=patch_tree(root,repo_root)
    print("patched",len(changed))
if __name__=="__main__":
    main()