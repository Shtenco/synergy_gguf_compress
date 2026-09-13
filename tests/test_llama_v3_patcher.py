from llama_cpp_v3.apply_patch import (
    A_DATA_SUFFIX,
    B_DATA_SUFFIX,
    patch_llama_graph_cpp,
    patch_llama_graph_h,
    patch_llama_model_cpp,
    patch_llama_model_h,
)


def test_v3_graph_registry_and_cache_key():
    src = '''#include <map>\nstruct ggml_cgraph;\nstruct ggml_context;\nstruct ggml_tensor;\n    const llama_adapter_loras    * loras;\n    const llama_adapter_loras    * loras;\n            loras == other.loras &&\n            cross == other.cross;\n'''
    out = patch_llama_graph_h(src)
    assert "llama_lowrank_q4_registry" in out
    assert out.count("const llama_lowrank_q4_registry * lowrank_q4 = nullptr;") == 2
    assert "lowrank_q4 == other.lowrank_q4" in out


def test_v3_model_loader_detects_packed_companions():
    src = '''ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {\n    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;\n    return ml.create_tensor(\n        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,\n        tn, ne, flags);\n}\nggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {\n    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);'''
    out = patch_llama_model_cpp(src)
    assert A_DATA_SUFFIX in out
    assert B_DATA_SUFFIX in out
    assert "lowrank_q4.by_proxy[proxy] = entry;" in out
    assert "params_lr.lowrank_q4 = &lowrank_q4;" in out


def test_v3_model_has_owned_registry():
    src = "struct llama_model {\n    llm_type type = LLM_TYPE_UNKNOWN;\n    llm_arch arch = LLM_ARCH_UNKNOWN;\n"
    out = patch_llama_model_h(src)
    assert "llama_lowrank_q4_registry lowrank_q4;" in out


def test_v3_graph_builds_single_custom_node():
    src = '''#include "llama-graph.h"\n    cvec             (params.cvec),\n    loras            (params.loras),\n    mctx             (params.mctx),\nggml_tensor * llm_graph_context::build_lora_mm(\n          ggml_tensor * w,\n          ggml_tensor * cur,\n          ggml_tensor * w_s) const {\n    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);\n'''
    out = patch_llama_graph_cpp(src)
    assert "gguf_compress_lowrank_q4_forward" in out
    assert "ggml_custom_4d" in out
    assert "lowrank_q4       (params.lowrank_q4)" in out
