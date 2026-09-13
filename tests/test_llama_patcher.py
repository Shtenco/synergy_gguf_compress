from llama_cpp_v2.apply_patch import (
    LR_A_SUFFIX,
    patch_llama_graph_cpp,
    patch_llama_graph_h,
    patch_llama_model_cpp,
    patch_llama_model_h,
)


def test_graph_h_registry_and_cache_key():
    src = '''#include <map>\nstruct ggml_cgraph;\nstruct ggml_context;\nstruct ggml_tensor;\n    const llama_adapter_loras    * loras;\n    const llama_adapter_loras    * loras;\n            loras == other.loras &&\n            cross == other.cross;\n'''
    out = patch_llama_graph_h(src)
    assert "llama_lowrank_registry" in out
    assert out.count("const llama_lowrank_registry * lowrank = nullptr;") == 2
    assert "lowrank == other.lowrank" in out


def test_graph_cpp_two_mul_mat_path():
    src = '''    cvec             (params.cvec),\n    loras            (params.loras),\n    mctx             (params.mctx),\nggml_tensor * llm_graph_context::build_lora_mm(\n          ggml_tensor * w,\n          ggml_tensor * cur,\n          ggml_tensor * w_s) const {\n    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);\n'''
    out = patch_llama_graph_cpp(src)
    assert "lowrank          (params.lowrank)" in out
    assert "ggml_tensor * hidden = ggml_mul_mat(ctx0, w, cur);" in out
    assert "res = ggml_mul_mat(ctx0, it->second, hidden);" in out


def test_model_h_adds_owned_registry():
    src = "struct llama_model {\n    llm_type type = LLM_TYPE_UNKNOWN;\n    llm_arch arch = LLM_ARCH_UNKNOWN;\n"
    out = patch_llama_model_h(src)
    assert "llama_lowrank_registry lowrank;" in out


def test_model_cpp_loader_and_graph_bridge():
    src = '''ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {\n    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;\n    return ml.create_tensor(\n        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,\n        tn, ne, flags);\n}\nggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {\n    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);'''
    out = patch_llama_model_cpp(src)
    assert LR_A_SUFFIX in out
    assert "lowrank.a_by_b[b] = a;" in out
    assert "params_lowrank.lowrank = &lowrank;" in out
