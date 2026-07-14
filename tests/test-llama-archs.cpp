#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-v/--verbose]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

enum class glm_dsa_indexshare_fixture {
    DEFAULT,
    CONFLICTING_METADATA,
    SHARED_FIRST,
    FULL_SHARED,
    FULL_SHARED_SHARED,
    PERIODIC_FULL_SHARED,
};

static gguf_context_ptr get_gguf_ctx(
        const llm_arch arch,
        const bool moe,
        const glm_dsa_indexshare_fixture glm_dsa_indexshare = glm_dsa_indexshare_fixture::DEFAULT,
        const uint32_t context_length = 128,
        const uint32_t glm_dsa_indexer_top_k = 8) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = context_length;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
        if (arch == LLM_ARCH_GLM_DSA && glm_dsa_indexshare == glm_dsa_indexshare_fixture::FULL_SHARED_SHARED) {
            n_layer = 3;
        } else if (arch == LLM_ARCH_GLM_DSA && glm_dsa_indexshare == glm_dsa_indexshare_fixture::PERIODIC_FULL_SHARED) {
            n_layer = 4;
        }
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    }

    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, arch == LLM_ARCH_GLM_DSA ? uint32_t(128) : uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      arch == LLM_ARCH_GLM_DSA ? glm_dsa_indexer_top_k : uint32_t(8));
    if (arch == LLM_ARCH_GLM_DSA) {
        if (glm_dsa_indexshare == glm_dsa_indexshare_fixture::CONFLICTING_METADATA) {
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K_FREQUENCY, uint32_t(1));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_SKIP_TOP_K_OFFSET, uint32_t(0));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, std::vector<std::string>({"full", "shared"}));
        } else if (glm_dsa_indexshare == glm_dsa_indexshare_fixture::SHARED_FIRST) {
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, std::vector<std::string>({"shared", "full"}));
        } else if (glm_dsa_indexshare == glm_dsa_indexshare_fixture::FULL_SHARED) {
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K_FREQUENCY, uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_SKIP_TOP_K_OFFSET, uint32_t(1));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, std::vector<std::string>({"full", "shared"}));
        } else if (glm_dsa_indexshare == glm_dsa_indexshare_fixture::FULL_SHARED_SHARED) {
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K_FREQUENCY, uint32_t(3));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_SKIP_TOP_K_OFFSET, uint32_t(1));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, std::vector<std::string>({"full", "shared", "shared"}));
        } else if (glm_dsa_indexshare == glm_dsa_indexshare_fixture::PERIODIC_FULL_SHARED) {
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K_FREQUENCY, uint32_t(3));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_SKIP_TOP_K_OFFSET, uint32_t(1));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, std::vector<std::string>({"full", "shared", "shared", "full"}));
        } else {
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K_FREQUENCY, uint32_t(1));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_SKIP_TOP_K_OFFSET, uint32_t(0));
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, std::vector<std::string>({"full", "full"}));
        }
    }
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(2)); // sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static void set_test_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, true);
#endif
}

static void unset_test_env(const char * name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

static bool test_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && strcmp(value, "0") != 0 && strcmp(value, "false") != 0 && strcmp(value, "FALSE") != 0;
}

class scoped_test_env {
public:
    scoped_test_env(const char * name, const char * value) : name(name) {
        const char * old_value = getenv(name);
        if (old_value != nullptr) {
            had_old_value = true;
            this->old_value = old_value;
        }
        set_test_env(name, value);
    }

    ~scoped_test_env() {
        if (had_old_value) {
            set_test_env(name, old_value.c_str());
        } else {
            unset_test_env(name);
        }
    }

private:
    const char * name;
    bool had_old_value = false;
    std::string old_value;
};

static bool test_glm_dsa_rejects_conflicting_indexshare_metadata(const size_t seed) {
    static constexpr const char * expected_error =
            "attention.indexer.types conflicts with attention.indexer.top_k_frequency";

    try {
        gguf_context_ptr gguf_ctx = get_gguf_ctx(
                LLM_ARCH_GLM_DSA,
                true,
                glm_dsa_indexshare_fixture::CONFLICTING_METADATA);

        llama_model_params model_params = llama_model_default_params();
        model_params.progress_callback = silent_model_load_progress;
        size_t tmp = seed;
        llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tmp, model_params));
        if (!model) {
            return true;
        }
        GGML_UNUSED(model);
    } catch (const std::exception & err) {
        const std::string message = err.what();
        return message.find(expected_error) != std::string::npos;
    }

    return false;
}

static bool test_glm_dsa_rejects_shared_first_indexshare_metadata(const size_t seed) {
    static constexpr const char * expected_error =
            "GLM_DSA IndexShare Shared layer 0 has no preceding Full layer";

    try {
        gguf_context_ptr gguf_ctx = get_gguf_ctx(
                LLM_ARCH_GLM_DSA,
                true,
                glm_dsa_indexshare_fixture::SHARED_FIRST);

        llama_model_params model_params = llama_model_default_params();
        model_params.progress_callback = silent_model_load_progress;
        size_t tmp = seed;
        llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tmp, model_params));
        if (!model) {
            return true;
        }
        GGML_UNUSED(model);
    } catch (const std::exception & err) {
        const std::string message = err.what();
        return message.find(expected_error) != std::string::npos;
    }

    return false;
}

struct glm_dsa_graph_op_counts {
    int total             = 0;
    int lightning_indexer = 0;
    int top_k             = 0;
    int top_k_consumers   = 0;
    int compact_get_rows  = 0;
    int compact_get_rows_ops = 0;
    int compact_k_get_rows   = 0;
    int compact_v_get_rows   = 0;
    int compact_v_views      = 0;
    int compact_flash_attn   = 0;
    int compact_flash_no_mask = 0;
    int compact_flash_v_view = 0;
    int flash_attn        = 0;
    int sparse_mask       = 0;
    int sparse_mask_nodes = 0;
    int dsa_sparse_attn   = 0;
    int64_t max_top_k_width = 0;
    int64_t max_compact_rows = 0;
};

static bool glm_dsa_tensor_name_starts_with(const ggml_tensor * tensor, const char * prefix) {
    return tensor != nullptr &&
            tensor->name[0] != '\0' &&
            strncmp(tensor->name, prefix, strlen(prefix)) == 0;
}

static bool glm_dsa_is_indexshare_top_k_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    if (strncmp(tensor->name, "ffn_moe_topk", strlen("ffn_moe_topk")) == 0) {
        return false;
    }
    return tensor->op == GGML_OP_TOP_K ||
            strncmp(tensor->name, "top_k", strlen("top_k")) == 0;
}

static bool glm_dsa_tensor_wraps_indexshare_top_k(const ggml_tensor * tensor, int depth = 0) {
    if (tensor == nullptr || depth > 4) {
        return false;
    }
    if (glm_dsa_is_indexshare_top_k_tensor(tensor)) {
        return true;
    }
    if (tensor->op == GGML_OP_VIEW ||
            tensor->op == GGML_OP_RESHAPE ||
            tensor->op == GGML_OP_CONT ||
            tensor->op == GGML_OP_PERMUTE) {
        return glm_dsa_tensor_wraps_indexshare_top_k(tensor->src[0], depth + 1);
    }
    return false;
}

static bool glm_dsa_tensor_consumes_indexshare_top_k(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    if (tensor->op == GGML_OP_DSA_SPARSE_MASK) {
        return glm_dsa_tensor_wraps_indexshare_top_k(tensor->src[1]);
    }
    if (tensor->op == GGML_OP_DSA_SPARSE_ATTN) {
        return glm_dsa_tensor_wraps_indexshare_top_k(tensor->src[4]);
    }
    if (tensor->op == GGML_OP_DSA_TOP1_ATTN) {
        return glm_dsa_tensor_wraps_indexshare_top_k(tensor->src[2]);
    }
    if (tensor->op == GGML_OP_GET_ROWS &&
            (strncmp(tensor->name, "dsa_compact_", strlen("dsa_compact_")) == 0 ||
             strncmp(tensor->name, "dsa_sparse_mask_topk", strlen("dsa_sparse_mask_topk")) == 0)) {
        return glm_dsa_tensor_wraps_indexshare_top_k(tensor->src[1]);
    }
    return false;
}

static bool glm_dsa_graph_op_counter_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    glm_dsa_graph_op_counts * counts = static_cast<glm_dsa_graph_op_counts *>(user_data);
    ++counts->total;
    if (!ask) {
        return true;
    }

    if (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
        ++counts->lightning_indexer;
    } else if (tensor->op == GGML_OP_TOP_K) {
        if (strncmp(tensor->name, "ffn_moe_topk", strlen("ffn_moe_topk")) != 0) {
            ++counts->top_k;
            counts->max_top_k_width = std::max(counts->max_top_k_width, tensor->ne[0]);
        }
    } else if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        ++counts->flash_attn;
        if (glm_dsa_tensor_name_starts_with(tensor->src[1], "dsa_compact_k") &&
                glm_dsa_tensor_name_starts_with(tensor->src[2], "dsa_compact_v")) {
            ++counts->compact_flash_attn;
            if (tensor->src[3] == nullptr) {
                ++counts->compact_flash_no_mask;
            }
            if (tensor->src[2]->op == GGML_OP_VIEW && tensor->src[2]->src[0] == tensor->src[1]) {
                ++counts->compact_flash_v_view;
            }
        }
    } else if (tensor->op == GGML_OP_DSA_SPARSE_MASK) {
        ++counts->sparse_mask;
    } else if (tensor->op == GGML_OP_DSA_SPARSE_ATTN) {
        ++counts->dsa_sparse_attn;
    }

    if (glm_dsa_tensor_consumes_indexshare_top_k(tensor)) {
        ++counts->top_k_consumers;
    }
    if (glm_dsa_tensor_name_starts_with(tensor, "dsa_compact_")) {
        ++counts->compact_get_rows;
        counts->max_compact_rows = std::max(counts->max_compact_rows, tensor->ne[1]);
        if (tensor->op == GGML_OP_GET_ROWS) {
            ++counts->compact_get_rows_ops;
            if (glm_dsa_tensor_name_starts_with(tensor, "dsa_compact_k")) {
                ++counts->compact_k_get_rows;
            } else if (glm_dsa_tensor_name_starts_with(tensor, "dsa_compact_v")) {
                ++counts->compact_v_get_rows;
            }
        } else if (tensor->op == GGML_OP_VIEW && glm_dsa_tensor_name_starts_with(tensor, "dsa_compact_v")) {
            ++counts->compact_v_views;
        }
    }
    if (strncmp(tensor->name, "dsa_sparse_mask", strlen("dsa_sparse_mask")) == 0) {
        ++counts->sparse_mask_nodes;
    }
    return true;
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode);

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_eval_user_data = nullptr,
        const llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    ctx_params.flash_attn_type = flash_attn_type;
    ctx_params.cb_eval = cb_eval;
    ctx_params.cb_eval_user_data = cb_eval_user_data;

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static bool test_glm_dsa_indexshare_graph(
        const size_t seed,
        const std::vector<ggml_backend_dev_t> & devs,
        const glm_dsa_indexshare_fixture fixture,
        const char * label,
        const int expected_producers = 1,
        const int expected_consumers = 2) {
    try {
        gguf_context_ptr gguf_ctx = get_gguf_ctx(
                LLM_ARCH_GLM_DSA,
                true,
                fixture);
        glm_dsa_graph_op_counts counts;
        auto model_ctx = get_model_and_ctx(
                gguf_ctx.get(),
                nullptr,
                seed,
                devs,
                LLAMA_SPLIT_MODE_LAYER,
                false,
                glm_dsa_graph_op_counter_callback,
                &counts);

        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ctx.first.get()));
        const std::vector<llama_token> tokens = get_tokens(8, n_vocab, seed);
        get_logits(model_ctx.first.get(), model_ctx.second.get(), tokens, false);

        if (counts.top_k != expected_producers ||
                counts.lightning_indexer != expected_producers ||
                counts.top_k_consumers != expected_consumers) {
            fprintf(stderr, "GLM_DSA %s graph counted top_k=%d lightning_indexer=%d top_k_consumers=%d\n",
                    label, counts.top_k, counts.lightning_indexer, counts.top_k_consumers);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA %s graph test failed: %s\n", label, err.what());
        return false;
    }
}

static bool test_glm_dsa_full_shared_indexshare_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    return test_glm_dsa_indexshare_graph(
            seed,
            devs,
            glm_dsa_indexshare_fixture::FULL_SHARED,
            "Full->Shared",
            1,
            2);
}

static bool test_glm_dsa_full_shared_shared_indexshare_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    return test_glm_dsa_indexshare_graph(
            seed,
            devs,
            glm_dsa_indexshare_fixture::FULL_SHARED_SHARED,
            "Full->Shared->Shared",
            1,
            3);
}

static bool test_glm_dsa_periodic_full_shared_indexshare_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    return test_glm_dsa_indexshare_graph(
            seed,
            devs,
            glm_dsa_indexshare_fixture::PERIODIC_FULL_SHARED,
            "Full->Shared->Shared->Full",
            2,
            4);
}

static void decode_tokens(llama_context * lctx, const std::vector<llama_token> & tokens, llama_pos start_pos, bool output_all = true) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const bool output = output_all || i + 1 == tokens.size();
        common_batch_add(batch, tokens[i], start_pos + (llama_pos) i, {0}, output);
    }
    batch.n_tokens = tokens.size();
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }
    llama_batch_free(batch);
}

static glm_dsa_graph_op_counts run_glm_dsa_decode_graph_after_prefill(
        const size_t seed,
        const std::vector<ggml_backend_dev_t> & devs,
        uint32_t n_prefill = 16,
        uint32_t context_length = 128,
        uint32_t indexer_top_k = 8) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(
            LLM_ARCH_GLM_DSA,
            true,
            glm_dsa_indexshare_fixture::FULL_SHARED,
            context_length,
            indexer_top_k);
    glm_dsa_graph_op_counts counts;
    auto model_ctx = get_model_and_ctx(
            gguf_ctx.get(),
            nullptr,
            seed,
            devs,
            LLAMA_SPLIT_MODE_LAYER,
            false,
            glm_dsa_graph_op_counter_callback,
            &counts,
            LLAMA_FLASH_ATTN_TYPE_ENABLED);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ctx.first.get()));
    const std::vector<llama_token> tokens = get_tokens(n_prefill + 1, n_vocab, seed);
    decode_tokens(model_ctx.second.get(), std::vector<llama_token>(tokens.begin(), tokens.begin() + n_prefill), 0);

    counts = {};
    decode_tokens(model_ctx.second.get(), std::vector<llama_token>{tokens[n_prefill]}, n_prefill);

    return counts;
}

static std::vector<ggml_backend_dev_t> glm_dsa_native_sparse_devs() {
    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_GPU:
            case GGML_BACKEND_DEVICE_TYPE_IGPU:
            case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                return {dev};
            case GGML_BACKEND_DEVICE_TYPE_CPU:
            case GGML_BACKEND_DEVICE_TYPE_META:
                break;
        }
    }
    return {};
}

static glm_dsa_graph_op_counts run_glm_dsa_verify_graph_after_prefill(
        const size_t seed,
        const std::vector<ggml_backend_dev_t> & devs,
        uint32_t n_prefill = 16,
        uint32_t n_verify = 4,
        uint32_t context_length = 128,
        uint32_t indexer_top_k = 8) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(
            LLM_ARCH_GLM_DSA,
            true,
            glm_dsa_indexshare_fixture::FULL_SHARED,
            context_length,
            indexer_top_k);
    glm_dsa_graph_op_counts counts;
    auto model_ctx = get_model_and_ctx(
            gguf_ctx.get(),
            nullptr,
            seed,
            devs,
            LLAMA_SPLIT_MODE_LAYER,
            false,
            glm_dsa_graph_op_counter_callback,
            &counts,
            LLAMA_FLASH_ATTN_TYPE_ENABLED);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ctx.first.get()));
    const std::vector<llama_token> tokens = get_tokens(n_prefill + n_verify, n_vocab, seed);
    decode_tokens(model_ctx.second.get(), std::vector<llama_token>(tokens.begin(), tokens.begin() + n_prefill), 0, false);

    counts = {};
    decode_tokens(model_ctx.second.get(), std::vector<llama_token>(tokens.begin() + n_prefill, tokens.end()), n_prefill, true);

    return counts;
}

static glm_dsa_graph_op_counts run_glm_dsa_prefill_graph(
        const size_t seed,
        const std::vector<ggml_backend_dev_t> & devs,
        uint32_t n_tokens,
        uint32_t context_length = 128,
        uint32_t indexer_top_k = 8,
        bool output_all = true) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(
            LLM_ARCH_GLM_DSA,
            true,
            glm_dsa_indexshare_fixture::FULL_SHARED,
            context_length,
            indexer_top_k);
    glm_dsa_graph_op_counts counts;
    auto model_ctx = get_model_and_ctx(
            gguf_ctx.get(),
            nullptr,
            seed,
            devs,
            LLAMA_SPLIT_MODE_LAYER,
            false,
            glm_dsa_graph_op_counter_callback,
            &counts,
            LLAMA_FLASH_ATTN_TYPE_ENABLED);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ctx.first.get()));
    const std::vector<llama_token> tokens = get_tokens(n_tokens, n_vocab, seed);
    decode_tokens(model_ctx.second.get(), tokens, 0, output_all);

    return counts;
}

static glm_dsa_graph_op_counts run_glm_dsa_prefill_graph_after_prefill(
        const size_t seed,
        const std::vector<ggml_backend_dev_t> & devs,
        uint32_t n_prefill,
        uint32_t n_tokens,
        uint32_t context_length,
        uint32_t indexer_top_k) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(
            LLM_ARCH_GLM_DSA,
            true,
            glm_dsa_indexshare_fixture::FULL_SHARED,
            context_length,
            indexer_top_k);
    glm_dsa_graph_op_counts counts;
    auto model_ctx = get_model_and_ctx(
            gguf_ctx.get(),
            nullptr,
            seed,
            devs,
            LLAMA_SPLIT_MODE_LAYER,
            false,
            glm_dsa_graph_op_counter_callback,
            &counts,
            LLAMA_FLASH_ATTN_TYPE_ENABLED);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ctx.first.get()));
    const std::vector<llama_token> tokens = get_tokens(n_prefill + n_tokens, n_vocab, seed);
    decode_tokens(model_ctx.second.get(), std::vector<llama_token>(tokens.begin(), tokens.begin() + n_prefill), 0, false);

    counts = {};
    decode_tokens(model_ctx.second.get(), std::vector<llama_token>(tokens.begin() + n_prefill, tokens.end()), n_prefill, false);

    return counts;
}

static bool validate_glm_dsa_compact_decode_graph_counts(
        const char * label,
        const glm_dsa_graph_op_counts & counts,
        int64_t expected_top_k_width = 0) {
    const bool top_k_width_ok = expected_top_k_width <= 0 ||
            (counts.top_k > 0 &&
             counts.max_top_k_width == expected_top_k_width &&
             counts.max_compact_rows == expected_top_k_width);
    const bool compact_kv_ok =
            counts.compact_k_get_rows > 0 &&
            (counts.compact_v_views > 0 || counts.compact_v_get_rows > 0) &&
            counts.top_k_consumers > 0;
    const bool flash_ok =
            counts.compact_flash_attn > 0 &&
            counts.compact_flash_no_mask > 0 &&
            counts.flash_attn > 0;
    const bool dense_sparse_absent =
            counts.sparse_mask == 0 &&
            counts.sparse_mask_nodes == 0 &&
            counts.dsa_sparse_attn == 0;

    if (!top_k_width_ok || !compact_kv_ok || !flash_ok || !dense_sparse_absent) {
        fprintf(stderr,
                "GLM_DSA %s compact decode graph counted total=%d top_k=%d max_top_k_width=%lld top_k_consumers=%d compact_get_rows=%d compact_get_rows_ops=%d compact_k_get_rows=%d compact_v_get_rows=%d compact_v_views=%d max_compact_rows=%lld flash_attn=%d compact_flash_attn=%d compact_flash_no_mask=%d compact_flash_v_view=%d sparse_mask=%d sparse_mask_nodes=%d dsa_sparse_attn=%d expected_top_k_width=%lld\n",
                label,
                counts.total,
                counts.top_k,
                (long long) counts.max_top_k_width,
                counts.top_k_consumers,
                counts.compact_get_rows,
                counts.compact_get_rows_ops,
                counts.compact_k_get_rows,
                counts.compact_v_get_rows,
                counts.compact_v_views,
                (long long) counts.max_compact_rows,
                counts.flash_attn,
                counts.compact_flash_attn,
                counts.compact_flash_no_mask,
                counts.compact_flash_v_view,
                counts.sparse_mask,
                counts.sparse_mask_nodes,
                counts.dsa_sparse_attn,
                (long long) expected_top_k_width);
        return false;
    }

    if (test_env_enabled("LLAMA_GLM_DSA_PRINT_COMPACT_DECODE_COUNTS")) {
        fprintf(stderr,
                "GLM_DSA compact-decode graph validated label=%s total=%d top_k=%d max_top_k_width=%lld top_k_consumers=%d compact_get_rows=%d compact_get_rows_ops=%d compact_k_get_rows=%d compact_v_get_rows=%d compact_v_views=%d max_compact_rows=%lld flash_attn=%d compact_flash_attn=%d compact_flash_no_mask=%d compact_flash_v_view=%d sparse_mask=%d sparse_mask_nodes=%d dsa_sparse_attn=%d expected_top_k_width=%lld\n",
                label,
                counts.total,
                counts.top_k,
                (long long) counts.max_top_k_width,
                counts.top_k_consumers,
                counts.compact_get_rows,
                counts.compact_get_rows_ops,
                counts.compact_k_get_rows,
                counts.compact_v_get_rows,
                counts.compact_v_views,
                (long long) counts.max_compact_rows,
                counts.flash_attn,
                counts.compact_flash_attn,
                counts.compact_flash_no_mask,
                counts.compact_flash_v_view,
                counts.sparse_mask,
                counts.sparse_mask_nodes,
                counts.dsa_sparse_attn,
                (long long) expected_top_k_width);
    }

    return true;
}

static bool test_glm_dsa_direct_sparse_decode_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        if (devs.empty()) {
            fprintf(stderr, "GLM_DSA direct sparse decode graph test skipped: no native sparse backend device available\n");
            return true;
        }
        scoped_test_env direct_decode_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K", "1024");
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        const glm_dsa_graph_op_counts counts = run_glm_dsa_decode_graph_after_prefill(seed, devs);

        if (counts.dsa_sparse_attn == 0 || counts.compact_get_rows != 0 || counts.sparse_mask != 0 || counts.sparse_mask_nodes != 0) {
            fprintf(stderr,
                    "GLM_DSA direct sparse decode graph counted total=%d dsa_sparse_attn=%d compact_get_rows=%d sparse_mask=%d sparse_mask_nodes=%d flash_attn=%d\n",
                    counts.total, counts.dsa_sparse_attn, counts.compact_get_rows, counts.sparse_mask, counts.sparse_mask_nodes, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA direct sparse decode graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_compact_decode_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env compact_decode_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K", "4");
        scoped_test_env compact_min_kv("LLAMA_GLM_DSA_COMPACT_FLASH_MIN_KV", "1");
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        const glm_dsa_graph_op_counts counts = run_glm_dsa_decode_graph_after_prefill(seed, devs);

        if (!validate_glm_dsa_compact_decode_graph_counts("compact", counts)) {
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA compact decode graph test failed: %s\n", err.what());
        return false;
    }
}

static std::vector<ggml_backend_dev_t> glm_dsa_cpu_devs() {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error("CPU backend device is unavailable");
    }
    return {cpu_dev};
}

static bool test_glm_dsa_cpu_compact_decode_graph(const size_t seed) {
    try {
        scoped_test_env compact_decode_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K", "4");
        scoped_test_env compact_min_kv("LLAMA_GLM_DSA_COMPACT_FLASH_MIN_KV", "1");
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        const glm_dsa_graph_op_counts counts = run_glm_dsa_decode_graph_after_prefill(seed, glm_dsa_cpu_devs());

        if (!validate_glm_dsa_compact_decode_graph_counts("CPU compact", counts)) {
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA CPU compact decode graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_cpu_direct_sparse_decode_falls_back_graph(const size_t seed) {
    try {
        scoped_test_env direct_decode_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K", "1024");
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        const glm_dsa_graph_op_counts counts = run_glm_dsa_decode_graph_after_prefill(seed, glm_dsa_cpu_devs());

        if (counts.sparse_mask_nodes == 0 || counts.dsa_sparse_attn != 0 || counts.compact_get_rows != 0) {
            fprintf(stderr,
                    "GLM_DSA CPU direct-sparse fallback graph counted total=%d sparse_mask=%d sparse_mask_nodes=%d dsa_sparse_attn=%d compact_get_rows=%d flash_attn=%d\n",
                    counts.total, counts.sparse_mask, counts.sparse_mask_nodes, counts.dsa_sparse_attn, counts.compact_get_rows, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA CPU direct-sparse fallback graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_compact_min_kv_decode_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env compact_decode_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K", "4");
        scoped_test_env compact_min_kv("LLAMA_GLM_DSA_COMPACT_FLASH_MIN_KV", "1024");
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        const glm_dsa_graph_op_counts counts = run_glm_dsa_decode_graph_after_prefill(seed, devs);

        if (counts.sparse_mask_nodes == 0 || counts.compact_get_rows != 0 || counts.dsa_sparse_attn != 0) {
            fprintf(stderr,
                    "GLM_DSA compact min-kv decode graph counted total=%d sparse_mask=%d sparse_mask_nodes=%d compact_get_rows=%d dsa_sparse_attn=%d flash_attn=%d\n",
                    counts.total, counts.sparse_mask, counts.sparse_mask_nodes, counts.compact_get_rows, counts.dsa_sparse_attn, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA compact min-kv decode graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_default_compact_decode_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        // GLM-5.2's native top_k=768 exceeds the default direct sparse decode
        // cap, so default decode policy should select compact selected-KV flash.
        const glm_dsa_graph_op_counts counts = run_glm_dsa_decode_graph_after_prefill(seed, devs, 768, 1024, 768);

        if (!validate_glm_dsa_compact_decode_graph_counts("default", counts, 768)) {
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA default compact decode graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_short_history_compact_decode_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env compact_decode_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K", "4");
        scoped_test_env compact_min_kv("LLAMA_GLM_DSA_COMPACT_FLASH_MIN_KV", "1");
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        // During one-token decode, IndexShare top-k is selected from the previous
        // KV state while attention sees previous+current KV. Even at short
        // history, the forced compact policy should exercise selected-KV flash.
        const glm_dsa_graph_op_counts counts = run_glm_dsa_decode_graph_after_prefill(seed, devs, 4);

        if (!validate_glm_dsa_compact_decode_graph_counts("short-history", counts)) {
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA short-history compact decode graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_native_top_k_compact_decode_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        // GLM-5.2 uses top_k=768. After a 768-token prefix, the next token sees
        // 769 visible KV entries while IndexShare produces 768 selected rows.
        // This is the model-native compact selected-KV decode shape.
        const glm_dsa_graph_op_counts counts = run_glm_dsa_decode_graph_after_prefill(seed, devs, 768, 1024, 768);

        if (!validate_glm_dsa_compact_decode_graph_counts("native top-k", counts, 768)) {
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA native top-k compact decode graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_compact_disabled_decode_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env compact_decode_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K", "4");
        scoped_test_env compact_disabled("LLAMA_GLM_DSA_DISABLE_COMPACT_FLASH_ATTN", "1");
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        const glm_dsa_graph_op_counts counts = run_glm_dsa_decode_graph_after_prefill(seed, devs);

        if (counts.sparse_mask_nodes == 0 || counts.compact_get_rows != 0 || counts.dsa_sparse_attn != 0) {
            fprintf(stderr,
                    "GLM_DSA compact-disabled decode graph counted total=%d sparse_mask=%d sparse_mask_nodes=%d compact_get_rows=%d dsa_sparse_attn=%d flash_attn=%d\n",
                    counts.total, counts.sparse_mask, counts.sparse_mask_nodes, counts.compact_get_rows, counts.dsa_sparse_attn, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA compact-disabled decode graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_verify_dense_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        scoped_test_env prefill_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_PREFILL_MAX_TOKENS", "8");
        const glm_dsa_graph_op_counts counts = run_glm_dsa_verify_graph_after_prefill(seed, devs);

        if (counts.sparse_mask_nodes == 0 || counts.compact_get_rows != 0 || counts.dsa_sparse_attn != 0) {
            fprintf(stderr,
                    "GLM_DSA verification graph counted total=%d dsa_sparse_attn=%d compact_get_rows=%d sparse_mask=%d sparse_mask_nodes=%d flash_attn=%d\n",
                    counts.total, counts.dsa_sparse_attn, counts.compact_get_rows, counts.sparse_mask, counts.sparse_mask_nodes, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA verification dense graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_native_top_k_verify_dense_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        scoped_test_env prefill_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_PREFILL_MAX_TOKENS", "8");
        // Verification requests logits for the whole candidate span. Even with
        // GLM-5.2's native top_k=768 shape, this phase is intentionally kept on
        // the dense/auto path until verifier-specific sparse parity is proven.
        const glm_dsa_graph_op_counts counts = run_glm_dsa_verify_graph_after_prefill(seed, devs, 768, 4, 1024, 768);

        if (counts.top_k == 0 || counts.max_top_k_width != 768 ||
                counts.sparse_mask_nodes == 0 || counts.compact_get_rows != 0 || counts.dsa_sparse_attn != 0) {
            fprintf(stderr,
                    "GLM_DSA native top-k verification graph counted total=%d top_k=%d max_top_k_width=%lld dsa_sparse_attn=%d compact_get_rows=%d sparse_mask=%d sparse_mask_nodes=%d flash_attn=%d\n",
                    counts.total, counts.top_k, (long long) counts.max_top_k_width, counts.dsa_sparse_attn, counts.compact_get_rows,
                    counts.sparse_mask, counts.sparse_mask_nodes, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA native top-k verification dense graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_short_prefill_dense_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        scoped_test_env prefill_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_PREFILL_MAX_TOKENS", "8");
        const glm_dsa_graph_op_counts counts = run_glm_dsa_prefill_graph(seed, devs, 8);

        if (counts.sparse_mask_nodes == 0 || counts.compact_get_rows != 0 || counts.dsa_sparse_attn != 0) {
            fprintf(stderr,
                    "GLM_DSA short prefill graph counted total=%d dsa_sparse_attn=%d compact_get_rows=%d sparse_mask=%d sparse_mask_nodes=%d flash_attn=%d\n",
                    counts.total, counts.dsa_sparse_attn, counts.compact_get_rows, counts.sparse_mask, counts.sparse_mask_nodes, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA short prefill dense graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_native_top_k_short_prefill_dense_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        scoped_test_env prefill_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_PREFILL_MAX_TOKENS", "2048");
        // GLM-5.2's package policy keeps short prefill dense below the
        // threshold. Use output_all=false so this remains prefill rather than
        // speculative verification, while still exercising native top_k=768.
        const glm_dsa_graph_op_counts counts = run_glm_dsa_prefill_graph(seed, devs, 768, 1024, 768, false);

        if (counts.top_k == 0 || counts.max_top_k_width != 768 ||
                counts.sparse_mask_nodes == 0 || counts.compact_get_rows != 0 || counts.dsa_sparse_attn != 0) {
            fprintf(stderr,
                    "GLM_DSA native top-k short prefill graph counted total=%d top_k=%d max_top_k_width=%lld dsa_sparse_attn=%d compact_get_rows=%d sparse_mask=%d sparse_mask_nodes=%d flash_attn=%d\n",
                    counts.total, counts.top_k, (long long) counts.max_top_k_width, counts.dsa_sparse_attn, counts.compact_get_rows,
                    counts.sparse_mask, counts.sparse_mask_nodes, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA native top-k short prefill dense graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_long_prefill_dense_mask_guard_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        if (devs.empty()) {
            fprintf(stderr, "GLM_DSA long prefill dense-mask guard graph test skipped: no native sparse backend device available\n");
            return true;
        }
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        scoped_test_env direct_sparse_prefill("LLAMA_GLM_DSA_ENABLE_DIRECT_SPARSE_PREFILL", "1");
        scoped_test_env prefill_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_PREFILL_MAX_TOKENS", "8");
        scoped_test_env dense_mask_limit("LLAMA_GLM_DSA_DENSE_SPARSE_MASK_MAX_BYTES", "1");
        scoped_test_env large_prefill_enabled("LLAMA_GLM_DSA_ENABLE_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL", "1");
        const glm_dsa_graph_op_counts counts = run_glm_dsa_prefill_graph(seed, devs, 16);

        if (counts.dsa_sparse_attn == 0 || counts.compact_get_rows != 0 || counts.sparse_mask != 0 || counts.sparse_mask_nodes != 0) {
            fprintf(stderr,
                    "GLM_DSA long prefill dense-mask guard graph counted total=%d dsa_sparse_attn=%d compact_get_rows=%d sparse_mask=%d sparse_mask_nodes=%d flash_attn=%d\n",
                    counts.total, counts.dsa_sparse_attn, counts.compact_get_rows, counts.sparse_mask, counts.sparse_mask_nodes, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA long prefill dense-mask guard graph test failed: %s\n", err.what());
        return false;
    }
}

static bool test_glm_dsa_native_top_k_long_prefill_dense_mask_guard_graph(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devs) {
    try {
        if (devs.empty()) {
            fprintf(stderr, "GLM_DSA native top-k long prefill dense-mask guard graph test skipped: no native sparse backend device available\n");
            return true;
        }
        scoped_test_env graph_reuse_disabled("LLAMA_GRAPH_REUSE_DISABLE", "1");
        scoped_test_env direct_sparse_prefill("LLAMA_GLM_DSA_ENABLE_DIRECT_SPARSE_PREFILL", "1");
        scoped_test_env prefill_cap("LLAMA_GLM_DSA_DIRECT_SPARSE_PREFILL_MAX_TOKENS", "8");
        scoped_test_env dense_mask_limit("LLAMA_GLM_DSA_DENSE_SPARSE_MASK_MAX_BYTES", "1");
        scoped_test_env large_prefill_enabled("LLAMA_GLM_DSA_ENABLE_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL", "1");
        // GLM-5.2's configured top_k=768 would otherwise require a large dense
        // sparse mask during long-context prefill. Warm the KV first, then
        // assert the measured continuation selects native direct sparse
        // attention and avoids materializing that mask.
        const glm_dsa_graph_op_counts counts = run_glm_dsa_prefill_graph_after_prefill(seed, devs, 768, 64, 1024, 768);

        if (counts.top_k == 0 || counts.max_top_k_width != 768 || counts.dsa_sparse_attn == 0 ||
                counts.compact_get_rows != 0 || counts.sparse_mask != 0 || counts.sparse_mask_nodes != 0) {
            fprintf(stderr,
                    "GLM_DSA native top-k long prefill dense-mask guard graph counted total=%d top_k=%d max_top_k_width=%lld dsa_sparse_attn=%d compact_get_rows=%d sparse_mask=%d sparse_mask_nodes=%d flash_attn=%d\n",
                    counts.total, counts.top_k, (long long) counts.max_top_k_width, counts.dsa_sparse_attn, counts.compact_get_rows,
                    counts.sparse_mask, counts.sparse_mask_nodes, counts.flash_attn);
            return false;
        }
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "GLM_DSA native top-k long prefill dense-mask guard graph test failed: %s\n", err.what());
        return false;
    }
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    if (arch == LLM_ARCH_DEEPSEEK4) {
        return false;
    }

    // FIXME some models are segfaulting with WebGPU:
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    return true;
}

static bool glm_dsa_backend_config_supported(const std::vector<ggml_backend_dev_t> & devs) {
    for (ggml_backend_dev_t dev : devs) {
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
            case GGML_BACKEND_DEVICE_TYPE_GPU:
            case GGML_BACKEND_DEVICE_TYPE_IGPU:
                break;
            default:
                return false;
        }
    }
    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    if (target_arch == LLM_ARCH_UNKNOWN || target_arch == LLM_ARCH_GLM_DSA) {
        const bool glm_dsa_contract_ok =
                test_glm_dsa_rejects_conflicting_indexshare_metadata(seed) &&
                test_glm_dsa_rejects_shared_first_indexshare_metadata(seed) &&
                test_glm_dsa_full_shared_indexshare_graph(seed, {}) &&
                test_glm_dsa_full_shared_shared_indexshare_graph(seed, {}) &&
                test_glm_dsa_periodic_full_shared_indexshare_graph(seed, {}) &&
                test_glm_dsa_direct_sparse_decode_graph(seed, glm_dsa_native_sparse_devs()) &&
                test_glm_dsa_compact_decode_graph(seed, glm_dsa_cpu_devs()) &&
                test_glm_dsa_cpu_compact_decode_graph(seed) &&
                test_glm_dsa_cpu_direct_sparse_decode_falls_back_graph(seed) &&
                test_glm_dsa_compact_min_kv_decode_graph(seed, glm_dsa_cpu_devs()) &&
                test_glm_dsa_default_compact_decode_graph(seed, glm_dsa_cpu_devs()) &&
                test_glm_dsa_short_history_compact_decode_graph(seed, glm_dsa_cpu_devs()) &&
                test_glm_dsa_native_top_k_compact_decode_graph(seed, glm_dsa_cpu_devs()) &&
                test_glm_dsa_compact_disabled_decode_graph(seed, glm_dsa_cpu_devs()) &&
                test_glm_dsa_verify_dense_graph(seed, {}) &&
                test_glm_dsa_native_top_k_verify_dense_graph(seed, {}) &&
                test_glm_dsa_short_prefill_dense_graph(seed, {}) &&
                test_glm_dsa_native_top_k_short_prefill_dense_graph(seed, {}) &&
                test_glm_dsa_long_prefill_dense_mask_guard_graph(seed, glm_dsa_native_sparse_devs()) &&
                test_glm_dsa_native_top_k_long_prefill_dense_mask_guard_graph(seed, glm_dsa_native_sparse_devs());
        if (!glm_dsa_contract_ok) {
            fprintf(stderr, "GLM_DSA IndexShare metadata contract test failed\n");
            all_ok = false;
        }
    }
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (arch == LLM_ARCH_GLM_DSA && !glm_dsa_backend_config_supported(dc.devs)) {
                    skip = true;
                }
#if defined(GGML_USE_WEBGPU)
                skip = true; // FIXME
#endif // GGML_USE_WEBGPU
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // FIXME these tests are disabled in the CI for macOS-latest-cmake-arm64 because they are segfaulting
    common_init();
    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;
    std::string out;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
            continue;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (!out.empty()) {
            return save_models(arch, seed, log_level, out);
        }
        return test_backends(arch, seed, log_level);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
