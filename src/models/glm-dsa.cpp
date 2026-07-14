#include "models.h"

#include "glm-dsa-block.h"

#include "llama-kv-cache-dsa.h"

#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

static bool llama_glm_dsa_disable_lightning_indexer() {
    const char * value = getenv("LLAMA_GLM_DSA_DISABLE_LIGHTNING_INDEXER");
    return value != nullptr && atoi(value) != 0;
}

static bool llama_glm_dsa_allow_indexshare_tensor_fallback() {
    const char * value = getenv("LLAMA_GLM_DSA_ALLOW_INDEXSHARE_TENSOR_FALLBACK");
    return value != nullptr && atoi(value) != 0;
}

static bool llama_glm_dsa_indexshare_exec_log_enabled() {
    const char * value = getenv("LLAMA_GLM_DSA_INDEXSHARE_EXEC_LOG");
    return value != nullptr && atoi(value) != 0;
}

static bool llama_glm_dsa_block_contract_log_enabled() {
    const char * value = getenv("LLAMA_GLM_DSA_BLOCK_CONTRACT_LOG");
    return value != nullptr && atoi(value) != 0;
}

static bool llama_glm_dsa_roofline_bypass_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && atoi(value) != 0;
}

static bool llama_glm_dsa_layer_has_indexer(const llama_layer & layer) {
    const bool has_any =
            layer.indexer_k_norm   ||
            layer.indexer_k_norm_b ||
            layer.indexer_proj     ||
            layer.indexer_attn_k   ||
            layer.indexer_attn_q_b;

    const bool has_all =
            layer.indexer_k_norm   &&
            layer.indexer_k_norm_b &&
            layer.indexer_proj     &&
            layer.indexer_attn_k   &&
            layer.indexer_attn_q_b;

    if (has_any != has_all) {
        throw std::runtime_error("GLM_DSA layer has incomplete indexer tensors");
    }

    return has_all;
}

static const char * llama_glm_dsa_indexshare_role_source(const llama_hparams & hparams) {
    if (hparams.indexer_types_present) {
        return "metadata_types";
    }
    if (hparams.indexer_top_k_freq > 0) {
        return "metadata_frequency";
    }
    return "tensor_fallback";
}

static int llama_glm_dsa_indexshare_freq() {
    const char * value = getenv("LLAMA_GLM_DSA_INDEXSHARE_FREQ");
    if (value == nullptr) {
        return 1;
    }

    const int freq = atoi(value);
    return freq > 0 ? freq : 1;
}

static bool llama_glm_dsa_indexshare_pattern_layer_is_full(const char * pattern, int il, bool * matched) {
    *matched = false;
    if (pattern == nullptr || pattern[0] == '\0') {
        return false;
    }

    int layer_index = 0;
    for (const char * p = pattern; *p != '\0'; ++p) {
        const char value = static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
        if (value != 'F' && value != 'S') {
            continue;
        }
        if (layer_index == il) {
            *matched = true;
            return value == 'F';
        }
        ++layer_index;
    }

    return false;
}

static int32_t llama_glm_dsa_indexer_type_from_string(const std::string & value) {
    if (value == "full") {
        return 1;
    }
    if (value == "shared") {
        return 0;
    }

    throw std::runtime_error("GLM_DSA attention.indexer.types values must be \"full\" or \"shared\"");
}

static bool llama_glm_dsa_frequency_layer_is_full(uint32_t layer, uint32_t offset, uint32_t freq) {
    return layer < offset || (layer >= offset && ((layer - offset + 1) % freq) == 0);
}

static void llama_glm_dsa_validate_hparams(const llama_hparams & hparams) {
    const uint32_t n_rot = hparams.n_rot();
    if (hparams.n_layer() == 0) {
        throw std::runtime_error("GLM_DSA effective decoder layer count must be positive");
    }
    if (n_rot == 0 || (n_rot % 2) != 0) {
        throw std::runtime_error("GLM_DSA rope.dimension_count must be positive and even");
    }
    if (hparams.n_embd_head_k_mla() <= n_rot) {
        throw std::runtime_error("GLM_DSA attention.key_length_mla must be greater than rope.dimension_count");
    }
    if (hparams.n_embd_head_v_mla() == 0) {
        throw std::runtime_error("GLM_DSA attention.value_length_mla must be positive");
    }
    if (hparams.n_lora_q == 0 || hparams.n_lora_kv == 0) {
        throw std::runtime_error("GLM_DSA q_lora_rank and kv_lora_rank must be positive");
    }
    if (hparams.n_expert == 0) {
        throw std::runtime_error("GLM_DSA expert_count must be positive");
    }
    if (hparams.n_expert_used == 0) {
        throw std::runtime_error("GLM_DSA expert_used_count must be positive");
    }
    if (hparams.n_expert_used > hparams.n_expert) {
        throw std::runtime_error("GLM_DSA expert_used_count must be less than or equal to expert_count");
    }
    if (hparams.n_expert_shared == 0) {
        throw std::runtime_error("GLM_DSA expert_shared_count must be positive");
    }
    if (hparams.n_ff_exp == 0) {
        throw std::runtime_error("GLM_DSA expert_feed_forward_length must be positive");
    }
    if (hparams.n_layer_dense_lead >= hparams.n_layer()) {
        throw std::runtime_error("GLM_DSA leading_dense_block_count must be less than effective decoder layer count");
    }
    if (hparams.indexer_n_head == 0) {
        throw std::runtime_error("GLM_DSA attention.indexer.head_count must be positive");
    }
    if (hparams.indexer_head_size == 0) {
        throw std::runtime_error("GLM_DSA attention.indexer.key_length must be positive");
    }
    if (hparams.indexer_head_size <= n_rot) {
        throw std::runtime_error("GLM_DSA attention.indexer.key_length must be greater than rope.dimension_count");
    }
    if (hparams.indexer_top_k == 0) {
        throw std::runtime_error("GLM_DSA attention.indexer.top_k must be positive");
    }
}

static void llama_glm_dsa_validate_indexer_type_frequency_consistency(const llama_hparams & hparams) {
    if (!hparams.indexer_types_present || hparams.indexer_top_k_freq == 0) {
        return;
    }

    int conflict_count = 0;
    int first_conflict = -1;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const bool role_full = hparams.indexer_types[il] == 1;
        const bool freq_full = llama_glm_dsa_frequency_layer_is_full(
                il,
                hparams.indexer_skip_top_k_offset,
                hparams.indexer_top_k_freq);
        if (role_full != freq_full) {
            ++conflict_count;
            if (first_conflict < 0) {
                first_conflict = (int) il;
            }
        }
    }

    if (conflict_count > 0) {
        std::ostringstream message;
        message << "GLM_DSA attention.indexer.types conflicts with attention.indexer.top_k_frequency at "
                << conflict_count << " layer(s), first=" << first_conflict;
        throw std::runtime_error(message.str());
    }
}

static bool llama_glm_dsa_metadata_layer_is_full(const llama_hparams & hparams, int il, bool * matched) {
    *matched = false;
    if (il < 0 || (uint32_t) il >= hparams.n_layer()) {
        return false;
    }

    if (hparams.indexer_types_present) {
        const int32_t role = hparams.indexer_types[il];
        if (role < 0) {
            throw std::runtime_error("GLM_DSA attention.indexer.types has an unset layer role");
        }
        *matched = true;
        return role == 1;
    }

    if (hparams.indexer_top_k_freq > 0) {
        const uint32_t layer = (uint32_t) il;
        *matched = true;
        return llama_glm_dsa_frequency_layer_is_full(
                layer,
                hparams.indexer_skip_top_k_offset,
                hparams.indexer_top_k_freq);
    }

    return false;
}

static void llama_glm_dsa_validate_indexshare_sequence(const llama_hparams & hparams) {
    bool saw_full = false;
    bool saw_role = false;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        bool metadata_matched = false;
        const bool full_layer = llama_glm_dsa_metadata_layer_is_full(hparams, il, &metadata_matched);
        if (!metadata_matched) {
            continue;
        }

        saw_role = true;
        if (full_layer) {
            saw_full = true;
            continue;
        }

        if (!saw_full) {
            std::ostringstream message;
            message << "GLM_DSA IndexShare Shared layer " << il << " has no preceding Full layer";
            throw std::runtime_error(message.str());
        }
    }

    if (saw_role && !saw_full) {
        throw std::runtime_error("GLM_DSA IndexShare metadata must declare at least one Full layer");
    }
}

static void llama_glm_dsa_validate_indexshare_layer_contract(int il, const llama_hparams & hparams, const llama_layer & layer) {
    const bool has_indexer = llama_glm_dsa_layer_has_indexer(layer);

    bool metadata_matched = false;
    const bool metadata_full = llama_glm_dsa_metadata_layer_is_full(hparams, il, &metadata_matched);
    if (!metadata_matched) {
        return;
    }

    if (metadata_full && !has_indexer) {
        std::ostringstream message;
        message << "GLM_DSA IndexShare metadata declares Full layer " << il << " without indexer tensors";
        throw std::runtime_error(message.str());
    }
    if (!metadata_full && has_indexer) {
        std::ostringstream message;
        message << "GLM_DSA IndexShare metadata declares Shared layer " << il << " with indexer tensors";
        throw std::runtime_error(message.str());
    }
}

static void llama_glm_dsa_validate_nextn_indexer_contract(int il, const llama_hparams & hparams, const llama_layer & layer) {
    if ((uint32_t) il < hparams.n_layer()) {
        return;
    }

    if (!llama_glm_dsa_layer_has_indexer(layer)) {
        std::ostringstream message;
        message << "GLM_DSA MTP/NextN layer " << il << " requires complete indexer tensors";
        throw std::runtime_error(message.str());
    }
}

static bool llama_glm_dsa_layer_uses_indexer(int il, const llama_hparams & hparams, const llama_layer & layer) {
    const bool has_indexer = llama_glm_dsa_layer_has_indexer(layer);

    bool metadata_matched = false;
    const bool metadata_full = llama_glm_dsa_metadata_layer_is_full(hparams, il, &metadata_matched);
    if (metadata_matched) {
        if (metadata_full && !has_indexer) {
            throw std::runtime_error("GLM_DSA IndexShare metadata selects a Full layer without indexer tensors");
        }
        return metadata_full;
    }

    bool pattern_matched = false;
    const bool pattern_full = llama_glm_dsa_indexshare_pattern_layer_is_full(
            getenv("LLAMA_GLM_DSA_INDEXSHARE_PATTERN"),
            il,
            &pattern_matched);
    if (pattern_matched) {
        if (pattern_full && !has_indexer) {
            throw std::runtime_error("GLM_DSA IndexShare pattern selects a Full layer without indexer tensors");
        }
        return pattern_full;
    }

    if (!has_indexer) {
        return false;
    }

    const int freq = llama_glm_dsa_indexshare_freq();
    return freq <= 1 || (il % freq) == 0;
}

static void llama_glm_dsa_log_indexshare_exec(
        int  il,
        bool full_layer,
        bool has_input_top_k) {
    if (!llama_glm_dsa_indexshare_exec_log_enabled()) {
        return;
    }

    LLAMA_LOG_INFO(
            "%s: GLM_DSA IndexShare exec layer=%d role=%s input_top_k=%d\n",
            __func__,
            il,
            full_layer ? "full" : "shared",
            has_input_top_k ? 1 : 0);
}

static void llama_glm_dsa_log_indexshare_top_k(int il, const ggml_tensor * top_k, const ggml_tensor * indexer_score) {
    if (!llama_glm_dsa_indexshare_exec_log_enabled()) {
        return;
    }

    LLAMA_LOG_INFO(
            "%s: GLM_DSA IndexShare top_k layer=%d source=indexer width=%lld score_width=%lld\n",
            __func__,
            il,
            (long long) (top_k ? top_k->ne[0] : 0),
            (long long) (indexer_score ? indexer_score->ne[0] : 0));
}

static void llama_glm_dsa_log_indexshare_consume(int il, const ggml_tensor * top_k) {
    if (!llama_glm_dsa_indexshare_exec_log_enabled()) {
        return;
    }

    LLAMA_LOG_INFO(
            "%s: GLM_DSA IndexShare consume layer=%d source=last_top_k width=%lld batch=%lld stream=%lld\n",
            __func__,
            il,
            (long long) (top_k ? top_k->ne[0] : 0),
            (long long) (top_k ? top_k->ne[1] : 0),
            (long long) (top_k ? top_k->ne[3] : 0));
}

static void llama_glm_dsa_log_block_plan(
        const llama_glm_dsa_block_plan & plan,
        int expected_group_size) {
    if (!llama_glm_dsa_block_contract_log_enabled()) {
        return;
    }

    for (const llama_glm_dsa_block_span & block : plan.blocks) {
        LLAMA_LOG_INFO(
            "%s: GLM_DSA block ordinal=%d full_layer=%d group=[%d,%d) execution=[%d,%d) repeating=%d expected_group_size=%d complete_execution=%d producer_in_execution=%d needs_input_top_k=%d\n",
            __func__,
            block.ordinal,
            block.full_layer,
            block.layer_begin,
            block.layer_end,
            block.execution_begin,
            block.execution_end,
            block.repeating_group ? 1 : 0,
            expected_group_size,
            block.complete_execution ? 1 : 0,
            block.producer_in_execution ? 1 : 0,
            block.needs_input_top_k ? 1 : 0);
    }
}

static void llama_glm_dsa_log_indexshare_contract(
        const llama_model & model,
        const llama_hparams & hparams) {
    if (!llama_glm_dsa_indexshare_exec_log_enabled()) {
        return;
    }

    int full_layers = 0;
    int shared_layers = 0;
    int total_indexer_tensor_layers = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        bool metadata_matched = false;
        bool full_layer = llama_glm_dsa_metadata_layer_is_full(hparams, il, &metadata_matched);
        if (!metadata_matched) {
            full_layer = llama_glm_dsa_layer_has_indexer(model.layers[il]);
        }

        if (full_layer) {
            ++full_layers;
        } else {
            ++shared_layers;
        }

        const bool has_indexer = llama_glm_dsa_layer_has_indexer(model.layers[il]);
        if (has_indexer) {
            ++total_indexer_tensor_layers;
        }

    }

    LLAMA_LOG_INFO(
            "%s: GLM_DSA IndexShare source=%s full_layers=%d shared_layers=%d indexer_tensor_layers=%d top_k=%u top_k_frequency=%u skip_top_k_offset=%u nextn_layers=%u\n",
            __func__,
            llama_glm_dsa_indexshare_role_source(hparams),
            full_layers,
            shared_layers,
            total_indexer_tensor_layers,
            hparams.indexer_top_k,
            hparams.indexer_top_k_freq,
            hparams.indexer_skip_top_k_offset,
            hparams.n_layer_nextn);
}

void llama_model_glm_dsa::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // deepseek MLA parameters
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);

    // DSA parameters
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    if (hparams.n_layer_nextn >= hparams.n_layer_all) {
        throw std::runtime_error("GLM_DSA nextn_predict_layers must be less than block_count");
    }
    llama_glm_dsa_validate_hparams(hparams);
    const bool has_indexer_top_k_freq =
            ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K_FREQUENCY, hparams.indexer_top_k_freq, false);
    const bool has_indexer_skip_top_k_offset =
            ml.get_key(LLM_KV_ATTENTION_INDEXER_SKIP_TOP_K_OFFSET, hparams.indexer_skip_top_k_offset, false);
    if (has_indexer_top_k_freq && hparams.indexer_top_k_freq == 0) {
        throw std::runtime_error("GLM_DSA attention.indexer.top_k_frequency must be positive when present");
    }
    if (has_indexer_top_k_freq && !has_indexer_skip_top_k_offset) {
        throw std::runtime_error("GLM_DSA attention.indexer.skip_top_k_offset is required when top_k_frequency is present");
    }

    std::vector<std::string> indexer_types;
    const bool has_indexer_types = ml.get_arr(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types, false);
    if (has_indexer_types) {
        if (indexer_types.size() != hparams.n_layer()) {
            throw std::runtime_error("GLM_DSA attention.indexer.types length must match effective decoder layer count");
        }
        hparams.indexer_types_present = true;
        for (size_t il = 0; il < indexer_types.size(); ++il) {
            hparams.indexer_types[il] = llama_glm_dsa_indexer_type_from_string(indexer_types[il]);
        }
    }
    if (!has_indexer_types && !has_indexer_top_k_freq && !llama_glm_dsa_allow_indexshare_tensor_fallback()) {
        throw std::runtime_error("GLM_DSA IndexShare metadata requires attention.indexer.types or attention.indexer.top_k_frequency");
    }
    llama_glm_dsa_validate_indexer_type_frequency_consistency(hparams);
    llama_glm_dsa_validate_indexshare_sequence(hparams);

    // Expert gating function (GLM-4.5 uses sigmoid)
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func =  LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    switch (hparams.n_layer()) {
        case 78: type = LLM_TYPE_744B_A40B; break; // GLM-5.2 target layers + 1 native MTP block
        case 79: type = LLM_TYPE_744B_A40B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_glm_dsa::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    uint32_t n_vocab_tensor = vocab.n_tokens();
    const bool has_vocab_size = ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab_tensor, false);
    if (!has_vocab_size) {
        const ggml_tensor * tok_embd_meta = ml.get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str());
        if (tok_embd_meta != nullptr && tok_embd_meta->ne[0] == n_embd && tok_embd_meta->ne[1] > n_vocab_tensor) {
            n_vocab_tensor = tok_embd_meta->ne[1];
        }
    }

    const bool is_mla = hparams.is_mla();
    if (!is_mla) {
        throw std::runtime_error("GLM_DSA architecture requires MLA");
    }

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab_tensor}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // try to load output.weight, if not found, use token_embd (tied embeddings)
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab_tensor}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab_tensor}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer_all; ++i) {
        int flags = 0;
        if (i >= n_layer) {
            // skip all tensors in the NextN layers
            // TODO @ngxson : TENSOR_NOT_REQUIRED was a hack, need to remove it later
            flags |= TENSOR_SKIP | TENSOR_NOT_REQUIRED;
        }

        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);

        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags | TENSOR_NOT_REQUIRED);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags | TENSOR_NOT_REQUIRED);
        layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i), {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v_mla)}, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        if (layer.wkv_b) {
            throw std::runtime_error("GLM_DSA sparse attention does not support unsplit attn_kv_b tensors");
        }
        if (!layer.wk_b || !layer.wv_b) {
            throw std::runtime_error("GLM_DSA sparse attention requires split attn_k_b and attn_v_b tensors");
        }

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // GLM-DSA IndexShare metadata describes target decoder layers only. Native MTP
        // blocks can still carry full decoder-shaped tensors and complete indexer tensors.
        bool indexer_metadata_matched = false;
        const bool indexer_metadata_full =
                i < (int) hparams.n_layer() &&
                llama_glm_dsa_metadata_layer_is_full(hparams, i, &indexer_metadata_matched);
        const bool skip_virtual_indexer =
                indexer_metadata_matched && !indexer_metadata_full;
        const int indexer_flags =
                flags | TENSOR_NOT_REQUIRED | (skip_virtual_indexer ? TENSOR_SKIP_IF_VIRTUAL : 0);
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {hparams.indexer_head_size}, indexer_flags);
        layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {hparams.indexer_head_size}, indexer_flags);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, indexer_flags);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, hparams.indexer_head_size}, indexer_flags);
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, indexer_flags);
        llama_glm_dsa_validate_indexshare_layer_contract(i, hparams, layer);
        llama_glm_dsa_validate_nextn_indexer_contract(i, hparams, layer);
        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        // NextN/MTP tensors (preserved but unused) - conditionally load for last n_layer_nextn
        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);

            // Optional tensors
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab_tensor }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab_tensor }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

llama_model_glm_dsa::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const bool is_mla = hparams.is_mla();
    GGML_ASSERT(is_mla);

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();
    GGML_UNUSED(n_embd_head_v);

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    GGML_ASSERT(n_embd_indexer_head_nope >= 0);
    const uint32_t n_indexer_top_k = hparams.indexer_top_k;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    const int effective_n_layers = n_layer;
    const int il_start = 0;
    const int il_end   = effective_n_layers;

    for (int il = il_start; il < il_end; ++il) {
        if (!model.layers[il].wk_b || !model.layers[il].wv_b) {
            throw std::runtime_error("GLM_DSA sparse attention requires split K_B and V_B tensors");
        }
    }

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_attn_k_dsa * inp_attn_dsa = build_attn_inp_k_dsa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * last_top_k = nullptr;

    const int expected_group_size = int(hparams.indexer_top_k_freq);
    const int repeating_group_begin = expected_group_size > 0 ?
        std::max(0, int(hparams.indexer_skip_top_k_offset) - 1) : 0;
    const llama_glm_dsa_block_plan block_plan = llama_glm_dsa_make_block_plan(
        effective_n_layers,
        il_start,
        il_end,
        repeating_group_begin,
        expected_group_size,
        [&](int il) {
            bool metadata_matched = false;
            const bool metadata_full = llama_glm_dsa_metadata_layer_is_full(
                hparams, il, &metadata_matched);
            return metadata_matched ? metadata_full :
                llama_glm_dsa_layer_uses_indexer(il, hparams, model.layers[il]);
        });
    llama_glm_dsa_log_block_plan(block_plan, expected_group_size);

    llama_glm_dsa_log_indexshare_contract(model, hparams);

    const bool roofline_bypass_attention = n_tokens == 1 && llama_glm_dsa_roofline_bypass_enabled(
        "LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTENTION");
    const bool roofline_bypass_after_q_b = n_tokens == 1 && llama_glm_dsa_roofline_bypass_enabled(
        "LLAMA_GLM_DSA_ROOFLINE_BYPASS_AFTER_Q_B");
    const bool roofline_bypass_before_cache = n_tokens == 1 && llama_glm_dsa_roofline_bypass_enabled(
        "LLAMA_GLM_DSA_ROOFLINE_BYPASS_BEFORE_CACHE");
    const bool roofline_bypass_dense_ffn = n_tokens == 1 && llama_glm_dsa_roofline_bypass_enabled(
        "LLAMA_GLM_DSA_ROOFLINE_BYPASS_DENSE_FFN");
    const bool roofline_bypass_routed_moe = n_tokens == 1 && llama_glm_dsa_roofline_bypass_enabled(
        "LLAMA_GLM_DSA_ROOFLINE_BYPASS_ROUTED_MOE");
    const bool roofline_bypass_shared_expert = n_tokens == 1 && llama_glm_dsa_roofline_bypass_enabled(
        "LLAMA_GLM_DSA_ROOFLINE_BYPASS_SHARED_EXPERT");

    const auto build_layer = [&](int il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * qr = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            cb(qr, "qr", il);

            qr = build_norm(qr, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "qr", il);

            ggml_tensor * top_k = last_top_k;
            const bool uses_indexer = llama_glm_dsa_layer_uses_indexer(il, hparams, model.layers[il]);
            llama_glm_dsa_log_indexshare_exec(il, uses_indexer, top_k != nullptr);

            if (uses_indexer) {
                ggml_tensor * indexer_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, qr);
                cb(indexer_q, "indexer_q", il);

                ggml_tensor * indexer_q_pe =
                    ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, n_tokens,
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head, 0);
                cb(indexer_q_pe, "indexer_q_pe", il);

                ggml_tensor * indexer_q_nope =
                    ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, n_tokens,
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head_rope));
                cb(indexer_q_nope, "indexer_q_nope", il);

                indexer_q_pe = ggml_rope_ext(
                    ctx0, indexer_q_pe, inp_pos, nullptr, n_rot, LLAMA_ROPE_TYPE_NEOX,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                    beta_fast, beta_slow);
                cb(indexer_q_pe, "indexer_q_pe", il);
                indexer_q = ggml_concat(ctx0, indexer_q_pe, indexer_q_nope, 0);
                cb(indexer_q, "indexer_q", il);

                ggml_tensor * indexer_k = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, cur);
                cb(indexer_k, "indexer_k", il);

                indexer_k = build_norm(indexer_k, model.layers[il].indexer_k_norm, model.layers[il].indexer_k_norm_b, LLM_NORM, il);
                cb(indexer_k, "indexer_k", il);

                ggml_tensor * indexer_k_pe =
                    ggml_view_3d(ctx0, indexer_k, n_embd_indexer_head_rope, 1, n_tokens,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1, 0);
                cb(indexer_k_pe, "indexer_k_pe", il);

                ggml_tensor * indexer_k_nope =
                    ggml_view_3d(ctx0, indexer_k, n_embd_indexer_head_nope, 1, n_tokens,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head_rope));
                cb(indexer_k_nope, "indexer_k_nope", il);

                indexer_k_pe = ggml_rope_ext(
                    ctx0, indexer_k_pe, inp_pos, nullptr, n_rot, LLAMA_ROPE_TYPE_NEOX,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                    beta_fast, beta_slow);
                cb(indexer_k_pe, "indexer_k_pe", il);
                indexer_k = ggml_concat(ctx0, indexer_k_pe, indexer_k_nope, 0);
                cb(indexer_k, "indexer_k", il);

                indexer_q = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_q);
                cb(indexer_q, "indexer_q", il);
                indexer_k = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_k);
                cb(indexer_k, "indexer_k", il);

                const auto * mctx_lid = inp_attn_dsa->mctx->get_lid();
                const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
                ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, indexer_k, k_idxs_lid, il));

                ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
                cb(indexer_weights, "indexer_weights", il);

                indexer_k = mctx_lid->get_k(ctx0, il);

                const auto n_stream = indexer_k->ne[3];
                indexer_q = ggml_view_4d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream, indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
                indexer_weights = ggml_view_4d(ctx0, indexer_weights, indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream, indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);
                indexer_weights = ggml_scale(
                    ctx0,
                    indexer_weights,
                    1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head)));
                cb(indexer_weights, "indexer_weights", il);

                ggml_tensor * indexer_score = nullptr;
                ggml_tensor * indexer_kq_mask = inp_attn_dsa->get_kq_mask_lid();
                if (llama_glm_dsa_disable_lightning_indexer()) {
                    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
                    cb(indexer_q, "indexer_q", il);
                    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
                    cb(indexer_k, "indexer_k", il);

                    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
                    cb(indexer_kq, "indexer_kq", il);

                    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
                    cb(indexer_kq, "indexer_kq", il);

                    indexer_score = ggml_relu(ctx0, indexer_kq);
                    cb(indexer_score, "indexer_score", il);

                    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
                    cb(indexer_score, "indexer_score", il);

                    indexer_score = ggml_sum_rows(ctx0, indexer_score);
                    cb(indexer_score, "indexer_score", il);

                    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
                    indexer_score = ggml_add(ctx0, indexer_score, indexer_kq_mask);
                } else {
                    // The fused indexer consumes cached K as [head_size, 1, n_kv, n_stream].
                    // Do not apply the dense KQ path's K permutation here.
                    cb(indexer_k, "indexer_k", il);

                    indexer_score = ggml_lightning_indexer(
                        ctx0, indexer_q, indexer_k, indexer_weights, indexer_kq_mask);
                }
                cb(indexer_score, "indexer_score", il);

                const uint32_t n_score = uint32_t(indexer_score->ne[0]);
                const uint32_t n_kv = inp_attn_dsa->mctx->get_lid()->get_n_kv_used();
                const uint32_t n_top_k = std::min(std::min(n_score, n_kv), n_indexer_top_k);
                if (n_top_k == 0) {
                    throw std::runtime_error("GLM_DSA IndexShare requires at least one KV entry before top-k selection");
                }
                top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                cb(top_k, "top_k", il);
                llama_glm_dsa_log_indexshare_top_k(il, top_k, indexer_score);
                last_top_k = top_k;
            } else if (!top_k) {
                throw std::runtime_error("GLM_DSA split starts inside an IndexShare consumer group without top-k sideband input");
            } else {
                llama_glm_dsa_log_indexshare_consume(il, top_k);
            }

            if (roofline_bypass_attention) {
                cur = ggml_scale(ctx0, inpSA, 0.0f);
                cb(cur, "attn_roofline_zero", il);
            } else {
                ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_b, qr);
                cb(q, "q", il);

                if (roofline_bypass_after_q_b) {
                    ggml_build_forward_expand(gf, q);
                    cur = ggml_scale(ctx0, inpSA, 0.0f);
                    cb(cur, "attn_after_q_b_roofline_zero", il);
                } else {
                ggml_tensor * q_nope =
                    ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                                 ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
                cb(q_nope, "q_nope", il);

                ggml_tensor * q_pe = ggml_view_3d(
                    ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                    ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "q_nope_perm", il);

            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

            q_pe = ggml_rope_ext(
                ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);
            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            cb(Qcur, "Qcur", il);

            kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
            cb(kv_cmpr, "kv_cmpr_reshape", il);

            k_pe = ggml_rope_ext(
                ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);
            ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = kv_cmpr;
            cb(Vcur, "Vcur", il);

                if (roofline_bypass_before_cache) {
                    ggml_build_forward_expand(gf, Qcur);
                    ggml_build_forward_expand(gf, Kcur);
                    ggml_build_forward_expand(gf, Vcur);
                    cur = ggml_scale(ctx0, inpSA, 0.0f);
                    cb(cur, "attn_before_cache_roofline_zero", il);
                } else {
                    cur = build_attn(inp_attn_dsa,
                            model.layers[il].wo, NULL, model.layers[il].wo_s,
                            Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, top_k, kq_scale, il);
                }
                }
            }
        }

        if (il == il_end - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            if (roofline_bypass_dense_ffn) {
                cur = ggml_scale(ctx0, cur, 0.0f);
                cb(cur, "ffn_dense_roofline_zero", il);
            } else {
                cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }
        } else {
            if (roofline_bypass_routed_moe && roofline_bypass_shared_expert) {
                cur = ggml_scale(ctx0, cur, 0.0f);
                cb(cur, "ffn_moe_all_roofline_zero", il);
            } else {
                ggml_tensor * moe_out = nullptr;
                if (roofline_bypass_routed_moe) {
                    moe_out = ggml_scale(ctx0, cur, 0.0f);
                    cb(moe_out, "ffn_moe_routed_roofline_zero", il);
                } else {
                    moe_out = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        hparams.expert_weights_scale,
                        (llama_expert_gating_func_type) hparams.expert_gating_func,
                        il,
                        nullptr,
                        model.layers[il].ffn_gate_up_exps,
                        model.layers[il].ffn_up_exps_s,
                        model.layers[il].ffn_gate_exps_s,
                        model.layers[il].ffn_down_exps_s);
                    cb(moe_out, "ffn_moe_out", il);
                }

                ggml_tensor * ffn_shexp = nullptr;
                if (roofline_bypass_shared_expert) {
                    ffn_shexp = ggml_scale(ctx0, cur, 0.0f);
                    cb(ffn_shexp, "ffn_shared_roofline_zero", il);
                } else {
                    ffn_shexp = build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                        model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                        model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);
                }

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    };

    for (const llama_glm_dsa_block_span & block : block_plan.blocks) {
        for (int il = block.execution_begin; il < block.execution_end; ++il) {
            build_layer(il);
        }
    }
    cur = inpL;

    if (roofline_bypass_attention || roofline_bypass_after_q_b || roofline_bypass_before_cache) {
        ggml_build_forward_expand(gf, inp_pos);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_glm_dsa::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}
