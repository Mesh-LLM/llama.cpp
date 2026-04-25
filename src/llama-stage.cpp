#include "llama_stage_abi.h"

#include "gguf.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <regex>
#include <string>

struct llama_stage_model {
    llama_model * model = nullptr;
    llama_stage_runtime_config config = {};
    bool executable = true;
};

struct llama_stage_session {
    llama_stage_model * stage_model = nullptr;
    llama_context * ctx = nullptr;
    int32_t n_past = 0;
};

struct llama_stage_model_info {
    gguf_context * ctx = nullptr;
};

static llama_stage_error * llama_stage_make_error(enum llama_stage_status status, const char * message) {
    llama_stage_error * error = new llama_stage_error{};
    error->status = status;

    if (message == nullptr) {
        error->message = nullptr;
        return error;
    }

    const size_t len = std::strlen(message);
    char * owned = static_cast<char *>(std::malloc(len + 1));
    if (owned == nullptr) {
        error->status = LLAMA_STAGE_STATUS_ERROR;
        error->message = nullptr;
        return error;
    }

    std::memcpy(owned, message, len + 1);
    error->message = owned;
    return error;
}

static void llama_stage_set_error(
        llama_stage_error ** out_error,
        enum llama_stage_status status,
        const char * message) {
    if (out_error != nullptr) {
        *out_error = llama_stage_make_error(status, message);
    }
}

static enum llama_stage_status llama_stage_success(llama_stage_error ** out_error) {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    return LLAMA_STAGE_STATUS_OK;
}

static int32_t llama_stage_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }

    static const std::regex blk_pattern("^blk\\.([0-9]+)\\.");
    static const std::regex cache_pattern("^cache_.*_l([0-9]+)$");

    std::cmatch match;
    if (std::regex_search(name, match, blk_pattern)) {
        return static_cast<int32_t>(std::stol(match[1].str()));
    }
    if (std::regex_search(name, match, cache_pattern)) {
        return static_cast<int32_t>(std::stol(match[1].str()));
    }

    return -1;
}

static enum llama_stage_tensor_role llama_stage_role_from_name(const char * name, int32_t layer_index) {
    if (name == nullptr) {
        return LLAMA_STAGE_TENSOR_ROLE_UNKNOWN;
    }

    const std::string tensor_name(name);
    if (layer_index >= 0) {
        return LLAMA_STAGE_TENSOR_ROLE_LAYER;
    }
    if (tensor_name == "token_embd.weight" || tensor_name == "per_layer_token_embd.weight") {
        return LLAMA_STAGE_TENSOR_ROLE_EMBEDDING;
    }
    if (tensor_name == "output.weight" || tensor_name == "output.bias") {
        return LLAMA_STAGE_TENSOR_ROLE_OUTPUT;
    }
    if (tensor_name == "output_norm.weight" || tensor_name == "output_norm.bias") {
        return LLAMA_STAGE_TENSOR_ROLE_FINAL_NORM;
    }

    return LLAMA_STAGE_TENSOR_ROLE_METADATA;
}

static bool llama_stage_is_full_model_config(const struct llama_stage_runtime_config * config) {
    if (config == nullptr) {
        return true;
    }

    return !config->filter_tensors_on_load && config->layer_start == 0;
}

struct llama_stage_filter_scope {
    explicit llama_stage_filter_scope(const llama_stage_runtime_config * config) {
        if (config != nullptr && config->filter_tensors_on_load) {
            llama_model_loader_stage_filter filter;
            filter.enabled = true;
            filter.layer_start = config->layer_start;
            filter.layer_end = config->layer_end;
            filter.include_embeddings = config->include_embeddings;
            filter.include_output = config->include_output;
            llama_model_loader_set_stage_filter(filter);
            enabled = true;
        }
    }

    ~llama_stage_filter_scope() {
        if (enabled) {
            llama_model_loader_clear_stage_filter();
        }
    }

    bool enabled = false;
};

static enum llama_stage_status llama_stage_decode_tokens(
        llama_stage_session * session,
        const llama_token * token_ids,
        size_t token_count,
        bool request_logits,
        struct llama_stage_error ** out_error) {
    (void) request_logits;

    if (session == nullptr || session->ctx == nullptr || token_ids == nullptr || token_count == 0) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "session and at least one token are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(token_ids),
            static_cast<int32_t>(token_count));

    const int32_t rc = llama_decode(session->ctx, batch);
    if (rc != 0) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_RUNTIME_ERROR, "llama_decode failed");
        return LLAMA_STAGE_STATUS_RUNTIME_ERROR;
    }

    session->n_past += static_cast<int32_t>(token_count);
    return llama_stage_success(out_error);
}

static llama_token llama_stage_greedy_sample(llama_stage_session * session) {
    const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(session->ctx, -1);

    llama_token best = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        if (logits[token] > best_logit) {
            best_logit = logits[token];
            best = token;
        }
    }

    return best;
}

extern "C" {

struct llama_stage_abi_version llama_stage_abi_version(void) {
    return {
        LLAMA_STAGE_ABI_VERSION_MAJOR,
        LLAMA_STAGE_ABI_VERSION_MINOR,
        LLAMA_STAGE_ABI_VERSION_PATCH,
    };
}

uint64_t llama_stage_abi_features(void) {
    return LLAMA_STAGE_FEATURE_MODEL_INTROSPECTION |
           LLAMA_STAGE_FEATURE_TOKENIZE_DETOKENIZE;
}

const char * llama_stage_status_string(enum llama_stage_status status) {
    switch (status) {
        case LLAMA_STAGE_STATUS_OK:               return "ok";
        case LLAMA_STAGE_STATUS_ERROR:            return "error";
        case LLAMA_STAGE_STATUS_INVALID_ARGUMENT: return "invalid_argument";
        case LLAMA_STAGE_STATUS_UNSUPPORTED:      return "unsupported";
        case LLAMA_STAGE_STATUS_BUFFER_TOO_SMALL: return "buffer_too_small";
        case LLAMA_STAGE_STATUS_IO_ERROR:         return "io_error";
        case LLAMA_STAGE_STATUS_MODEL_ERROR:      return "model_error";
        case LLAMA_STAGE_STATUS_RUNTIME_ERROR:    return "runtime_error";
    }

    return "unknown";
}

void llama_stage_error_free(struct llama_stage_error * error) {
    if (error == nullptr) {
        return;
    }

    std::free(const_cast<char *>(error->message));
    delete error;
}

enum llama_stage_status llama_stage_model_open(
        const char * path,
        const struct llama_stage_runtime_config * config,
        struct llama_stage_model ** out_model,
        struct llama_stage_error ** out_error) {
    if (path == nullptr || out_model == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "path and out_model are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    *out_model = nullptr;

    if (config != nullptr && config->filter_tensors_on_load && config->layer_start >= config->layer_end) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "layer_start must be less than layer_end");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    if (!llama_stage_is_full_model_config(config) && (config == nullptr || !config->filter_tensors_on_load)) {
        llama_stage_set_error(
                out_error,
                LLAMA_STAGE_STATUS_UNSUPPORTED,
                "runtime tensor filtering is not implemented yet; use a full-model single-stage config");
        return LLAMA_STAGE_STATUS_UNSUPPORTED;
    }

    llama_model_params params = llama_model_default_params();
    if (config != nullptr) {
        params.n_gpu_layers = config->n_gpu_layers;
        if (config->disable_repack || config->filter_tensors_on_load) {
            params.use_extra_bufts = false;
        }
    }

    llama_backend_init();
    llama_stage_filter_scope filter_scope(config);
    llama_model * model = llama_model_load_from_file(path, params);
    if (model == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_MODEL_ERROR, "failed to load llama model");
        return LLAMA_STAGE_STATUS_MODEL_ERROR;
    }

    llama_stage_model * stage_model = new llama_stage_model{};
    stage_model->model = model;
    if (config != nullptr) {
        stage_model->config = *config;
        stage_model->executable = !config->filter_tensors_on_load;
    }

    *out_model = stage_model;
    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_model_free(
        struct llama_stage_model * model,
        struct llama_stage_error ** out_error) {
    if (model != nullptr) {
        llama_model_free(model->model);
        delete model;
    }
    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_session_create(
        struct llama_stage_model * model,
        struct llama_stage_session ** out_session,
        struct llama_stage_error ** out_error) {
    if (model == nullptr || model->model == nullptr || out_session == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "model and out_session are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (!model->executable) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_UNSUPPORTED, "filtered runtime-slice handles are not executable yet");
        return LLAMA_STAGE_STATUS_UNSUPPORTED;
    }

    *out_session = nullptr;

    llama_context_params params = llama_context_default_params();
    params.n_ctx = model->config.ctx_size > 0 ? static_cast<uint32_t>(model->config.ctx_size) : 512;
    params.n_batch = params.n_ctx;

    llama_context * ctx = llama_init_from_model(model->model, params);
    if (ctx == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_RUNTIME_ERROR, "failed to create llama context");
        return LLAMA_STAGE_STATUS_RUNTIME_ERROR;
    }

    llama_stage_session * session = new llama_stage_session{};
    session->stage_model = model;
    session->ctx = ctx;
    session->n_past = 0;
    *out_session = session;
    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_session_free(
        struct llama_stage_session * session,
        struct llama_stage_error ** out_error) {
    if (session != nullptr) {
        llama_free(session->ctx);
        delete session;
    }
    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_prefill_chunk(
        struct llama_stage_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const void * input_activations,
        size_t input_activation_bytes,
        void * output_activations,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        struct llama_stage_error ** out_error) {
    (void) input_activations;
    (void) input_activation_bytes;
    (void) output_activations;
    (void) output_activation_capacity;

    if (out_output_activation_bytes != nullptr) {
        *out_output_activation_bytes = 0;
    }

    return llama_stage_decode_tokens(session, token_ids, token_count, false, out_error);
}

enum llama_stage_status llama_stage_decode_step(
        struct llama_stage_session * session,
        llama_token token_id,
        const void * input_activation,
        size_t input_activation_bytes,
        void * output_activation,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        llama_token * out_predicted_token,
        struct llama_stage_error ** out_error) {
    (void) input_activation;
    (void) input_activation_bytes;
    (void) output_activation;
    (void) output_activation_capacity;

    if (out_output_activation_bytes != nullptr) {
        *out_output_activation_bytes = 0;
    }

    enum llama_stage_status status = llama_stage_decode_tokens(session, &token_id, 1, true, out_error);
    if (status != LLAMA_STAGE_STATUS_OK) {
        return status;
    }

    if (out_predicted_token != nullptr) {
        *out_predicted_token = llama_stage_greedy_sample(session);
    }

    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_export_state(
        struct llama_stage_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct llama_stage_error ** out_error) {
    (void) session;
    (void) layer_start;
    (void) layer_end;
    (void) output;
    (void) output_capacity;
    (void) out_bytes;
    llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_UNSUPPORTED, "state export is not implemented yet");
    return LLAMA_STAGE_STATUS_UNSUPPORTED;
}

enum llama_stage_status llama_stage_import_state(
        struct llama_stage_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct llama_stage_error ** out_error) {
    (void) session;
    (void) layer_start;
    (void) layer_end;
    (void) input;
    (void) input_bytes;
    llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_UNSUPPORTED, "state import is not implemented yet");
    return LLAMA_STAGE_STATUS_UNSUPPORTED;
}

enum llama_stage_status llama_stage_tokenize(
        struct llama_stage_model * model,
        const char * text,
        bool add_special,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct llama_stage_error ** out_error) {
    if (model == nullptr || model->model == nullptr || text == nullptr || out_token_count == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "model, text, and out_token_count are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (output_token_capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "token output capacity exceeds int32_t range");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model->model);
    const int32_t result = llama_tokenize(
            vocab,
            text,
            static_cast<int32_t>(std::strlen(text)),
            output_tokens,
            static_cast<int32_t>(output_token_capacity),
            add_special,
            true);
    if (result < 0) {
        *out_token_count = static_cast<size_t>(-result);
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_BUFFER_TOO_SMALL, "token output buffer is too small");
        return LLAMA_STAGE_STATUS_BUFFER_TOO_SMALL;
    }

    *out_token_count = static_cast<size_t>(result);
    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_detokenize(
        struct llama_stage_model * model,
        const llama_token * tokens,
        size_t token_count,
        char * output_text,
        size_t output_text_capacity,
        size_t * out_text_bytes,
        struct llama_stage_error ** out_error) {
    if (model == nullptr || model->model == nullptr || tokens == nullptr || out_text_bytes == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "model, tokens, and out_text_bytes are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            output_text_capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "detokenize inputs exceed int32_t range");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model->model);
    const int32_t result = llama_detokenize(
            vocab,
            tokens,
            static_cast<int32_t>(token_count),
            output_text,
            static_cast<int32_t>(output_text_capacity),
            true,
            false);
    if (result < 0) {
        *out_text_bytes = static_cast<size_t>(-result);
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_BUFFER_TOO_SMALL, "text output buffer is too small");
        return LLAMA_STAGE_STATUS_BUFFER_TOO_SMALL;
    }

    *out_text_bytes = static_cast<size_t>(result);
    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_model_info_open(
        const char * path,
        struct llama_stage_model_info ** out_info,
        struct llama_stage_error ** out_error) {
    if (path == nullptr || out_info == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "path and out_info are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    *out_info = nullptr;

    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (ctx == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_MODEL_ERROR, "failed to open GGUF model metadata");
        return LLAMA_STAGE_STATUS_MODEL_ERROR;
    }

    llama_stage_model_info * info = new llama_stage_model_info{};
    info->ctx = ctx;
    *out_info = info;
    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_model_info_free(
        struct llama_stage_model_info * info,
        struct llama_stage_error ** out_error) {
    if (info != nullptr) {
        gguf_free(info->ctx);
        delete info;
    }
    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_model_info_tensor_count(
        struct llama_stage_model_info * info,
        size_t * out_count,
        struct llama_stage_error ** out_error) {
    if (info == nullptr || info->ctx == nullptr || out_count == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "info and out_count are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    const int64_t n_tensors = gguf_get_n_tensors(info->ctx);
    if (n_tensors < 0) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_MODEL_ERROR, "GGUF reported negative tensor count");
        return LLAMA_STAGE_STATUS_MODEL_ERROR;
    }

    *out_count = static_cast<size_t>(n_tensors);
    return llama_stage_success(out_error);
}

enum llama_stage_status llama_stage_model_info_tensor_at(
        struct llama_stage_model_info * info,
        size_t index,
        struct llama_stage_tensor_info * out_tensor,
        struct llama_stage_error ** out_error) {
    if (info == nullptr || info->ctx == nullptr || out_tensor == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "info and out_tensor are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    const int64_t n_tensors = gguf_get_n_tensors(info->ctx);
    if (index >= static_cast<size_t>(n_tensors)) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "tensor index is out of range");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    const int64_t tensor_id = static_cast<int64_t>(index);
    const char * name = gguf_get_tensor_name(info->ctx, tensor_id);
    const int32_t layer_index = llama_stage_layer_from_name(name);

    out_tensor->name = name;
    out_tensor->layer_index = layer_index;
    out_tensor->role = llama_stage_role_from_name(name, layer_index);
    out_tensor->ggml_type = static_cast<uint32_t>(gguf_get_tensor_type(info->ctx, tensor_id));
    out_tensor->byte_size = static_cast<uint64_t>(gguf_get_tensor_size(info->ctx, tensor_id));
    out_tensor->element_count = 0;
    return llama_stage_success(out_error);
}

}
