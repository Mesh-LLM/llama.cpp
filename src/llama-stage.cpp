#include "llama_stage_abi.h"

#include "gguf.h"
#include "llama-graph.h"
#include "llama-model.h"
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

struct llama_stage_graph_filter_scope {
    explicit llama_stage_graph_filter_scope(const llama_stage_runtime_config * config) {
        if (config != nullptr && config->filter_tensors_on_load) {
            llama_stage_graph_filter filter;
            filter.enabled = true;
            filter.layer_start = config->layer_start;
            filter.layer_end = config->layer_end;
            filter.include_embeddings = config->include_embeddings;
            filter.include_output = config->include_output;
            llama_stage_graph_set_filter(filter);
            enabled = true;
        }
    }

    ~llama_stage_graph_filter_scope() {
        if (enabled) {
            llama_stage_graph_clear_filter();
        }
    }

    bool enabled = false;
};

static bool llama_stage_is_filtered(const llama_stage_session * session) {
    return session != nullptr &&
           session->stage_model != nullptr &&
           session->stage_model->config.filter_tensors_on_load;
}

static bool llama_stage_emits_activation_frame(const llama_stage_session * session) {
    return llama_stage_is_filtered(session) && !session->stage_model->config.include_output;
}

static size_t llama_stage_activation_payload_bytes(const llama_stage_session * session, size_t token_count) {
    if (session == nullptr || session->stage_model == nullptr || session->stage_model->model == nullptr) {
        return 0;
    }

    return token_count *
           static_cast<size_t>(llama_model_n_embd(session->stage_model->model)) *
           sizeof(float);
}

static bool llama_stage_has_activation_payload(const llama_stage_activation_desc * desc, const void * payload) {
    return desc != nullptr && desc->payload_bytes > 0 && payload != nullptr;
}

static enum llama_stage_status llama_stage_validate_frame_input(
        llama_stage_session * session,
        const llama_stage_activation_desc * input_desc,
        const void * input_payload,
        size_t expected_token_count,
        struct llama_stage_error ** out_error) {
    if (session == nullptr || session->stage_model == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "session is required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    const llama_stage_runtime_config & config = session->stage_model->config;
    if (!config.filter_tensors_on_load || config.layer_start == 0) {
        if (llama_stage_has_activation_payload(input_desc, input_payload)) {
            llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "this stage expects token input, not activation input");
            return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
        }
        return LLAMA_STAGE_STATUS_OK;
    }

    if (input_desc == nullptr || input_payload == nullptr || input_desc->payload_bytes == 0) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "non-first runtime slices require an activation frame input");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (input_desc->version != 1 ||
        input_desc->dtype != LLAMA_STAGE_ACTIVATION_DTYPE_F32 ||
        input_desc->layout != LLAMA_STAGE_ACTIVATION_LAYOUT_TOKEN_MAJOR) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "activation frame must be version 1 F32 token-major");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (input_desc->sequence_count != 1 || input_desc->token_count != expected_token_count) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "activation frame token or sequence count does not match request");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (input_desc->layer_end != config.layer_start) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "activation frame layer_end must match this stage layer_start");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    const size_t expected_bytes = llama_stage_activation_payload_bytes(session, expected_token_count);
    if (input_desc->payload_bytes != expected_bytes) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "activation frame payload size does not match model hidden size");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    return LLAMA_STAGE_STATUS_OK;
}

static enum llama_stage_status llama_stage_prepare_output_activation_frame(
        llama_stage_session * session,
        size_t token_count,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_stage_activation_desc * output_desc,
        struct llama_stage_error ** out_error) {
    const size_t payload_bytes = llama_stage_emits_activation_frame(session) ?
            llama_stage_activation_payload_bytes(session, token_count) : 0;

    if (out_output_payload_bytes != nullptr) {
        *out_output_payload_bytes = payload_bytes;
    }

    if (payload_bytes > 0) {
        if (output_payload == nullptr) {
            llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "output_payload is required for runtime-slice activation output");
            return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
        }
        if (output_payload_capacity < payload_bytes) {
            llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_BUFFER_TOO_SMALL, "output activation buffer is too small");
            return LLAMA_STAGE_STATUS_BUFFER_TOO_SMALL;
        }
    } else if (output_payload_capacity > 0 && output_payload == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "output_payload is required when output capacity is non-zero");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    if (output_desc != nullptr) {
        *output_desc = {};
        output_desc->version = 1;
        output_desc->dtype = payload_bytes > 0 ? LLAMA_STAGE_ACTIVATION_DTYPE_F32 : LLAMA_STAGE_ACTIVATION_DTYPE_UNKNOWN;
        output_desc->layout = payload_bytes > 0 ? LLAMA_STAGE_ACTIVATION_LAYOUT_TOKEN_MAJOR : LLAMA_STAGE_ACTIVATION_LAYOUT_OPAQUE;
        output_desc->producer_stage_index = session != nullptr ? session->stage_model->config.stage_index : -1;
        output_desc->layer_start = session != nullptr ? session->stage_model->config.layer_start : 0;
        output_desc->layer_end = session != nullptr ? session->stage_model->config.layer_end : 0;
        output_desc->token_count = static_cast<uint32_t>(std::min<size_t>(token_count, std::numeric_limits<uint32_t>::max()));
        output_desc->sequence_count = token_count > 0 ? 1 : 0;
        output_desc->payload_bytes = payload_bytes;
        output_desc->flags = 0;
    }

    return LLAMA_STAGE_STATUS_OK;
}

static enum llama_stage_status llama_stage_decode_batch(
        llama_stage_session * session,
        llama_batch batch,
        size_t token_count,
        struct llama_stage_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || token_count == 0) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "session and at least one token are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    llama_stage_graph_filter_scope graph_filter_scope(&session->stage_model->config);
    const int32_t rc = llama_decode(session->ctx, batch);
    if (rc != 0) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_RUNTIME_ERROR, "llama_decode failed");
        return LLAMA_STAGE_STATUS_RUNTIME_ERROR;
    }

    session->n_past += static_cast<int32_t>(token_count);
    return llama_stage_success(out_error);
}

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
    if (llama_stage_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "non-first runtime slices require activation input");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(token_ids),
            static_cast<int32_t>(token_count));

    return llama_stage_decode_batch(session, batch, token_count, out_error);
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

static enum llama_stage_status llama_stage_prepare_empty_activation_frame(
        llama_stage_session * session,
        size_t token_count,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_stage_activation_desc * output_desc,
        struct llama_stage_error ** out_error) {
    if (output_payload_capacity > 0 && output_payload == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "output_payload is required when output capacity is non-zero");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    if (out_output_payload_bytes != nullptr) {
        *out_output_payload_bytes = 0;
    }

    if (output_desc != nullptr) {
        *output_desc = {};
        output_desc->version = 1;
        output_desc->dtype = LLAMA_STAGE_ACTIVATION_DTYPE_UNKNOWN;
        output_desc->layout = LLAMA_STAGE_ACTIVATION_LAYOUT_OPAQUE;
        output_desc->producer_stage_index = session != nullptr ? session->stage_model->config.stage_index : -1;
        output_desc->layer_start = session != nullptr ? session->stage_model->config.layer_start : 0;
        output_desc->layer_end = session != nullptr ? session->stage_model->config.layer_end : 0;
        output_desc->token_count = static_cast<uint32_t>(std::min<size_t>(token_count, std::numeric_limits<uint32_t>::max()));
        output_desc->sequence_count = token_count > 0 ? 1 : 0;
        output_desc->payload_bytes = 0;
        output_desc->flags = 0;
    }

    return LLAMA_STAGE_STATUS_OK;
}

static enum llama_stage_status llama_stage_copy_output_activation_frame(
        llama_stage_session * session,
        size_t token_count,
        void * output_payload,
        struct llama_stage_error ** out_error) {
    if (!llama_stage_emits_activation_frame(session)) {
        return llama_stage_success(out_error);
    }

    float * embeddings = llama_get_embeddings(session->ctx);
    if (embeddings == nullptr) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_RUNTIME_ERROR, "llama embeddings output was not available");
        return LLAMA_STAGE_STATUS_RUNTIME_ERROR;
    }

    const size_t payload_bytes = llama_stage_activation_payload_bytes(session, token_count);
    std::memcpy(output_payload, embeddings, payload_bytes);
    return llama_stage_success(out_error);
}

static enum llama_stage_status llama_stage_decode_activation_frame(
        llama_stage_session * session,
        const llama_stage_activation_desc * input_desc,
        const void * input_payload,
        size_t token_count,
        bool request_logits,
        struct llama_stage_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || input_desc == nullptr || input_payload == nullptr || token_count == 0) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "session and activation frame input are required");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    const int32_t n_tokens = static_cast<int32_t>(token_count);
    const int32_t n_embd = llama_model_n_embd(session->stage_model->model);
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.n_tokens = n_tokens;
    std::memcpy(batch.embd, input_payload, static_cast<size_t>(input_desc->payload_bytes));

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i] = session->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = request_logits && i == n_tokens - 1 ? 1 : 0;
    }

    enum llama_stage_status status = llama_stage_decode_batch(session, batch, token_count, out_error);
    llama_batch_free(batch);
    return status;
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
    return LLAMA_STAGE_FEATURE_RUNTIME_SLICE |
           LLAMA_STAGE_FEATURE_MODEL_INTROSPECTION |
           LLAMA_STAGE_FEATURE_TOKENIZE_DETOKENIZE |
           LLAMA_STAGE_FEATURE_ACTIVATION_FRAME;
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

    if (config != nullptr && config->filter_tensors_on_load && (config->layer_start < 0 || config->layer_start >= config->layer_end)) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "layer_start must be non-negative and less than layer_end");
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

    if (config != nullptr && config->filter_tensors_on_load) {
        const int32_t n_layer = llama_model_n_layer(model);
        if (model->arch != LLM_ARCH_LLAMA) {
            llama_model_free(model);
            llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_UNSUPPORTED, "runtime-slice execution is currently supported for LLaMA-family graphs only");
            return LLAMA_STAGE_STATUS_UNSUPPORTED;
        }
        if (config->layer_end > n_layer) {
            llama_model_free(model);
            llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "layer_end exceeds model layer count");
            return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
        }
        if (config->include_embeddings && config->layer_start != 0) {
            llama_model_free(model);
            llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "only the first runtime slice may include token embeddings");
            return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
        }
        if (config->layer_start == 0 && !config->include_embeddings) {
            llama_model_free(model);
            llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "the first runtime slice must include token embeddings");
            return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
        }
        if (config->include_output && config->layer_end != n_layer) {
            llama_model_free(model);
            llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "only the final runtime slice may include output tensors");
            return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
        }
    }

    llama_stage_model * stage_model = new llama_stage_model{};
    stage_model->model = model;
    if (config != nullptr) {
        stage_model->config = *config;
        stage_model->executable = true;
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
    *out_session = nullptr;

    llama_context_params params = llama_context_default_params();
    params.n_ctx = model->config.ctx_size > 0 ? static_cast<uint32_t>(model->config.ctx_size) : 512;
    params.n_batch = params.n_ctx;
    params.embeddings = model->config.filter_tensors_on_load && !model->config.include_output;

    llama_stage_graph_filter_scope graph_filter_scope(&model->config);
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

enum llama_stage_status llama_stage_prefill_chunk_frame(
        struct llama_stage_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const struct llama_stage_activation_desc * input_desc,
        const void * input_payload,
        struct llama_stage_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct llama_stage_error ** out_error) {
    if (token_count == 0) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "token_count must be greater than zero");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        llama_stage_set_error(out_error, LLAMA_STAGE_STATUS_INVALID_ARGUMENT, "token_count exceeds activation descriptor range");
        return LLAMA_STAGE_STATUS_INVALID_ARGUMENT;
    }

    enum llama_stage_status status = llama_stage_validate_frame_input(
            session,
            input_desc,
            input_payload,
            token_count,
            out_error);
    if (status != LLAMA_STAGE_STATUS_OK) {
        return status;
    }

    status = llama_stage_prepare_output_activation_frame(
            session,
            token_count,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            output_desc,
            out_error);
    if (status != LLAMA_STAGE_STATUS_OK) {
        return status;
    }

    if (llama_stage_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        status = llama_stage_decode_activation_frame(session, input_desc, input_payload, token_count, false, out_error);
    } else {
        status = llama_stage_decode_tokens(session, token_ids, token_count, false, out_error);
    }
    if (status != LLAMA_STAGE_STATUS_OK) {
        return status;
    }

    return llama_stage_copy_output_activation_frame(session, token_count, output_payload, out_error);
}

enum llama_stage_status llama_stage_decode_step_frame(
        struct llama_stage_session * session,
        llama_token token_id,
        const struct llama_stage_activation_desc * input_desc,
        const void * input_payload,
        struct llama_stage_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct llama_stage_error ** out_error) {
    enum llama_stage_status status = llama_stage_validate_frame_input(
            session,
            input_desc,
            input_payload,
            1,
            out_error);
    if (status != LLAMA_STAGE_STATUS_OK) {
        return status;
    }

    status = llama_stage_prepare_output_activation_frame(
            session,
            1,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            output_desc,
            out_error);
    if (status != LLAMA_STAGE_STATUS_OK) {
        return status;
    }

    if (llama_stage_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        status = llama_stage_decode_activation_frame(session, input_desc, input_payload, 1, true, out_error);
    } else {
        status = llama_stage_decode_tokens(session, &token_id, 1, true, out_error);
    }
    if (status != LLAMA_STAGE_STATUS_OK) {
        return status;
    }

    if (out_predicted_token != nullptr) {
        *out_predicted_token = session->stage_model->config.include_output ? llama_stage_greedy_sample(session) : -1;
    }

    return llama_stage_copy_output_activation_frame(session, 1, output_payload, out_error);
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
