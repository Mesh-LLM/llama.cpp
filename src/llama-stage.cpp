#include "llama_stage_abi.h"

#include "gguf.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <regex>
#include <string>

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

extern "C" {

struct llama_stage_abi_version llama_stage_abi_version(void) {
    return {
        LLAMA_STAGE_ABI_VERSION_MAJOR,
        LLAMA_STAGE_ABI_VERSION_MINOR,
        LLAMA_STAGE_ABI_VERSION_PATCH,
    };
}

uint64_t llama_stage_abi_features(void) {
    return LLAMA_STAGE_FEATURE_MODEL_INTROSPECTION;
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
