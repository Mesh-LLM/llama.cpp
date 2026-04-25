#ifndef LLAMA_STAGE_ABI_H
#define LLAMA_STAGE_ABI_H

#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Experimental ABI for layer-range staged inference.
//
// This header is intentionally a capability boundary. It exposes opaque handles
// and plain C structs so Rust orchestration code can use llama.cpp model
// execution, tokenization, GGUF metadata, and GGUF writing without depending on
// private C++ layouts.
//
// Stability note:
// - ABI version 0 is experimental.
// - Callers must feature-probe before use.
// - Names, structs, and ownership rules may change until cross-family staged
//   runtime validation has completed.

#define LLAMA_STAGE_ABI_VERSION_MAJOR 0
#define LLAMA_STAGE_ABI_VERSION_MINOR 1
#define LLAMA_STAGE_ABI_VERSION_PATCH 0

enum llama_stage_feature {
    LLAMA_STAGE_FEATURE_RUNTIME_SLICE          = 1 << 0,
    LLAMA_STAGE_FEATURE_LAYER_PACKAGE          = 1 << 1,
    LLAMA_STAGE_FEATURE_ARTIFACT_SLICE         = 1 << 2,
    LLAMA_STAGE_FEATURE_MODEL_INTROSPECTION    = 1 << 3,
    LLAMA_STAGE_FEATURE_GGUF_SLICE_WRITE       = 1 << 4,
    LLAMA_STAGE_FEATURE_STATE_IMPORT_EXPORT    = 1 << 5,
    LLAMA_STAGE_FEATURE_TOKENIZE_DETOKENIZE    = 1 << 6,
};

enum llama_stage_status {
    LLAMA_STAGE_STATUS_OK                      = 0,
    LLAMA_STAGE_STATUS_ERROR                   = 1,
    LLAMA_STAGE_STATUS_INVALID_ARGUMENT        = 2,
    LLAMA_STAGE_STATUS_UNSUPPORTED             = 3,
    LLAMA_STAGE_STATUS_BUFFER_TOO_SMALL        = 4,
    LLAMA_STAGE_STATUS_IO_ERROR                = 5,
    LLAMA_STAGE_STATUS_MODEL_ERROR             = 6,
    LLAMA_STAGE_STATUS_RUNTIME_ERROR           = 7,
};

enum llama_stage_load_mode {
    LLAMA_STAGE_LOAD_MODE_RUNTIME_SLICE        = 0,
    LLAMA_STAGE_LOAD_MODE_LAYER_PACKAGE        = 1,
    LLAMA_STAGE_LOAD_MODE_ARTIFACT_SLICE       = 2,
};

enum llama_stage_tensor_role {
    LLAMA_STAGE_TENSOR_ROLE_UNKNOWN            = 0,
    LLAMA_STAGE_TENSOR_ROLE_METADATA           = 1,
    LLAMA_STAGE_TENSOR_ROLE_TOKENIZER          = 2,
    LLAMA_STAGE_TENSOR_ROLE_EMBEDDING          = 3,
    LLAMA_STAGE_TENSOR_ROLE_LAYER              = 4,
    LLAMA_STAGE_TENSOR_ROLE_FINAL_NORM         = 5,
    LLAMA_STAGE_TENSOR_ROLE_OUTPUT             = 6,
};

struct llama_stage_model;
struct llama_stage_session;
struct llama_stage_model_info;
struct llama_stage_slice_plan;

struct llama_stage_error {
    enum llama_stage_status status;
    const char * message;
};

struct llama_stage_runtime_config {
    int32_t stage_index;
    int32_t layer_start;
    int32_t layer_end;
    int32_t ctx_size;
    int32_t n_gpu_layers;

    enum llama_stage_load_mode load_mode;

    bool disable_repack;
    bool filter_tensors_on_load;
    bool include_embeddings;
    bool include_output;
};

struct llama_stage_tensor_info {
    const char * name;
    int32_t layer_index;
    enum llama_stage_tensor_role role;
    uint32_t ggml_type;
    uint64_t byte_size;
    uint64_t element_count;
};

struct llama_stage_abi_version {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

LLAMA_API struct llama_stage_abi_version llama_stage_abi_version(void);

LLAMA_API uint64_t llama_stage_abi_features(void);

LLAMA_API const char * llama_stage_status_string(enum llama_stage_status status);

LLAMA_API void llama_stage_error_free(struct llama_stage_error * error);

LLAMA_API enum llama_stage_status llama_stage_model_open(
        const char * path,
        const struct llama_stage_runtime_config * config,
        struct llama_stage_model ** out_model,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_model_free(
        struct llama_stage_model * model,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_session_create(
        struct llama_stage_model * model,
        struct llama_stage_session ** out_session,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_session_free(
        struct llama_stage_session * session,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_prefill_chunk(
        struct llama_stage_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const void * input_activations,
        size_t input_activation_bytes,
        void * output_activations,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_decode_step(
        struct llama_stage_session * session,
        llama_token token_id,
        const void * input_activation,
        size_t input_activation_bytes,
        void * output_activation,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        llama_token * out_predicted_token,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_export_state(
        struct llama_stage_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_import_state(
        struct llama_stage_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_tokenize(
        struct llama_stage_model * model,
        const char * text,
        bool add_special,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_detokenize(
        struct llama_stage_model * model,
        const llama_token * tokens,
        size_t token_count,
        char * output_text,
        size_t output_text_capacity,
        size_t * out_text_bytes,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_model_info_open(
        const char * path,
        struct llama_stage_model_info ** out_info,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_model_info_free(
        struct llama_stage_model_info * info,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_model_info_tensor_count(
        struct llama_stage_model_info * info,
        size_t * out_count,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_model_info_tensor_at(
        struct llama_stage_model_info * info,
        size_t index,
        struct llama_stage_tensor_info * out_tensor,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_slice_plan_create(
        struct llama_stage_model_info * info,
        struct llama_stage_slice_plan ** out_plan,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_slice_plan_free(
        struct llama_stage_slice_plan * plan,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_slice_plan_add_layer_range(
        struct llama_stage_slice_plan * plan,
        int32_t stage_index,
        int32_t layer_start,
        int32_t layer_end,
        bool include_embeddings,
        bool include_output,
        struct llama_stage_error ** out_error);

LLAMA_API enum llama_stage_status llama_stage_write_slice_gguf(
        struct llama_stage_model_info * info,
        const struct llama_stage_slice_plan * plan,
        int32_t stage_index,
        const char * output_path,
        struct llama_stage_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_STAGE_ABI_H
