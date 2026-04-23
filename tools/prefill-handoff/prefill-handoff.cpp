#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <mutex>
#include <deque>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static std::string model_architecture_name(const llama_model * model) {
    char buf[128] = {0};
    const int rc = llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf));
    if (rc <= 0) {
        return "";
    }
    return std::string(buf);
}

static bool staged_same_topology_requires_token_sideband(const llama_model * model) {
    const std::string arch = model_architecture_name(model);
    return arch == "gemma4" || arch == "qwen35moe";
}

static bool staged_same_topology_prompt_boundary_unsupported(const llama_model * model) {
    const std::string arch = model_architecture_name(model);
    return arch == "gemma4";
}

static void populate_embd_batch_tokens(
    llama_batch & batch,
    const llama_token * tokens,
    int token_count
) {
    if (tokens == nullptr || token_count <= 0) {
        return;
    }
    batch.token = (llama_token *) malloc(sizeof(llama_token) * (size_t) token_count);
    GGML_ASSERT(batch.token != nullptr);
    std::memcpy(batch.token, tokens, sizeof(llama_token) * (size_t) token_count);
}

struct prefill_handoff_metrics {
    bool enabled = true;
    std::string state_kind = "full_context";
    std::string readiness_path = "pending_import";
    int prompt_tokens = 0;
    int requested_verify_tokens = 0;
    int generated_tokens = 0;
    int compared_token_steps = 0;
    int verified_decode_tokens = 0;
    int token_mismatch_count = 0;
    int first_replayed_token = -1;
    int first_mismatch_index = -1;
    int first_baseline_token = -1;
    int first_shadow_token = -1;
    int shadow_decode_failure_index = -1;
    uint64_t serialized_bytes = 0;
    double prefill_elapsed_s = 0.0;
    double export_elapsed_s = 0.0;
    double import_elapsed_s = 0.0;
    double shadow_ready_elapsed_s = -1.0;
    double first_baseline_decode_elapsed_s = 0.0;
    double first_shadow_decode_elapsed_s = 0.0;
    double import_plus_first_replay_elapsed_s = -1.0;
    double first_token_replay_delta_s = 0.0;
    double ready_vs_baseline_first_token_delta_s = -1.0;
    bool shadow_context_created = false;
    bool import_succeeded = false;
    bool logits_available_after_import = false;
    bool logits_available_after_first_shadow_decode = false;
    bool shadow_decode_succeeded = false;
    int speculative_tokens = 1;
    int speculative_rounds = 0;
    int speculative_drafted_tokens = 0;
    int speculative_accepted_tokens = 0;
    int speculative_verified_tokens = 0;
    int speculative_rollback_count = 0;
    std::string speculative_draft_mode = "disabled";
    std::string output_text;
    std::vector<int> generated_token_ids_trace;
};

struct split_compute_metrics {
    bool enabled = false;
    int split_layer = -1;
    int prompt_tokens = 0;
    int requested_decode_tokens = 0;
    int generated_tokens = 0;
    int compared_token_steps = 0;
    int token_mismatch_count = 0;
    int first_mismatch_step = -1;
    int first_mismatch_baseline_token = -1;
    int first_mismatch_split_token = -1;
    int first_generated_token = -1;
    int baseline_next_token = -1;
    int split_next_token = -1;
    uint64_t late_state_bytes = 0;
    double baseline_prefill_elapsed_s = 0.0;
    double baseline_first_decode_elapsed_s = 0.0;
    double baseline_total_decode_elapsed_s = 0.0;
    double stage1_prefill_elapsed_s = 0.0;
    double stage1_first_decode_elapsed_s = 0.0;
    double stage1_total_decode_elapsed_s = 0.0;
    double stage2_import_elapsed_s = 0.0;
    double stage2_first_decode_elapsed_s = 0.0;
    double stage2_total_decode_elapsed_s = 0.0;
    double prompt_boundary_stage1_prefill_elapsed_s = 0.0;
    double prompt_boundary_stage2_prefill_elapsed_s = 0.0;
    double split_total_elapsed_s = 0.0;
    double split_delta_vs_baseline_s = 0.0;
    bool stage2_import_succeeded = false;
    bool token_match = false;
    bool prompt_boundary_seed_match = false;
    int prompt_boundary_seed_baseline_token = -1;
    int prompt_boundary_seed_split_token = -1;
    std::string output_text;
    std::vector<int> decode_input_tokens;
    std::vector<int> baseline_predicted_tokens;
    std::vector<int> split_predicted_tokens;
    std::vector<uint64_t> debug_stage1_boundary_hashes;
};

struct multistage_compute_metrics {
    bool enabled = false;
    int stage_count = 0;
    int prompt_tokens = 0;
    int requested_decode_tokens = 0;
    int generated_tokens = 0;
    int compared_token_steps = 0;
    int token_mismatch_count = 0;
    int first_mismatch_step = -1;
    int first_mismatch_baseline_token = -1;
    int first_mismatch_staged_token = -1;
    double first_mismatch_baseline_margin = 0.0;
    double first_mismatch_staged_margin = 0.0;
    int seed_token = -1;
    uint64_t baseline_state_bytes = 0;
    double baseline_prefill_elapsed_s = 0.0;
    std::vector<double> baseline_prefill_layer_elapsed_s;
    std::vector<int32_t> baseline_prefill_layer_observation_count;
    double baseline_first_decode_elapsed_s = 0.0;
    double baseline_total_decode_elapsed_s = 0.0;
    double staged_prefill_elapsed_s = 0.0;
    double staged_first_decode_elapsed_s = 0.0;
    double staged_total_decode_elapsed_s = 0.0;
    double staged_delta_vs_baseline_s = 0.0;
    double staged_total_runtime_s = 0.0;
    double staged_total_delta_vs_baseline_runtime_s = 0.0;
    bool token_match = false;
    std::string baseline_output_text;
    std::string output_text;
    uint64_t boundary_embedding_bytes = 0;
    std::vector<int32_t> boundaries;
    std::vector<uint64_t> stage_state_bytes;
    bool live_prefill_enabled = false;
    int live_prefill_chunk_tokens = 0;
    int live_prefill_downstream_chunk_tokens = 0;
    int live_prefill_tile_tokens = 0;
    int live_prefill_fill_chunk_tokens = 0;
    int live_prefill_fill_chunk_count = 0;
    int live_prefill_max_inflight = 0;
    bool live_prefill_source_buffer_enabled = false;
    bool keep_stage_logits = false;
    double live_prefill_hop_latency_ms = 0.0;
    double live_prefill_hop_jitter_ms = 0.0;
    double live_prefill_hop_bandwidth_mbps = 0.0;
    std::vector<int32_t> live_prefill_stage_chunk_counts;
    std::vector<double> live_prefill_stage_compute_elapsed_s;
    std::vector<double> live_prefill_stage_send_elapsed_s;
    std::vector<double> live_prefill_stage_reply_wait_elapsed_s;
    std::vector<double> live_prefill_stage_input_copy_elapsed_s;
    std::vector<double> live_prefill_stage_output_extract_elapsed_s;
    std::vector<double> live_prefill_stage_runtime_lock_wait_elapsed_s;
    std::vector<double> live_prefill_stage_runtime_lock_hold_elapsed_s;
    std::vector<double> live_prefill_stage_input_wait_elapsed_s;
    std::vector<double> live_prefill_stage_output_wait_elapsed_s;
    std::vector<int32_t> live_prefill_stage_input_max_queue_depth;
    std::vector<int32_t> live_prefill_stage_output_max_queue_depth;
    std::vector<int32_t> live_prefill_edge_chunk_counts;
    std::vector<double> live_prefill_edge_transport_elapsed_s;
    std::vector<int32_t> live_prefill_edge_max_queue_depth;
    std::vector<std::vector<uint64_t>> debug_decode_stage_boundary_hashes;
    std::vector<int> decode_input_tokens;
    std::vector<int> baseline_predicted_tokens;
    std::vector<int> staged_predicted_tokens;
};

struct baseline_prefill_layer_timing_probe {
    bool enabled = false;
    bool active = false;
    double prefill_start_s = 0.0;
    double last_event_s = 0.0;
    bool has_last_event = false;
    std::vector<double> layer_elapsed_s;
    std::vector<int32_t> layer_observation_count;

    void reset(int32_t n_layer) {
        enabled = true;
        active = false;
        prefill_start_s = 0.0;
        last_event_s = 0.0;
        has_last_event = false;
        layer_elapsed_s.assign((size_t) n_layer, 0.0);
        layer_observation_count.assign((size_t) n_layer, 0);
    }

    void begin() {
        active = true;
        prefill_start_s = ggml_time_us() / 1e6;
        last_event_s = prefill_start_s;
        has_last_event = false;
        std::fill(layer_elapsed_s.begin(), layer_elapsed_s.end(), 0.0);
        std::fill(layer_observation_count.begin(), layer_observation_count.end(), 0);
    }

    void end() {
        active = false;
    }
};

static bool parse_l_out_layer_index(const ggml_tensor * t, int32_t & layer_index) {
    const char * name = ggml_get_name(t);
    if (name == nullptr) {
        return false;
    }
    static constexpr const char * prefix = "l_out-";
    static constexpr size_t prefix_len = 6;
    if (std::strncmp(name, prefix, prefix_len) != 0) {
        return false;
    }
    char * end = nullptr;
    const long value = std::strtol(name + prefix_len, &end, 10);
    if (end == name + prefix_len || end == nullptr || *end != '\0' || value < 0) {
        return false;
    }
    layer_index = (int32_t) value;
    return true;
}

static bool baseline_prefill_layer_timing_eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * probe = static_cast<baseline_prefill_layer_timing_probe *>(user_data);
    if (probe == nullptr || !probe->enabled || !probe->active) {
        return false;
    }
    int32_t layer_index = -1;
    if (!parse_l_out_layer_index(t, layer_index)) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (layer_index < 0 || (size_t) layer_index >= probe->layer_elapsed_s.size()) {
        return true;
    }
    const double now_s = ggml_time_us() / 1e6;
    const double delta_s = probe->has_last_event ? (now_s - probe->last_event_s) : (now_s - probe->prefill_start_s);
    probe->layer_elapsed_s[(size_t) layer_index] += std::max(0.0, delta_s);
    probe->layer_observation_count[(size_t) layer_index] += 1;
    probe->last_event_s = now_s;
    probe->has_last_event = true;
    return true;
}

static std::pair<std::string, bool> build_generation_prompt(
    const llama_model * model,
    const std::string & raw_prompt
) {
    auto chat_templates = common_chat_templates_init(model, "");
    if (!chat_templates) {
        return { raw_prompt, false };
    }

    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = raw_prompt;

    common_chat_templates_inputs inputs;
    inputs.messages = { std::move(user_msg) };
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    inputs.enable_thinking = false;

    return { common_chat_templates_apply(chat_templates.get(), inputs).prompt, true };
}

struct multistage_stage_server_session_summary {
    int session_index = 0;
    int prefill_chunk_count = 0;
    double prefill_compute_elapsed_s = 0.0;
    double prefill_recv_elapsed_s = 0.0;
    double prefill_forward_elapsed_s = 0.0;
    double prefill_forward_send_elapsed_s = 0.0;
    double prefill_forward_reply_wait_elapsed_s = 0.0;
    double prefill_reply_elapsed_s = 0.0;
    double prefill_input_copy_elapsed_s = 0.0;
    double prefill_output_extract_elapsed_s = 0.0;
    uint64_t prefill_stage_state_bytes = 0;
    uint64_t prefill_input_bytes = 0;
    uint64_t prefill_forward_bytes = 0;
    uint64_t prefill_reply_bytes = 0;
    int decode_step_count = 0;
    double decode_compute_elapsed_s = 0.0;
    std::vector<uint64_t> decode_input_hashes;
    std::vector<uint64_t> decode_output_hashes;
    std::vector<int32_t> decode_predicted_tokens;
    std::string last_error;
};

static std::string json_out_path;
static std::string state_out_path;
static std::string state_in_path;
static std::string state_chunks_dir_path;
static std::string shadow_model_path;
static std::string replay_tokens_in_path;
static std::string replay_tokens_out_path;
static std::string split_embd_out_path;
static std::string split_embd_in_path;
static std::string split_embd_stream_out_path;
static std::string split_embd_stream_in_path;
static std::string split_embd_stream_connect_addr;
static std::string split_embd_stream_listen_addr;
static int split_seed_token = -1;
static std::string state_mode = "full";
static int state_il_start = -1;
static int state_il_end = -1;
static int verify_tokens = 8;
static double baseline_first_token_elapsed_s = -1.0;
static int split_compute_layer = -1;
static int split_stage2_layer = -1;
static int split_prompt_tokens = -1;
static int split_expected_token = -1;
static bool split_live_transport = false;
static int split_speculative_tokens = 1;
static std::string split_speculative_draft_mode = "ngram";
static std::string split_draft_tokens_in_path;
static int split_draft_skip_stride = 2;
static std::string multistage_compute_boundaries_csv;
static std::string multistage_models_csv;
static bool multistage_live_prefill = false;
static int multistage_live_prefill_chunk_tokens = 1;
static int multistage_live_prefill_downstream_chunk_tokens = 0;
static int multistage_live_prefill_tile_tokens = 0;
static int multistage_live_prefill_fill_chunk_tokens = 0;
static int multistage_live_prefill_fill_chunk_count = 0;
static int multistage_live_prefill_max_inflight = 0;
static int multistage_live_prefill_reply_credit_limit = -1;
static bool multistage_live_decode_replay = false;
static int multistage_live_decode_replay_window = 1;
static bool multistage_live_prefill_source_buffer = false;
static bool multistage_keep_stage_logits = false;
static double multistage_live_prefill_hop_latency_ms = 0.0;
static double multistage_live_prefill_hop_jitter_ms = 0.0;
static double multistage_live_prefill_hop_bandwidth_mbps = 0.0;
static int multistage_live_prefill_seed = 0;
static bool multistage_debug_trace = false;
static bool multistage_live_prefill_import_baseline_state = false;
static bool multistage_live_decode_import_baseline_state = false;
static bool multistage_live_decode_auto_pos = false;
static int multistage_live_decode_light_context_tokens = 0;
static bool baseline_prefill_layer_timing = false;
static std::mutex multistage_live_prefill_runtime_mutex;
static std::string multistage_stage_server_listen_addr;
static std::string multistage_stage_next_connect_addr;
static std::string multistage_stage_driver_connect_addr;
static int multistage_stage_index = -1;
static int multistage_stage_session_count = 1;
static int multistage_stage_il_start = -1;
static int multistage_stage_il_end = -1;

enum multistage_stage_message_kind : int32_t {
    MULTISTAGE_STAGE_MSG_PREFILL_EMBD = 1,
    MULTISTAGE_STAGE_MSG_DECODE_EMBD = 2,
    MULTISTAGE_STAGE_MSG_STOP = 3,
    MULTISTAGE_STAGE_MSG_PREFILL_FINAL_EMBD = 4,
    MULTISTAGE_STAGE_MSG_DECODE_REPLAY_EMBD = 5,
    MULTISTAGE_STAGE_MSG_DECODE_REPLAY_FINAL_EMBD = 6,
    MULTISTAGE_STAGE_MSG_STATE_IMPORT = 7,
    MULTISTAGE_STAGE_MSG_DECODE_READOUT = 8,
    MULTISTAGE_STAGE_MSG_DECODE_LIGHT_CTX = 9,
};

enum multistage_stage_reply_kind : int32_t {
    MULTISTAGE_STAGE_REPLY_ACK = 1,
    MULTISTAGE_STAGE_REPLY_PREDICTED_TOKEN = 2,
};

enum multistage_stage_phase_kind : int32_t {
    MULTISTAGE_STAGE_PHASE_PREFILL = 1,
    MULTISTAGE_STAGE_PHASE_DECODE = 2,
    MULTISTAGE_STAGE_PHASE_DECODE_REPLAY = 3,
    MULTISTAGE_STAGE_PHASE_DECODE_LIGHT = 4,
};

enum multistage_stage_state_flags : int32_t {
    MULTISTAGE_STAGE_STATE_FLAG_FINAL_CHUNK = 1 << 0,
    MULTISTAGE_STAGE_STATE_FLAG_LIGHT_CONTEXT = 1 << 1,
};

static constexpr int32_t MULTISTAGE_STAGE_STATE_VERSION = 1;

struct multistage_stage_state_header {
    int32_t version = MULTISTAGE_STAGE_STATE_VERSION;
    int32_t seq_id = 0;
    int32_t phase = 0;
    int32_t flags = 0;
    int32_t checkpoint_generation = 0;
    int32_t prompt_token_count = 0;
    int32_t decode_step = -1;
    int32_t current_token = LLAMA_TOKEN_NULL;
    int32_t source_stage_index = -1;
    int32_t reserved = 0;
};

static bool multistage_stage_msg_is_prefill(int32_t kind) {
    return kind == MULTISTAGE_STAGE_MSG_PREFILL_EMBD ||
           kind == MULTISTAGE_STAGE_MSG_PREFILL_FINAL_EMBD;
}

static bool multistage_stage_msg_requires_predicted_reply(int32_t kind) {
    return kind == MULTISTAGE_STAGE_MSG_DECODE_EMBD ||
           kind == MULTISTAGE_STAGE_MSG_DECODE_READOUT ||
           kind == MULTISTAGE_STAGE_MSG_DECODE_LIGHT_CTX ||
           kind == MULTISTAGE_STAGE_MSG_PREFILL_FINAL_EMBD ||
           kind == MULTISTAGE_STAGE_MSG_DECODE_REPLAY_FINAL_EMBD;
}

static bool multistage_stage_msg_is_decode_replay(int32_t kind) {
    return kind == MULTISTAGE_STAGE_MSG_DECODE_REPLAY_EMBD ||
           kind == MULTISTAGE_STAGE_MSG_DECODE_REPLAY_FINAL_EMBD;
}

static bool multistage_stage_msg_is_decode_light_context(int32_t kind) {
    return kind == MULTISTAGE_STAGE_MSG_DECODE_LIGHT_CTX;
}

static bool multistage_stage_state_matches_kind(
    int32_t kind,
    const multistage_stage_state_header & stage_state
) {
    if (kind == MULTISTAGE_STAGE_MSG_STATE_IMPORT) {
        return true;
    }
    int32_t expected_phase = MULTISTAGE_STAGE_PHASE_DECODE;
    if (multistage_stage_msg_is_prefill(kind)) {
        expected_phase = MULTISTAGE_STAGE_PHASE_PREFILL;
    } else if (multistage_stage_msg_is_decode_replay(kind)) {
        expected_phase = MULTISTAGE_STAGE_PHASE_DECODE_REPLAY;
    } else if (multistage_stage_msg_is_decode_light_context(kind)) {
        expected_phase = MULTISTAGE_STAGE_PHASE_DECODE_LIGHT;
    }
    if (stage_state.phase != expected_phase) {
        return false;
    }
    const bool expected_final =
        kind == MULTISTAGE_STAGE_MSG_PREFILL_FINAL_EMBD ||
        kind == MULTISTAGE_STAGE_MSG_DECODE_REPLAY_FINAL_EMBD;
    const bool actual_final =
        (stage_state.flags & MULTISTAGE_STAGE_STATE_FLAG_FINAL_CHUNK) != 0;
    if (expected_final != actual_final) {
        return false;
    }
    const bool expected_light_context = multistage_stage_msg_is_decode_light_context(kind);
    const bool actual_light_context =
        (stage_state.flags & MULTISTAGE_STAGE_STATE_FLAG_LIGHT_CONTEXT) != 0;
    return expected_light_context == actual_light_context;
}

static multistage_stage_state_header make_stage_state_header(
    int32_t kind,
    int32_t checkpoint_generation = 0,
    int32_t prompt_token_count = 0,
    int32_t decode_step = -1,
    int32_t current_token = LLAMA_TOKEN_NULL,
    int32_t source_stage_index = -1
) {
    multistage_stage_state_header header;
    if (kind == MULTISTAGE_STAGE_MSG_STATE_IMPORT || multistage_stage_msg_is_prefill(kind)) {
        header.phase = MULTISTAGE_STAGE_PHASE_PREFILL;
    } else if (multistage_stage_msg_is_decode_replay(kind)) {
        header.phase = MULTISTAGE_STAGE_PHASE_DECODE_REPLAY;
    } else if (multistage_stage_msg_is_decode_light_context(kind)) {
        header.phase = MULTISTAGE_STAGE_PHASE_DECODE_LIGHT;
        header.flags |= MULTISTAGE_STAGE_STATE_FLAG_LIGHT_CONTEXT;
    } else {
        header.phase = MULTISTAGE_STAGE_PHASE_DECODE;
    }
    if (kind == MULTISTAGE_STAGE_MSG_PREFILL_FINAL_EMBD ||
        kind == MULTISTAGE_STAGE_MSG_DECODE_REPLAY_FINAL_EMBD) {
        header.flags |= MULTISTAGE_STAGE_STATE_FLAG_FINAL_CHUNK;
    }
    header.checkpoint_generation = checkpoint_generation;
    header.prompt_token_count = prompt_token_count;
    header.decode_step = decode_step;
    header.current_token = current_token;
    header.source_stage_index = source_stage_index;
    return header;
}

static uint64_t fnv1a64_bytes(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= (uint64_t) bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

template<typename T>
struct blocking_queue {
    std::mutex mutex;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
    std::deque<T> items;
    size_t capacity = 0;
    size_t max_depth = 0;
    double total_push_wait_s = 0.0;
    double total_pop_wait_s = 0.0;
    int push_wait_count = 0;
    int pop_wait_count = 0;

    void set_capacity(size_t new_capacity) {
        std::lock_guard<std::mutex> lock(mutex);
        capacity = new_capacity;
    }

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto wait_start = std::chrono::steady_clock::now();
        cv_not_full.wait(lock, [&]() { return capacity == 0 || items.size() < capacity; });
        const double wait_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
        if (wait_s > 0.0) {
            total_push_wait_s += wait_s;
            push_wait_count += 1;
        }
        items.push_back(std::move(item));
        max_depth = std::max(max_depth, items.size());
        lock.unlock();
        cv_not_empty.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        const auto wait_start = std::chrono::steady_clock::now();
        cv_not_empty.wait(lock, [&]() { return !items.empty(); });
        const double wait_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
        if (wait_s > 0.0) {
            total_pop_wait_s += wait_s;
            pop_wait_count += 1;
        }
        T item = std::move(items.front());
        items.pop_front();
        lock.unlock();
        cv_not_full.notify_one();
        return item;
    }
};

struct multistage_prefill_chunk {
    int chunk_index = 0;
    int pos_start = 0;
    int token_count = 0;
    std::vector<llama_token> tokens;
    std::vector<float> embeddings;
    bool stop = false;
};

static bool send_all(int fd, const void * data, size_t size) {
    const char * ptr = reinterpret_cast<const char *>(data);
    size_t sent = 0;
    while (sent < size) {
        const ssize_t rc = ::send(fd, ptr + sent, size - sent, 0);
        if (rc <= 0) {
            return false;
        }
        sent += (size_t) rc;
    }
    return true;
}

static bool recv_all(int fd, void * data, size_t size) {
    char * ptr = reinterpret_cast<char *>(data);
    size_t recvd = 0;
    while (recvd < size) {
        const ssize_t rc = ::recv(fd, ptr + recvd, size - recvd, 0);
        if (rc <= 0) {
            return false;
        }
        recvd += (size_t) rc;
    }
    return true;
}

static bool send_stage_message(
    int fd,
    int32_t kind,
    int32_t pos_start,
    int32_t token_count,
    const multistage_stage_state_header & stage_state,
    const llama_token * tokens,
    size_t token_payload_count,
    const float * embeddings,
    size_t embedding_count
) {
    const int32_t token_sideband_count = (int32_t) token_payload_count;
    const int32_t header[4] = {kind, pos_start, token_count, token_sideband_count};
    if (!send_all(fd, header, sizeof(header))) {
        return false;
    }
    if (!send_all(fd, &stage_state, sizeof(stage_state))) {
        return false;
    }
    if (token_payload_count > 0 && tokens != nullptr) {
        if (!send_all(fd, tokens, sizeof(llama_token) * token_payload_count)) {
            return false;
        }
    }
    if (embedding_count > 0 && embeddings != nullptr) {
        return send_all(fd, embeddings, sizeof(float) * embedding_count);
    }
    return true;
}

static bool send_stage_state_import_message(
    int fd,
    int32_t target_stage_index,
    const multistage_stage_state_header & stage_state,
    const uint8_t * state_bytes,
    size_t state_byte_count
) {
    const int32_t header[4] = {
        MULTISTAGE_STAGE_MSG_STATE_IMPORT,
        target_stage_index,
        (int32_t) state_byte_count,
        0,
    };
    if (!send_all(fd, header, sizeof(header))) {
        return false;
    }
    if (!send_all(fd, &stage_state, sizeof(stage_state))) {
        return false;
    }
    if (state_byte_count > 0 && state_bytes != nullptr) {
        return send_all(fd, state_bytes, state_byte_count);
    }
    return true;
}

static bool send_stage_reply_ack(int fd) {
    const int32_t payload[2] = {MULTISTAGE_STAGE_REPLY_ACK, 0};
    return send_all(fd, payload, sizeof(payload));
}

static bool send_stage_reply_predicted(int fd, int32_t predicted) {
    const int32_t payload[2] = {MULTISTAGE_STAGE_REPLY_PREDICTED_TOKEN, predicted};
    return send_all(fd, payload, sizeof(payload));
}

static bool recv_stage_reply(
    int fd,
    int32_t & reply_kind,
    int32_t & predicted
) {
    int32_t payload[2] = {0, 0};
    if (!recv_all(fd, payload, sizeof(payload))) {
        return false;
    }
    reply_kind = payload[0];
    predicted = payload[1];
    return reply_kind == MULTISTAGE_STAGE_REPLY_ACK ||
           reply_kind == MULTISTAGE_STAGE_REPLY_PREDICTED_TOKEN;
}

static bool recv_stage_message(
    int fd,
    int32_t & kind,
    int32_t & pos_start,
    int32_t & token_count,
    multistage_stage_state_header & stage_state,
    std::vector<llama_token> & tokens,
    std::vector<float> & embeddings,
    std::vector<uint8_t> & raw_bytes,
    int32_t n_embd
) {
    int32_t header[4] = {0, 0, 0, 0};
    if (!recv_all(fd, header, sizeof(header))) {
        return false;
    }
    kind = header[0];
    pos_start = header[1];
    token_count = header[2];
    const int32_t token_sideband_count = header[3];
    if (!recv_all(fd, &stage_state, sizeof(stage_state))) {
        return false;
    }
    if (stage_state.version != MULTISTAGE_STAGE_STATE_VERSION) {
        return false;
    }
    if (kind == MULTISTAGE_STAGE_MSG_STOP) {
        tokens.clear();
        embeddings.clear();
        raw_bytes.clear();
        return true;
    }
    if (token_count < 0 || token_sideband_count < 0) {
        return false;
    }
    if (kind == MULTISTAGE_STAGE_MSG_STATE_IMPORT) {
        tokens.clear();
        embeddings.clear();
        raw_bytes.resize((size_t) token_count);
        if (!raw_bytes.empty()) {
            return recv_all(fd, raw_bytes.data(), raw_bytes.size());
        }
        return true;
    }
    tokens.resize((size_t) token_sideband_count);
    if (!tokens.empty()) {
        if (!recv_all(fd, tokens.data(), sizeof(llama_token) * tokens.size())) {
            return false;
        }
    }
    embeddings.resize((size_t) token_count * (size_t) n_embd);
    raw_bytes.clear();
    if (!embeddings.empty()) {
        return recv_all(fd, embeddings.data(), sizeof(float) * embeddings.size());
    }
    return true;
}

static void write_json_string(std::ofstream & out, const std::string & value) {
    out << '"';
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if ((unsigned char) c < 0x20) {
                    out << '?';
                } else {
                    out << c;
                }
        }
    }
    out << '"';
}

static std::vector<int32_t> parse_int32_csv(const std::string & text) {
    std::vector<int32_t> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        values.push_back((int32_t) std::atoi(item.c_str()));
    }
    return values;
}

static int32_t multistage_stage_expected_pos_start(
    int32_t kind,
    const multistage_stage_state_header & stage_state,
    int32_t token_count
) {
    if (multistage_stage_msg_is_decode_light_context(kind)) {
        return stage_state.prompt_token_count + stage_state.decode_step + 1 - token_count;
    }
    return stage_state.prompt_token_count + stage_state.decode_step;
}

static std::vector<std::string> parse_string_csv(const std::string & text) {
    std::vector<std::string> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        values.push_back(item);
    }
    return values;
}

static double sample_transport_delay_s(
    std::mt19937 & rng,
    double hop_latency_ms,
    double hop_jitter_ms,
    size_t transfer_bytes,
    double hop_bandwidth_mbps
) {
    double latency_ms = std::max(hop_latency_ms, 0.0);
    const double jitter_ms = std::max(hop_jitter_ms, 0.0);
    if (jitter_ms > 0.0) {
        std::uniform_real_distribution<double> dist(-jitter_ms, jitter_ms);
        latency_ms = std::max(0.0, latency_ms + dist(rng));
    }
    double bandwidth_s = 0.0;
    if (hop_bandwidth_mbps > 0.0) {
        bandwidth_s = (double(transfer_bytes) * 8.0) / (hop_bandwidth_mbps * 1000000.0);
    }
    return latency_ms / 1000.0 + bandwidth_s;
}

static llama_model_params clamp_stage_model_gpu_layers(
    llama_model_params params,
    int32_t stage_layer_count
) {
    if (params.n_gpu_layers > 0 && stage_layer_count > 0) {
        params.n_gpu_layers = std::min(params.n_gpu_layers, stage_layer_count);
    }
    return params;
}

static std::vector<std::pair<int, int>> make_prompt_chunks(int total_tokens, int chunk_tokens) {
    std::vector<std::pair<int, int>> chunks;
    if (total_tokens <= 0) {
        return chunks;
    }
    const int chunk_size = std::max(chunk_tokens, 1);
    for (int start = 0; start < total_tokens; start += chunk_size) {
        const int count = std::min(chunk_size, total_tokens - start);
        chunks.emplace_back(start, count);
    }
    return chunks;
}

static std::vector<std::pair<int, int>> make_adaptive_prompt_chunks(
    int total_tokens,
    int fill_chunk_tokens,
    int fill_chunk_count,
    int stream_chunk_tokens
) {
    if (total_tokens <= 0) {
        return {};
    }
    if (fill_chunk_tokens <= 0 || fill_chunk_count <= 0) {
        return make_prompt_chunks(total_tokens, stream_chunk_tokens);
    }

    fill_chunk_tokens = std::max(1, fill_chunk_tokens);
    stream_chunk_tokens = std::max(1, stream_chunk_tokens);
    std::vector<std::pair<int, int>> chunks;
    int cursor = 0;
    for (int chunk_index = 0; chunk_index < fill_chunk_count && cursor < total_tokens; ++chunk_index) {
        const int count = std::min(fill_chunk_tokens, total_tokens - cursor);
        chunks.emplace_back(cursor, count);
        cursor += count;
    }
    while (cursor < total_tokens) {
        const int count = std::min(stream_chunk_tokens, total_tokens - cursor);
        chunks.emplace_back(cursor, count);
        cursor += count;
    }
    return chunks;
}

static llama_token select_greedy_token(const float * logits, int n_vocab) {
    llama_token next = 0;
    float max_logit = -INFINITY;
    for (int token_id = 0; token_id < n_vocab; ++token_id) {
        if (logits[token_id] > max_logit) {
            max_logit = logits[token_id];
            next = token_id;
        }
    }
    return next;
}

static std::pair<llama_token, double> select_greedy_token_with_margin(const float * logits, int n_vocab) {
    llama_token best = 0;
    float best_logit = -INFINITY;
    float second_logit = -INFINITY;
    for (int token = 0; token < n_vocab; ++token) {
        const float value = logits[token];
        if (value > best_logit) {
            second_logit = best_logit;
            best_logit = value;
            best = token;
        } else if (value > second_logit) {
            second_logit = value;
        }
    }
    return {best, (double) (best_logit - second_logit)};
}

static llama_token propose_ngram_token(
    const std::vector<llama_token> & history,
    int max_n = 2
) {
    if (history.empty()) {
        return LLAMA_TOKEN_NULL;
    }

    const int history_size = (int) history.size();
    for (int n = std::min(max_n, history_size); n >= 1; --n) {
        for (int start = history_size - n - 1; start >= 0; --start) {
            bool match = true;
            for (int j = 0; j < n; ++j) {
                if (history[start + j] != history[history_size - n + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return history[start + n];
            }
        }
    }

    return LLAMA_TOKEN_NULL;
}

static void write_binary_file(const std::string & path, const std::vector<uint8_t> & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open binary output");
    }
    out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
}

static std::vector<uint8_t> read_binary_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open binary input");
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static int parse_chunk_index(const std::string & name) {
    const size_t dot = name.find('.');
    const std::string stem = dot == std::string::npos ? name : name.substr(0, dot);
    return std::atoi(stem.c_str());
}

static std::vector<uint8_t> read_chunk_directory(const std::string & path) {
    namespace fs = std::filesystem;

    std::vector<fs::path> chunk_paths;
    for (const auto & entry : fs::directory_iterator(path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".bin") {
            continue;
        }
        chunk_paths.push_back(entry.path());
    }
    std::sort(chunk_paths.begin(), chunk_paths.end(), [](const fs::path & a, const fs::path & b) {
        return parse_chunk_index(a.filename().string()) < parse_chunk_index(b.filename().string());
    });

    std::vector<uint8_t> combined;
    for (const auto & chunk_path : chunk_paths) {
        std::vector<uint8_t> chunk = read_binary_file(chunk_path.string());
        combined.insert(combined.end(), chunk.begin(), chunk.end());
    }
    return combined;
}

static void write_token_file(const std::string & path, const std::vector<llama_token> & tokens) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open token output");
    }
    for (llama_token token : tokens) {
        out << token << "\n";
    }
}

static void write_float_file(const std::string & path, const float * data, int count) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open float output");
    }
    out.write(reinterpret_cast<const char *>(data), sizeof(float) * count);
}

static std::vector<float> read_float_file(const std::string & path, int expected_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open float input");
    }
    in.seekg(0, std::ios::end);
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size != std::streamsize(sizeof(float) * expected_count)) {
        throw std::runtime_error("unexpected float input size");
    }
    std::vector<float> data(expected_count);
    in.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

static std::vector<float> read_float_stream_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open float stream input");
    }
    in.seekg(0, std::ios::end);
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0 || size % std::streamsize(sizeof(float)) != 0) {
        throw std::runtime_error("unexpected float stream size");
    }
    std::vector<float> data((size_t) size / sizeof(float));
    in.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

static std::vector<llama_token> read_token_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open token input");
    }
    std::vector<llama_token> tokens;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        tokens.push_back((llama_token) std::strtol(line.c_str(), nullptr, 10));
    }
    return tokens;
}

static void write_result_json(
    const std::string & path,
    const std::string & prompt,
    const prefill_handoff_metrics & metrics
) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open JSON output");
    }

    out << "{\n";
    out << "  \"prompt\": ";
    write_json_string(out, prompt);
    out << ",\n";
    out << "  \"real_prefill_handoff\": {\n";
    out << "    \"enabled\": " << (metrics.enabled ? "true" : "false") << ",\n";
    out << "    \"state_kind\": ";
    write_json_string(out, metrics.state_kind);
    out << ",\n";
    out << "    \"readiness_path\": ";
    write_json_string(out, metrics.readiness_path);
    out << ",\n";
    out << "    \"prompt_tokens\": " << metrics.prompt_tokens << ",\n";
    out << "    \"requested_verify_tokens\": " << metrics.requested_verify_tokens << ",\n";
    out << "    \"generated_tokens\": " << metrics.generated_tokens << ",\n";
    out << "    \"compared_token_steps\": " << metrics.compared_token_steps << ",\n";
    out << "    \"verified_decode_tokens\": " << metrics.verified_decode_tokens << ",\n";
    out << "    \"token_mismatch_count\": " << metrics.token_mismatch_count << ",\n";
    out << "    \"first_replayed_token\": " << metrics.first_replayed_token << ",\n";
    out << "    \"first_mismatch_index\": " << metrics.first_mismatch_index << ",\n";
    out << "    \"first_baseline_token\": " << metrics.first_baseline_token << ",\n";
    out << "    \"first_shadow_token\": " << metrics.first_shadow_token << ",\n";
    out << "    \"shadow_decode_failure_index\": " << metrics.shadow_decode_failure_index << ",\n";
    out << "    \"serialized_bytes\": " << metrics.serialized_bytes << ",\n";
    out << "    \"prefill_elapsed_s\": " << metrics.prefill_elapsed_s << ",\n";
    out << "    \"export_elapsed_s\": " << metrics.export_elapsed_s << ",\n";
    out << "    \"import_elapsed_s\": " << metrics.import_elapsed_s << ",\n";
    out << "    \"shadow_ready_elapsed_s\": " << metrics.shadow_ready_elapsed_s << ",\n";
    out << "    \"first_baseline_decode_elapsed_s\": " << metrics.first_baseline_decode_elapsed_s << ",\n";
    out << "    \"first_shadow_decode_elapsed_s\": " << metrics.first_shadow_decode_elapsed_s << ",\n";
    out << "    \"import_plus_first_replay_elapsed_s\": " << metrics.import_plus_first_replay_elapsed_s << ",\n";
    out << "    \"first_token_replay_delta_s\": " << metrics.first_token_replay_delta_s << ",\n";
    out << "    \"ready_vs_baseline_first_token_delta_s\": " << metrics.ready_vs_baseline_first_token_delta_s << ",\n";
    out << "    \"shadow_context_created\": " << (metrics.shadow_context_created ? "true" : "false") << ",\n";
    out << "    \"import_succeeded\": " << (metrics.import_succeeded ? "true" : "false") << ",\n";
    out << "    \"logits_available_after_import\": " << (metrics.logits_available_after_import ? "true" : "false") << ",\n";
    out << "    \"logits_available_after_first_shadow_decode\": " << (metrics.logits_available_after_first_shadow_decode ? "true" : "false") << ",\n";
    out << "    \"shadow_decode_succeeded\": " << (metrics.shadow_decode_succeeded ? "true" : "false") << ",\n";
    out << "    \"speculative_tokens\": " << metrics.speculative_tokens << ",\n";
    out << "    \"speculative_rounds\": " << metrics.speculative_rounds << ",\n";
    out << "    \"speculative_drafted_tokens\": " << metrics.speculative_drafted_tokens << ",\n";
    out << "    \"speculative_accepted_tokens\": " << metrics.speculative_accepted_tokens << ",\n";
    out << "    \"speculative_verified_tokens\": " << metrics.speculative_verified_tokens << ",\n";
    out << "    \"speculative_rollback_count\": " << metrics.speculative_rollback_count << ",\n";
    out << "    \"speculative_draft_mode\": ";
    write_json_string(out, metrics.speculative_draft_mode);
    out << ",\n";
    out << "    \"generated_token_ids\": [";
    for (size_t i = 0; i < metrics.generated_token_ids_trace.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.generated_token_ids_trace[i];
    }
    out << "]\n";
    out << "  },\n";
    out << "  \"output_text\": ";
    write_json_string(out, metrics.output_text);
    out << "\n";
    out << "}\n";
}

static void write_split_compute_json(
    const std::string & path,
    const std::string & prompt,
    const split_compute_metrics & metrics
) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open JSON output");
    }

    out << "{\n";
    out << "  \"prompt\": ";
    write_json_string(out, prompt);
    out << ",\n";
    out << "  \"split_compute_proof\": {\n";
    out << "    \"enabled\": " << (metrics.enabled ? "true" : "false") << ",\n";
    out << "    \"split_layer\": " << metrics.split_layer << ",\n";
    out << "    \"prompt_tokens\": " << metrics.prompt_tokens << ",\n";
    out << "    \"requested_decode_tokens\": " << metrics.requested_decode_tokens << ",\n";
    out << "    \"generated_tokens\": " << metrics.generated_tokens << ",\n";
    out << "    \"compared_token_steps\": " << metrics.compared_token_steps << ",\n";
    out << "    \"token_mismatch_count\": " << metrics.token_mismatch_count << ",\n";
    out << "    \"first_mismatch_step\": " << metrics.first_mismatch_step << ",\n";
    out << "    \"first_mismatch_baseline_token\": " << metrics.first_mismatch_baseline_token << ",\n";
    out << "    \"first_mismatch_split_token\": " << metrics.first_mismatch_split_token << ",\n";
    out << "    \"first_generated_token\": " << metrics.first_generated_token << ",\n";
    out << "    \"baseline_next_token\": " << metrics.baseline_next_token << ",\n";
    out << "    \"split_next_token\": " << metrics.split_next_token << ",\n";
    out << "    \"late_state_bytes\": " << metrics.late_state_bytes << ",\n";
    out << "    \"baseline_prefill_elapsed_s\": " << metrics.baseline_prefill_elapsed_s << ",\n";
    out << "    \"baseline_first_decode_elapsed_s\": " << metrics.baseline_first_decode_elapsed_s << ",\n";
    out << "    \"baseline_total_decode_elapsed_s\": " << metrics.baseline_total_decode_elapsed_s << ",\n";
    out << "    \"stage1_prefill_elapsed_s\": " << metrics.stage1_prefill_elapsed_s << ",\n";
    out << "    \"stage1_first_decode_elapsed_s\": " << metrics.stage1_first_decode_elapsed_s << ",\n";
    out << "    \"stage1_total_decode_elapsed_s\": " << metrics.stage1_total_decode_elapsed_s << ",\n";
    out << "    \"stage2_import_elapsed_s\": " << metrics.stage2_import_elapsed_s << ",\n";
    out << "    \"stage2_first_decode_elapsed_s\": " << metrics.stage2_first_decode_elapsed_s << ",\n";
    out << "    \"stage2_total_decode_elapsed_s\": " << metrics.stage2_total_decode_elapsed_s << ",\n";
    out << "    \"prompt_boundary_stage1_prefill_elapsed_s\": " << metrics.prompt_boundary_stage1_prefill_elapsed_s << ",\n";
    out << "    \"prompt_boundary_stage2_prefill_elapsed_s\": " << metrics.prompt_boundary_stage2_prefill_elapsed_s << ",\n";
    out << "    \"split_total_elapsed_s\": " << metrics.split_total_elapsed_s << ",\n";
    out << "    \"split_delta_vs_baseline_s\": " << metrics.split_delta_vs_baseline_s << ",\n";
    out << "    \"stage2_import_succeeded\": " << (metrics.stage2_import_succeeded ? "true" : "false") << ",\n";
    out << "    \"token_match\": " << (metrics.token_match ? "true" : "false") << ",\n";
    out << "    \"prompt_boundary_seed_match\": " << (metrics.prompt_boundary_seed_match ? "true" : "false") << ",\n";
    out << "    \"prompt_boundary_seed_baseline_token\": " << metrics.prompt_boundary_seed_baseline_token << ",\n";
    out << "    \"prompt_boundary_seed_split_token\": " << metrics.prompt_boundary_seed_split_token << ",\n";
    out << "    \"decode_input_tokens\": [";
    for (size_t i = 0; i < metrics.decode_input_tokens.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.decode_input_tokens[i];
    }
    out << "],\n";
    out << "    \"baseline_predicted_tokens\": [";
    for (size_t i = 0; i < metrics.baseline_predicted_tokens.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.baseline_predicted_tokens[i];
    }
    out << "],\n";
    out << "    \"split_predicted_tokens\": [";
    for (size_t i = 0; i < metrics.split_predicted_tokens.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.split_predicted_tokens[i];
    }
    out << "],\n";
    out << "    \"debug_stage1_boundary_hashes\": [";
    for (size_t i = 0; i < metrics.debug_stage1_boundary_hashes.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.debug_stage1_boundary_hashes[i];
    }
    out << "]\n";
    out << "  }\n";
    out << "}\n";
}

static void write_multistage_compute_json(
    const std::string & path,
    const std::string & prompt,
    const multistage_compute_metrics & metrics
) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open JSON output");
    }

    out << "{\n";
    out << "  \"prompt\": ";
    write_json_string(out, prompt);
    out << ",\n";
    out << "  \"multistage_compute_proof\": {\n";
    out << "    \"enabled\": " << (metrics.enabled ? "true" : "false") << ",\n";
    out << "    \"stage_count\": " << metrics.stage_count << ",\n";
    out << "    \"prompt_tokens\": " << metrics.prompt_tokens << ",\n";
    out << "    \"requested_decode_tokens\": " << metrics.requested_decode_tokens << ",\n";
    out << "    \"generated_tokens\": " << metrics.generated_tokens << ",\n";
    out << "    \"compared_token_steps\": " << metrics.compared_token_steps << ",\n";
    out << "    \"token_mismatch_count\": " << metrics.token_mismatch_count << ",\n";
    out << "    \"first_mismatch_step\": " << metrics.first_mismatch_step << ",\n";
    out << "    \"first_mismatch_baseline_token\": " << metrics.first_mismatch_baseline_token << ",\n";
    out << "    \"first_mismatch_staged_token\": " << metrics.first_mismatch_staged_token << ",\n";
    out << "    \"first_mismatch_baseline_margin\": " << metrics.first_mismatch_baseline_margin << ",\n";
    out << "    \"first_mismatch_staged_margin\": " << metrics.first_mismatch_staged_margin << ",\n";
    out << "    \"seed_token\": " << metrics.seed_token << ",\n";
    out << "    \"baseline_state_bytes\": " << metrics.baseline_state_bytes << ",\n";
    out << "    \"baseline_prefill_elapsed_s\": " << metrics.baseline_prefill_elapsed_s << ",\n";
    out << "    \"baseline_prefill_layer_elapsed_s\": [";
    for (size_t i = 0; i < metrics.baseline_prefill_layer_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.baseline_prefill_layer_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"baseline_prefill_layer_observation_count\": [";
    for (size_t i = 0; i < metrics.baseline_prefill_layer_observation_count.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.baseline_prefill_layer_observation_count[i];
    }
    out << "],\n";
    out << "    \"baseline_first_decode_elapsed_s\": " << metrics.baseline_first_decode_elapsed_s << ",\n";
    out << "    \"baseline_total_decode_elapsed_s\": " << metrics.baseline_total_decode_elapsed_s << ",\n";
    out << "    \"staged_prefill_elapsed_s\": " << metrics.staged_prefill_elapsed_s << ",\n";
    out << "    \"staged_first_decode_elapsed_s\": " << metrics.staged_first_decode_elapsed_s << ",\n";
    out << "    \"staged_total_decode_elapsed_s\": " << metrics.staged_total_decode_elapsed_s << ",\n";
    out << "    \"staged_delta_vs_baseline_s\": " << metrics.staged_delta_vs_baseline_s << ",\n";
    out << "    \"staged_total_runtime_s\": " << metrics.staged_total_runtime_s << ",\n";
    out << "    \"staged_total_delta_vs_baseline_runtime_s\": " << metrics.staged_total_delta_vs_baseline_runtime_s << ",\n";
    out << "    \"token_match\": " << (metrics.token_match ? "true" : "false") << ",\n";
    out << "    \"boundary_embedding_bytes\": " << metrics.boundary_embedding_bytes << ",\n";
    out << "    \"live_prefill_enabled\": " << (metrics.live_prefill_enabled ? "true" : "false") << ",\n";
    out << "    \"live_prefill_chunk_tokens\": " << metrics.live_prefill_chunk_tokens << ",\n";
    out << "    \"live_prefill_downstream_chunk_tokens\": " << metrics.live_prefill_downstream_chunk_tokens << ",\n";
    out << "    \"live_prefill_tile_tokens\": " << metrics.live_prefill_tile_tokens << ",\n";
    out << "    \"live_prefill_fill_chunk_tokens\": " << metrics.live_prefill_fill_chunk_tokens << ",\n";
    out << "    \"live_prefill_fill_chunk_count\": " << metrics.live_prefill_fill_chunk_count << ",\n";
    out << "    \"live_prefill_max_inflight\": " << metrics.live_prefill_max_inflight << ",\n";
    out << "    \"live_prefill_reply_credit_limit\": " << multistage_live_prefill_reply_credit_limit << ",\n";
    out << "    \"live_prefill_source_buffer_enabled\": " << (metrics.live_prefill_source_buffer_enabled ? "true" : "false") << ",\n";
    out << "    \"keep_stage_logits\": " << (metrics.keep_stage_logits ? "true" : "false") << ",\n";
    out << "    \"live_prefill_hop_latency_ms\": " << metrics.live_prefill_hop_latency_ms << ",\n";
    out << "    \"live_prefill_hop_jitter_ms\": " << metrics.live_prefill_hop_jitter_ms << ",\n";
    out << "    \"live_prefill_hop_bandwidth_mbps\": " << metrics.live_prefill_hop_bandwidth_mbps << ",\n";
    out << "    \"boundaries\": [";
    for (size_t i = 0; i < metrics.boundaries.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.boundaries[i];
    }
    out << "],\n";
    out << "    \"stage_state_bytes\": [";
    for (size_t i = 0; i < metrics.stage_state_bytes.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.stage_state_bytes[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_chunk_counts\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_chunk_counts.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_chunk_counts[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_compute_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_compute_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_compute_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_send_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_send_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_send_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_reply_wait_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_reply_wait_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_reply_wait_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_input_copy_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_input_copy_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_input_copy_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_output_extract_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_output_extract_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_output_extract_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_runtime_lock_wait_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_runtime_lock_wait_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_runtime_lock_wait_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_runtime_lock_hold_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_runtime_lock_hold_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_runtime_lock_hold_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_input_wait_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_input_wait_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_input_wait_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_output_wait_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_output_wait_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_output_wait_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_input_max_queue_depth\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_input_max_queue_depth.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_input_max_queue_depth[i];
    }
    out << "],\n";
    out << "    \"live_prefill_stage_output_max_queue_depth\": [";
    for (size_t i = 0; i < metrics.live_prefill_stage_output_max_queue_depth.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_stage_output_max_queue_depth[i];
    }
    out << "],\n";
    out << "    \"live_prefill_edge_chunk_counts\": [";
    for (size_t i = 0; i < metrics.live_prefill_edge_chunk_counts.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_edge_chunk_counts[i];
    }
    out << "],\n";
    out << "    \"live_prefill_edge_transport_elapsed_s\": [";
    for (size_t i = 0; i < metrics.live_prefill_edge_transport_elapsed_s.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_edge_transport_elapsed_s[i];
    }
    out << "],\n";
    out << "    \"live_prefill_edge_max_queue_depth\": [";
    for (size_t i = 0; i < metrics.live_prefill_edge_max_queue_depth.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.live_prefill_edge_max_queue_depth[i];
    }
    out << "],\n";
    out << "    \"debug_decode_stage_boundary_hashes\": [";
    for (size_t step = 0; step < metrics.debug_decode_stage_boundary_hashes.size(); ++step) {
        if (step > 0) out << ", ";
        out << "[";
        const auto & row = metrics.debug_decode_stage_boundary_hashes[step];
        for (size_t stage = 0; stage < row.size(); ++stage) {
            if (stage > 0) out << ", ";
            out << row[stage];
        }
        out << "]";
    }
    out << "],\n";
    out << "    \"decode_input_tokens\": [";
    for (size_t i = 0; i < metrics.decode_input_tokens.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.decode_input_tokens[i];
    }
    out << "],\n";
    out << "    \"baseline_predicted_tokens\": [";
    for (size_t i = 0; i < metrics.baseline_predicted_tokens.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.baseline_predicted_tokens[i];
    }
    out << "],\n";
    out << "    \"staged_predicted_tokens\": [";
    for (size_t i = 0; i < metrics.staged_predicted_tokens.size(); ++i) {
        if (i > 0) out << ", ";
        out << metrics.staged_predicted_tokens[i];
    }
    out << "]\n";
    out << "  },\n";
    out << "  \"baseline_output_text\": ";
    write_json_string(out, metrics.baseline_output_text);
    out << ",\n";
    out << "  \"output_text\": ";
    write_json_string(out, metrics.output_text);
    out << "\n";
    out << "}\n";
}

static void write_multistage_stage_server_json(
    const std::string & path,
    int stage_index,
    bool has_next_stage,
    const std::vector<multistage_stage_server_session_summary> & session_summaries
) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open multistage stage server JSON output");
    }

    out << "{\n";
    out << "  \"multistage_stage_server\": {\n";
    out << "    \"stage_index\": " << stage_index << ",\n";
    out << "    \"has_next_stage\": " << (has_next_stage ? "true" : "false") << ",\n";
    out << "    \"session_summaries\": [";
    for (size_t session_index = 0; session_index < session_summaries.size(); ++session_index) {
        if (session_index > 0) out << ", ";
        const auto & session = session_summaries[session_index];
        out << "{";
        out << "\"session_index\": " << session.session_index << ", ";
        out << "\"prefill_chunk_count\": " << session.prefill_chunk_count << ", ";
        out << "\"prefill_compute_elapsed_s\": " << session.prefill_compute_elapsed_s << ", ";
        out << "\"prefill_recv_elapsed_s\": " << session.prefill_recv_elapsed_s << ", ";
        out << "\"prefill_forward_elapsed_s\": " << session.prefill_forward_elapsed_s << ", ";
        out << "\"prefill_forward_send_elapsed_s\": " << session.prefill_forward_send_elapsed_s << ", ";
        out << "\"prefill_forward_reply_wait_elapsed_s\": " << session.prefill_forward_reply_wait_elapsed_s << ", ";
        out << "\"prefill_reply_elapsed_s\": " << session.prefill_reply_elapsed_s << ", ";
        out << "\"prefill_input_copy_elapsed_s\": " << session.prefill_input_copy_elapsed_s << ", ";
        out << "\"prefill_output_extract_elapsed_s\": " << session.prefill_output_extract_elapsed_s << ", ";
        out << "\"prefill_stage_state_bytes\": " << session.prefill_stage_state_bytes << ", ";
        out << "\"prefill_input_bytes\": " << session.prefill_input_bytes << ", ";
        out << "\"prefill_forward_bytes\": " << session.prefill_forward_bytes << ", ";
        out << "\"prefill_reply_bytes\": " << session.prefill_reply_bytes << ", ";
        out << "\"decode_step_count\": " << session.decode_step_count << ", ";
        out << "\"decode_compute_elapsed_s\": " << session.decode_compute_elapsed_s << ", ";
        out << "\"decode_input_hashes\": [";
        for (size_t i = 0; i < session.decode_input_hashes.size(); ++i) {
            if (i > 0) out << ", ";
            out << session.decode_input_hashes[i];
        }
        out << "], ";
        out << "\"decode_output_hashes\": [";
        for (size_t i = 0; i < session.decode_output_hashes.size(); ++i) {
            if (i > 0) out << ", ";
            out << session.decode_output_hashes[i];
        }
        out << "], ";
        out << "\"decode_predicted_tokens\": [";
        for (size_t i = 0; i < session.decode_predicted_tokens.size(); ++i) {
            if (i > 0) out << ", ";
            out << session.decode_predicted_tokens[i];
        }
        out << "], ";
        out << "\"last_error\": ";
        write_json_string(out, session.last_error);
        out << "}";
    }
    out << "]\n";
    out << "  }\n";
    out << "}\n";
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_predict = 8;
    bool apply_chat_template = false;

    std::vector<const char *> filtered_argv;
    filtered_argv.reserve(argc);

    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json-out") == 0 && i + 1 < argc) {
            json_out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--state-out") == 0 && i + 1 < argc) {
            state_out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--state-in") == 0 && i + 1 < argc) {
            state_in_path = argv[++i];
        } else if (std::strcmp(argv[i], "--state-chunks-dir") == 0 && i + 1 < argc) {
            state_chunks_dir_path = argv[++i];
        } else if (std::strcmp(argv[i], "--shadow-model") == 0 && i + 1 < argc) {
            shadow_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--replay-tokens-in") == 0 && i + 1 < argc) {
            replay_tokens_in_path = argv[++i];
        } else if (std::strcmp(argv[i], "--replay-tokens-out") == 0 && i + 1 < argc) {
            replay_tokens_out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--split-embd-out") == 0 && i + 1 < argc) {
            split_embd_out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--split-embd-in") == 0 && i + 1 < argc) {
            split_embd_in_path = argv[++i];
        } else if (std::strcmp(argv[i], "--split-embd-stream-out") == 0 && i + 1 < argc) {
            split_embd_stream_out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--split-embd-stream-in") == 0 && i + 1 < argc) {
            split_embd_stream_in_path = argv[++i];
        } else if (std::strcmp(argv[i], "--split-embd-stream-connect") == 0 && i + 1 < argc) {
            split_embd_stream_connect_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--split-embd-stream-listen") == 0 && i + 1 < argc) {
            split_embd_stream_listen_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--verify-tokens") == 0 && i + 1 < argc) {
            verify_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--state-mode") == 0 && i + 1 < argc) {
            state_mode = argv[++i];
        } else if (std::strcmp(argv[i], "--state-il-start") == 0 && i + 1 < argc) {
            state_il_start = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--state-il-end") == 0 && i + 1 < argc) {
            state_il_end = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--baseline-first-token-elapsed-s") == 0 && i + 1 < argc) {
            baseline_first_token_elapsed_s = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--split-compute-layer") == 0 && i + 1 < argc) {
            split_compute_layer = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--split-stage2-layer") == 0 && i + 1 < argc) {
            split_stage2_layer = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--split-prompt-tokens") == 0 && i + 1 < argc) {
            split_prompt_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--split-expected-token") == 0 && i + 1 < argc) {
            split_expected_token = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--split-seed-token") == 0 && i + 1 < argc) {
            split_seed_token = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--split-live-transport") == 0) {
            split_live_transport = true;
        } else if (std::strcmp(argv[i], "--apply-chat-template") == 0) {
            apply_chat_template = true;
        } else if (std::strcmp(argv[i], "--split-speculative-tokens") == 0 && i + 1 < argc) {
            split_speculative_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--split-speculative-draft-mode") == 0 && i + 1 < argc) {
            split_speculative_draft_mode = argv[++i];
        } else if (std::strcmp(argv[i], "--split-draft-tokens-in") == 0 && i + 1 < argc) {
            split_draft_tokens_in_path = argv[++i];
        } else if (std::strcmp(argv[i], "--split-draft-skip-stride") == 0 && i + 1 < argc) {
            split_draft_skip_stride = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-compute-boundaries") == 0 && i + 1 < argc) {
            multistage_compute_boundaries_csv = argv[++i];
        } else if (std::strcmp(argv[i], "--multistage-models") == 0 && i + 1 < argc) {
            multistage_models_csv = argv[++i];
        } else if (std::strcmp(argv[i], "--multistage-live-prefill") == 0) {
            multistage_live_prefill = true;
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-chunk-tokens") == 0 && i + 1 < argc) {
            multistage_live_prefill_chunk_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-downstream-chunk-tokens") == 0 && i + 1 < argc) {
            multistage_live_prefill_downstream_chunk_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-tile-tokens") == 0 && i + 1 < argc) {
            multistage_live_prefill_tile_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-fill-chunk-tokens") == 0 && i + 1 < argc) {
            multistage_live_prefill_fill_chunk_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-fill-chunk-count") == 0 && i + 1 < argc) {
            multistage_live_prefill_fill_chunk_count = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-max-inflight") == 0 && i + 1 < argc) {
            multistage_live_prefill_max_inflight = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-reply-credit-limit") == 0 && i + 1 < argc) {
            multistage_live_prefill_reply_credit_limit = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-decode-replay") == 0) {
            multistage_live_decode_replay = true;
        } else if (std::strcmp(argv[i], "--multistage-live-decode-replay-window") == 0 && i + 1 < argc) {
            multistage_live_decode_replay_window = std::max(std::atoi(argv[++i]), 1);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-source-buffer") == 0) {
            multistage_live_prefill_source_buffer = true;
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-import-baseline-state") == 0) {
            multistage_live_prefill_import_baseline_state = true;
        } else if (std::strcmp(argv[i], "--multistage-live-decode-import-baseline-state") == 0) {
            multistage_live_decode_import_baseline_state = true;
        } else if (std::strcmp(argv[i], "--multistage-live-decode-auto-pos") == 0) {
            multistage_live_decode_auto_pos = true;
        } else if (std::strcmp(argv[i], "--multistage-live-decode-light-context-tokens") == 0 && i + 1 < argc) {
            multistage_live_decode_light_context_tokens = std::max(std::atoi(argv[++i]), 0);
        } else if (std::strcmp(argv[i], "--multistage-keep-stage-logits") == 0) {
            multistage_keep_stage_logits = true;
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-hop-latency-ms") == 0 && i + 1 < argc) {
            multistage_live_prefill_hop_latency_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-hop-jitter-ms") == 0 && i + 1 < argc) {
            multistage_live_prefill_hop_jitter_ms = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-hop-bandwidth-mbps") == 0 && i + 1 < argc) {
            multistage_live_prefill_hop_bandwidth_mbps = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-live-prefill-seed") == 0 && i + 1 < argc) {
            multistage_live_prefill_seed = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-debug-trace") == 0) {
            multistage_debug_trace = true;
        } else if (std::strcmp(argv[i], "--baseline-prefill-layer-timing") == 0) {
            baseline_prefill_layer_timing = true;
        } else if (std::strcmp(argv[i], "--multistage-stage-server-listen") == 0 && i + 1 < argc) {
            multistage_stage_server_listen_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--multistage-stage-next-connect") == 0 && i + 1 < argc) {
            multistage_stage_next_connect_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--multistage-stage-driver-connect") == 0 && i + 1 < argc) {
            multistage_stage_driver_connect_addr = argv[++i];
        } else if (std::strcmp(argv[i], "--multistage-stage-index") == 0 && i + 1 < argc) {
            multistage_stage_index = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-stage-session-count") == 0 && i + 1 < argc) {
            multistage_stage_session_count = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-stage-il-start") == 0 && i + 1 < argc) {
            multistage_stage_il_start = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--multistage-stage-il-end") == 0 && i + 1 < argc) {
            multistage_stage_il_end = std::atoi(argv[++i]);
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    common_init();

    if (!common_params_parse((int) filtered_argv.size(), const_cast<char **>(filtered_argv.data()), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const bool multistage_stage_server_mode = !multistage_stage_server_listen_addr.empty();
    const bool multistage_stage_driver_mode = !multistage_stage_driver_connect_addr.empty();
    if (params.prompt.empty() && state_in_path.empty() && state_chunks_dir_path.empty() && !multistage_stage_server_mode) {
        LOG_ERR("prompt is required (use --prompt or --file)\n");
        return 1;
    }
    if (verify_tokens < 0) {
        LOG_ERR("--verify-tokens must be >= 0\n");
        return 1;
    }
    if (split_speculative_tokens < 1) {
        LOG_ERR("--split-speculative-tokens must be >= 1\n");
        return 1;
    }
    if (split_speculative_draft_mode != "ngram" &&
        split_speculative_draft_mode != "oracle" &&
        split_speculative_draft_mode != "self-skip") {
        LOG_ERR("--split-speculative-draft-mode must be one of: ngram, oracle, self-skip\n");
        return 1;
    }
    if (split_draft_skip_stride < 1) {
        LOG_ERR("--split-draft-skip-stride must be >= 1\n");
        return 1;
    }
    if (multistage_live_prefill && multistage_live_prefill_chunk_tokens < 1) {
        LOG_ERR("--multistage-live-prefill-chunk-tokens must be >= 1\n");
        return 1;
    }
    if (multistage_live_prefill_downstream_chunk_tokens < 0) {
        LOG_ERR("--multistage-live-prefill-downstream-chunk-tokens must be >= 0\n");
        return 1;
    }
    if (multistage_live_prefill_tile_tokens < 0) {
        LOG_ERR("--multistage-live-prefill-tile-tokens must be >= 0\n");
        return 1;
    }
    if (multistage_live_prefill_fill_chunk_tokens < 0) {
        LOG_ERR("--multistage-live-prefill-fill-chunk-tokens must be >= 0\n");
        return 1;
    }
    if (multistage_live_prefill_fill_chunk_count < 0) {
        LOG_ERR("--multistage-live-prefill-fill-chunk-count must be >= 0\n");
        return 1;
    }
    if (multistage_live_prefill && multistage_live_prefill_max_inflight < 0) {
        LOG_ERR("--multistage-live-prefill-max-inflight must be >= 0\n");
        return 1;
    }
    if (multistage_live_prefill && multistage_live_prefill_reply_credit_limit < -1) {
        LOG_ERR("--multistage-live-prefill-reply-credit-limit must be >= -1\n");
        return 1;
    }
    if (!state_in_path.empty() && !state_chunks_dir_path.empty()) {
        LOG_ERR("use either --state-in or --state-chunks-dir, not both\n");
        return 1;
    }
    const bool split_stage2_mode =
        !split_embd_in_path.empty() ||
        !split_embd_stream_in_path.empty() ||
        !split_embd_stream_connect_addr.empty();
    const bool split_stage1_server_mode = !split_embd_stream_listen_addr.empty();
    if ((!state_in_path.empty() || !state_chunks_dir_path.empty()) &&
        replay_tokens_in_path.empty() &&
        !split_stage2_mode &&
        !split_stage1_server_mode) {
        LOG_ERR("--replay-tokens-in is required when --state-in is used\n");
        return 1;
    }
    if (state_mode != "full" && state_mode != "seq" && state_mode != "seq-partial") {
        LOG_ERR("--state-mode must be one of: full, seq, seq-partial\n");
        return 1;
    }
    if ((state_il_start >= 0 || state_il_end >= 0) && state_mode == "full") {
        LOG_ERR("--state-il-start/--state-il-end only support seq and seq-partial modes\n");
        return 1;
    }
    if (!multistage_compute_boundaries_csv.empty() && (!state_in_path.empty() || !state_chunks_dir_path.empty())) {
        LOG_ERR("--multistage-compute-boundaries does not support imported-state mode\n");
        return 1;
    }
    if (multistage_compute_boundaries_csv.empty() && !multistage_models_csv.empty()) {
        LOG_ERR("--multistage-models requires --multistage-compute-boundaries\n");
        return 1;
    }
    if (!multistage_stage_next_connect_addr.empty() && !multistage_stage_server_mode) {
        LOG_ERR("--multistage-stage-next-connect requires --multistage-stage-server-listen\n");
        return 1;
    }
    if (multistage_stage_server_mode && multistage_stage_session_count <= 0) {
        LOG_ERR("--multistage-stage-session-count must be > 0\n");
        return 1;
    }
    if ((multistage_stage_il_start >= 0 || multistage_stage_il_end >= 0) && !multistage_stage_server_mode) {
        LOG_ERR("--multistage-stage-il-start/--multistage-stage-il-end require --multistage-stage-server-listen\n");
        return 1;
    }
    if (multistage_stage_server_mode &&
        (multistage_stage_il_start >= 0 || multistage_stage_il_end >= 0) &&
        (multistage_stage_il_start < 0 || multistage_stage_il_end < 0 || multistage_stage_il_start >= multistage_stage_il_end)) {
        LOG_ERR("--multistage-stage-il-start/--multistage-stage-il-end must define a non-empty range\n");
        return 1;
    }
    if (multistage_stage_driver_mode && multistage_compute_boundaries_csv.empty()) {
        LOG_ERR("--multistage-stage-driver-connect requires --multistage-compute-boundaries\n");
        return 1;
    }
    if (split_stage2_mode) {
        if (state_in_path.empty() && state_chunks_dir_path.empty()) {
            LOG_ERR("--split-embd-in requires imported state\n");
            return 1;
        }
        if (split_stage2_layer < 0 || split_prompt_tokens < 0) {
            LOG_ERR("--split-embd-in requires --split-stage2-layer and --split-prompt-tokens\n");
            return 1;
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_context_params ctx_params = common_context_params_to_llama(params);
    baseline_prefill_layer_timing_probe baseline_layer_probe;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("failed to load model\n");
        return 1;
    }
    llama_model * shadow_model = nullptr;

    if (multistage_live_prefill &&
        !multistage_stage_server_mode &&
        !multistage_compute_boundaries_csv.empty() &&
        staged_same_topology_prompt_boundary_unsupported(model)) {
        LOG_ERR(
            "same-topology live prefill is unsupported for %s: prompt-boundary handoff diverges even though late-state split remains exact; use split late-state handoff instead\n",
            model_architecture_name(model).c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    if (baseline_prefill_layer_timing) {
        baseline_layer_probe.reset(llama_model_n_layer(model));
        ctx_params.cb_eval = baseline_prefill_layer_timing_eval_cb;
        ctx_params.cb_eval_user_data = &baseline_layer_probe;
    }

    llama_context * ctx = nullptr;
    if (!multistage_stage_server_mode) {
        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            LOG_ERR("failed to create context\n");
            llama_model_free(model);
            return 1;
        }
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (multistage_stage_server_mode) {
        const int32_t n_embd_inp = llama_model_n_embd_inp(model);
        const size_t colon = multistage_stage_server_listen_addr.rfind(':');
        if (colon == std::string::npos) {
            LOG_ERR("--multistage-stage-server-listen must be host:port\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const int listen_port = std::atoi(multistage_stage_server_listen_addr.substr(colon + 1).c_str());
        const bool has_next_stage = !multistage_stage_next_connect_addr.empty();
        std::string next_host = "127.0.0.1";
        int next_port = -1;
        if (has_next_stage) {
            const size_t next_colon = multistage_stage_next_connect_addr.rfind(':');
            if (next_colon == std::string::npos) {
                LOG_ERR("--multistage-stage-next-connect must be host:port\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            next_host = multistage_stage_next_connect_addr.substr(0, next_colon);
            next_port = std::atoi(multistage_stage_next_connect_addr.substr(next_colon + 1).c_str());
        }

        llama_context_params server_ctx_params = ctx_params;
        server_ctx_params.cb_eval = nullptr;
        server_ctx_params.cb_eval_user_data = nullptr;

        int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) {
            LOG_ERR("failed to create multistage stage server socket\n");
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        int opt = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        sockaddr_in listen_addr{};
        listen_addr.sin_family = AF_INET;
        listen_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        listen_addr.sin_port = htons((uint16_t) listen_port);
        if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&listen_addr), sizeof(listen_addr)) != 0 ||
            ::listen(listen_fd, 1) != 0) {
            ::close(listen_fd);
            LOG_ERR("failed to bind/listen multistage stage server socket\n");
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        std::vector<multistage_stage_server_session_summary> session_summaries;
        session_summaries.reserve((size_t) multistage_stage_session_count);

        for (int session_index = 0; session_index < multistage_stage_session_count; ++session_index) {
            const int conn_fd = ::accept(listen_fd, nullptr, nullptr);
            if (conn_fd < 0) {
                ::close(listen_fd);
                LOG_ERR("failed to accept multistage stage server connection\n");
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            int next_fd = -1;
            if (has_next_stage) {
                next_fd = ::socket(AF_INET, SOCK_STREAM, 0);
                if (next_fd < 0) {
                    ::close(conn_fd);
                    ::close(listen_fd);
                    LOG_ERR("failed to create multistage next-stage socket\n");
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                sockaddr_in next_addr{};
                next_addr.sin_family = AF_INET;
                next_addr.sin_port = htons((uint16_t) next_port);
                next_addr.sin_addr.s_addr = (next_host == "127.0.0.1" || next_host == "localhost")
                    ? htonl(INADDR_LOOPBACK)
                    : htonl(INADDR_LOOPBACK);
                if (::connect(next_fd, reinterpret_cast<sockaddr *>(&next_addr), sizeof(next_addr)) != 0) {
                    ::close(next_fd);
                    ::close(conn_fd);
                    ::close(listen_fd);
                    LOG_ERR("failed to connect multistage next-stage socket\n");
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
            }

            llama_context * session_ctx = llama_init_from_model(model, server_ctx_params);
            if (!session_ctx) {
                if (next_fd >= 0) {
                    ::close(next_fd);
                }
                ::close(conn_fd);
                ::close(listen_fd);
                LOG_ERR("failed to create multistage stage server context\n");
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            if (multistage_stage_il_start >= 0 && multistage_stage_il_end >= 0) {
                llama_set_compute_range(session_ctx, multistage_stage_il_start, multistage_stage_il_end);
            }
            llama_set_embeddings(session_ctx, true);
            llama_set_logits(session_ctx, !has_next_stage);
            llama_context * replay_ctx = nullptr;
            auto init_replay_ctx = [&]() -> bool {
                if (replay_ctx) {
                    return true;
                }
                replay_ctx = llama_init_from_model(model, server_ctx_params);
                if (!replay_ctx) {
                    return false;
                }
                if (multistage_stage_il_start >= 0 && multistage_stage_il_end >= 0) {
                    llama_set_compute_range(replay_ctx, multistage_stage_il_start, multistage_stage_il_end);
                }
                llama_set_embeddings(replay_ctx, true);
                llama_set_logits(replay_ctx, !has_next_stage);
                return true;
            };
            auto reset_replay_ctx = [&]() -> bool {
                if (replay_ctx) {
                    llama_free(replay_ctx);
                    replay_ctx = nullptr;
                }
                return init_replay_ctx();
            };

            multistage_stage_server_session_summary session_summary;
            session_summary.session_index = session_index;
            std::vector<llama_token> recv_tokens;
            std::vector<float> recv_embeddings;
            std::vector<uint8_t> recv_raw_bytes;
            std::vector<float> next_embeddings;
            std::deque<std::vector<uint8_t>> decode_replay_checkpoint_history;
            bool have_decode_replay_checkpoint = false;
            bool decode_replay_active = false;
            int decode_replay_start_pos = -1;
            int decode_replay_latest_committed_pos = -1;
            int pending_prefill_replies = 0;
            int expected_prompt_token_count = -1;
            int last_decode_step = -1;
            const int max_deferred_prefill_replies =
                multistage_live_prefill_reply_credit_limit >= 0
                    ? multistage_live_prefill_reply_credit_limit
                    : std::max(multistage_live_prefill_max_inflight - 1, 0);
            auto set_last_error = [&](const std::string & value) {
                if (session_summary.last_error.empty()) {
                    session_summary.last_error = value;
                }
            };
            auto recv_checked_downstream_reply = [&](int32_t expected_reply_kind, int32_t & predicted) {
                int32_t reply_kind = 0;
                if (!recv_stage_reply(next_fd, reply_kind, predicted)) {
                    set_last_error("forward_send_recv_failed");
                    return false;
                }
                if (reply_kind != expected_reply_kind) {
                    set_last_error(
                        expected_reply_kind == MULTISTAGE_STAGE_REPLY_ACK
                        ? "forward_reply_expected_ack"
                        : "forward_reply_missing_predicted_token");
                    return false;
                }
                return true;
            };

            while (true) {
                int32_t message_kind = 0;
                int32_t pos_start = 0;
                int32_t token_count = 0;
                multistage_stage_state_header recv_stage_state;
                recv_tokens.clear();
                recv_embeddings.clear();
                const double recv_t0 = ggml_time_us() / 1e6;
                if (!recv_stage_message(
                        conn_fd,
                        message_kind,
                        pos_start,
                        token_count,
                        recv_stage_state,
                        recv_tokens,
                        recv_embeddings,
                        recv_raw_bytes,
                        n_embd_inp)) {
                    set_last_error("recv_stage_message_failed");
                    break;
                }
                const double recv_t1 = ggml_time_us() / 1e6;
                const bool is_prefill_message = multistage_stage_msg_is_prefill(message_kind);
                const bool is_decode_replay_message = multistage_stage_msg_is_decode_replay(message_kind);
                const bool is_decode_light_context_message =
                    multistage_stage_msg_is_decode_light_context(message_kind);
                const bool reply_requires_predicted = multistage_stage_msg_requires_predicted_reply(message_kind);
                const bool is_control_message =
                    message_kind == MULTISTAGE_STAGE_MSG_STOP ||
                    message_kind == MULTISTAGE_STAGE_MSG_STATE_IMPORT;
                if (recv_stage_state.seq_id != 0) {
                    set_last_error("stage_state_seq_id_mismatch");
                    break;
                }
                if (message_kind != MULTISTAGE_STAGE_MSG_STOP &&
                    recv_stage_state.source_stage_index >= 0 &&
                    recv_stage_state.source_stage_index != multistage_stage_index - 1) {
                    set_last_error("stage_state_source_stage_mismatch");
                    break;
                }
                if (!is_control_message && !is_prefill_message && recv_stage_state.decode_step < 0) {
                    set_last_error("stage_state_decode_step_missing");
                    break;
                }
                if (!is_control_message && !is_prefill_message && recv_stage_state.prompt_token_count <= 0) {
                    set_last_error("stage_state_prompt_token_count_missing");
                    break;
                }
                if (!is_control_message &&
                    !is_prefill_message &&
                    pos_start != multistage_stage_expected_pos_start(message_kind, recv_stage_state, token_count)) {
                    set_last_error("stage_state_decode_pos_mismatch");
                    break;
                }
                if ((message_kind == MULTISTAGE_STAGE_MSG_DECODE_EMBD ||
                     is_decode_light_context_message) &&
                    !recv_tokens.empty() &&
                    recv_stage_state.current_token != LLAMA_TOKEN_NULL &&
                    recv_stage_state.current_token != recv_tokens.back()) {
                    set_last_error("stage_state_current_token_mismatch");
                    break;
                }
                if (!multistage_stage_state_matches_kind(message_kind, recv_stage_state)) {
                    set_last_error("stage_state_kind_mismatch");
                    break;
                }
                if (!is_control_message && !is_prefill_message) {
                    if (expected_prompt_token_count < 0) {
                        expected_prompt_token_count = recv_stage_state.prompt_token_count;
                    } else if (recv_stage_state.prompt_token_count != expected_prompt_token_count) {
                        set_last_error("stage_state_prompt_token_count_changed");
                        break;
                    }
                }
                if (!is_control_message && !is_prefill_message && !is_decode_replay_message) {
                    if (last_decode_step >= 0 && recv_stage_state.decode_step != last_decode_step + 1) {
                        set_last_error("stage_state_decode_step_nonmonotonic");
                        break;
                    }
                    last_decode_step = recv_stage_state.decode_step;
                }
                llama_context * active_ctx = session_ctx;
                if (is_prefill_message) {
                    session_summary.prefill_recv_elapsed_s += recv_t1 - recv_t0;
                    session_summary.prefill_input_bytes +=
                        (uint64_t) (sizeof(int32_t) * 4 +
                            sizeof(multistage_stage_state_header) +
                            (sizeof(llama_token) * recv_tokens.size()) +
                            (sizeof(float) * recv_embeddings.size()));
                }
                if (message_kind == MULTISTAGE_STAGE_MSG_STOP) {
                    if (pending_prefill_replies != 0) {
                        set_last_error("stop_with_pending_prefill_replies");
                    }
                    if (next_fd >= 0) {
                        (void) send_stage_message(
                            next_fd,
                            message_kind,
                            0,
                            0,
                            make_stage_state_header(message_kind),
                            nullptr,
                            0,
                            nullptr,
                            0);
                    }
                    break;
                }

                if (message_kind == MULTISTAGE_STAGE_MSG_STATE_IMPORT) {
                    const bool target_this_stage = pos_start == multistage_stage_index;
                    if (target_this_stage) {
                        if (multistage_stage_il_start < 0 || multistage_stage_il_end < 0) {
                            set_last_error("state_import_range_unavailable");
                            break;
                        }
                        llama_memory_t session_mem = llama_get_memory(session_ctx);
                        if (!session_mem) {
                            set_last_error("state_import_memory_unavailable");
                            break;
                        }
                        llama_memory_clear(session_mem, true);
                        if (!llama_memory_seq_rm(session_mem, 0, -1, -1)) {
                            set_last_error("state_import_memory_clear_failed");
                            break;
                        }
                        const size_t imported = llama_state_seq_set_data_range(
                            session_ctx,
                            recv_raw_bytes.data(),
                            recv_raw_bytes.size(),
                            0,
                            multistage_stage_il_start,
                            multistage_stage_il_end,
                            0);
                        if (imported != recv_raw_bytes.size()) {
                            set_last_error("state_import_failed");
                            break;
                        }
                    } else if (next_fd >= 0) {
                        multistage_stage_state_header forward_stage_state = recv_stage_state;
                        forward_stage_state.source_stage_index = multistage_stage_index;
                        const bool send_ok = send_stage_state_import_message(
                            next_fd,
                            pos_start,
                            forward_stage_state,
                            recv_raw_bytes.data(),
                            recv_raw_bytes.size());
                        if (!send_ok) {
                            set_last_error("state_import_forward_failed");
                            break;
                        }
                        int32_t ignored_predicted = -1;
                        if (!recv_checked_downstream_reply(MULTISTAGE_STAGE_REPLY_ACK, ignored_predicted)) {
                            break;
                        }
                    } else {
                        set_last_error("state_import_missing_target_stage");
                        break;
                    }

                    if (!send_stage_reply_ack(conn_fd)) {
                        set_last_error("reply_send_failed");
                        break;
                    }
                    continue;
                }

                if (message_kind == MULTISTAGE_STAGE_MSG_DECODE_READOUT) {
                    int32_t predicted = -1;
                    if (next_fd >= 0) {
                        multistage_stage_state_header forward_stage_state = recv_stage_state;
                        forward_stage_state.source_stage_index = multistage_stage_index;
                        const bool send_ok = send_stage_message(
                            next_fd,
                            message_kind,
                            pos_start,
                            0,
                            forward_stage_state,
                            nullptr,
                            0,
                            nullptr,
                            0);
                        if (!send_ok) {
                            set_last_error("readout_forward_failed");
                            break;
                        }
                        int32_t ignored_predicted = -1;
                        if (!recv_checked_downstream_reply(MULTISTAGE_STAGE_REPLY_PREDICTED_TOKEN, ignored_predicted)) {
                            break;
                        }
                        predicted = ignored_predicted;
                    } else {
                        const float * stage_logits = llama_get_logits_ith(active_ctx, -1);
                        if (!stage_logits) {
                            set_last_error("stage_logits_unavailable");
                            break;
                        }
                        predicted = (int32_t) select_greedy_token(stage_logits, n_vocab);
                    }
                    if (!send_stage_reply_predicted(conn_fd, predicted)) {
                        set_last_error("reply_send_failed");
                        break;
                    }
                    session_summary.decode_predicted_tokens.push_back(predicted);
                    continue;
                }

                const double stage_compute_t0 = ggml_time_us() / 1e6;
                if (is_prefill_message) {
                    session_summary.prefill_chunk_count += 1;
                } else if (message_kind == MULTISTAGE_STAGE_MSG_DECODE_EMBD) {
                    session_summary.decode_step_count += token_count;
                    session_summary.decode_input_hashes.push_back(
                        recv_embeddings.empty() ? 0ULL : fnv1a64_bytes(recv_embeddings.data(), sizeof(float) * recv_embeddings.size()));
                }

                const llama_model * stage_model = llama_get_model(active_ctx);
                const bool stage_requires_sequential_embd =
                    !is_prefill_message &&
                    stage_model != nullptr &&
                    (llama_model_is_recurrent(stage_model) || llama_model_is_hybrid(stage_model));
                const bool stage_requires_token_sideband =
                    stage_model != nullptr && staged_same_topology_requires_token_sideband(stage_model);
                next_embeddings.clear();
                bool next_embeddings_ready = false;
                if (is_decode_replay_message &&
                    multistage_live_decode_replay &&
                    token_count > 0 &&
                    !decode_replay_active) {
                    if (!init_replay_ctx()) {
                        set_last_error("decode_replay_context_init_failed");
                        break;
                    }
                    if (!have_decode_replay_checkpoint || decode_replay_checkpoint_history.empty()) {
                        set_last_error("decode_replay_checkpoint_unavailable");
                        break;
                    }
                    size_t checkpoints_back = 0;
                    if (decode_replay_start_pos >= 0 &&
                        decode_replay_latest_committed_pos >= pos_start) {
                        checkpoints_back = (size_t) (decode_replay_latest_committed_pos - pos_start + 1);
                    }
                    if (checkpoints_back >= decode_replay_checkpoint_history.size()) {
                        set_last_error("decode_replay_checkpoint_history_unavailable");
                        break;
                    }
                    if (checkpoints_back > 0) {
                        if (!reset_replay_ctx()) {
                            set_last_error("decode_replay_context_reset_failed");
                            break;
                        }
                    }
                    const auto & restore_checkpoint =
                        decode_replay_checkpoint_history[decode_replay_checkpoint_history.size() - 1 - checkpoints_back];
                    llama_memory_t replay_mem = llama_get_memory(replay_ctx);
                    if (!replay_mem) {
                        set_last_error("decode_replay_checkpoint_clear_failed");
                        break;
                    }
                    llama_memory_clear(replay_mem, true);
                    if (!llama_memory_seq_rm(replay_mem, 0, -1, -1)) {
                        set_last_error("decode_replay_checkpoint_clear_failed");
                        break;
                    }
                    const size_t restored = llama_state_seq_set_data(
                        replay_ctx,
                        restore_checkpoint.data(),
                        restore_checkpoint.size(),
                        0);
                    if (restored != restore_checkpoint.size()) {
                        set_last_error("decode_replay_checkpoint_restore_failed");
                        break;
                    }
                    decode_replay_active = true;
                }
                if (is_decode_light_context_message &&
                    multistage_live_decode_light_context_tokens > 0 &&
                    token_count > 0 &&
                    !decode_replay_active) {
                    if (!init_replay_ctx()) {
                        set_last_error("light_context_context_init_failed");
                        break;
                    }
                    if (!have_decode_replay_checkpoint || decode_replay_checkpoint_history.empty()) {
                        set_last_error("light_context_checkpoint_unavailable");
                        break;
                    }
                    size_t checkpoints_back = 0;
                    if (decode_replay_start_pos >= 0 &&
                        decode_replay_latest_committed_pos >= pos_start) {
                        checkpoints_back = (size_t) (decode_replay_latest_committed_pos - pos_start + 1);
                    }
                    if (checkpoints_back >= decode_replay_checkpoint_history.size()) {
                        set_last_error("light_context_checkpoint_history_unavailable");
                        break;
                    }
                    if (checkpoints_back > 0) {
                        if (!reset_replay_ctx()) {
                            set_last_error("light_context_context_reset_failed");
                            break;
                        }
                    }
                    const auto & restore_checkpoint =
                        decode_replay_checkpoint_history[decode_replay_checkpoint_history.size() - 1 - checkpoints_back];
                    llama_memory_t replay_mem = llama_get_memory(replay_ctx);
                    if (!replay_mem) {
                        set_last_error("light_context_checkpoint_clear_failed");
                        break;
                    }
                    llama_memory_clear(replay_mem, true);
                    if (!llama_memory_seq_rm(replay_mem, 0, -1, -1)) {
                        set_last_error("light_context_checkpoint_clear_failed");
                        break;
                    }
                    const size_t restored = llama_state_seq_set_data(
                        replay_ctx,
                        restore_checkpoint.data(),
                        restore_checkpoint.size(),
                        0);
                    if (restored != restore_checkpoint.size()) {
                        set_last_error("light_context_checkpoint_restore_failed");
                        break;
                    }
                    decode_replay_active = true;
                }
                if (is_decode_replay_message && multistage_live_decode_replay) {
                    if (!replay_ctx) {
                        set_last_error("decode_replay_context_unavailable");
                        break;
                    }
                    active_ctx = replay_ctx;
                } else if (is_decode_light_context_message && multistage_live_decode_light_context_tokens > 0) {
                    if (!replay_ctx) {
                        set_last_error("light_context_unavailable");
                        break;
                    }
                    active_ctx = replay_ctx;
                }

                if (stage_requires_sequential_embd &&
                    token_count > 1) {
                    bool stage_ok = true;
                    if (next_fd >= 0) {
                        next_embeddings.reserve((size_t) token_count * (size_t) n_embd_inp);
                    }
                    for (int token_index = 0; token_index < token_count; ++token_index) {
                        llama_batch embd_batch = llama_batch_init(1, n_embd_inp, 1);
                        embd_batch.n_tokens = 1;
                        if (!recv_embeddings.empty()) {
                            const double input_copy_t0 = ggml_time_us() / 1e6;
                            std::memcpy(
                                embd_batch.embd,
                                recv_embeddings.data() + ((size_t) token_index * (size_t) n_embd_inp),
                                sizeof(float) * (size_t) n_embd_inp);
                            const double input_copy_t1 = ggml_time_us() / 1e6;
                            session_summary.prefill_input_copy_elapsed_s += input_copy_t1 - input_copy_t0;
                        }
                        if (stage_requires_token_sideband && (size_t) token_index < recv_tokens.size()) {
                            populate_embd_batch_tokens(embd_batch, recv_tokens.data() + token_index, 1);
                        }
                        if (is_prefill_message || !multistage_live_decode_auto_pos) {
                            embd_batch.pos[0] = (llama_pos) (pos_start + token_index);
                            embd_batch.n_seq_id[0] = 1;
                            embd_batch.seq_id[0][0] = 0;
                        } else {
                            embd_batch.pos = nullptr;
                            embd_batch.n_seq_id = nullptr;
                            embd_batch.seq_id = nullptr;
                        }
                        embd_batch.logits[0] = (next_fd >= 0) || token_index == token_count - 1;
                        if (llama_decode(active_ctx, embd_batch)) {
                            stage_ok = false;
                            set_last_error("sequential_stage_decode_failed");
                        }
                        llama_batch_free(embd_batch);
                        if (!stage_ok) {
                            break;
                        }
                        if (next_fd >= 0) {
                            const double output_extract_t0 = ggml_time_us() / 1e6;
                            float * token_embeddings = llama_get_embeddings_ith(active_ctx, 0);
                            if (!token_embeddings) {
                                stage_ok = false;
                                set_last_error("sequential_token_embeddings_unavailable");
                                break;
                            }
                            next_embeddings.insert(next_embeddings.end(), token_embeddings, token_embeddings + n_embd_inp);
                            const double output_extract_t1 = ggml_time_us() / 1e6;
                            session_summary.prefill_output_extract_elapsed_s += output_extract_t1 - output_extract_t0;
                        }
                    }
                    if (!stage_ok) {
                        break;
                    }
                    next_embeddings_ready = next_fd >= 0;
                } else {
                    llama_batch embd_batch = llama_batch_init(token_count, n_embd_inp, 1);
                    embd_batch.n_tokens = token_count;
                    if (!recv_embeddings.empty()) {
                        const double input_copy_t0 = ggml_time_us() / 1e6;
                        std::memcpy(embd_batch.embd, recv_embeddings.data(), sizeof(float) * recv_embeddings.size());
                        const double input_copy_t1 = ggml_time_us() / 1e6;
                        session_summary.prefill_input_copy_elapsed_s += input_copy_t1 - input_copy_t0;
                    }
                    if (stage_requires_token_sideband && !recv_tokens.empty()) {
                        populate_embd_batch_tokens(embd_batch, recv_tokens.data(), token_count);
                    }
                    for (int token_index = 0; token_index < token_count; ++token_index) {
                        if (is_prefill_message || !multistage_live_decode_auto_pos) {
                            embd_batch.pos[token_index] = (llama_pos) (pos_start + token_index);
                            embd_batch.n_seq_id[token_index] = 1;
                            embd_batch.seq_id[token_index][0] = 0;
                        }
                        if (is_prefill_message) {
                            embd_batch.logits[token_index] = (next_fd >= 0) || token_index == token_count - 1;
                        } else {
                            embd_batch.logits[token_index] = (next_fd < 0) && token_index == token_count - 1;
                        }
                    }
                    if (!is_prefill_message && multistage_live_decode_auto_pos) {
                        embd_batch.pos = nullptr;
                        embd_batch.n_seq_id = nullptr;
                        embd_batch.seq_id = nullptr;
                    }
                    if (llama_decode(active_ctx, embd_batch)) {
                        llama_batch_free(embd_batch);
                        set_last_error("stage_decode_failed");
                        break;
                    }
                    llama_batch_free(embd_batch);
                }

                int32_t predicted = -1;
                if (next_fd >= 0) {
                    if (!next_embeddings_ready) {
                        next_embeddings.reserve((size_t) token_count * (size_t) n_embd_inp);
                        bool extraction_ok = true;
                        const double output_extract_t0 = ggml_time_us() / 1e6;
                        for (int token_index = 0; token_index < token_count; ++token_index) {
                            // Prefill needs row-accurate extraction for multi-token batches, but
                            // single-token decode has historically matched the monolithic path more
                            // closely when we read the latest row via -1.
                            const int embd_index =
                                (!is_prefill_message && token_count == 1) ? -1 : token_index;
                            float * token_embeddings = llama_get_embeddings_ith(active_ctx, embd_index);
                            if (!token_embeddings) {
                                extraction_ok = false;
                                set_last_error("forward_token_embeddings_unavailable");
                                break;
                            }
                            next_embeddings.insert(next_embeddings.end(), token_embeddings, token_embeddings + n_embd_inp);
                        }
                        const double output_extract_t1 = ggml_time_us() / 1e6;
                        if (is_prefill_message && extraction_ok) {
                            session_summary.prefill_output_extract_elapsed_s += output_extract_t1 - output_extract_t0;
                        }
                        next_embeddings_ready = extraction_ok;
                    }
                    if (next_embeddings_ready && token_count > 0) {
                        if (message_kind == MULTISTAGE_STAGE_MSG_DECODE_EMBD) {
                            session_summary.decode_output_hashes.push_back(
                                fnv1a64_bytes(next_embeddings.data(), sizeof(float) * next_embeddings.size()));
                        }
                        multistage_stage_state_header forward_stage_state = recv_stage_state;
                        forward_stage_state.source_stage_index = multistage_stage_index;
                        const double forward_t0 = ggml_time_us() / 1e6;
                        const bool send_ok = send_stage_message(
                            next_fd,
                            message_kind,
                            pos_start,
                            token_count,
                            forward_stage_state,
                            recv_tokens.empty() ? nullptr : recv_tokens.data(),
                            recv_tokens.size(),
                            next_embeddings.data(),
                            next_embeddings.size());
                        const double forward_send_t1 = ggml_time_us() / 1e6;
                        if (!send_ok) {
                            set_last_error("forward_send_recv_failed");
                            break;
                        }
                        double forward_t1 = forward_send_t1;
                        if (reply_requires_predicted) {
                            int32_t ignored_predicted = -1;
                            while (pending_prefill_replies > 0) {
                                if (!recv_checked_downstream_reply(MULTISTAGE_STAGE_REPLY_ACK, ignored_predicted)) {
                                    break;
                                }
                                pending_prefill_replies -= 1;
                            }
                            if (!session_summary.last_error.empty()) {
                                break;
                            }
                            if (!recv_checked_downstream_reply(MULTISTAGE_STAGE_REPLY_PREDICTED_TOKEN, predicted)) {
                                break;
                            }
                            forward_t1 = ggml_time_us() / 1e6;
                        } else {
                            if (max_deferred_prefill_replies <= 0) {
                                int32_t ignored_predicted = -1;
                                if (!recv_checked_downstream_reply(MULTISTAGE_STAGE_REPLY_ACK, ignored_predicted)) {
                                    break;
                                }
                                forward_t1 = ggml_time_us() / 1e6;
                            } else {
                                while (pending_prefill_replies >= max_deferred_prefill_replies) {
                                    int32_t ignored_predicted = -1;
                                    if (!recv_checked_downstream_reply(MULTISTAGE_STAGE_REPLY_ACK, ignored_predicted)) {
                                        break;
                                    }
                                    pending_prefill_replies -= 1;
                                }
                                if (!session_summary.last_error.empty()) {
                                    break;
                                }
                                pending_prefill_replies += 1;
                            }
                        }
                        if (is_prefill_message) {
                            session_summary.prefill_forward_elapsed_s += forward_t1 - forward_t0;
                            session_summary.prefill_forward_send_elapsed_s += forward_send_t1 - forward_t0;
                            session_summary.prefill_forward_reply_wait_elapsed_s += forward_t1 - forward_send_t1;
                            session_summary.prefill_forward_bytes +=
                                (uint64_t) (sizeof(int32_t) * 4 +
                                    sizeof(multistage_stage_state_header) +
                                    (sizeof(llama_token) * recv_tokens.size()) +
                                    (sizeof(float) * next_embeddings.size()));
                        }
                    } else {
                        set_last_error("forward_embeddings_not_ready");
                        break;
                    }
                } else if (reply_requires_predicted) {
                    const float * stage_logits = llama_get_logits_ith(active_ctx, -1);
                    if (!stage_logits) {
                        set_last_error("stage_logits_unavailable");
                        break;
                    }
                    predicted = (int32_t) select_greedy_token(stage_logits, n_vocab);
                }

                const double stage_compute_t1 = ggml_time_us() / 1e6;
                if (is_prefill_message) {
                    session_summary.prefill_compute_elapsed_s += stage_compute_t1 - stage_compute_t0;
                    session_summary.prefill_stage_state_bytes =
                        std::max(
                            session_summary.prefill_stage_state_bytes,
                            (uint64_t) llama_state_seq_get_size(session_ctx, 0));
                    if (message_kind == MULTISTAGE_STAGE_MSG_PREFILL_FINAL_EMBD &&
                        (multistage_live_decode_replay ||
                         multistage_live_decode_light_context_tokens > 0)) {
                        const size_t checkpoint_size = llama_state_seq_get_size(session_ctx, 0);
                        if (checkpoint_size == 0) {
                            set_last_error("decode_replay_checkpoint_empty");
                            break;
                        }
                        std::vector<uint8_t> decode_replay_checkpoint;
                        decode_replay_checkpoint.resize(checkpoint_size);
                        const size_t serialized = llama_state_seq_get_data(
                            session_ctx,
                            decode_replay_checkpoint.data(),
                            decode_replay_checkpoint.size(),
                            0);
                        if (serialized != checkpoint_size) {
                            set_last_error("decode_replay_checkpoint_export_failed");
                            break;
                        }
                        decode_replay_checkpoint_history.clear();
                        decode_replay_checkpoint_history.push_back(std::move(decode_replay_checkpoint));
                        have_decode_replay_checkpoint = true;
                        decode_replay_start_pos = pos_start + token_count;
                        decode_replay_latest_committed_pos = decode_replay_start_pos - 1;
                        if (recv_stage_state.checkpoint_generation > 0) {
                            decode_replay_latest_committed_pos = decode_replay_start_pos - 1;
                        }
                    }
                } else if (message_kind == MULTISTAGE_STAGE_MSG_DECODE_EMBD ||
                           is_decode_light_context_message ||
                           is_decode_replay_message) {
                    session_summary.decode_compute_elapsed_s += stage_compute_t1 - stage_compute_t0;
                    session_summary.decode_predicted_tokens.push_back(predicted);
                    if (is_decode_replay_message &&
                        recv_stage_state.checkpoint_generation <= 0) {
                        set_last_error("decode_replay_generation_missing");
                        break;
                    }
                    if (message_kind == MULTISTAGE_STAGE_MSG_DECODE_REPLAY_FINAL_EMBD) {
                        const size_t checkpoint_size = llama_state_seq_get_size(active_ctx, 0);
                        if (checkpoint_size == 0) {
                            set_last_error("decode_replay_checkpoint_advance_empty");
                            break;
                        }
                        std::vector<uint8_t> decode_replay_checkpoint;
                        decode_replay_checkpoint.resize(checkpoint_size);
                        const size_t serialized = llama_state_seq_get_data(
                            active_ctx,
                            decode_replay_checkpoint.data(),
                            decode_replay_checkpoint.size(),
                            0);
                        if (serialized != checkpoint_size) {
                            set_last_error("decode_replay_checkpoint_advance_failed");
                            break;
                        }
                        decode_replay_checkpoint_history.push_back(std::move(decode_replay_checkpoint));
                        const size_t max_checkpoint_history =
                            (size_t) std::max(multistage_live_decode_replay_window, 1) + 1;
                        while (decode_replay_checkpoint_history.size() > max_checkpoint_history) {
                            decode_replay_checkpoint_history.pop_front();
                        }
                        have_decode_replay_checkpoint = true;
                        decode_replay_latest_committed_pos = pos_start + token_count - 1;
                        decode_replay_active = false;
                    } else if (is_decode_light_context_message) {
                        if (recv_stage_state.checkpoint_generation <= 0) {
                            set_last_error("light_context_generation_missing");
                            break;
                        }
                        const size_t checkpoint_size = llama_state_seq_get_size(active_ctx, 0);
                        if (checkpoint_size == 0) {
                            set_last_error("light_context_checkpoint_advance_empty");
                            break;
                        }
                        std::vector<uint8_t> decode_replay_checkpoint;
                        decode_replay_checkpoint.resize(checkpoint_size);
                        const size_t serialized = llama_state_seq_get_data(
                            active_ctx,
                            decode_replay_checkpoint.data(),
                            decode_replay_checkpoint.size(),
                            0);
                        if (serialized != checkpoint_size) {
                            set_last_error("light_context_checkpoint_advance_failed");
                            break;
                        }
                        decode_replay_checkpoint_history.push_back(std::move(decode_replay_checkpoint));
                        const size_t max_checkpoint_history =
                            (size_t) std::max(multistage_live_decode_light_context_tokens, 1) + 1;
                        while (decode_replay_checkpoint_history.size() > max_checkpoint_history) {
                            decode_replay_checkpoint_history.pop_front();
                        }
                        have_decode_replay_checkpoint = true;
                        decode_replay_latest_committed_pos = pos_start + token_count - 1;
                        decode_replay_active = false;
                    }
                }

                const double reply_t0 = ggml_time_us() / 1e6;
                const bool reply_ok = reply_requires_predicted
                    ? send_stage_reply_predicted(conn_fd, predicted)
                    : send_stage_reply_ack(conn_fd);
                if (!reply_ok) {
                    set_last_error("reply_send_failed");
                    break;
                }
                const double reply_t1 = ggml_time_us() / 1e6;
                if (is_prefill_message) {
                    session_summary.prefill_reply_elapsed_s += reply_t1 - reply_t0;
                    session_summary.prefill_reply_bytes += sizeof(int32_t) * 2;
                }
            }

            if (next_fd >= 0) {
                ::close(next_fd);
            }
            ::close(conn_fd);
            llama_free(session_ctx);
            if (replay_ctx) {
                llama_free(replay_ctx);
            }
            session_summaries.push_back(std::move(session_summary));
        }

        ::close(listen_fd);
        if (!json_out_path.empty()) {
            write_multistage_stage_server_json(
                json_out_path,
                multistage_stage_index,
                has_next_stage,
                session_summaries);
        }
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    if (split_stage1_server_mode) {
        const int32_t n_layer = llama_model_n_layer(model);
        const int32_t n_embd_inp = llama_model_n_embd_inp(model);
        if (split_compute_layer <= 0 || split_compute_layer >= n_layer) {
            LOG_ERR("--split-embd-stream-listen requires --split-compute-layer in (0, %d)\n", n_layer);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        if (split_seed_token < 0) {
            LOG_ERR("--split-embd-stream-listen requires --split-seed-token\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const size_t colon = split_embd_stream_listen_addr.rfind(':');
        if (colon == std::string::npos) {
            LOG_ERR("--split-embd-stream-listen must be host:port\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const int port = std::atoi(split_embd_stream_listen_addr.substr(colon + 1).c_str());

        const std::string effective_prompt = apply_chat_template
            ? build_generation_prompt(model, params.prompt).first
            : params.prompt;
        std::vector<llama_token> prompt_tokens = common_tokenize(ctx, effective_prompt, add_bos, true);
        llama_context_params stage1_ctx_params = ctx_params;
        stage1_ctx_params.cb_eval = nullptr;
        stage1_ctx_params.cb_eval_user_data = nullptr;
        llama_context * stage1_ctx = llama_init_from_model(model, stage1_ctx_params);
        if (!stage1_ctx) {
            LOG_ERR("failed to create split stage1 server context\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        llama_set_embeddings(stage1_ctx, true);
        llama_set_logits(stage1_ctx, false);
        llama_set_compute_range(stage1_ctx, 0, split_compute_layer);
        if (llama_decode(stage1_ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
            LOG_ERR("split stage1 server prompt prefill failed\n");
            llama_free(stage1_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        llama_context * draft_ctx = nullptr;
        if (split_speculative_tokens > 1 && split_speculative_draft_mode == "self-skip") {
            if ((state_in_path.empty() && state_chunks_dir_path.empty()) ||
                state_mode != "seq" ||
                state_il_start < 0 ||
                state_il_end < 0) {
                LOG_ERR("self-skip speculative mode requires seq late state via --state-in/--state-chunks-dir and --state-il-start/--state-il-end\n");
                llama_free(stage1_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            draft_ctx = ctx;
            llama_set_embeddings(draft_ctx, true);
            std::vector<uint8_t> draft_state = !state_in_path.empty()
                ? read_binary_file(state_in_path)
                : read_chunk_directory(state_chunks_dir_path);
            const size_t imported = llama_state_seq_set_data_range(
                draft_ctx,
                draft_state.data(),
                draft_state.size(),
                0,
                state_il_start,
                state_il_end,
                0);
            if (imported != draft_state.size()) {
                LOG_ERR("failed to import late state for self-skip draft mode\n");
                llama_free(stage1_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            llama_set_compute_range(draft_ctx, split_compute_layer, llama_model_n_layer(model));
            llama_set_compute_skip_stride(draft_ctx, split_draft_skip_stride);
        }

        int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) {
            LOG_ERR("failed to create split stage1 server socket\n");
            llama_free(stage1_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        int opt = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons((uint16_t) port);
        if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || ::listen(listen_fd, 1) != 0) {
            LOG_ERR("failed to bind/listen split stage1 server socket\n");
            ::close(listen_fd);
            llama_free(stage1_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const int conn_fd = ::accept(listen_fd, nullptr, nullptr);
        if (conn_fd < 0) {
            LOG_ERR("failed to accept split stage1 server connection\n");
            ::close(listen_fd);
            llama_free(stage1_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        llama_token current_token = (llama_token) split_seed_token;
        std::vector<llama_token> history_tokens = prompt_tokens;
        history_tokens.push_back(current_token);
        std::vector<llama_token> oracle_tokens;
        if (split_speculative_tokens > 1 && split_speculative_draft_mode == "oracle") {
            if (split_draft_tokens_in_path.empty()) {
                LOG_ERR("oracle speculative mode requires --split-draft-tokens-in\n");
                ::close(conn_fd);
                ::close(listen_fd);
                llama_free(stage1_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            oracle_tokens = read_token_file(split_draft_tokens_in_path);
        }
        std::string output_text;
        int generated_tokens = 0;
        prefill_handoff_metrics metrics;
        metrics.enabled = true;
        metrics.state_kind = "split_stage1_live_server";
        metrics.speculative_tokens = split_speculative_tokens;
        metrics.speculative_draft_mode =
            split_speculative_tokens > 1 ? split_speculative_draft_mode : "disabled";
        for (int step = 0; step < params.n_predict;) {
            if (llama_vocab_is_eog(vocab, current_token)) {
                break;
            }
            if (split_speculative_tokens <= 1) {
                if (llama_decode(stage1_ctx, llama_batch_get_one(&current_token, 1))) {
                    break;
                }
                float * boundary_embd = llama_get_embeddings_ith(stage1_ctx, -1);
                if (!boundary_embd) {
                    break;
                }
                const int32_t step_i32 = (int32_t) step;
                if (!send_all(conn_fd, &step_i32, sizeof(step_i32)) ||
                    !send_all(conn_fd, boundary_embd, sizeof(float) * (size_t) n_embd_inp)) {
                    break;
                }
                int32_t predicted = -1;
                if (!recv_all(conn_fd, &predicted, sizeof(predicted))) {
                    break;
                }
                current_token = (llama_token) predicted;
                history_tokens.push_back(current_token);
                output_text += common_token_to_piece(stage1_ctx, current_token, false);
                generated_tokens++;
                ++step;
                continue;
            }

            const size_t checkpoint_size =
                llama_state_seq_get_size_range(stage1_ctx, 0, 0, split_compute_layer, 0);
            std::vector<uint8_t> checkpoint(checkpoint_size);
            const size_t checkpoint_written =
                llama_state_seq_get_data_range(stage1_ctx, checkpoint.data(), checkpoint.size(), 0, 0, split_compute_layer, 0);
            checkpoint.resize(checkpoint_written);
            const llama_token checkpoint_token = current_token;
            const size_t checkpoint_history_size = history_tokens.size();
            std::vector<uint8_t> draft_checkpoint;
            if (draft_ctx) {
                const size_t draft_checkpoint_size =
                    llama_state_seq_get_size_range(draft_ctx, 0, split_compute_layer, llama_model_n_layer(model), 0);
                draft_checkpoint.resize(draft_checkpoint_size);
                const size_t draft_written =
                    llama_state_seq_get_data_range(draft_ctx, draft_checkpoint.data(), draft_checkpoint.size(), 0, split_compute_layer, llama_model_n_layer(model), 0);
                draft_checkpoint.resize(draft_written);
            }

            const int remaining_steps = params.n_predict - step;
            const int block_len = std::min(split_speculative_tokens, remaining_steps);
            std::vector<float> block_embd((size_t) block_len * (size_t) n_embd_inp);
            std::vector<llama_token> draft_tokens;
            draft_tokens.reserve((size_t) block_len);

            llama_token speculative_current = current_token;
            std::vector<llama_token> speculative_history = history_tokens;
            int produced = 0;
            for (; produced < block_len; ++produced) {
                if (llama_vocab_is_eog(vocab, speculative_current)) {
                    break;
                }
                if (llama_decode(stage1_ctx, llama_batch_get_one(&speculative_current, 1))) {
                    break;
                }
                float * boundary_embd = llama_get_embeddings_ith(stage1_ctx, -1);
                if (!boundary_embd) {
                    break;
                }
                std::memcpy(
                    block_embd.data() + (size_t) produced * (size_t) n_embd_inp,
                    boundary_embd,
                    sizeof(float) * (size_t) n_embd_inp
                );
                llama_token draft = LLAMA_TOKEN_NULL;
                if (split_speculative_draft_mode == "oracle" &&
                    (size_t) (step + produced + 1) < oracle_tokens.size()) {
                    draft = oracle_tokens[(size_t) (step + produced + 1)];
                } else if (draft_ctx) {
                    llama_batch embd_batch = llama_batch_init(1, n_embd_inp, 1);
                    embd_batch.n_tokens = 1;
                    std::memcpy(
                        embd_batch.embd,
                        block_embd.data() + (size_t) produced * (size_t) n_embd_inp,
                        sizeof(float) * (size_t) n_embd_inp
                    );
                    embd_batch.pos[0] = (llama_pos) (prompt_tokens.size() + step + produced);
                    embd_batch.n_seq_id[0] = 1;
                    embd_batch.seq_id[0][0] = 0;
                    embd_batch.logits[0] = 1;
                    if (llama_decode(draft_ctx, embd_batch) == 0) {
                        const float * draft_logits = llama_get_logits_ith(draft_ctx, -1);
                        if (draft_logits) {
                            draft = select_greedy_token(draft_logits, n_vocab);
                        }
                    }
                    llama_batch_free(embd_batch);
                } else {
                    draft = propose_ngram_token(speculative_history);
                }
                if (draft == LLAMA_TOKEN_NULL) {
                    draft = speculative_current;
                }
                draft_tokens.push_back(draft);
                speculative_history.push_back(draft);
                speculative_current = draft;
            }
            if (produced <= 0) {
                break;
            }

            const int32_t block_step_i32 = (int32_t) step;
            const int32_t block_len_i32 = produced;
            if (!send_all(conn_fd, &block_step_i32, sizeof(block_step_i32)) ||
                !send_all(conn_fd, &block_len_i32, sizeof(block_len_i32))) {
                break;
            }
            bool send_ok = true;
            for (int i = 0; i < produced; ++i) {
                const int32_t draft_i32 = (int32_t) draft_tokens[(size_t) i];
                if (!send_all(conn_fd, &draft_i32, sizeof(draft_i32)) ||
                    !send_all(
                        conn_fd,
                        block_embd.data() + (size_t) i * (size_t) n_embd_inp,
                        sizeof(float) * (size_t) n_embd_inp
                    )) {
                    send_ok = false;
                    break;
                }
            }
            if (!send_ok) {
                break;
            }

            int32_t accepted_prefix_len = 0;
            int32_t returned_token_count = 0;
            if (!recv_all(conn_fd, &accepted_prefix_len, sizeof(accepted_prefix_len)) ||
                !recv_all(conn_fd, &returned_token_count, sizeof(returned_token_count))) {
                break;
            }
            if (accepted_prefix_len < 0 || accepted_prefix_len > produced ||
                returned_token_count <= 0 || returned_token_count > produced) {
                break;
            }
            std::vector<int32_t> returned_i32((size_t) returned_token_count);
            if (!recv_all(conn_fd, returned_i32.data(), sizeof(int32_t) * returned_i32.size())) {
                break;
            }

            metrics.speculative_rounds++;
            metrics.speculative_drafted_tokens += produced;
            metrics.speculative_accepted_tokens += accepted_prefix_len;
            metrics.speculative_verified_tokens += returned_token_count;
            if (accepted_prefix_len < produced) {
                metrics.speculative_rollback_count++;
                current_token = checkpoint_token;
                history_tokens.resize(checkpoint_history_size);
                if (llama_state_seq_set_data_range(
                        stage1_ctx,
                        checkpoint.data(),
                        checkpoint.size(),
                        0,
                        0,
                        split_compute_layer,
                        0) != checkpoint.size()) {
                    break;
                }
                if (draft_ctx && llama_state_seq_set_data_range(
                        draft_ctx,
                        draft_checkpoint.data(),
                        draft_checkpoint.size(),
                        0,
                        split_compute_layer,
                        llama_model_n_layer(model),
                        0) != draft_checkpoint.size()) {
                    break;
                }
                for (int i = 0; i < returned_token_count; ++i) {
                    if (llama_decode(stage1_ctx, llama_batch_get_one(&current_token, 1))) {
                        returned_token_count = i;
                        break;
                    }
                    float * exact_boundary_embd = llama_get_embeddings_ith(stage1_ctx, -1);
                    current_token = (llama_token) returned_i32[(size_t) i];
                    history_tokens.push_back(current_token);
                    output_text += common_token_to_piece(stage1_ctx, current_token, false);
                    generated_tokens++;
                    if (draft_ctx && exact_boundary_embd) {
                        llama_batch embd_batch = llama_batch_init(1, n_embd_inp, 1);
                        embd_batch.n_tokens = 1;
                        std::memcpy(embd_batch.embd, exact_boundary_embd, sizeof(float) * (size_t) n_embd_inp);
                        embd_batch.pos[0] = (llama_pos) (prompt_tokens.size() + step + i);
                        embd_batch.n_seq_id[0] = 1;
                        embd_batch.seq_id[0][0] = 0;
                        embd_batch.logits[0] = 1;
                        (void) llama_decode(draft_ctx, embd_batch);
                        llama_batch_free(embd_batch);
                    }
                }
            } else {
                for (int i = 0; i < returned_token_count; ++i) {
                    current_token = (llama_token) returned_i32[(size_t) i];
                    history_tokens.push_back(current_token);
                    output_text += common_token_to_piece(stage1_ctx, current_token, false);
                    generated_tokens++;
                }
            }
            step += returned_token_count;
        }

        ::close(conn_fd);
        ::close(listen_fd);
        metrics.generated_tokens = generated_tokens;
        metrics.output_text = output_text;
        if (!json_out_path.empty()) {
            write_result_json(json_out_path, params.prompt, metrics);
        }

        llama_free(stage1_ctx);
        if (!draft_ctx) {
            llama_free(ctx);
        } else {
            llama_free(draft_ctx);
        }
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    if (!multistage_compute_boundaries_csv.empty()) {
        const int32_t n_layer = llama_model_n_layer(model);
        const int32_t n_embd_inp = llama_model_n_embd_inp(model);
        std::vector<int32_t> boundaries = parse_int32_csv(multistage_compute_boundaries_csv);
        if (boundaries.empty()) {
            LOG_ERR("--multistage-compute-boundaries requires at least one boundary\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        int32_t prev_boundary = 0;
        for (int32_t boundary : boundaries) {
            if (boundary <= prev_boundary || boundary >= n_layer) {
                LOG_ERR("--multistage-compute-boundaries must be strictly increasing in (0, %d)\n", n_layer);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            prev_boundary = boundary;
        }

        std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, add_bos, true);
        if (prompt_tokens.empty()) {
            LOG_ERR("multistage compute proof requires a non-empty prompt\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        multistage_compute_metrics metrics;
        metrics.enabled = true;
        metrics.boundaries = boundaries;
        metrics.stage_count = (int) boundaries.size() + 1;
        metrics.prompt_tokens = (int) prompt_tokens.size();
        metrics.requested_decode_tokens = params.n_predict;
        metrics.boundary_embedding_bytes = (uint64_t) n_embd_inp * sizeof(float);

        std::vector<std::string> multistage_model_paths;
        if (!multistage_models_csv.empty()) {
            multistage_model_paths = parse_string_csv(multistage_models_csv);
            if ((int) multistage_model_paths.size() != metrics.stage_count) {
                LOG_ERR("--multistage-models count (%zu) must match stage count (%d)\n",
                    multistage_model_paths.size(), metrics.stage_count);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }

        llama_context_params proof_ctx_params = ctx_params;
        proof_ctx_params.cb_eval = nullptr;
        proof_ctx_params.cb_eval_user_data = nullptr;

        std::vector<std::pair<int32_t, int32_t>> stage_ranges;
        stage_ranges.reserve((size_t) metrics.stage_count);
        int32_t stage_start = 0;
        for (int32_t boundary : boundaries) {
            stage_ranges.emplace_back(stage_start, boundary);
            stage_start = boundary;
        }
        stage_ranges.emplace_back(stage_start, n_layer);

        if (multistage_live_prefill && !multistage_stage_driver_connect_addr.empty()) {
            const size_t colon = multistage_stage_driver_connect_addr.rfind(':');
            if (colon == std::string::npos) {
                LOG_ERR("--multistage-stage-driver-connect must be host:port\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            const std::string host = multistage_stage_driver_connect_addr.substr(0, colon);
            const int port = std::atoi(multistage_stage_driver_connect_addr.substr(colon + 1).c_str());

            if (baseline_prefill_layer_timing) {
                baseline_layer_probe.begin();
            }
            const double base_prefill_t0 = ggml_time_us() / 1e6;
            if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
                LOG_ERR("baseline prompt prefill failed\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            const double base_prefill_t1 = ggml_time_us() / 1e6;
            if (baseline_prefill_layer_timing) {
                baseline_layer_probe.end();
                metrics.baseline_prefill_layer_elapsed_s = baseline_layer_probe.layer_elapsed_s;
                metrics.baseline_prefill_layer_observation_count = baseline_layer_probe.layer_observation_count;
            }
            metrics.baseline_prefill_elapsed_s = base_prefill_t1 - base_prefill_t0;
            metrics.baseline_state_bytes = llama_state_seq_get_size(ctx, 0);

            llama_model * stage0_model = model;
            if (!multistage_model_paths.empty()) {
                const auto [il_start, il_end] = stage_ranges[0];
                const auto stage0_model_params = clamp_stage_model_gpu_layers(model_params, il_end - il_start);
                stage0_model = llama_model_load_from_file(multistage_model_paths[0].c_str(), stage0_model_params);
            }
            if (!stage0_model) {
                LOG_ERR("failed to load multistage stage0 model\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            llama_context * stage0_ctx = llama_init_from_model(stage0_model, proof_ctx_params);
            if (!stage0_ctx) {
                if (stage0_model != model) {
                    llama_model_free(stage0_model);
                }
                LOG_ERR("failed to create multistage stage0 context\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            llama_set_embeddings(stage0_ctx, true);
            llama_set_logits(stage0_ctx, false);
            const bool stage0_uses_shard_model =
                !multistage_model_paths.empty() &&
                multistage_model_paths[0] != params.model.path;
            if (!stage0_uses_shard_model) {
                llama_set_compute_range(stage0_ctx, stage_ranges[0].first, stage_ranges[0].second);
            }

            int downstream_fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (downstream_fd < 0) {
                llama_free(stage0_ctx);
                if (stage0_model != model) {
                    llama_model_free(stage0_model);
                }
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                LOG_ERR("failed to create multistage downstream socket\n");
                return 1;
            }
            sockaddr_in downstream_addr{};
            downstream_addr.sin_family = AF_INET;
            downstream_addr.sin_port = htons((uint16_t) port);
            downstream_addr.sin_addr.s_addr = (host == "127.0.0.1" || host == "localhost")
                ? htonl(INADDR_LOOPBACK)
                : htonl(INADDR_LOOPBACK);
            if (::connect(downstream_fd, reinterpret_cast<sockaddr *>(&downstream_addr), sizeof(downstream_addr)) != 0) {
                ::close(downstream_fd);
                llama_free(stage0_ctx);
                if (stage0_model != model) {
                    llama_model_free(stage0_model);
                }
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                LOG_ERR("failed to connect multistage downstream socket\n");
                return 1;
            }

            metrics.live_prefill_enabled = true;
            metrics.live_prefill_chunk_tokens = multistage_live_prefill_chunk_tokens;
            metrics.live_prefill_downstream_chunk_tokens =
                multistage_live_prefill_downstream_chunk_tokens > 0
                ? multistage_live_prefill_downstream_chunk_tokens
                : multistage_live_prefill_chunk_tokens;
            metrics.live_prefill_tile_tokens = multistage_live_prefill_tile_tokens;
            metrics.live_prefill_fill_chunk_tokens = multistage_live_prefill_fill_chunk_tokens;
            metrics.live_prefill_fill_chunk_count = multistage_live_prefill_fill_chunk_count;
            metrics.live_prefill_max_inflight = multistage_live_prefill_max_inflight;
            metrics.live_prefill_source_buffer_enabled = multistage_live_prefill_source_buffer;
            metrics.keep_stage_logits = multistage_keep_stage_logits;
            metrics.live_prefill_hop_latency_ms = multistage_live_prefill_hop_latency_ms;
            metrics.live_prefill_hop_jitter_ms = multistage_live_prefill_hop_jitter_ms;
            metrics.live_prefill_hop_bandwidth_mbps = multistage_live_prefill_hop_bandwidth_mbps;
            metrics.live_prefill_stage_chunk_counts.resize((size_t) metrics.stage_count, 0);
            metrics.live_prefill_stage_compute_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
            metrics.live_prefill_stage_send_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
            metrics.live_prefill_stage_reply_wait_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
            metrics.live_prefill_stage_input_copy_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
            metrics.live_prefill_stage_output_extract_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
            metrics.live_prefill_stage_runtime_lock_wait_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
            metrics.live_prefill_stage_runtime_lock_hold_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
            metrics.live_prefill_stage_input_wait_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
            metrics.live_prefill_stage_output_wait_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
            metrics.live_prefill_stage_input_max_queue_depth.resize((size_t) metrics.stage_count, 0);
            metrics.live_prefill_stage_output_max_queue_depth.resize((size_t) metrics.stage_count, 0);
            metrics.live_prefill_edge_chunk_counts.resize((size_t) std::max(metrics.stage_count - 1, 0), 0);
            metrics.live_prefill_edge_transport_elapsed_s.resize((size_t) std::max(metrics.stage_count - 1, 0), 0.0);
            metrics.live_prefill_edge_max_queue_depth.resize((size_t) std::max(metrics.stage_count - 1, 0), 0);

            std::mt19937 hop_rng((uint32_t) multistage_live_prefill_seed);
            const std::vector<std::pair<int, int>> prompt_chunks =
                make_adaptive_prompt_chunks(
                    (int) prompt_tokens.size(),
                    multistage_live_prefill_fill_chunk_tokens,
                    multistage_live_prefill_fill_chunk_count,
                    multistage_live_prefill_chunk_tokens);
            llama_token current_token = LLAMA_TOKEN_NULL;
            const double staged_prefill_t0 = ggml_time_us() / 1e6;
            int prefill_chunk_count = 0;
            double stage0_prefill_compute_elapsed_s = 0.0;
            double stage0_transport_elapsed_s = 0.0;
            double stage0_send_elapsed_s = 0.0;
            double stage0_reply_wait_elapsed_s = 0.0;
            double stage0_output_extract_elapsed_s = 0.0;
            for (const auto & prompt_chunk : prompt_chunks) {
                const int chunk_start = prompt_chunk.first;
                const int chunk_count = prompt_chunk.second;
                // Keep prompt prefill on the chunked path so the stage0 recurrent
                // state and boundary embeddings match monolithic prefill.
                const bool stage0_requires_sequential_prefill = false;
                std::vector<float> chunk_embeddings;
                chunk_embeddings.reserve((size_t) chunk_count * (size_t) n_embd_inp);
                const double stage0_prefill_t0 = ggml_time_us() / 1e6;
                bool stage0_ok = true;
                if (stage0_requires_sequential_prefill && chunk_count > 1) {
                    for (int token_index = 0; token_index < chunk_count; ++token_index) {
                        const llama_token token = prompt_tokens[(size_t) chunk_start + (size_t) token_index];
                        llama_batch token_batch = llama_batch_init(1, 0, 1);
                        token_batch.n_tokens = 1;
                        token_batch.token[0] = token;
                        token_batch.pos[0] = (llama_pos) (chunk_start + token_index);
                        token_batch.n_seq_id[0] = 1;
                        token_batch.seq_id[0][0] = 0;
                        token_batch.logits[0] = 1;
                        const int rc = llama_decode(stage0_ctx, token_batch);
                        llama_batch_free(token_batch);
                        if (rc) {
                            stage0_ok = false;
                            break;
                        }
                        const double output_extract_t0 = ggml_time_us() / 1e6;
                        float * token_embeddings = llama_get_embeddings_ith(stage0_ctx, 0);
                        if (!token_embeddings) {
                            LOG_ERR("multistage process stage0 embeddings unavailable\n");
                            stage0_ok = false;
                            break;
                        }
                        chunk_embeddings.insert(chunk_embeddings.end(), token_embeddings, token_embeddings + n_embd_inp);
                        const double output_extract_t1 = ggml_time_us() / 1e6;
                        stage0_output_extract_elapsed_s += output_extract_t1 - output_extract_t0;
                    }
                } else {
                    llama_batch token_batch = llama_batch_init(chunk_count, 0, 1);
                    token_batch.n_tokens = chunk_count;
                    for (int token_index = 0; token_index < chunk_count; ++token_index) {
                        token_batch.token[token_index] = prompt_tokens[(size_t) chunk_start + (size_t) token_index];
                        token_batch.pos[token_index] = (llama_pos) (chunk_start + token_index);
                        token_batch.n_seq_id[token_index] = 1;
                        token_batch.seq_id[token_index][0] = 0;
                        token_batch.logits[token_index] = 1;
                    }
                    const int rc = llama_decode(stage0_ctx, token_batch);
                    llama_batch_free(token_batch);
                    if (rc) {
                        stage0_ok = false;
                    } else {
                        const double output_extract_t0 = ggml_time_us() / 1e6;
                        for (int token_index = 0; token_index < chunk_count; ++token_index) {
                            float * token_embeddings = llama_get_embeddings_ith(stage0_ctx, token_index);
                            if (!token_embeddings) {
                                LOG_ERR("multistage process stage0 embeddings unavailable\n");
                                stage0_ok = false;
                                break;
                            }
                            chunk_embeddings.insert(chunk_embeddings.end(), token_embeddings, token_embeddings + n_embd_inp);
                        }
                        const double output_extract_t1 = ggml_time_us() / 1e6;
                        if (stage0_ok) {
                            stage0_output_extract_elapsed_s += output_extract_t1 - output_extract_t0;
                        }
                    }
                }
                const double stage0_prefill_t1 = ggml_time_us() / 1e6;
                if (!stage0_ok) {
                    LOG_ERR("multistage process stage0 prefill failed\n");
                    break;
                }
                stage0_prefill_compute_elapsed_s += stage0_prefill_t1 - stage0_prefill_t0;
                if (chunk_embeddings.empty()) {
                    break;
                }
                const double hop_delay_s = sample_transport_delay_s(
                    hop_rng,
                    multistage_live_prefill_hop_latency_ms,
                    multistage_live_prefill_hop_jitter_ms,
                    sizeof(float) * chunk_embeddings.size(),
                    multistage_live_prefill_hop_bandwidth_mbps);
                if (hop_delay_s > 0.0) {
                    std::this_thread::sleep_for(std::chrono::duration<double>(hop_delay_s));
                }
                stage0_transport_elapsed_s += hop_delay_s;
                const bool is_final_prefill_chunk =
                    (size_t) (prefill_chunk_count + 1) == prompt_chunks.size();
                const int32_t message_kind = is_final_prefill_chunk
                    ? MULTISTAGE_STAGE_MSG_PREFILL_FINAL_EMBD
                    : MULTISTAGE_STAGE_MSG_PREFILL_EMBD;
                const multistage_stage_state_header stage0_stage_state =
                    make_stage_state_header(
                        message_kind,
                        multistage_live_decode_replay && is_final_prefill_chunk ? 1 : 0,
                        (int32_t) prompt_tokens.size(),
                        -1,
                        LLAMA_TOKEN_NULL,
                        0);
                int32_t predicted_i32 = -1;
                const double stage0_send_t0 = ggml_time_us() / 1e6;
                const bool stage0_send_ok = send_stage_message(
                    downstream_fd,
                    message_kind,
                    chunk_start,
                    chunk_count,
                    stage0_stage_state,
                    prompt_tokens.data() + chunk_start,
                    (size_t) chunk_count,
                    chunk_embeddings.data(),
                    chunk_embeddings.size());
                const double stage0_send_t1 = ggml_time_us() / 1e6;
                int32_t reply_kind = 0;
                const bool stage0_recv_ok = stage0_send_ok && recv_stage_reply(downstream_fd, reply_kind, predicted_i32);
                const double stage0_recv_t1 = ggml_time_us() / 1e6;
                if (!stage0_send_ok || !stage0_recv_ok) {
                    LOG_ERR("multistage process stage0 prefill send/recv failed\n");
                    break;
                }
                if (is_final_prefill_chunk) {
                    if (reply_kind != MULTISTAGE_STAGE_REPLY_PREDICTED_TOKEN) {
                        LOG_ERR("multistage process stage0 final prefill expected predicted token reply\n");
                        break;
                    }
                    current_token = (llama_token) predicted_i32;
                } else if (reply_kind != MULTISTAGE_STAGE_REPLY_ACK) {
                    LOG_ERR("multistage process stage0 non-final prefill expected ack reply\n");
                    break;
                }
                stage0_send_elapsed_s += stage0_send_t1 - stage0_send_t0;
                stage0_reply_wait_elapsed_s += stage0_recv_t1 - stage0_send_t1;
                prefill_chunk_count += 1;
            }
            const double staged_prefill_t1 = ggml_time_us() / 1e6;
            metrics.staged_prefill_elapsed_s = staged_prefill_t1 - staged_prefill_t0;
            metrics.stage_state_bytes.assign((size_t) metrics.stage_count, 0);
            metrics.stage_state_bytes[0] = llama_state_seq_get_size(stage0_ctx, 0);
            metrics.live_prefill_stage_chunk_counts[0] = prefill_chunk_count;
            metrics.live_prefill_stage_compute_elapsed_s[0] = stage0_prefill_compute_elapsed_s;
            metrics.live_prefill_stage_send_elapsed_s[0] = stage0_send_elapsed_s;
            metrics.live_prefill_stage_reply_wait_elapsed_s[0] = stage0_reply_wait_elapsed_s;
            metrics.live_prefill_stage_output_extract_elapsed_s[0] = stage0_output_extract_elapsed_s;
            if (!metrics.live_prefill_edge_transport_elapsed_s.empty()) {
                metrics.live_prefill_edge_transport_elapsed_s[0] = stage0_transport_elapsed_s;
                metrics.live_prefill_edge_chunk_counts[0] = prefill_chunk_count;
            }

            auto import_baseline_stage_states = [&](
                const char * phase_label,
                int decode_step = -1,
                llama_token import_token = LLAMA_TOKEN_NULL
            ) -> bool {
                for (int stage_index = 1; stage_index < metrics.stage_count; ++stage_index) {
                    const auto [il_start, il_end] = stage_ranges[(size_t) stage_index];
                    const size_t stage_state_size = llama_state_seq_get_size_range(ctx, 0, il_start, il_end, 0);
                    std::vector<uint8_t> stage_state(stage_state_size);
                    const size_t stage_state_bytes = llama_state_seq_get_data_range(
                        ctx,
                        stage_state.data(),
                        stage_state.size(),
                        0,
                        il_start,
                        il_end,
                        0);
                    stage_state.resize(stage_state_bytes);
                    const bool send_ok = send_stage_state_import_message(
                        downstream_fd,
                        stage_index,
                        make_stage_state_header(
                            MULTISTAGE_STAGE_MSG_STATE_IMPORT,
                            0,
                            (int32_t) prompt_tokens.size(),
                            decode_step,
                            import_token,
                            0),
                        stage_state.data(),
                        stage_state.size());
                    int32_t reply_kind = 0;
                    int32_t ignored_predicted = -1;
                    const bool recv_ok = send_ok && recv_stage_reply(downstream_fd, reply_kind, ignored_predicted);
                    if (!send_ok || !recv_ok || reply_kind != MULTISTAGE_STAGE_REPLY_ACK) {
                        LOG_ERR("multistage process %s state import send/recv failed for stage %d\n", phase_label, stage_index);
                        return false;
                    }
                }
                return true;
            };
            bool baseline_state_import_ok = true;
            if (multistage_live_prefill_import_baseline_state) {
                baseline_state_import_ok = import_baseline_stage_states("prefill");
            }
            if (!baseline_state_import_ok) {
                (void) send_stage_message(
                    downstream_fd,
                    MULTISTAGE_STAGE_MSG_STOP,
                    0,
                    0,
                    make_stage_state_header(MULTISTAGE_STAGE_MSG_STOP),
                    nullptr,
                    0,
                    nullptr,
                    0);
                ::close(downstream_fd);
                llama_free(stage0_ctx);
                if (stage0_model != model) {
                    llama_model_free(stage0_model);
                }
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            std::vector<llama_token> decode_context_recent_tokens;
            std::vector<float> decode_context_recent_embeddings;
            const int decode_context_window = std::max(
                multistage_live_decode_replay
                    ? multistage_live_decode_replay_window
                    : 0,
                multistage_live_decode_light_context_tokens);
            decode_context_recent_tokens.reserve((size_t) std::max(decode_context_window, 1));
            decode_context_recent_embeddings.reserve((size_t) std::max(decode_context_window, 1) * (size_t) n_embd_inp);
            int32_t decode_context_checkpoint_generation =
                (multistage_live_decode_replay || multistage_live_decode_light_context_tokens > 0) ? 1 : 0;

            for (int step = 0; step < params.n_predict; ++step) {
                if (current_token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, current_token)) {
                    break;
                }

                const double staged_t0 = ggml_time_us() / 1e6;
                int32_t predicted_i32 = -1;
                int32_t reply_kind = 0;
                bool decode_send_ok = true;
                llama_token expected = LLAMA_TOKEN_NULL;
                float baseline_margin = 0.0f;

                if (multistage_live_decode_import_baseline_state) {
                    if (!import_baseline_stage_states("decode", step, current_token)) {
                        break;
                    }
                }

                    llama_batch stage0_decode_batch = llama_batch_init(1, 0, 1);
                    stage0_decode_batch.n_tokens = 1;
                    stage0_decode_batch.token[0] = current_token;
                    if (!multistage_live_decode_auto_pos) {
                        stage0_decode_batch.pos[0] = (llama_pos) ((int) prompt_tokens.size() + step);
                        stage0_decode_batch.n_seq_id[0] = 1;
                        stage0_decode_batch.seq_id[0][0] = 0;
                    } else {
                        stage0_decode_batch.pos = nullptr;
                        stage0_decode_batch.n_seq_id = nullptr;
                        stage0_decode_batch.seq_id = nullptr;
                    }
                    stage0_decode_batch.logits[0] = 0;
                    if (llama_decode(stage0_ctx, stage0_decode_batch)) {
                        llama_batch_free(stage0_decode_batch);
                        LOG_ERR("multistage process stage0 decode failed at step %d\n", step);
                        break;
                    }
                    llama_batch_free(stage0_decode_batch);
                    float * boundary_embd = llama_get_embeddings_ith(stage0_ctx, -1);
                    if (!boundary_embd) {
                        LOG_ERR("multistage process stage0 boundary embeddings unavailable at step %d\n", step);
                        break;
                    }
                    if (multistage_debug_trace) {
                        std::vector<uint64_t> step_boundary_hashes;
                        step_boundary_hashes.push_back(
                            fnv1a64_bytes(boundary_embd, sizeof(float) * (size_t) n_embd_inp));
                        metrics.debug_decode_stage_boundary_hashes.push_back(std::move(step_boundary_hashes));
                    }
                    const double hop_delay_s = sample_transport_delay_s(
                        hop_rng,
                        multistage_live_prefill_hop_latency_ms,
                        multistage_live_prefill_hop_jitter_ms,
                        sizeof(float) * (size_t) n_embd_inp,
                        multistage_live_prefill_hop_bandwidth_mbps);
                    if (hop_delay_s > 0.0) {
                        std::this_thread::sleep_for(std::chrono::duration<double>(hop_delay_s));
                    }
                    if (multistage_live_decode_replay || multistage_live_decode_light_context_tokens > 0) {
                        decode_context_recent_tokens.push_back(current_token);
                        decode_context_recent_embeddings.insert(
                            decode_context_recent_embeddings.end(),
                            boundary_embd,
                            boundary_embd + n_embd_inp);
                        const size_t context_window = (size_t) std::max(decode_context_window, 1);
                        while (decode_context_recent_tokens.size() > context_window) {
                            decode_context_recent_tokens.erase(decode_context_recent_tokens.begin());
                        }
                        while (decode_context_recent_embeddings.size() >
                               decode_context_recent_tokens.size() * (size_t) n_embd_inp) {
                            decode_context_recent_embeddings.erase(
                                decode_context_recent_embeddings.begin(),
                                decode_context_recent_embeddings.begin() + n_embd_inp);
                        }
                        if (multistage_live_decode_replay) {
                            const int replay_token_count = (int) decode_context_recent_tokens.size();
                            const int replay_pos_start =
                                (int) prompt_tokens.size() + step + 1 - replay_token_count;
                            for (int replay_index = 0; replay_index < replay_token_count; ++replay_index) {
                                const bool is_final_replay = replay_index == replay_token_count - 1;
                                const int32_t replay_kind = is_final_replay
                                    ? MULTISTAGE_STAGE_MSG_DECODE_REPLAY_FINAL_EMBD
                                    : MULTISTAGE_STAGE_MSG_DECODE_REPLAY_EMBD;
                                const llama_token replay_token = decode_context_recent_tokens[(size_t) replay_index];
                                const float * replay_embd =
                                    decode_context_recent_embeddings.data() + ((size_t) replay_index * (size_t) n_embd_inp);
                                const int replay_step =
                                    replay_pos_start + replay_index - (int) prompt_tokens.size();
                                const multistage_stage_state_header replay_stage_state =
                                    make_stage_state_header(
                                        replay_kind,
                                        decode_context_checkpoint_generation,
                                        (int32_t) prompt_tokens.size(),
                                        replay_step,
                                        replay_token,
                                        0);
                                if (!send_stage_message(
                                        downstream_fd,
                                        replay_kind,
                                        replay_pos_start + replay_index,
                                        1,
                                        replay_stage_state,
                                        &replay_token,
                                        1,
                                        replay_embd,
                                        (size_t) n_embd_inp) ||
                                    !recv_stage_reply(downstream_fd, reply_kind, predicted_i32)) {
                                    decode_send_ok = false;
                                    break;
                                }
                                const int32_t expected_reply_kind = is_final_replay
                                    ? MULTISTAGE_STAGE_REPLY_PREDICTED_TOKEN
                                    : MULTISTAGE_STAGE_REPLY_ACK;
                                if (reply_kind != expected_reply_kind) {
                                    decode_send_ok = false;
                                    break;
                                }
                                if (is_final_replay) {
                                    decode_context_checkpoint_generation += 1;
                                }
                            }
                        } else {
                            const int light_token_count = (int) decode_context_recent_tokens.size();
                            const int light_pos_start =
                                (int) prompt_tokens.size() + step + 1 - light_token_count;
                            const multistage_stage_state_header light_stage_state =
                                make_stage_state_header(
                                    MULTISTAGE_STAGE_MSG_DECODE_LIGHT_CTX,
                                    decode_context_checkpoint_generation,
                                    (int32_t) prompt_tokens.size(),
                                    step,
                                    current_token,
                                    0);
                            if (!send_stage_message(
                                    downstream_fd,
                                    MULTISTAGE_STAGE_MSG_DECODE_LIGHT_CTX,
                                    light_pos_start,
                                    light_token_count,
                                    light_stage_state,
                                    decode_context_recent_tokens.data(),
                                    decode_context_recent_tokens.size(),
                                    decode_context_recent_embeddings.data(),
                                    decode_context_recent_embeddings.size()) ||
                                !recv_stage_reply(downstream_fd, reply_kind, predicted_i32)) {
                                decode_send_ok = false;
                            } else if (reply_kind != MULTISTAGE_STAGE_REPLY_PREDICTED_TOKEN) {
                                decode_send_ok = false;
                            } else {
                                decode_context_checkpoint_generation += 1;
                            }
                        }
                    } else {
                        const multistage_stage_state_header decode_stage_state =
                            make_stage_state_header(
                                MULTISTAGE_STAGE_MSG_DECODE_EMBD,
                                0,
                                (int32_t) prompt_tokens.size(),
                                step,
                                current_token,
                                0);
                        if (!send_stage_message(
                                downstream_fd,
                                MULTISTAGE_STAGE_MSG_DECODE_EMBD,
                                (int32_t) prompt_tokens.size() + step,
                                1,
                                decode_stage_state,
                                &current_token,
                                1,
                                boundary_embd,
                                (size_t) n_embd_inp) ||
                            !recv_stage_reply(downstream_fd, reply_kind, predicted_i32)) {
                            decode_send_ok = false;
                        } else if (reply_kind != MULTISTAGE_STAGE_REPLY_PREDICTED_TOKEN) {
                            decode_send_ok = false;
                        }
                    }
                if (!decode_send_ok) {
                    LOG_ERR("multistage process stage0 decode send/recv failed at step %d\n", step);
                    break;
                }
                const double staged_t1 = ggml_time_us() / 1e6;
                metrics.staged_total_decode_elapsed_s += staged_t1 - staged_t0;
                if (step == 0) {
                    metrics.staged_first_decode_elapsed_s = staged_t1 - staged_t0;
                }

                const double base_decode_t0 = ggml_time_us() / 1e6;
                llama_token baseline_token = current_token;
                if (llama_decode(ctx, llama_batch_get_one(&baseline_token, 1))) {
                    LOG_ERR("baseline multistage decode failed at step %d\n", step);
                    break;
                }
                const double base_decode_t1 = ggml_time_us() / 1e6;
                metrics.baseline_total_decode_elapsed_s += base_decode_t1 - base_decode_t0;
                if (step == 0) {
                    metrics.baseline_first_decode_elapsed_s = base_decode_t1 - base_decode_t0;
                }

                const float * baseline_logits = llama_get_logits_ith(ctx, -1);
                if (!baseline_logits) {
                    LOG_ERR("baseline multistage logits unavailable at step %d\n", step);
                    break;
                }
                std::tie(expected, baseline_margin) = select_greedy_token_with_margin(baseline_logits, n_vocab);
                metrics.baseline_output_text += common_token_to_piece(ctx, expected, false);

                current_token = (llama_token) predicted_i32;

                metrics.generated_tokens++;
                metrics.output_text += common_token_to_piece(ctx, current_token, false);
                metrics.compared_token_steps++;
                if (current_token != expected) {
                    metrics.token_mismatch_count++;
                    if (metrics.first_mismatch_step < 0) {
                        metrics.first_mismatch_step = step;
                        metrics.first_mismatch_baseline_token = expected;
                        metrics.first_mismatch_staged_token = current_token;
                        metrics.first_mismatch_baseline_margin = baseline_margin;
                        metrics.first_mismatch_staged_margin = 0.0;
                    }
                    break;
                }
            }

            (void) send_stage_message(
                downstream_fd,
                MULTISTAGE_STAGE_MSG_STOP,
                0,
                0,
                make_stage_state_header(MULTISTAGE_STAGE_MSG_STOP),
                nullptr,
                0,
                nullptr,
                0);
            ::close(downstream_fd);
            metrics.token_match = metrics.token_mismatch_count == 0;
            metrics.staged_delta_vs_baseline_s =
                metrics.staged_total_decode_elapsed_s - metrics.baseline_total_decode_elapsed_s;
            metrics.staged_total_runtime_s = metrics.staged_prefill_elapsed_s + metrics.staged_total_decode_elapsed_s;
            metrics.staged_total_delta_vs_baseline_runtime_s =
                metrics.staged_total_runtime_s -
                (metrics.baseline_prefill_elapsed_s + metrics.baseline_total_decode_elapsed_s);

            if (!json_out_path.empty()) {
                write_multistage_compute_json(json_out_path, params.prompt, metrics);
            }

            llama_free(stage0_ctx);
            if (stage0_model != model) {
                llama_model_free(stage0_model);
            }
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }

        std::vector<llama_model *> stage_models((size_t) metrics.stage_count, nullptr);
        std::vector<llama_context *> stage_ctxs((size_t) metrics.stage_count, nullptr);
        auto cleanup_stage_ctxs = [&]() {
            for (llama_context * stage_ctx : stage_ctxs) {
                if (stage_ctx) {
                    llama_free(stage_ctx);
                }
            }
            for (llama_model * stage_model : stage_models) {
                if (stage_model) {
                    llama_model_free(stage_model);
                }
            }
        };

        for (int stage_index = 0; stage_index < metrics.stage_count; ++stage_index) {
            llama_model * stage_model = model;
            if (!multistage_model_paths.empty()) {
                const auto [il_start, il_end] = stage_ranges[(size_t) stage_index];
                const auto stage_model_params = clamp_stage_model_gpu_layers(model_params, il_end - il_start);
                stage_model = llama_model_load_from_file(multistage_model_paths[(size_t) stage_index].c_str(), stage_model_params);
                stage_models[(size_t) stage_index] = stage_model;
            }
            if (!stage_model) {
                LOG_ERR("failed to load multistage model %d\n", stage_index);
                cleanup_stage_ctxs();
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            stage_ctxs[(size_t) stage_index] = llama_init_from_model(stage_model, proof_ctx_params);
            if (!stage_ctxs[(size_t) stage_index]) {
                LOG_ERR("failed to create multistage proof context %d\n", stage_index);
                cleanup_stage_ctxs();
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            llama_set_embeddings(
                stage_ctxs[(size_t) stage_index],
                !multistage_live_prefill || stage_index < metrics.stage_count - 1);
            llama_set_logits(
                stage_ctxs[(size_t) stage_index],
                multistage_keep_stage_logits || stage_index == metrics.stage_count - 1);
        }

        if (baseline_prefill_layer_timing) {
            baseline_layer_probe.begin();
        }
        const double base_prefill_t0 = ggml_time_us() / 1e6;
        if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
            LOG_ERR("baseline prompt prefill failed\n");
            cleanup_stage_ctxs();
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const double base_prefill_t1 = ggml_time_us() / 1e6;
        if (baseline_prefill_layer_timing) {
            baseline_layer_probe.end();
            metrics.baseline_prefill_layer_elapsed_s = baseline_layer_probe.layer_elapsed_s;
            metrics.baseline_prefill_layer_observation_count = baseline_layer_probe.layer_observation_count;
        }
        metrics.baseline_prefill_elapsed_s = base_prefill_t1 - base_prefill_t0;
        metrics.baseline_state_bytes = llama_state_seq_get_size(ctx, 0);

        for (int stage_index = 0; stage_index < metrics.stage_count; ++stage_index) {
            const auto [il_start, il_end] = stage_ranges[(size_t) stage_index];
            const bool stage_uses_shard_model =
                !multistage_model_paths.empty() &&
                (size_t) stage_index < multistage_model_paths.size() &&
                multistage_model_paths[(size_t) stage_index] != params.model.path;
            if (!stage_uses_shard_model) {
                llama_set_compute_range(stage_ctxs[(size_t) stage_index], il_start, il_end);
            }
            if (stage_index == 0) {
                metrics.stage_state_bytes.push_back(0);
                continue;
            }
            const size_t stage_state_size = llama_state_seq_get_size_range(ctx, 0, il_start, il_end, 0);
            metrics.stage_state_bytes.push_back(stage_state_size);
        }

        metrics.live_prefill_enabled = multistage_live_prefill;
        metrics.live_prefill_chunk_tokens = multistage_live_prefill ? multistage_live_prefill_chunk_tokens : 0;
        metrics.live_prefill_downstream_chunk_tokens = multistage_live_prefill ?
            (multistage_live_prefill_downstream_chunk_tokens > 0 ? multistage_live_prefill_downstream_chunk_tokens : multistage_live_prefill_chunk_tokens) : 0;
        metrics.live_prefill_tile_tokens = multistage_live_prefill ? multistage_live_prefill_tile_tokens : 0;
        metrics.live_prefill_fill_chunk_tokens = multistage_live_prefill ? multistage_live_prefill_fill_chunk_tokens : 0;
        metrics.live_prefill_fill_chunk_count = multistage_live_prefill ? multistage_live_prefill_fill_chunk_count : 0;
        metrics.live_prefill_max_inflight = multistage_live_prefill ? multistage_live_prefill_max_inflight : 0;
        metrics.live_prefill_source_buffer_enabled = multistage_live_prefill ? multistage_live_prefill_source_buffer : false;
        metrics.keep_stage_logits = multistage_keep_stage_logits;
        metrics.live_prefill_hop_latency_ms = multistage_live_prefill ? multistage_live_prefill_hop_latency_ms : 0.0;
        metrics.live_prefill_hop_jitter_ms = multistage_live_prefill ? multistage_live_prefill_hop_jitter_ms : 0.0;
        metrics.live_prefill_hop_bandwidth_mbps = multistage_live_prefill ? multistage_live_prefill_hop_bandwidth_mbps : 0.0;
        metrics.live_prefill_stage_chunk_counts.resize((size_t) metrics.stage_count, 0);
        metrics.live_prefill_stage_compute_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
        metrics.live_prefill_stage_send_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
        metrics.live_prefill_stage_reply_wait_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
        metrics.live_prefill_stage_input_copy_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
        metrics.live_prefill_stage_output_extract_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
        metrics.live_prefill_stage_input_wait_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
        metrics.live_prefill_stage_output_wait_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
        metrics.live_prefill_stage_input_max_queue_depth.resize((size_t) metrics.stage_count, 0);
        metrics.live_prefill_stage_output_max_queue_depth.resize((size_t) metrics.stage_count, 0);
        metrics.live_prefill_edge_chunk_counts.resize((size_t) std::max(metrics.stage_count - 1, 0), 0);
        metrics.live_prefill_edge_transport_elapsed_s.resize((size_t) std::max(metrics.stage_count - 1, 0), 0.0);
        metrics.live_prefill_edge_max_queue_depth.resize((size_t) std::max(metrics.stage_count - 1, 0), 0);
        metrics.live_prefill_stage_runtime_lock_wait_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
        metrics.live_prefill_stage_runtime_lock_hold_elapsed_s.resize((size_t) metrics.stage_count, 0.0);
        // Metal-backed hybrid runs still appear to require serialized eval across
        // stage contexts; removing this lock deadlocks the local shard-backed
        // benchmark even though the models are separate.
        const bool live_prefill_requires_runtime_mutex = true;
        const bool live_prefill_uses_transport_threads =
            multistage_live_prefill_hop_latency_ms > 0.0 ||
            multistage_live_prefill_hop_jitter_ms > 0.0 ||
            multistage_live_prefill_hop_bandwidth_mbps > 0.0;

        if (!multistage_live_prefill) {
            if (llama_decode(stage_ctxs[0], llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
                LOG_ERR("multistage first-stage prompt prefill failed\n");
                cleanup_stage_ctxs();
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            for (int stage_index = 1; stage_index < metrics.stage_count; ++stage_index) {
                const auto [il_start, il_end] = stage_ranges[(size_t) stage_index];
                const size_t stage_state_size = metrics.stage_state_bytes[(size_t) stage_index];
                std::vector<uint8_t> stage_state(stage_state_size);
                const size_t stage_state_bytes = llama_state_seq_get_data_range(
                    ctx,
                    stage_state.data(),
                    stage_state.size(),
                    0,
                    il_start,
                    il_end,
                    0);
                metrics.stage_state_bytes[(size_t) stage_index] = stage_state_bytes;
                if (llama_state_seq_set_data_range(
                        stage_ctxs[(size_t) stage_index],
                        stage_state.data(),
                        stage_state_bytes,
                        0,
                        il_start,
                        il_end,
                        0) != stage_state_bytes) {
                    LOG_ERR("multistage stage-state import failed for stage %d\n", stage_index);
                    cleanup_stage_ctxs();
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
            }
        } else {
            std::vector<blocking_queue<multistage_prefill_chunk>> stage_inputs((size_t) metrics.stage_count);
            std::vector<blocking_queue<multistage_prefill_chunk>> stage_outputs(
                live_prefill_uses_transport_threads ? (size_t) std::max(metrics.stage_count - 1, 0) : 0);
            blocking_queue<multistage_prefill_chunk> source_ready;
            std::vector<std::thread> transport_threads;
            std::vector<std::thread> stage_threads;
            std::vector<int32_t> direct_edge_chunk_counts((size_t) std::max(metrics.stage_count - 1, 0), 0);
            if (multistage_live_prefill_max_inflight > 0) {
                for (blocking_queue<multistage_prefill_chunk> & queue : stage_inputs) {
                    queue.set_capacity((size_t) multistage_live_prefill_max_inflight);
                }
                if (live_prefill_uses_transport_threads) {
                    for (blocking_queue<multistage_prefill_chunk> & queue : stage_outputs) {
                        queue.set_capacity((size_t) multistage_live_prefill_max_inflight);
                    }
                }
            }
            std::mutex error_mutex;
            std::string error_message;
            auto set_error = [&](const std::string & message) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (error_message.empty()) {
                    error_message = message;
                }
            };

            if (live_prefill_uses_transport_threads) {
                for (int edge_index = 0; edge_index < metrics.stage_count - 1; ++edge_index) {
                    transport_threads.emplace_back([&, edge_index]() {
                        std::mt19937 rng((uint32_t) (multistage_live_prefill_seed + edge_index));
                        int chunk_count = 0;
                        double total_transport_elapsed_s = 0.0;
                        while (true) {
                            multistage_prefill_chunk chunk = stage_outputs[(size_t) edge_index].pop();
                            if (chunk.stop) {
                                stage_inputs[(size_t) edge_index + 1].push(std::move(chunk));
                                break;
                            }
                            const double delay_s = sample_transport_delay_s(
                                rng,
                                multistage_live_prefill_hop_latency_ms,
                                multistage_live_prefill_hop_jitter_ms,
                                sizeof(float) * chunk.embeddings.size(),
                                multistage_live_prefill_hop_bandwidth_mbps);
                            if (delay_s > 0.0) {
                                std::this_thread::sleep_for(std::chrono::duration<double>(delay_s));
                            }
                            total_transport_elapsed_s += delay_s;
                            chunk_count += 1;
                            stage_inputs[(size_t) edge_index + 1].push(std::move(chunk));
                        }
                        metrics.live_prefill_edge_chunk_counts[(size_t) edge_index] = chunk_count;
                        metrics.live_prefill_edge_transport_elapsed_s[(size_t) edge_index] = total_transport_elapsed_s;
                    });
                }
            }

            for (int stage_index = 0; stage_index < metrics.stage_count; ++stage_index) {
                stage_threads.emplace_back([&, stage_index]() {
                    const int32_t n_embd = n_embd_inp;
                    llama_context * stage_ctx = stage_ctxs[(size_t) stage_index];
                    auto forward_chunk = [&](multistage_prefill_chunk && next_chunk) {
                        if (stage_index >= metrics.stage_count - 1) {
                            return;
                        }
                        if (live_prefill_uses_transport_threads) {
                            stage_outputs[(size_t) stage_index].push(std::move(next_chunk));
                        } else {
                            if (!next_chunk.stop) {
                                direct_edge_chunk_counts[(size_t) stage_index] += 1;
                            }
                            stage_inputs[(size_t) stage_index + 1].push(std::move(next_chunk));
                        }
                    };
                    int chunk_count = 0;
                    double total_compute_elapsed_s = 0.0;
                    double total_runtime_lock_wait_elapsed_s = 0.0;
                    double total_runtime_lock_hold_elapsed_s = 0.0;
                    while (true) {
                        multistage_prefill_chunk chunk = stage_inputs[(size_t) stage_index].pop();
                        if (chunk.stop) {
                            if (stage_index < metrics.stage_count - 1) {
                                forward_chunk(std::move(chunk));
                            }
                            break;
                        }

                        bool stage_ok = true;
                        const double compute_t0 = ggml_time_us() / 1e6;
                        const llama_model * stage_model = llama_get_model(stage_ctx);
                        // Prompt prefill must use the chunked recurrent kernels to match
                        // monolithic prefill state. Decode replay/light-context paths are
                        // handled by the stage-server path above where sequential replay is
                        // still intentional.
                        const bool stage_requires_sequential_embd = false;
                        const int tile_tokens = stage_requires_sequential_embd
                            ? 1
                            : (multistage_live_prefill_tile_tokens > 0
                                ? std::min(multistage_live_prefill_tile_tokens, chunk.token_count)
                                : chunk.token_count);
                        int processed = 0;
                        int tile_index = 0;
                        while (processed < chunk.token_count) {
                            const int current_tokens = std::min(tile_tokens, chunk.token_count - processed);
                            std::vector<multistage_prefill_chunk> produced_chunks;
                            std::vector<float> tile_embeddings;
                            auto run_stage_tile = [&]() {
                                if (stage_index == 0) {
                                    llama_batch token_batch = llama_batch_init(current_tokens, 0, 1);
                                    token_batch.n_tokens = current_tokens;
                                    for (int token_index = 0; token_index < current_tokens; ++token_index) {
                                        token_batch.token[token_index] = chunk.tokens[(size_t) processed + (size_t) token_index];
                                        token_batch.pos[token_index] = (llama_pos) (chunk.pos_start + processed + token_index);
                                        token_batch.n_seq_id[token_index] = 1;
                                        token_batch.seq_id[token_index][0] = 0;
                                        token_batch.logits[token_index] =
                                            stage_index < metrics.stage_count - 1 || token_index == current_tokens - 1;
                                    }
                                    if (llama_decode(stage_ctx, token_batch)) {
                                        stage_ok = false;
                                    }
                                    llama_batch_free(token_batch);
                                } else {
                                    llama_batch embd_batch = llama_batch_init(current_tokens, n_embd, 1);
                                    embd_batch.n_tokens = current_tokens;
                                    const size_t embd_offset = (size_t) processed * (size_t) n_embd;
                                    const size_t embd_count = (size_t) current_tokens * (size_t) n_embd;
                                    std::memcpy(
                                        embd_batch.embd,
                                        chunk.embeddings.data() + embd_offset,
                                        sizeof(float) * embd_count);
                                    if (staged_same_topology_requires_token_sideband(stage_model) &&
                                        chunk.tokens.size() >= (size_t) processed + (size_t) current_tokens) {
                                        populate_embd_batch_tokens(
                                            embd_batch,
                                            chunk.tokens.data() + processed,
                                            current_tokens);
                                    }
                                    for (int token_index = 0; token_index < current_tokens; ++token_index) {
                                        embd_batch.pos[token_index] = (llama_pos) (chunk.pos_start + processed + token_index);
                                        embd_batch.n_seq_id[token_index] = 1;
                                        embd_batch.seq_id[token_index][0] = 0;
                                        embd_batch.logits[token_index] =
                                            stage_index < metrics.stage_count - 1 || token_index == current_tokens - 1;
                                    }
                                    if (llama_decode(stage_ctx, embd_batch)) {
                                        stage_ok = false;
                                    }
                                    llama_batch_free(embd_batch);
                                }

                                if (!stage_ok) {
                                    return;
                                }

                                if (stage_index < metrics.stage_count - 1) {
                                    tile_embeddings.reserve((size_t) current_tokens * (size_t) n_embd);
                                    for (int token_index = 0; token_index < current_tokens; ++token_index) {
                                        float * token_embeddings = llama_get_embeddings_ith(
                                            stage_ctx,
                                            token_index);
                                        if (!token_embeddings) {
                                            set_error("multistage live prefill token embeddings unavailable");
                                            stage_ok = false;
                                            break;
                                        }
                                        tile_embeddings.insert(
                                            tile_embeddings.end(),
                                            token_embeddings,
                                            token_embeddings + n_embd);
                                    }
                                    if (!stage_ok) {
                                        return;
                                    }
                                }
                            };
                            if (live_prefill_requires_runtime_mutex) {
                                const double runtime_lock_wait_t0 = ggml_time_us() / 1e6;
                                std::lock_guard<std::mutex> runtime_lock(multistage_live_prefill_runtime_mutex);
                                const double runtime_lock_hold_t0 = ggml_time_us() / 1e6;
                                total_runtime_lock_wait_elapsed_s += runtime_lock_hold_t0 - runtime_lock_wait_t0;
                                run_stage_tile();
                                const double runtime_lock_hold_t1 = ggml_time_us() / 1e6;
                                total_runtime_lock_hold_elapsed_s += runtime_lock_hold_t1 - runtime_lock_hold_t0;
                            } else {
                                run_stage_tile();
                            }

                            if (!stage_ok) {
                                break;
                            }

                            if (stage_index < metrics.stage_count - 1) {
                                const int downstream_chunk_tokens =
                                    (stage_index == 0 && metrics.live_prefill_downstream_chunk_tokens > 0)
                                    ? std::min(metrics.live_prefill_downstream_chunk_tokens, current_tokens)
                                    : current_tokens;
                                if (downstream_chunk_tokens < current_tokens) {
                                    int emitted = 0;
                                    int subchunk_index = 0;
                                    while (emitted < current_tokens) {
                                        const int subchunk_tokens = std::min(downstream_chunk_tokens, current_tokens - emitted);
                                        multistage_prefill_chunk next_chunk;
                                        next_chunk.chunk_index = chunk.chunk_index * 100000 + tile_index * 1000 + subchunk_index;
                                        next_chunk.pos_start = chunk.pos_start + processed + emitted;
                                        next_chunk.token_count = subchunk_tokens;
                                        next_chunk.tokens.assign(
                                            chunk.tokens.begin() + (ptrdiff_t) processed + emitted,
                                            chunk.tokens.begin() + (ptrdiff_t) processed + emitted + subchunk_tokens);
                                        const size_t embd_offset = (size_t) emitted * (size_t) n_embd;
                                        next_chunk.embeddings.assign(
                                            tile_embeddings.begin() + (ptrdiff_t) embd_offset,
                                            tile_embeddings.begin() + (ptrdiff_t) embd_offset + ((size_t) subchunk_tokens * (size_t) n_embd));
                                        produced_chunks.push_back(std::move(next_chunk));
                                        emitted += subchunk_tokens;
                                        subchunk_index += 1;
                                    }
                                } else {
                                    multistage_prefill_chunk next_chunk;
                                    next_chunk.chunk_index = chunk.chunk_index * 1000 + tile_index;
                                    next_chunk.pos_start = chunk.pos_start + processed;
                                    next_chunk.token_count = current_tokens;
                                    next_chunk.tokens.assign(
                                        chunk.tokens.begin() + (ptrdiff_t) processed,
                                        chunk.tokens.begin() + (ptrdiff_t) processed + current_tokens);
                                    next_chunk.embeddings = std::move(tile_embeddings);
                                    produced_chunks.push_back(std::move(next_chunk));
                                }
                            }

                            for (multistage_prefill_chunk & next_chunk : produced_chunks) {
                                forward_chunk(std::move(next_chunk));
                            }
                            processed += current_tokens;
                            tile_index += 1;
                        }

                        if (!stage_ok) {
                            set_error("multistage live prefill stage decode failed");
                            multistage_prefill_chunk stop_chunk;
                            stop_chunk.stop = true;
                            if (stage_index < metrics.stage_count - 1) {
                                forward_chunk(std::move(stop_chunk));
                            }
                            break;
                        }
                        const double compute_t1 = ggml_time_us() / 1e6;
                        total_compute_elapsed_s += (compute_t1 - compute_t0);
                        chunk_count += 1;
                    }
                    metrics.live_prefill_stage_chunk_counts[(size_t) stage_index] = chunk_count;
                    metrics.live_prefill_stage_compute_elapsed_s[(size_t) stage_index] = total_compute_elapsed_s;
                    metrics.live_prefill_stage_runtime_lock_wait_elapsed_s[(size_t) stage_index] = total_runtime_lock_wait_elapsed_s;
                    metrics.live_prefill_stage_runtime_lock_hold_elapsed_s[(size_t) stage_index] = total_runtime_lock_hold_elapsed_s;
                });
            }

            const std::vector<std::pair<int, int>> prompt_chunks =
                make_adaptive_prompt_chunks(
                    (int) prompt_tokens.size(),
                    multistage_live_prefill_fill_chunk_tokens,
                    multistage_live_prefill_fill_chunk_count,
                    multistage_live_prefill_chunk_tokens);
            std::vector<multistage_prefill_chunk> prebuilt_chunks;
            if (multistage_live_prefill_source_buffer) {
                prebuilt_chunks.reserve(prompt_chunks.size());
                for (size_t chunk_index = 0; chunk_index < prompt_chunks.size(); ++chunk_index) {
                    const auto [chunk_start, chunk_count] = prompt_chunks[chunk_index];
                    multistage_prefill_chunk chunk;
                    chunk.chunk_index = (int) chunk_index;
                    chunk.pos_start = chunk_start;
                    chunk.token_count = chunk_count;
                    chunk.tokens.insert(
                        chunk.tokens.end(),
                        prompt_tokens.begin() + chunk_start,
                        prompt_tokens.begin() + chunk_start + chunk_count);
                    prebuilt_chunks.push_back(std::move(chunk));
                }
                for (multistage_prefill_chunk & chunk : prebuilt_chunks) {
                    source_ready.push(std::move(chunk));
                }
                multistage_prefill_chunk stop_chunk;
                stop_chunk.stop = true;
                source_ready.push(std::move(stop_chunk));
            }
            const double staged_prefill_t0 = ggml_time_us() / 1e6;
            if (multistage_live_prefill_source_buffer) {
                while (true) {
                    multistage_prefill_chunk chunk = source_ready.pop();
                    const bool stop = chunk.stop;
                    stage_inputs[0].push(std::move(chunk));
                    if (stop) {
                        break;
                    }
                }
            } else {
                for (size_t chunk_index = 0; chunk_index < prompt_chunks.size(); ++chunk_index) {
                    const auto [chunk_start, chunk_count] = prompt_chunks[chunk_index];
                    multistage_prefill_chunk chunk;
                    chunk.chunk_index = (int) chunk_index;
                    chunk.pos_start = chunk_start;
                    chunk.token_count = chunk_count;
                    chunk.tokens.insert(
                        chunk.tokens.end(),
                        prompt_tokens.begin() + chunk_start,
                        prompt_tokens.begin() + chunk_start + chunk_count);
                    stage_inputs[0].push(std::move(chunk));
                }
                multistage_prefill_chunk stop_chunk;
                stop_chunk.stop = true;
                stage_inputs[0].push(std::move(stop_chunk));
            }

            for (std::thread & thread : stage_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            for (std::thread & thread : transport_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            if (!live_prefill_uses_transport_threads) {
                for (size_t edge_index = 0; edge_index < direct_edge_chunk_counts.size(); ++edge_index) {
                    metrics.live_prefill_edge_chunk_counts[edge_index] = direct_edge_chunk_counts[edge_index];
                    metrics.live_prefill_edge_transport_elapsed_s[edge_index] = 0.0;
                    metrics.live_prefill_edge_max_queue_depth[edge_index] = 0;
                }
            }
            for (size_t stage_index = 0; stage_index < stage_inputs.size(); ++stage_index) {
                metrics.live_prefill_stage_input_wait_elapsed_s[stage_index] = stage_inputs[stage_index].total_pop_wait_s;
                metrics.live_prefill_stage_input_max_queue_depth[stage_index] = (int32_t) stage_inputs[stage_index].max_depth;
            }
            if (live_prefill_uses_transport_threads) {
                for (size_t stage_index = 0; stage_index < stage_outputs.size(); ++stage_index) {
                    metrics.live_prefill_stage_output_wait_elapsed_s[stage_index] = stage_outputs[stage_index].total_push_wait_s;
                    metrics.live_prefill_stage_output_max_queue_depth[stage_index] = (int32_t) stage_outputs[stage_index].max_depth;
                    metrics.live_prefill_edge_max_queue_depth[stage_index] = (int32_t) stage_outputs[stage_index].max_depth;
                }
            }
            const double staged_prefill_t1 = ggml_time_us() / 1e6;
            metrics.staged_prefill_elapsed_s = staged_prefill_t1 - staged_prefill_t0;

            // Record the actual per-stage sequence-state footprint after live
            // prefill completes so benchmark summaries can track KV residency
            // across all stage owners instead of the earlier placeholder sizes.
            metrics.stage_state_bytes.clear();
            metrics.stage_state_bytes.reserve((size_t) metrics.stage_count);
            for (int stage_index = 0; stage_index < metrics.stage_count; ++stage_index) {
                const auto [il_start, il_end] = stage_ranges[(size_t) stage_index];
                const bool stage_uses_shard_model =
                    !multistage_model_paths.empty() &&
                    (size_t) stage_index < multistage_model_paths.size() &&
                    multistage_model_paths[(size_t) stage_index] != params.model.path;
                const size_t stage_state_size = stage_uses_shard_model
                    ? llama_state_seq_get_size(stage_ctxs[(size_t) stage_index], 0)
                    : llama_state_seq_get_size_range(stage_ctxs[(size_t) stage_index], 0, il_start, il_end, 0);
                metrics.stage_state_bytes.push_back(stage_state_size);
            }

            if (!error_message.empty()) {
                LOG_ERR("%s\n", error_message.c_str());
                cleanup_stage_ctxs();
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }

        auto import_local_baseline_stage_states = [&](const char * phase_label) -> bool {
            for (int stage_index = 1; stage_index < metrics.stage_count; ++stage_index) {
                const auto [il_start, il_end] = stage_ranges[(size_t) stage_index];
                llama_context * stage_ctx = stage_ctxs[(size_t) stage_index];
                if (!stage_ctx) {
                    LOG_ERR("multistage local %s state import missing stage context %d\n", phase_label, stage_index);
                    return false;
                }
                llama_memory_t stage_mem = llama_get_memory(stage_ctx);
                if (!stage_mem) {
                    LOG_ERR("multistage local %s state import missing stage memory %d\n", phase_label, stage_index);
                    return false;
                }
                const size_t stage_state_size = llama_state_seq_get_size_range(ctx, 0, il_start, il_end, 0);
                std::vector<uint8_t> stage_state(stage_state_size);
                const size_t serialized = llama_state_seq_get_data_range(
                    ctx,
                    stage_state.data(),
                    stage_state.size(),
                    0,
                    il_start,
                    il_end,
                    0);
                stage_state.resize(serialized);
                llama_memory_clear(stage_mem, true);
                if (!llama_memory_seq_rm(stage_mem, 0, -1, -1)) {
                    LOG_ERR("multistage local %s state import failed to clear stage memory %d\n", phase_label, stage_index);
                    return false;
                }
                const size_t imported = llama_state_seq_set_data_range(
                    stage_ctx,
                    stage_state.data(),
                    stage_state.size(),
                    0,
                    il_start,
                    il_end,
                    0);
                if (imported != stage_state.size()) {
                    LOG_ERR("multistage local %s state import failed for stage %d\n", phase_label, stage_index);
                    return false;
                }
            }
            return true;
        };

        if (multistage_live_prefill_import_baseline_state) {
            if (!import_local_baseline_stage_states("prefill")) {
                cleanup_stage_ctxs();
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }

        const float * initial_logits = llama_get_logits_ith(ctx, -1);
        if (!initial_logits) {
            LOG_ERR("baseline logits unavailable after prefill\n");
            cleanup_stage_ctxs();
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const auto [baseline_seed_token, baseline_seed_margin] = select_greedy_token_with_margin(initial_logits, n_vocab);
        llama_token current_token = baseline_seed_token;
        metrics.seed_token = current_token;
        if (!multistage_live_prefill) {
            metrics.staged_prefill_elapsed_s = metrics.baseline_prefill_elapsed_s;
        }

        const float * staged_prefill_logits = llama_get_logits_ith(stage_ctxs.back(), -1);
        if (!staged_prefill_logits) {
            LOG_ERR("multistage final logits unavailable after prefill\n");
            cleanup_stage_ctxs();
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const auto [staged_seed_token, staged_seed_margin] = select_greedy_token_with_margin(staged_prefill_logits, n_vocab);
        if (staged_seed_token != current_token) {
            metrics.token_mismatch_count = 1;
            metrics.first_mismatch_step = 0;
            metrics.first_mismatch_baseline_token = current_token;
            metrics.first_mismatch_staged_token = staged_seed_token;
            metrics.first_mismatch_baseline_margin = baseline_seed_margin;
            metrics.first_mismatch_staged_margin = staged_seed_margin;
            metrics.token_match = false;
            metrics.staged_total_runtime_s = metrics.staged_prefill_elapsed_s;
            metrics.staged_total_delta_vs_baseline_runtime_s =
                metrics.staged_total_runtime_s - metrics.baseline_prefill_elapsed_s;
            if (!json_out_path.empty()) {
                write_multistage_compute_json(json_out_path, params.prompt, metrics);
            }
            cleanup_stage_ctxs();
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }

        for (int step = 0; step < params.n_predict; ++step) {
            if (llama_vocab_is_eog(vocab, current_token)) {
                break;
            }

            metrics.decode_input_tokens.push_back((int) current_token);
            const double base_decode_t0 = ggml_time_us() / 1e6;
            llama_token baseline_token = current_token;
            if (llama_decode(ctx, llama_batch_get_one(&baseline_token, 1))) {
                LOG_ERR("baseline multistage decode failed at step %d\n", step);
                break;
            }
            const double base_decode_t1 = ggml_time_us() / 1e6;
            metrics.baseline_total_decode_elapsed_s += base_decode_t1 - base_decode_t0;
            if (step == 0) {
                metrics.baseline_first_decode_elapsed_s = base_decode_t1 - base_decode_t0;
            }

            const float * baseline_logits = llama_get_logits_ith(ctx, -1);
            if (!baseline_logits) {
                LOG_ERR("baseline multistage logits unavailable at step %d\n", step);
                break;
            }
            const auto [expected, baseline_margin] = select_greedy_token_with_margin(baseline_logits, n_vocab);

            std::vector<float> boundary_embd;
            const double staged_t0 = ggml_time_us() / 1e6;
            bool staged_ok = true;
            std::vector<uint64_t> step_boundary_hashes;
            if (multistage_debug_trace) {
                step_boundary_hashes.reserve((size_t) std::max(0, metrics.stage_count - 1));
            }
            if (multistage_live_decode_import_baseline_state) {
                staged_ok = import_local_baseline_stage_states("decode");
            }
            for (int stage_index = 0; stage_index < metrics.stage_count; ++stage_index) {
                llama_context * stage_ctx = stage_ctxs[(size_t) stage_index];
                if (stage_index == 0) {
                    llama_token stage_token = current_token;
                    if (llama_decode(stage_ctx, llama_batch_get_one(&stage_token, 1))) {
                        staged_ok = false;
                        break;
                    }
                } else {
                    llama_batch embd_batch = llama_batch_init(1, n_embd_inp, 1);
                    embd_batch.n_tokens = 1;
                    std::memcpy(embd_batch.embd, boundary_embd.data(), sizeof(float) * (size_t) n_embd_inp);
                    if (staged_same_topology_requires_token_sideband(llama_get_model(stage_ctx))) {
                        populate_embd_batch_tokens(embd_batch, &current_token, 1);
                    }
                    embd_batch.pos[0] = (llama_pos) (prompt_tokens.size() + step);
                    embd_batch.n_seq_id[0] = 1;
                    embd_batch.seq_id[0][0] = 0;
                    embd_batch.logits[0] = stage_index == metrics.stage_count - 1 ? 1 : 0;
                    const int rc = llama_decode(stage_ctx, embd_batch);
                    llama_batch_free(embd_batch);
                    if (rc) {
                        staged_ok = false;
                        break;
                    }
                }

                if (stage_index < metrics.stage_count - 1) {
                    float * stage_boundary = llama_get_embeddings_ith(stage_ctx, -1);
                    if (!stage_boundary) {
                        staged_ok = false;
                        break;
                    }
                    boundary_embd.assign(stage_boundary, stage_boundary + n_embd_inp);
                    if (multistage_debug_trace) {
                        step_boundary_hashes.push_back(
                            fnv1a64_bytes(boundary_embd.data(), sizeof(float) * (size_t) n_embd_inp));
                    }
                }
            }
            if (multistage_debug_trace) {
                metrics.debug_decode_stage_boundary_hashes.push_back(std::move(step_boundary_hashes));
            }
            const double staged_t1 = ggml_time_us() / 1e6;
            metrics.staged_total_decode_elapsed_s += staged_t1 - staged_t0;
            if (step == 0) {
                metrics.staged_first_decode_elapsed_s = staged_t1 - staged_t0;
            }
            if (!staged_ok) {
                break;
            }

            const float * staged_logits = llama_get_logits_ith(stage_ctxs.back(), -1);
            if (!staged_logits) {
                LOG_ERR("multistage final logits unavailable at step %d\n", step);
                break;
            }
            const auto [predicted, staged_margin] = select_greedy_token_with_margin(staged_logits, n_vocab);
            metrics.baseline_predicted_tokens.push_back((int) expected);
            metrics.staged_predicted_tokens.push_back((int) predicted);
            metrics.baseline_output_text += common_token_to_piece(ctx, expected, false);

            metrics.generated_tokens++;
            metrics.output_text += common_token_to_piece(ctx, predicted, false);
            metrics.compared_token_steps++;
            if (predicted != expected) {
                metrics.token_mismatch_count++;
                if (metrics.first_mismatch_step < 0) {
                    metrics.first_mismatch_step = step;
                    metrics.first_mismatch_baseline_token = expected;
                    metrics.first_mismatch_staged_token = predicted;
                    metrics.first_mismatch_baseline_margin = baseline_margin;
                    metrics.first_mismatch_staged_margin = staged_margin;
                }
                break;
            }

            current_token = predicted;
        }

        metrics.token_match = metrics.token_mismatch_count == 0;
        metrics.staged_delta_vs_baseline_s =
            metrics.staged_total_decode_elapsed_s - metrics.baseline_total_decode_elapsed_s;
        metrics.staged_total_runtime_s = metrics.staged_prefill_elapsed_s + metrics.staged_total_decode_elapsed_s;
        metrics.staged_total_delta_vs_baseline_runtime_s =
            metrics.staged_total_runtime_s -
            (metrics.baseline_prefill_elapsed_s + metrics.baseline_total_decode_elapsed_s);

        if (!json_out_path.empty()) {
            write_multistage_compute_json(json_out_path, params.prompt, metrics);
        }

        cleanup_stage_ctxs();
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    const bool direct_split_proof_mode =
        split_compute_layer >= 0 &&
        split_embd_out_path.empty() &&
        split_embd_in_path.empty();

    if (direct_split_proof_mode) {
        const int32_t n_layer = llama_model_n_layer(model);
        const int32_t n_embd_inp = llama_model_n_embd_inp(model);
        if (split_compute_layer <= 0 || split_compute_layer >= n_layer) {
            LOG_ERR("--split-compute-layer must be in (0, %d)\n", n_layer);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, add_bos, true);
        if (prompt_tokens.empty()) {
            LOG_ERR("split compute proof requires a non-empty prompt\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        split_compute_metrics split_metrics;
        split_metrics.enabled = true;
        split_metrics.split_layer = split_compute_layer;
        split_metrics.prompt_tokens = (int) prompt_tokens.size();
        split_metrics.requested_decode_tokens = params.n_predict;

        llama_context_params proof_ctx_params = ctx_params;
        proof_ctx_params.cb_eval = nullptr;
        proof_ctx_params.cb_eval_user_data = nullptr;

        if (!shadow_model_path.empty() && shadow_model == nullptr) {
            shadow_model = llama_model_load_from_file(shadow_model_path.c_str(), model_params);
            if (!shadow_model) {
                LOG_ERR("failed to load shadow model\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }

        llama_context * stage1_ctx = llama_init_from_model(model, proof_ctx_params);
        llama_context * stage2_ctx = llama_init_from_model(shadow_model ? shadow_model : model, proof_ctx_params);
        if (!stage1_ctx || !stage2_ctx) {
            LOG_ERR("failed to create split proof contexts\n");
            if (stage1_ctx) llama_free(stage1_ctx);
            if (stage2_ctx) llama_free(stage2_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        llama_set_embeddings(stage1_ctx, true);
        llama_set_logits(stage1_ctx, false);
        llama_set_compute_range(stage1_ctx, 0, split_compute_layer);
        llama_set_embeddings(stage2_ctx, true);
        const bool split_stage2_requires_token_sideband =
            staged_same_topology_requires_token_sideband(llama_get_model(stage2_ctx));

        const double base_prefill_t0 = ggml_time_us() / 1e6;
        if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
            LOG_ERR("baseline prompt prefill failed\n");
            llama_free(stage1_ctx);
            llama_free(stage2_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const double base_prefill_t1 = ggml_time_us() / 1e6;
        split_metrics.baseline_prefill_elapsed_s = base_prefill_t1 - base_prefill_t0;

        llama_context * baseline_ctx = llama_init_from_model(model, proof_ctx_params);
        if (!baseline_ctx) {
            LOG_ERR("failed to create isolated baseline context\n");
            llama_free(stage1_ctx);
            llama_free(stage2_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        if (llama_decode(baseline_ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
            LOG_ERR("failed to prefill isolated baseline context\n");
            llama_free(baseline_ctx);
            llama_free(stage1_ctx);
            llama_free(stage2_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        const double stage1_prefill_t0 = ggml_time_us() / 1e6;
        if (llama_decode(stage1_ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
            LOG_ERR("stage1 prompt prefill failed\n");
            llama_free(stage1_ctx);
            llama_free(stage2_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const double stage1_prefill_t1 = ggml_time_us() / 1e6;
        split_metrics.stage1_prefill_elapsed_s = stage1_prefill_t1 - stage1_prefill_t0;

        size_t late_state_size = llama_state_seq_get_size_range(ctx, 0, split_compute_layer, n_layer, 0);
        std::vector<uint8_t> late_state(late_state_size);
        size_t late_state_bytes = llama_state_seq_get_data_range(ctx, late_state.data(), late_state.size(), 0, split_compute_layer, n_layer, 0);
        split_metrics.late_state_bytes = late_state_bytes;
        late_state.resize(late_state_bytes);

        const double stage2_import_t0 = ggml_time_us() / 1e6;
        size_t imported = llama_state_seq_set_data_range(stage2_ctx, late_state.data(), late_state.size(), 0, split_compute_layer, n_layer, 0);
        const double stage2_import_t1 = ggml_time_us() / 1e6;
        split_metrics.stage2_import_elapsed_s = stage2_import_t1 - stage2_import_t0;
        split_metrics.stage2_import_succeeded = imported == late_state.size();
        if (!split_metrics.stage2_import_succeeded) {
            LOG_ERR("stage2 late-state import failed\n");
            llama_free(stage1_ctx);
            llama_free(stage2_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        llama_set_compute_range(stage2_ctx, split_compute_layer, n_layer);
        int client_fd = -1;
        int listen_fd = -1;
        std::thread stage2_thread;
        std::mutex server_mu;
        std::condition_variable server_cv;
        bool server_ready = false;
        bool server_failed = false;
        int server_port = 0;
        std::string server_error;

        if (split_live_transport) {
            stage2_thread = std::thread([&]() {
                listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
                if (listen_fd < 0) {
                    std::lock_guard<std::mutex> lock(server_mu);
                    server_failed = true;
                    server_error = "failed to create split live transport socket";
                    server_ready = true;
                    server_cv.notify_one();
                    return;
                }
                int opt = 1;
                setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

                sockaddr_in addr{};
                addr.sin_family = AF_INET;
                addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
                addr.sin_port = 0;
                if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
                    ::listen(listen_fd, 1) != 0) {
                    std::lock_guard<std::mutex> lock(server_mu);
                    server_failed = true;
                    server_error = "failed to bind/listen split live transport socket";
                    server_ready = true;
                    server_cv.notify_one();
                    if (listen_fd >= 0) {
                        ::close(listen_fd);
                        listen_fd = -1;
                    }
                    return;
                }

                sockaddr_in bound{};
                socklen_t bound_len = sizeof(bound);
                if (::getsockname(listen_fd, reinterpret_cast<sockaddr *>(&bound), &bound_len) != 0) {
                    std::lock_guard<std::mutex> lock(server_mu);
                    server_failed = true;
                    server_error = "failed to query split live transport socket";
                    server_ready = true;
                    server_cv.notify_one();
                    ::close(listen_fd);
                    listen_fd = -1;
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(server_mu);
                    server_port = ntohs(bound.sin_port);
                    server_ready = true;
                }
                server_cv.notify_one();

                const int conn_fd = ::accept(listen_fd, nullptr, nullptr);
                if (conn_fd < 0) {
                    std::lock_guard<std::mutex> lock(server_mu);
                    server_failed = true;
                    server_error = "failed to accept split live transport connection";
                    ::close(listen_fd);
                    listen_fd = -1;
                    return;
                }

                while (true) {
                    int32_t step = -1;
                    if (!recv_all(conn_fd, &step, sizeof(step))) {
                        break;
                    }
                    if (step < 0) {
                        break;
                    }
                    llama_token recv_token = LLAMA_TOKEN_NULL;
                    if (split_stage2_requires_token_sideband) {
                        int32_t recv_token_i32 = LLAMA_TOKEN_NULL;
                        if (!recv_all(conn_fd, &recv_token_i32, sizeof(recv_token_i32))) {
                            break;
                        }
                        recv_token = (llama_token) recv_token_i32;
                    }
                    std::vector<float> embd((size_t) n_embd_inp);
                    if (!recv_all(conn_fd, embd.data(), sizeof(float) * embd.size())) {
                        break;
                    }
                    llama_batch embd_batch = llama_batch_init(1, n_embd_inp, 1);
                    embd_batch.n_tokens = 1;
                    std::memcpy(embd_batch.embd, embd.data(), sizeof(float) * embd.size());
                    embd_batch.pos[0] = (llama_pos) (prompt_tokens.size() + step);
                    embd_batch.n_seq_id[0] = 1;
                    embd_batch.seq_id[0][0] = 0;
                    embd_batch.logits[0] = 1;
                    if (split_stage2_requires_token_sideband) {
                        populate_embd_batch_tokens(embd_batch, &recv_token, 1);
                    }
                    if (llama_decode(stage2_ctx, embd_batch)) {
                        llama_batch_free(embd_batch);
                        break;
                    }
                    llama_batch_free(embd_batch);
                    const float * split_logits = llama_get_logits_ith(stage2_ctx, -1);
                    if (!split_logits) {
                        break;
                    }
                    const int32_t predicted = (int32_t) select_greedy_token(split_logits, n_vocab);
                    if (!send_all(conn_fd, &predicted, sizeof(predicted))) {
                        break;
                    }
                }

                ::close(conn_fd);
                ::close(listen_fd);
                listen_fd = -1;
            });

            {
                std::unique_lock<std::mutex> lock(server_mu);
                server_cv.wait(lock, [&]() { return server_ready; });
                if (server_failed) {
                    if (stage2_thread.joinable()) {
                        stage2_thread.join();
                    }
                    LOG_ERR("%s\n", server_error.c_str());
                    llama_free(stage1_ctx);
                    llama_free(stage2_ctx);
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
            }

            client_fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (client_fd < 0) {
                if (stage2_thread.joinable()) {
                    stage2_thread.join();
                }
                LOG_ERR("failed to create split live transport client socket\n");
                llama_free(stage1_ctx);
                llama_free(stage2_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            sockaddr_in client_addr{};
            client_addr.sin_family = AF_INET;
            client_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            client_addr.sin_port = htons((uint16_t) server_port);
            if (::connect(client_fd, reinterpret_cast<sockaddr *>(&client_addr), sizeof(client_addr)) != 0) {
                ::close(client_fd);
                if (stage2_thread.joinable()) {
                    stage2_thread.join();
                }
                LOG_ERR("failed to connect split live transport client socket\n");
                llama_free(stage1_ctx);
                llama_free(stage2_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }

        const float * baseline_logits = llama_get_logits_ith(baseline_ctx, -1);
        if (!baseline_logits) {
            LOG_ERR("baseline logits unavailable after prompt prefill\n");
            if (client_fd >= 0) {
                ::close(client_fd);
            }
            if (stage2_thread.joinable()) {
                stage2_thread.join();
            }
            llama_free(baseline_ctx);
            llama_free(stage1_ctx);
            llama_free(stage2_ctx);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        llama_token baseline_current_token = select_greedy_token(baseline_logits, n_vocab);
        split_metrics.prompt_boundary_seed_baseline_token = baseline_current_token;

        {
            llama_context * prompt_stage1_ctx = llama_init_from_model(model, proof_ctx_params);
            llama_context * prompt_stage2_ctx = llama_init_from_model(shadow_model ? shadow_model : model, proof_ctx_params);
            if (prompt_stage1_ctx && prompt_stage2_ctx) {
                llama_set_embeddings(prompt_stage1_ctx, true);
                llama_set_logits(prompt_stage1_ctx, false);
                llama_set_embeddings(prompt_stage2_ctx, true);
                llama_set_compute_range(prompt_stage1_ctx, 0, split_compute_layer);
                llama_set_compute_range(prompt_stage2_ctx, split_compute_layer, n_layer);

                const double prompt_stage1_t0 = ggml_time_us() / 1e6;
                const int prompt_stage1_rc = llama_decode(prompt_stage1_ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()));
                const double prompt_stage1_t1 = ggml_time_us() / 1e6;
                split_metrics.prompt_boundary_stage1_prefill_elapsed_s = prompt_stage1_t1 - prompt_stage1_t0;

                if (prompt_stage1_rc == 0) {
                    llama_batch prompt_embd_batch = llama_batch_init((int32_t) prompt_tokens.size(), n_embd_inp, 1);
                    prompt_embd_batch.n_tokens = (int32_t) prompt_tokens.size();
                    bool prompt_boundary_ok = true;
                    for (size_t token_index = 0; token_index < prompt_tokens.size(); ++token_index) {
                        float * token_embd = llama_get_embeddings_ith(prompt_stage1_ctx, (int32_t) token_index);
                        if (!token_embd) {
                            prompt_boundary_ok = false;
                            break;
                        }
                        std::memcpy(
                            prompt_embd_batch.embd + token_index * (size_t) n_embd_inp,
                            token_embd,
                            sizeof(float) * (size_t) n_embd_inp);
                        prompt_embd_batch.pos[token_index] = (llama_pos) token_index;
                        prompt_embd_batch.n_seq_id[token_index] = 1;
                        prompt_embd_batch.seq_id[token_index][0] = 0;
                        prompt_embd_batch.logits[token_index] = token_index + 1 == prompt_tokens.size();
                    }
                    if (prompt_boundary_ok && staged_same_topology_requires_token_sideband(llama_get_model(prompt_stage2_ctx))) {
                        populate_embd_batch_tokens(prompt_embd_batch, prompt_tokens.data(), (int) prompt_tokens.size());
                    }
                    const double prompt_stage2_t0 = ggml_time_us() / 1e6;
                    const int prompt_stage2_rc = prompt_boundary_ok ? llama_decode(prompt_stage2_ctx, prompt_embd_batch) : -1;
                    const double prompt_stage2_t1 = ggml_time_us() / 1e6;
                    split_metrics.prompt_boundary_stage2_prefill_elapsed_s = prompt_stage2_t1 - prompt_stage2_t0;
                    if (prompt_stage2_rc == 0) {
                        const float * prompt_stage2_logits = llama_get_logits_ith(prompt_stage2_ctx, -1);
                        if (prompt_stage2_logits) {
                            split_metrics.prompt_boundary_seed_split_token =
                                select_greedy_token(prompt_stage2_logits, n_vocab);
                            split_metrics.prompt_boundary_seed_match =
                                split_metrics.prompt_boundary_seed_split_token == split_metrics.prompt_boundary_seed_baseline_token;
                        }
                    }
                    llama_batch_free(prompt_embd_batch);
                }
            }
            if (prompt_stage1_ctx) {
                llama_free(prompt_stage1_ctx);
            }
            if (prompt_stage2_ctx) {
                llama_free(prompt_stage2_ctx);
            }
        }

        llama_token split_current_token = baseline_current_token;
        split_metrics.first_generated_token = baseline_current_token;
        split_metrics.token_match = true;

        for (int step = 0; step < params.n_predict; ++step) {
            if (llama_vocab_is_eog(vocab, baseline_current_token) || llama_vocab_is_eog(vocab, split_current_token)) {
                break;
            }

            llama_token stage1_token = split_current_token;
            split_metrics.decode_input_tokens.push_back((int) stage1_token);
            const double stage1_decode_t0 = ggml_time_us() / 1e6;
            if (llama_decode(stage1_ctx, llama_batch_get_one(&stage1_token, 1))) {
                LOG_ERR("stage1 split decode failed at step %d\n", step);
                if (client_fd >= 0) {
                    ::close(client_fd);
                }
                if (stage2_thread.joinable()) {
                    stage2_thread.join();
                }
                llama_free(stage1_ctx);
                llama_free(stage2_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            const double stage1_decode_t1 = ggml_time_us() / 1e6;
            const double stage1_elapsed_s = stage1_decode_t1 - stage1_decode_t0;
            split_metrics.stage1_total_decode_elapsed_s += stage1_elapsed_s;
            if (step == 0) {
                split_metrics.stage1_first_decode_elapsed_s = stage1_elapsed_s;
            }

            float * boundary_embd = llama_get_embeddings_ith(stage1_ctx, -1);
            if (!boundary_embd) {
                LOG_ERR("stage1 boundary embeddings unavailable at step %d\n", step);
                if (client_fd >= 0) {
                    ::close(client_fd);
                }
                if (stage2_thread.joinable()) {
                    stage2_thread.join();
                }
                llama_free(stage1_ctx);
                llama_free(stage2_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            split_metrics.debug_stage1_boundary_hashes.push_back(
                fnv1a64_bytes(boundary_embd, sizeof(float) * (size_t) n_embd_inp));

            const double stage2_decode_t0 = ggml_time_us() / 1e6;
            llama_token split_next = -1;
            if (split_live_transport) {
                const int32_t step_i32 = (int32_t) step;
                const int32_t stage1_token_i32 = (int32_t) stage1_token;
                if (!send_all(client_fd, &step_i32, sizeof(step_i32)) ||
                    (split_stage2_requires_token_sideband &&
                        !send_all(client_fd, &stage1_token_i32, sizeof(stage1_token_i32))) ||
                    !send_all(client_fd, boundary_embd, sizeof(float) * (size_t) n_embd_inp)) {
                    LOG_ERR("split live transport send failed at step %d\n", step);
                    ::close(client_fd);
                    if (stage2_thread.joinable()) {
                        stage2_thread.join();
                    }
                    llama_free(stage1_ctx);
                    llama_free(stage2_ctx);
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                int32_t predicted = -1;
                if (!recv_all(client_fd, &predicted, sizeof(predicted))) {
                    LOG_ERR("split live transport recv failed at step %d\n", step);
                    ::close(client_fd);
                    if (stage2_thread.joinable()) {
                        stage2_thread.join();
                    }
                    llama_free(stage1_ctx);
                    llama_free(stage2_ctx);
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                split_next = (llama_token) predicted;
            } else {
                llama_batch embd_batch = llama_batch_init(1, n_embd_inp, 1);
                embd_batch.n_tokens = 1;
                std::memcpy(embd_batch.embd, boundary_embd, sizeof(float) * n_embd_inp);
                embd_batch.pos[0] = (llama_pos) (prompt_tokens.size() + step);
                embd_batch.n_seq_id[0] = 1;
                embd_batch.seq_id[0][0] = 0;
                embd_batch.logits[0] = 1;
                if (split_stage2_requires_token_sideband) {
                    populate_embd_batch_tokens(embd_batch, &stage1_token, 1);
                }
                if (llama_decode(stage2_ctx, embd_batch)) {
                    llama_batch_free(embd_batch);
                    LOG_ERR("stage2 split decode failed at step %d\n", step);
                    if (client_fd >= 0) {
                        ::close(client_fd);
                    }
                    if (stage2_thread.joinable()) {
                        stage2_thread.join();
                    }
                    llama_free(stage1_ctx);
                    llama_free(stage2_ctx);
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                llama_batch_free(embd_batch);
                const float * split_next_logits = llama_get_logits_ith(stage2_ctx, -1);
                if (!split_next_logits) {
                    LOG_ERR("missing split logits after local stage2 decode at step %d\n", step);
                    if (client_fd >= 0) {
                        ::close(client_fd);
                    }
                    if (stage2_thread.joinable()) {
                        stage2_thread.join();
                    }
                    llama_free(stage1_ctx);
                    llama_free(stage2_ctx);
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                split_next = select_greedy_token(split_next_logits, n_vocab);
            }
            const double stage2_decode_t1 = ggml_time_us() / 1e6;
            const double stage2_elapsed_s = stage2_decode_t1 - stage2_decode_t0;
            split_metrics.stage2_total_decode_elapsed_s += stage2_elapsed_s;
            if (step == 0) {
                split_metrics.stage2_first_decode_elapsed_s = stage2_elapsed_s;
            }

            llama_token baseline_token = baseline_current_token;
            const double baseline_decode_t0 = ggml_time_us() / 1e6;
            if (llama_decode(baseline_ctx, llama_batch_get_one(&baseline_token, 1))) {
                LOG_ERR("baseline decode failed at step %d\n", step);
                if (client_fd >= 0) {
                    ::close(client_fd);
                }
                if (stage2_thread.joinable()) {
                    stage2_thread.join();
                }
                llama_free(baseline_ctx);
                llama_free(stage1_ctx);
                llama_free(stage2_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            const double baseline_decode_t1 = ggml_time_us() / 1e6;
            const double baseline_elapsed_s = baseline_decode_t1 - baseline_decode_t0;
            split_metrics.baseline_total_decode_elapsed_s += baseline_elapsed_s;
            if (step == 0) {
                split_metrics.baseline_first_decode_elapsed_s = baseline_elapsed_s;
            }

            const float * baseline_next_logits = llama_get_logits_ith(baseline_ctx, -1);
            if (!baseline_next_logits) {
                LOG_ERR("missing baseline logits after split-compute decode at step %d\n", step);
                if (client_fd >= 0) {
                    ::close(client_fd);
                }
                if (stage2_thread.joinable()) {
                    stage2_thread.join();
                }
                llama_free(baseline_ctx);
                llama_free(stage1_ctx);
                llama_free(stage2_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            const llama_token baseline_next = select_greedy_token(baseline_next_logits, n_vocab);
            split_metrics.baseline_predicted_tokens.push_back((int) baseline_next);
            split_metrics.split_predicted_tokens.push_back((int) split_next);
            if (step == 0) {
                split_metrics.baseline_next_token = baseline_next;
                split_metrics.split_next_token = split_next;
            }
            split_metrics.compared_token_steps++;
            if (baseline_next != split_next) {
                split_metrics.token_mismatch_count++;
                split_metrics.token_match = false;
                if (split_metrics.first_mismatch_step < 0) {
                    split_metrics.first_mismatch_step = step;
                    split_metrics.first_mismatch_baseline_token = baseline_next;
                    split_metrics.first_mismatch_split_token = split_next;
                }
            }
            split_metrics.generated_tokens++;
            split_metrics.output_text += common_token_to_piece(ctx, split_next, false);
            baseline_current_token = baseline_next;
            split_current_token = split_next;
        }

        if (client_fd >= 0) {
            const int32_t stop = -1;
            send_all(client_fd, &stop, sizeof(stop));
            ::close(client_fd);
        }
        if (stage2_thread.joinable()) {
            stage2_thread.join();
        }
        llama_free(baseline_ctx);

        split_metrics.split_total_elapsed_s =
            split_metrics.stage1_total_decode_elapsed_s +
            split_metrics.stage2_import_elapsed_s +
            split_metrics.stage2_total_decode_elapsed_s;
        split_metrics.split_delta_vs_baseline_s =
            split_metrics.split_total_elapsed_s -
            split_metrics.baseline_total_decode_elapsed_s;

        if (!json_out_path.empty()) {
            write_split_compute_json(json_out_path, params.prompt, split_metrics);
        }

        LOG_INF("split layer: %d of %d\n", split_compute_layer, n_layer);
        LOG_INF("prompt tokens: %d\n", split_metrics.prompt_tokens);
        LOG_INF("late state: %.2f MiB\n", split_metrics.late_state_bytes / (1024.0 * 1024.0));
        LOG_INF("generated tokens: %d compared: %d mismatches: %d\n",
            split_metrics.generated_tokens,
            split_metrics.compared_token_steps,
            split_metrics.token_mismatch_count);
        LOG_INF("baseline first decode: %.6fs\n", split_metrics.baseline_first_decode_elapsed_s);
        LOG_INF("baseline total decode: %.6fs\n", split_metrics.baseline_total_decode_elapsed_s);
        LOG_INF("stage1 first decode: %.6fs\n", split_metrics.stage1_first_decode_elapsed_s);
        LOG_INF("stage1 total decode: %.6fs\n", split_metrics.stage1_total_decode_elapsed_s);
        LOG_INF("stage2 import: %.6fs\n", split_metrics.stage2_import_elapsed_s);
        LOG_INF("stage2 first decode: %.6fs\n", split_metrics.stage2_first_decode_elapsed_s);
        LOG_INF("stage2 total decode: %.6fs\n", split_metrics.stage2_total_decode_elapsed_s);
        LOG_INF("split total: %.6fs\n", split_metrics.split_total_elapsed_s);
        LOG_INF("split delta vs baseline: %.6fs\n", split_metrics.split_delta_vs_baseline_s);
        LOG_INF("overall token match: %s (first %d vs %d)\n",
            split_metrics.token_match ? "yes" : "no",
            split_metrics.baseline_next_token,
            split_metrics.split_next_token);

        llama_free(stage1_ctx);
        llama_free(stage2_ctx);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return split_metrics.token_match ? 0 : 2;
    }

    prefill_handoff_metrics metrics;
    metrics.requested_verify_tokens = std::max(0, std::min(params.n_predict, verify_tokens));
    std::vector<llama_token> generated_token_ids;
    std::vector<llama_token> replay_tokens;
    std::vector<uint8_t> state_buffer;
    llama_context * shadow_ctx = nullptr;
    const llama_seq_id state_seq_id = 0;

    const bool has_import_state = !state_in_path.empty() || !state_chunks_dir_path.empty();

    if (!has_import_state) {
        std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, add_bos, true);
        metrics.prompt_tokens = (int) prompt_tokens.size();

        const double prefill_t0 = ggml_time_us() / 1e6;
        if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
            LOG_ERR("prefill failed\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        const double prefill_t1 = ggml_time_us() / 1e6;
        metrics.prefill_elapsed_s = prefill_t1 - prefill_t0;

        if (!split_embd_out_path.empty()) {
            const int32_t n_layer = llama_model_n_layer(model);
            const int32_t n_embd_inp = llama_model_n_embd_inp(model);
            if (split_compute_layer <= 0 || split_compute_layer >= n_layer) {
                LOG_ERR("--split-embd-out requires --split-compute-layer in (0, %d)\n", n_layer);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            const float * split_stage1_logits = llama_get_logits_ith(ctx, -1);
            if (!split_stage1_logits) {
                LOG_ERR("baseline logits unavailable for split stage1 export\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            const llama_token first_generated = select_greedy_token(split_stage1_logits, n_vocab);

            llama_context_params stage1_ctx_params = ctx_params;
            stage1_ctx_params.cb_eval = nullptr;
            stage1_ctx_params.cb_eval_user_data = nullptr;
            llama_context * stage1_ctx = llama_init_from_model(model, stage1_ctx_params);
            if (!stage1_ctx) {
                LOG_ERR("failed to create split stage1 context\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            llama_set_embeddings(stage1_ctx, true);
            llama_set_logits(stage1_ctx, false);
            llama_set_compute_range(stage1_ctx, 0, split_compute_layer);
            if (llama_decode(stage1_ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
                LOG_ERR("split stage1 prompt prefill failed\n");
                llama_free(stage1_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            llama_token first_generated_mut = first_generated;
            if (llama_decode(stage1_ctx, llama_batch_get_one(&first_generated_mut, 1))) {
                LOG_ERR("split stage1 decode failed\n");
                llama_free(stage1_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            float * boundary_embd = llama_get_embeddings_ith(stage1_ctx, -1);
            if (!boundary_embd) {
                LOG_ERR("split stage1 boundary embeddings unavailable\n");
                llama_free(stage1_ctx);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            write_float_file(split_embd_out_path, boundary_embd, n_embd_inp);
            llama_free(stage1_ctx);
        }

        size_t state_size = 0;
        if (state_mode == "full") {
            state_size = llama_state_get_size(ctx);
            metrics.state_kind = "full_context";
        } else if (state_mode == "seq") {
            state_size = state_il_start >= 0 && state_il_end >= 0
                ? llama_state_seq_get_size_range(ctx, state_seq_id, state_il_start, state_il_end, 0)
                : llama_state_seq_get_size(ctx, state_seq_id);
            metrics.state_kind = "sequence_state";
        } else {
            state_size = state_il_start >= 0 && state_il_end >= 0
                ? llama_state_seq_get_size_range(ctx, state_seq_id, state_il_start, state_il_end, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
                : llama_state_seq_get_size_ext(ctx, state_seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            metrics.state_kind = "sequence_partial_state";
        }
        size_t serialized = 0;
        if (state_size > 0) {
            state_buffer.resize(state_size);
            const double export_t0 = ggml_time_us() / 1e6;
            if (state_mode == "full") {
                serialized = llama_state_get_data(ctx, state_buffer.data(), state_buffer.size());
            } else if (state_mode == "seq") {
                serialized = state_il_start >= 0 && state_il_end >= 0
                    ? llama_state_seq_get_data_range(ctx, state_buffer.data(), state_buffer.size(), state_seq_id, state_il_start, state_il_end, 0)
                    : llama_state_seq_get_data(ctx, state_buffer.data(), state_buffer.size(), state_seq_id);
            } else {
                serialized = state_il_start >= 0 && state_il_end >= 0
                    ? llama_state_seq_get_data_range(ctx, state_buffer.data(), state_buffer.size(), state_seq_id, state_il_start, state_il_end, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
                    : llama_state_seq_get_data_ext(ctx, state_buffer.data(), state_buffer.size(), state_seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            const double export_t1 = ggml_time_us() / 1e6;
            metrics.serialized_bytes = serialized;
            metrics.export_elapsed_s = export_t1 - export_t0;

            if (!state_out_path.empty() && serialized > 0) {
                state_buffer.resize(serialized);
                write_binary_file(state_out_path, state_buffer);
            }
        }

        if (metrics.requested_verify_tokens > 0 && serialized > 0) {
            llama_context_params shadow_ctx_params = ctx_params;
            shadow_ctx_params.cb_eval = nullptr;
            shadow_ctx_params.cb_eval_user_data = nullptr;
            if (!shadow_model_path.empty() && shadow_model == nullptr) {
                shadow_model = llama_model_load_from_file(shadow_model_path.c_str(), model_params);
                if (!shadow_model) {
                    LOG_ERR("failed to load shadow model\n");
                    if (shadow_ctx) {
                        llama_free(shadow_ctx);
                    }
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
            }
            shadow_ctx = llama_init_from_model(shadow_model ? shadow_model : model, shadow_ctx_params);
            if (shadow_ctx) {
                if (shadow_model && state_il_start >= 0 && state_il_end > state_il_start) {
                    llama_set_compute_range(shadow_ctx, state_il_start, state_il_end);
                }
                metrics.shadow_context_created = true;
                const double import_t0 = ggml_time_us() / 1e6;
                size_t imported = 0;
                if (state_mode == "full") {
                    imported = llama_state_set_data(shadow_ctx, state_buffer.data(), serialized);
                } else if (state_mode == "seq") {
                    imported = state_il_start >= 0 && state_il_end >= 0
                        ? llama_state_seq_set_data_range(shadow_ctx, state_buffer.data(), serialized, state_seq_id, state_il_start, state_il_end, 0)
                        : llama_state_seq_set_data(shadow_ctx, state_buffer.data(), serialized, state_seq_id);
                } else {
                    imported = state_il_start >= 0 && state_il_end >= 0
                        ? llama_state_seq_set_data_range(shadow_ctx, state_buffer.data(), serialized, state_seq_id, state_il_start, state_il_end, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
                        : llama_state_seq_set_data_ext(shadow_ctx, state_buffer.data(), serialized, state_seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                const double import_t1 = ggml_time_us() / 1e6;
                metrics.import_elapsed_s = import_t1 - import_t0;
                metrics.import_succeeded = imported == serialized;
                metrics.logits_available_after_import = llama_get_logits_ith(shadow_ctx, -1) != nullptr;
                if (metrics.import_succeeded) {
                    if (metrics.logits_available_after_import) {
                        metrics.readiness_path = "import_only";
                        metrics.shadow_ready_elapsed_s = metrics.import_elapsed_s;
                        metrics.import_plus_first_replay_elapsed_s = metrics.import_elapsed_s;
                    } else {
                        metrics.readiness_path = "needs_first_decode";
                    }
                } else {
                    metrics.readiness_path = "import_failed";
                }
            }
        }

        for (int i = 0; i < params.n_predict; ++i) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) {
                LOG_ERR("baseline logits unavailable at token %d\n", i);
                break;
            }

            const llama_token next = select_greedy_token(logits, n_vocab);
            if (llama_vocab_is_eog(vocab, next)) {
                break;
            }

            if (shadow_ctx &&
                metrics.import_succeeded &&
                metrics.verified_decode_tokens < metrics.requested_verify_tokens) {
                const float * shadow_logits = llama_get_logits_ith(shadow_ctx, -1);
                if (shadow_logits != nullptr) {
                    const llama_token shadow_next = select_greedy_token(shadow_logits, n_vocab);
                    metrics.compared_token_steps++;
                    if (shadow_next != next) {
                        metrics.token_mismatch_count++;
                        if (metrics.first_mismatch_index < 0) {
                            metrics.first_mismatch_index = i;
                            metrics.first_baseline_token = next;
                            metrics.first_shadow_token = shadow_next;
                        }
                    }
                }
            }

            generated_token_ids.push_back(next);
            metrics.generated_token_ids_trace.push_back((int) next);
            metrics.output_text += common_token_to_piece(ctx, next, false);
            metrics.generated_tokens++;

            llama_token next_mut = next;
            const double base_decode_t0 = ggml_time_us() / 1e6;
            if (llama_decode(ctx, llama_batch_get_one(&next_mut, 1))) {
                LOG_ERR("baseline decode failed at token %d\n", i);
                break;
            }
            const double base_decode_t1 = ggml_time_us() / 1e6;
            if (i == 0) {
                metrics.first_baseline_decode_elapsed_s = base_decode_t1 - base_decode_t0;
                if (metrics.readiness_path == "import_only" && metrics.shadow_ready_elapsed_s >= 0.0) {
                    metrics.ready_vs_baseline_first_token_delta_s =
                        metrics.shadow_ready_elapsed_s - metrics.first_baseline_decode_elapsed_s;
                }
            }

            if (shadow_ctx &&
                metrics.import_succeeded &&
                metrics.verified_decode_tokens < metrics.requested_verify_tokens) {
                const double shadow_decode_t0 = ggml_time_us() / 1e6;
                if (llama_decode(shadow_ctx, llama_batch_get_one(&next_mut, 1))) {
                    metrics.shadow_decode_failure_index = i;
                    break;
                }
                const double shadow_decode_t1 = ggml_time_us() / 1e6;

                if (metrics.verified_decode_tokens == 0) {
                    metrics.first_shadow_decode_elapsed_s = shadow_decode_t1 - shadow_decode_t0;
                    metrics.first_token_replay_delta_s =
                        metrics.first_shadow_decode_elapsed_s - metrics.first_baseline_decode_elapsed_s;
                    metrics.import_plus_first_replay_elapsed_s =
                        metrics.import_elapsed_s + metrics.first_shadow_decode_elapsed_s;
                    metrics.first_replayed_token = next;
                    metrics.logits_available_after_first_shadow_decode =
                        llama_get_logits_ith(shadow_ctx, -1) != nullptr;
                    if (metrics.logits_available_after_first_shadow_decode) {
                        metrics.readiness_path = "after_first_decode";
                        metrics.shadow_ready_elapsed_s =
                            metrics.import_plus_first_replay_elapsed_s;
                        metrics.ready_vs_baseline_first_token_delta_s =
                            metrics.shadow_ready_elapsed_s - metrics.first_baseline_decode_elapsed_s;
                    } else if (!metrics.logits_available_after_import) {
                        metrics.readiness_path = "not_ready_after_first_decode";
                    }
                }

                metrics.verified_decode_tokens++;
            }
        }

        if (!split_embd_stream_out_path.empty() && split_compute_layer > 0 && generated_token_ids.size() > 1) {
            const int32_t n_layer = llama_model_n_layer(model);
            const int32_t n_embd_inp = llama_model_n_embd_inp(model);
            if (split_compute_layer >= n_layer) {
                LOG_ERR("--split-compute-layer must be in (0, %d) for split embedding stream\n", n_layer);
                if (shadow_ctx) {
                    llama_free(shadow_ctx);
                }
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            llama_context_params stage1_ctx_params = ctx_params;
            stage1_ctx_params.cb_eval = nullptr;
            stage1_ctx_params.cb_eval_user_data = nullptr;
            llama_context * stage1_ctx = llama_init_from_model(model, stage1_ctx_params);
            if (!stage1_ctx) {
                LOG_ERR("failed to create split stage1 context for stream export\n");
                if (shadow_ctx) {
                    llama_free(shadow_ctx);
                }
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            llama_set_embeddings(stage1_ctx, true);
            llama_set_logits(stage1_ctx, false);
            llama_set_compute_range(stage1_ctx, 0, split_compute_layer);
            if (llama_decode(stage1_ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
                LOG_ERR("split stage1 prompt prefill failed for stream export\n");
                llama_free(stage1_ctx);
                if (shadow_ctx) {
                    llama_free(shadow_ctx);
                }
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            const size_t stream_steps = generated_token_ids.size() - 1;
            std::vector<float> embd_stream(stream_steps * (size_t) n_embd_inp);
            for (size_t step = 0; step < stream_steps; ++step) {
                llama_token token = generated_token_ids[step];
                if (llama_decode(stage1_ctx, llama_batch_get_one(&token, 1))) {
                    LOG_ERR("split stage1 decode failed during stream export at step %zu\n", step);
                    llama_free(stage1_ctx);
                    if (shadow_ctx) {
                        llama_free(shadow_ctx);
                    }
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                float * boundary_embd = llama_get_embeddings_ith(stage1_ctx, -1);
                if (!boundary_embd) {
                    LOG_ERR("split stage1 boundary embeddings unavailable during stream export at step %zu\n", step);
                    llama_free(stage1_ctx);
                    if (shadow_ctx) {
                        llama_free(shadow_ctx);
                    }
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                std::memcpy(
                    embd_stream.data() + step * (size_t) n_embd_inp,
                    boundary_embd,
                    sizeof(float) * (size_t) n_embd_inp
                );
            }
            llama_free(stage1_ctx);

            std::ofstream out(split_embd_stream_out_path, std::ios::binary);
            if (!out) {
                LOG_ERR("failed to open split embedding stream output\n");
                if (shadow_ctx) {
                    llama_free(shadow_ctx);
                }
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            out.write(reinterpret_cast<const char *>(embd_stream.data()), std::streamsize(sizeof(float) * embd_stream.size()));
        }
    } else {
        if (state_mode == "full") {
            metrics.state_kind = "imported_full_context";
        } else if (state_mode == "seq") {
            metrics.state_kind = "imported_sequence_state";
        } else {
            metrics.state_kind = "imported_sequence_partial_state";
        }
        metrics.prompt_tokens = 0;
        if (!replay_tokens_in_path.empty()) {
            replay_tokens = read_token_file(replay_tokens_in_path);
        }
        if (!state_in_path.empty()) {
            state_buffer = read_binary_file(state_in_path);
        } else {
            state_buffer = read_chunk_directory(state_chunks_dir_path);
        }
        metrics.serialized_bytes = state_buffer.size();
        metrics.speculative_tokens = split_speculative_tokens;
        metrics.speculative_draft_mode =
            split_speculative_tokens > 1 ? split_speculative_draft_mode : "disabled";
        const double import_t0 = ggml_time_us() / 1e6;
        size_t imported = 0;
        if (state_mode == "full") {
            imported = llama_state_set_data(ctx, state_buffer.data(), state_buffer.size());
        } else if (state_mode == "seq") {
            imported = state_il_start >= 0 && state_il_end >= 0
                ? llama_state_seq_set_data_range(ctx, state_buffer.data(), state_buffer.size(), state_seq_id, state_il_start, state_il_end, 0)
                : llama_state_seq_set_data(ctx, state_buffer.data(), state_buffer.size(), state_seq_id);
        } else {
            imported = state_il_start >= 0 && state_il_end >= 0
                ? llama_state_seq_set_data_range(ctx, state_buffer.data(), state_buffer.size(), state_seq_id, state_il_start, state_il_end, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
                : llama_state_seq_set_data_ext(ctx, state_buffer.data(), state_buffer.size(), state_seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
        const double import_t1 = ggml_time_us() / 1e6;
        metrics.import_elapsed_s = import_t1 - import_t0;
        metrics.import_succeeded = imported == state_buffer.size();
        metrics.logits_available_after_import = llama_get_logits_ith(ctx, -1) != nullptr;
        metrics.first_baseline_decode_elapsed_s = baseline_first_token_elapsed_s;
        if (metrics.import_succeeded) {
            if (metrics.logits_available_after_import) {
                metrics.readiness_path = "import_only";
                metrics.shadow_ready_elapsed_s = metrics.import_elapsed_s;
                metrics.import_plus_first_replay_elapsed_s = metrics.import_elapsed_s;
            } else {
                metrics.readiness_path = "needs_first_decode";
            }
        } else {
            metrics.readiness_path = "import_failed";
        }

        if (split_stage2_mode) {
            const int32_t n_layer = llama_model_n_layer(model);
            const int32_t n_embd_inp = llama_model_n_embd_inp(model);
            if (split_stage2_layer <= 0 || split_stage2_layer >= n_layer) {
                LOG_ERR("--split-stage2-layer must be in (0, %d)\n", n_layer);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            if (!metrics.import_succeeded) {
                LOG_ERR("split stage2 requires successful state import\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            llama_set_compute_range(ctx, split_stage2_layer, n_layer);
            if (!split_embd_stream_in_path.empty() || !split_embd_stream_connect_addr.empty()) {
                std::vector<float> embd_stream;
                int socket_fd = -1;
                int max_steps = 0;
                if (!split_embd_stream_in_path.empty()) {
                    embd_stream = read_float_stream_file(split_embd_stream_in_path);
                    if (embd_stream.size() % (size_t) n_embd_inp != 0) {
                        LOG_ERR("split embedding stream size is not aligned to embedding width\n");
                        llama_free(ctx);
                        llama_model_free(model);
                        llama_backend_free();
                        return 1;
                    }
                    const size_t available_steps = embd_stream.size() / (size_t) n_embd_inp;
                    const int replay_steps = replay_tokens.empty() ? 0 : std::max(0, (int) replay_tokens.size() - 1);
                    max_steps = std::min<int>({params.n_predict, (int) available_steps, replay_steps});
                } else {
                    const size_t colon = split_embd_stream_connect_addr.rfind(':');
                    if (colon == std::string::npos) {
                        LOG_ERR("--split-embd-stream-connect must be host:port\n");
                        llama_free(ctx);
                        llama_model_free(model);
                        llama_backend_free();
                        return 1;
                    }
                    const std::string host = split_embd_stream_connect_addr.substr(0, colon);
                    const int port = std::atoi(split_embd_stream_connect_addr.substr(colon + 1).c_str());
                    socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
                    if (socket_fd < 0) {
                        LOG_ERR("failed to create split embedding stream socket\n");
                        llama_free(ctx);
                        llama_model_free(model);
                        llama_backend_free();
                        return 1;
                    }
                    sockaddr_in addr{};
                    addr.sin_family = AF_INET;
                    addr.sin_port = htons((uint16_t) port);
                    addr.sin_addr.s_addr = (host == "127.0.0.1" || host == "localhost")
                        ? htonl(INADDR_LOOPBACK)
                        : htonl(INADDR_LOOPBACK);
                    if (::connect(socket_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
                        ::close(socket_fd);
                        LOG_ERR("failed to connect split embedding stream socket\n");
                        llama_free(ctx);
                        llama_model_free(model);
                        llama_backend_free();
                        return 1;
                    }
                    const int replay_steps = replay_tokens.empty() ? 0 : std::max(0, (int) replay_tokens.size() - 1);
                    max_steps = std::min<int>(params.n_predict, replay_steps);
                }
                metrics.readiness_path = "split_stage2_decode_stream";
                metrics.shadow_decode_succeeded = true;
                for (int step = 0; step < max_steps;) {
                    int block_len = 1;
                    std::vector<int32_t> draft_tokens;
                    std::vector<float> block_embd;
                    if (!split_embd_stream_in_path.empty()) {
                        block_len = 1;
                        block_embd.resize((size_t) n_embd_inp);
                        std::memcpy(
                            block_embd.data(),
                            embd_stream.data() + (size_t) step * (size_t) n_embd_inp,
                            sizeof(float) * (size_t) n_embd_inp
                        );
                    } else if (split_speculative_tokens > 1) {
                        int32_t stream_step = -1;
                        int32_t stream_block_len = 0;
                        if (!recv_all(socket_fd, &stream_step, sizeof(stream_step)) ||
                            !recv_all(socket_fd, &stream_block_len, sizeof(stream_block_len)) ||
                            stream_step != step || stream_block_len <= 0) {
                            metrics.shadow_decode_failure_index = step;
                            metrics.shadow_decode_succeeded = false;
                            break;
                        }
                        block_len = std::min(stream_block_len, max_steps - step);
                        draft_tokens.resize((size_t) block_len);
                        block_embd.resize((size_t) block_len * (size_t) n_embd_inp);
                        bool block_recv_ok = true;
                        for (int i = 0; i < block_len; ++i) {
                            if (!recv_all(socket_fd, &draft_tokens[(size_t) i], sizeof(int32_t)) ||
                                !recv_all(
                                    socket_fd,
                                    block_embd.data() + (size_t) i * (size_t) n_embd_inp,
                                    sizeof(float) * (size_t) n_embd_inp)) {
                                block_recv_ok = false;
                                break;
                            }
                        }
                        if (!block_recv_ok) {
                            metrics.shadow_decode_failure_index = step;
                            metrics.shadow_decode_succeeded = false;
                            break;
                        }
                    } else {
                        int32_t stream_step = -1;
                        if (!recv_all(socket_fd, &stream_step, sizeof(stream_step)) || stream_step != step) {
                            metrics.shadow_decode_failure_index = step;
                            metrics.shadow_decode_succeeded = false;
                            break;
                        }
                        block_embd.resize((size_t) n_embd_inp);
                        if (!recv_all(socket_fd, block_embd.data(), sizeof(float) * block_embd.size())) {
                            metrics.shadow_decode_failure_index = step;
                            metrics.shadow_decode_succeeded = false;
                            break;
                        }
                    }

                    std::vector<int32_t> returned_tokens;
                    int accepted_prefix_len = 0;
                    for (int block_index = 0; block_index < block_len; ++block_index) {
                        llama_batch embd_batch = llama_batch_init(1, n_embd_inp, 1);
                        embd_batch.n_tokens = 1;
                        std::memcpy(
                            embd_batch.embd,
                            block_embd.data() + (size_t) block_index * (size_t) n_embd_inp,
                            sizeof(float) * (size_t) n_embd_inp
                        );
                        embd_batch.pos[0] = split_prompt_tokens + step + block_index;
                        embd_batch.n_seq_id[0] = 1;
                        embd_batch.seq_id[0][0] = 0;
                        embd_batch.logits[0] = 1;

                        const double decode_t0 = ggml_time_us() / 1e6;
                        if (llama_decode(ctx, embd_batch)) {
                            llama_batch_free(embd_batch);
                            LOG_ERR("split stage2 stream decode failed at step %d\n", step + block_index);
                            metrics.shadow_decode_failure_index = step + block_index;
                            metrics.shadow_decode_succeeded = false;
                            break;
                        }
                        const double decode_t1 = ggml_time_us() / 1e6;
                        llama_batch_free(embd_batch);

                        const float * split_logits = llama_get_logits_ith(ctx, -1);
                        if (!split_logits) {
                            LOG_ERR("split stage2 stream logits unavailable at step %d\n", step + block_index);
                            llama_free(ctx);
                            llama_model_free(model);
                            llama_backend_free();
                            return 1;
                        }
                        const llama_token predicted = select_greedy_token(split_logits, n_vocab);
                        const llama_token expected = replay_tokens[step + block_index + 1];
                        returned_tokens.push_back((int32_t) predicted);
                        metrics.generated_tokens++;
                        metrics.verified_decode_tokens++;
                        metrics.compared_token_steps++;
                        metrics.generated_token_ids_trace.push_back((int) predicted);
                        metrics.output_text += common_token_to_piece(ctx, predicted, false);
                        if (step == 0 && block_index == 0) {
                            metrics.first_shadow_token = predicted;
                            metrics.first_baseline_token = expected;
                            metrics.first_shadow_decode_elapsed_s = decode_t1 - decode_t0;
                            metrics.import_plus_first_replay_elapsed_s =
                                metrics.import_elapsed_s + metrics.first_shadow_decode_elapsed_s;
                            metrics.first_token_replay_delta_s =
                                metrics.first_baseline_decode_elapsed_s >= 0.0
                                ? metrics.first_shadow_decode_elapsed_s - metrics.first_baseline_decode_elapsed_s
                                : 0.0;
                            metrics.shadow_ready_elapsed_s = metrics.import_plus_first_replay_elapsed_s;
                            metrics.ready_vs_baseline_first_token_delta_s =
                                metrics.first_baseline_decode_elapsed_s >= 0.0
                                ? metrics.shadow_ready_elapsed_s - metrics.first_baseline_decode_elapsed_s
                                : -1.0;
                            metrics.logits_available_after_first_shadow_decode = true;
                        }
                        if (predicted != expected) {
                            metrics.token_mismatch_count++;
                            if (metrics.first_mismatch_index < 0) {
                                metrics.first_mismatch_index = step + block_index;
                                metrics.first_baseline_token = expected;
                                metrics.first_shadow_token = predicted;
                            }
                        }
                        if (socket_fd >= 0 && split_speculative_tokens > 1) {
                            if ((int32_t) predicted == draft_tokens[(size_t) block_index] &&
                                accepted_prefix_len == block_index) {
                                accepted_prefix_len++;
                            } else {
                                break;
                            }
                        }
                    }
                    if (!metrics.shadow_decode_succeeded) {
                        break;
                    }
                    if (socket_fd >= 0 && split_speculative_tokens > 1) {
                        const int32_t accepted_i32 = accepted_prefix_len;
                        const int32_t returned_count_i32 = (int32_t) returned_tokens.size();
                        if (!send_all(socket_fd, &accepted_i32, sizeof(accepted_i32)) ||
                            !send_all(socket_fd, &returned_count_i32, sizeof(returned_count_i32)) ||
                            !send_all(socket_fd, returned_tokens.data(), sizeof(int32_t) * returned_tokens.size())) {
                            metrics.shadow_decode_failure_index = step;
                            metrics.shadow_decode_succeeded = false;
                            break;
                        }
                        metrics.speculative_rounds++;
                        metrics.speculative_drafted_tokens += block_len;
                        metrics.speculative_accepted_tokens += accepted_prefix_len;
                        metrics.speculative_verified_tokens += returned_count_i32;
                        if (accepted_prefix_len < block_len) {
                            metrics.speculative_rollback_count++;
                        }
                        step += returned_count_i32;
                    } else {
                        if (socket_fd >= 0) {
                            const int32_t predicted_i32 = returned_tokens.empty() ? -1 : returned_tokens[0];
                            if (!send_all(socket_fd, &predicted_i32, sizeof(predicted_i32))) {
                                metrics.shadow_decode_failure_index = step;
                                metrics.shadow_decode_succeeded = false;
                                break;
                            }
                        }
                        step += 1;
                    }
                }
                if (socket_fd >= 0) {
                    ::close(socket_fd);
                }
            } else {
                std::vector<float> boundary_embd = read_float_file(split_embd_in_path, n_embd_inp);

                llama_batch embd_batch = llama_batch_init(1, n_embd_inp, 1);
                embd_batch.n_tokens = 1;
                std::memcpy(embd_batch.embd, boundary_embd.data(), sizeof(float) * n_embd_inp);
                embd_batch.pos[0] = split_prompt_tokens;
                embd_batch.n_seq_id[0] = 1;
                embd_batch.seq_id[0][0] = 0;
                embd_batch.logits[0] = 1;

                const double decode_t0 = ggml_time_us() / 1e6;
                if (llama_decode(ctx, embd_batch)) {
                    llama_batch_free(embd_batch);
                    LOG_ERR("split stage2 decode failed\n");
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                const double decode_t1 = ggml_time_us() / 1e6;
                llama_batch_free(embd_batch);

                const float * split_logits = llama_get_logits_ith(ctx, -1);
                if (!split_logits) {
                    LOG_ERR("split stage2 logits unavailable\n");
                    llama_free(ctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                const llama_token predicted = select_greedy_token(split_logits, n_vocab);
                metrics.first_shadow_token = predicted;
                metrics.first_shadow_decode_elapsed_s = decode_t1 - decode_t0;
                metrics.import_plus_first_replay_elapsed_s =
                    metrics.import_elapsed_s + metrics.first_shadow_decode_elapsed_s;
                metrics.first_token_replay_delta_s =
                    metrics.first_baseline_decode_elapsed_s >= 0.0
                    ? metrics.first_shadow_decode_elapsed_s - metrics.first_baseline_decode_elapsed_s
                    : 0.0;
                metrics.shadow_ready_elapsed_s = metrics.import_plus_first_replay_elapsed_s;
                metrics.ready_vs_baseline_first_token_delta_s =
                    metrics.first_baseline_decode_elapsed_s >= 0.0
                    ? metrics.shadow_ready_elapsed_s - metrics.first_baseline_decode_elapsed_s
                    : -1.0;
                metrics.logits_available_after_first_shadow_decode = true;
                metrics.readiness_path = "split_stage2_decode";
                metrics.shadow_decode_succeeded = true;
                metrics.verified_decode_tokens = 1;
                metrics.generated_token_ids_trace.push_back((int) predicted);
                metrics.output_text = common_token_to_piece(ctx, predicted, false);
                if (split_expected_token >= 0) {
                    metrics.token_mismatch_count = predicted == split_expected_token ? 0 : 1;
                    metrics.first_baseline_token = split_expected_token;
                }
            }

            if (!json_out_path.empty()) {
                write_result_json(json_out_path, params.prompt, metrics);
            }

            LOG_INF("readiness path: %s\n", metrics.readiness_path.c_str());
            LOG_INF("state import: %s\n", metrics.import_succeeded ? "ok" : "failed");
            LOG_INF("first shadow decode: %.6fs\n", metrics.first_shadow_decode_elapsed_s);
            LOG_INF("import + first replay: %.6fs\n", metrics.import_plus_first_replay_elapsed_s);
            LOG_INF("ready delta vs baseline first token: %.6fs\n", metrics.ready_vs_baseline_first_token_delta_s);
            LOG_INF("split predicted token: %d\n", metrics.first_shadow_token);

            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return metrics.token_mismatch_count == 0 ? 0 : 2;
        }

        const int replay_limit = std::min<int>((int) replay_tokens.size(), params.n_predict);
        for (int i = 0; i < replay_limit; ++i) {
            llama_token token = replay_tokens[i];
            metrics.output_text += common_token_to_piece(ctx, token, false);
            metrics.generated_token_ids_trace.push_back((int) token);
            metrics.generated_tokens++;
            const double decode_t0 = ggml_time_us() / 1e6;
            if (llama_decode(ctx, llama_batch_get_one(&token, 1))) {
                metrics.shadow_decode_failure_index = i;
                break;
            }
            const double decode_t1 = ggml_time_us() / 1e6;
            if (metrics.verified_decode_tokens == 0) {
                metrics.first_shadow_decode_elapsed_s = decode_t1 - decode_t0;
                metrics.import_plus_first_replay_elapsed_s =
                    metrics.import_elapsed_s + metrics.first_shadow_decode_elapsed_s;
                metrics.first_replayed_token = replay_tokens[i];
                metrics.logits_available_after_first_shadow_decode =
                    llama_get_logits_ith(ctx, -1) != nullptr;
                if (metrics.first_baseline_decode_elapsed_s >= 0.0) {
                    metrics.first_token_replay_delta_s =
                        metrics.first_shadow_decode_elapsed_s - metrics.first_baseline_decode_elapsed_s;
                }
                if (metrics.logits_available_after_first_shadow_decode) {
                    metrics.readiness_path = "after_first_decode";
                    metrics.shadow_ready_elapsed_s = metrics.import_plus_first_replay_elapsed_s;
                    if (metrics.first_baseline_decode_elapsed_s >= 0.0) {
                        metrics.ready_vs_baseline_first_token_delta_s =
                            metrics.shadow_ready_elapsed_s - metrics.first_baseline_decode_elapsed_s;
                    }
                } else if (!metrics.logits_available_after_import) {
                    metrics.readiness_path = "not_ready_after_first_decode";
                }
            }
            metrics.verified_decode_tokens++;
        }
    }

    if (!replay_tokens_out_path.empty() && !generated_token_ids.empty()) {
        write_token_file(replay_tokens_out_path, generated_token_ids);
    }

    const int expected_verified_tokens = !has_import_state
        ? std::min(metrics.requested_verify_tokens, metrics.generated_tokens)
        : std::min<int>(metrics.requested_verify_tokens, (int) replay_tokens.size());
    metrics.shadow_decode_succeeded =
        metrics.import_succeeded &&
        metrics.shadow_decode_failure_index < 0 &&
        metrics.verified_decode_tokens == expected_verified_tokens;

    if (!json_out_path.empty()) {
        write_result_json(json_out_path, params.prompt, metrics);
    }

    LOG_INF("prompt tokens: %d\n", metrics.prompt_tokens);
    LOG_INF("prefill elapsed: %.6fs\n", metrics.prefill_elapsed_s);
    LOG_INF("serialized state: %.2f MiB\n", metrics.serialized_bytes / (1024.0 * 1024.0));
    LOG_INF("state import: %s\n", metrics.import_succeeded ? "ok" : "failed");
    LOG_INF("readiness path: %s\n", metrics.readiness_path.c_str());
    LOG_INF("first baseline decode: %.6fs\n", metrics.first_baseline_decode_elapsed_s);
    LOG_INF("first shadow decode: %.6fs\n", metrics.first_shadow_decode_elapsed_s);
    LOG_INF("import + first replay: %.6fs\n", metrics.import_plus_first_replay_elapsed_s);
    LOG_INF("first replay delta vs baseline: %.6fs\n", metrics.first_token_replay_delta_s);
    LOG_INF("ready delta vs baseline first token: %.6fs\n", metrics.ready_vs_baseline_first_token_delta_s);
    LOG_INF("compared steps: %d mismatches: %d\n", metrics.compared_token_steps, metrics.token_mismatch_count);

    if (shadow_ctx) {
        llama_free(shadow_ctx);
    }
    if (shadow_model) {
        llama_model_free(shadow_model);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
