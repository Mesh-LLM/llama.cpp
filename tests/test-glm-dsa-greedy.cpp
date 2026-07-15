#include "test-glm-dsa-greedy.h"

#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama-cpp.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)
#    include <mach/mach.h>
#endif

namespace {

constexpr int    k_decode_steps     = 256;
constexpr int    k_warmup_steps     = 16;
constexpr int    k_route_probe_step = 130;
constexpr size_t k_layer_count      = 7;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static uint64_t resident_bytes() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t      count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t         status =
        task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count);
    return status == KERN_SUCCESS ? info.resident_size : 0;
#else
    return 0;
#endif
}

static bool is_metal_backend(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    const char * backend_name = ggml_backend_name(backend);
    const char * device_name  = ggml_backend_dev_name(ggml_backend_get_device(backend));
    return (backend_name && std::strncmp(backend_name, "Metal", 5) == 0) ||
           (device_name && std::strncmp(device_name, "MTL", 3) == 0);
}

static bool is_named_get_rows(const ggml_tensor * tensor, const char * name) {
    return tensor->op == GGML_OP_GET_ROWS && std::strstr(tensor->name, name);
}

struct op_signature {
    int lightning_indexer = 0;
    int selected_k        = 0;
    int selected_v        = 0;
    int dense_mask        = 0;
    int rope              = 0;
    int flash_attention   = 0;
    int heavy             = 0;
    int heavy_metal       = 0;
};

static bool same_shape(const op_signature & lhs, const op_signature & rhs) {
    return lhs.lightning_indexer == rhs.lightning_indexer && lhs.selected_k == rhs.selected_k &&
           lhs.selected_v == rhs.selected_v && lhs.dense_mask == rhs.dense_mask && lhs.rope == rhs.rope &&
           lhs.flash_attention == rhs.flash_attention && lhs.heavy == rhs.heavy && lhs.heavy_metal == rhs.heavy_metal;
}

struct route_capture {
    std::array<std::vector<int32_t>, k_layer_count> top_k;
    std::array<std::vector<float>, k_layer_count>   scores;
};

struct decode_observer {
    llama_context *          context        = nullptr;
    bool                     active         = false;
    bool                     capture_routes = false;
    op_signature             current;
    std::vector<std::string> non_metal_ops;
    route_capture            routes;

    void begin_step() {
        active  = true;
        current = {};
        non_metal_ops.clear();
    }

    op_signature finish_step() {
        active = false;
        return current;
    }
};

static int tensor_layer(const ggml_tensor * tensor, const char * prefix) {
    const size_t prefix_length = std::strlen(prefix);
    if (std::strncmp(tensor->name, prefix, prefix_length) != 0 || tensor->name[prefix_length] != '-') {
        return -1;
    }

    char *     end   = nullptr;
    const long layer = std::strtol(tensor->name + prefix_length + 1, &end, 10);
    return end != tensor->name + prefix_length + 1 && *end == '\0' && layer >= 0 && layer < (long) k_layer_count ?
               layer :
               -1;
}

static bool should_capture_route(const ggml_tensor * tensor) {
    return tensor_layer(tensor, "ffn_moe_topk") >= 0 || tensor_layer(tensor, "ffn_moe_probs_biased") >= 0;
}

static void capture_route_tensor(decode_observer & observer, ggml_tensor * tensor) {
    if (const int layer = tensor_layer(tensor, "ffn_moe_topk"); layer >= 0) {
        require(tensor->type == GGML_TYPE_I32, "captured GLM expert route is not i32");
        observer.routes.top_k[layer].resize(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, observer.routes.top_k[layer].data(), 0,
                                observer.routes.top_k[layer].size() * sizeof(int32_t));
        return;
    }

    if (const int layer = tensor_layer(tensor, "ffn_moe_probs_biased"); layer >= 0) {
        require(tensor->type == GGML_TYPE_F32, "captured GLM expert scores are not f32");
        observer.routes.scores[layer].resize(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, observer.routes.scores[layer].data(), 0,
                                observer.routes.scores[layer].size() * sizeof(float));
    }
}

static bool is_dense_sparse_mask(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_SET_ROWS && tensor->ne[0] == 1 && tensor->ne[1] > 1;
}

static bool observe_decode(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * observer = static_cast<decode_observer *>(user_data);
    if (!ask) {
        capture_route_tensor(*observer, tensor);
        return true;
    }
    if (!observer->active) {
        return false;
    }

    const bool selected_k = is_named_get_rows(tensor, "k_selected");
    const bool selected_v = is_named_get_rows(tensor, "v_selected");
    const bool dense_mask = is_dense_sparse_mask(tensor);
    observer->current.lightning_indexer += tensor->op == GGML_OP_LIGHTNING_INDEXER;
    observer->current.selected_k += selected_k;
    observer->current.selected_v += selected_v;
    observer->current.dense_mask += dense_mask;
    observer->current.rope += tensor->op == GGML_OP_ROPE;
    observer->current.flash_attention += tensor->op == GGML_OP_FLASH_ATTN_EXT;

    const bool heavy = tensor->op == GGML_OP_LIGHTNING_INDEXER || tensor->op == GGML_OP_ROPE ||
                       tensor->op == GGML_OP_FLASH_ATTN_EXT || selected_k || selected_v;
    if (heavy) {
        ++observer->current.heavy;
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(observer->context->get_sched(), tensor);
        if (is_metal_backend(backend)) {
            ++observer->current.heavy_metal;
        } else {
            const char * backend_name = backend ? ggml_backend_name(backend) : "unassigned";
            observer->non_metal_ops.emplace_back(std::string(tensor->name) + " (" + backend_name + ")");
        }
    }
    return observer->capture_routes && should_capture_route(tensor);
}

static bool silent_model_load_progress(float, void *) {
    return true;
}

static llama_model_ptr load_model(const char * model_path, ggml_backend_dev_t device) {
    std::array<ggml_backend_dev_t, 2> devices      = { device, nullptr };
    llama_model_params                model_params = llama_model_default_params();
    model_params.devices                           = devices.data();
    model_params.n_gpu_layers                      = -1;
    model_params.split_mode                        = LLAMA_SPLIT_MODE_NONE;
    model_params.progress_callback                 = silent_model_load_progress;
    llama_model_ptr model(llama_model_load_from_file(model_path, model_params));
    require(model != nullptr, "failed to load GLM long-parity fixture");
    return model;
}

static llama_context_ptr make_context(llama_model *     model,
                                      uint32_t          context_length,
                                      decode_observer & observer,
                                      bool              native_metal) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = context_length;
    params.n_batch              = 32;
    params.n_ubatch             = 32;
    params.n_threads            = 8;
    params.n_threads_batch      = 8;
    params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cb_eval              = observe_decode;
    params.cb_eval_user_data    = &observer;

    llama_context_ptr context(llama_init_from_model(model, params));
    require(context != nullptr, "failed to create GLM long-parity context");
    observer.context = context.get();

    llama_cparams & cparams = const_cast<llama_cparams &>(context->get_cparams());
    cparams.fused_lid       = native_metal;
    cparams.auto_flid       = false;
    cparams.flash_attn      = native_metal;
    cparams.auto_fa         = false;
    return context;
}

static void enable_cpu_reference(llama_context * context) {
    ggml_backend_sched_t scheduler = context->get_sched();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(scheduler); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(scheduler, i);
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        using set_use_ref_fn   = void (*)(ggml_backend_t, bool);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto               set_use_ref =
            reinterpret_cast<set_use_ref_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref"));
        require(set_use_ref != nullptr, "CPU backend has no reference-mode control");
        set_use_ref(backend, true);
    }
}

struct batch_guard {
    llama_batch value;

    batch_guard() : value(llama_batch_init(16, 0, 1)) {}

    ~batch_guard() { llama_batch_free(value); }
};

static const float * decode_tokens(llama_context *     context,
                                   batch_guard &       batch,
                                   const llama_token * tokens,
                                   size_t              count,
                                   int32_t             position,
                                   decode_observer *   observer,
                                   op_signature *      signature) {
    common_batch_clear(batch.value);
    for (size_t i = 0; i < count; ++i) {
        common_batch_add(batch.value, tokens[i], position + i, { 0 }, i + 1 == count);
    }
    if (observer) {
        observer->begin_step();
    }
    require(llama_decode(context, batch.value) == 0, "GLM long-parity decode failed");
    if (observer && signature) {
        *signature = observer->finish_step();
    }
    const float * logits = llama_get_logits_ith(context, batch.value.n_tokens - 1);
    require(logits != nullptr, "GLM long-parity decode produced no logits");
    return logits;
}

static llama_token greedy_token(const float * logits, int32_t n_vocab) {
    llama_token result     = -1;
    float       best_value = -std::numeric_limits<float>::infinity();
    for (llama_token token = 0; token < n_vocab; ++token) {
        if (!std::isnan(logits[token]) && (result < 0 || logits[token] > best_value)) {
            result     = token;
            best_value = logits[token];
        }
    }
    require(result >= 0, "GLM long-parity logits contain no finite argmax");
    return result;
}

static double logit_nmse(const float * expected, const float * actual, int32_t n_vocab) {
    double squared_error     = 0.0;
    double squared_reference = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
            if (std::isinf(expected[i]) && std::isinf(actual[i]) &&
                std::signbit(expected[i]) == std::signbit(actual[i])) {
                continue;
            }
            return std::numeric_limits<double>::infinity();
        }
        const double difference = static_cast<double>(expected[i]) - actual[i];
        squared_error += difference * difference;
        squared_reference += static_cast<double>(expected[i]) * expected[i];
    }
    return squared_reference == 0.0 ? squared_error : squared_error / squared_reference;
}

struct nmse_accumulator {
    long double squared_error     = 0.0;
    long double squared_reference = 0.0;

    void add(const float * expected, const float * actual, int32_t count) {
        for (int32_t i = 0; i < count; ++i) {
            if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
                continue;
            }
            const long double difference = static_cast<long double>(expected[i]) - actual[i];
            squared_error += difference * difference;
            squared_reference += static_cast<long double>(expected[i]) * expected[i];
        }
    }

    double value() const {
        return squared_reference == 0.0 ? static_cast<double>(squared_error) :
                                          static_cast<double>(squared_error / squared_reference);
    }
};

static bool same_indices(const std::vector<int32_t> & lhs, const std::vector<int32_t> & rhs) {
    std::vector<int32_t> lhs_sorted = lhs;
    std::vector<int32_t> rhs_sorted = rhs;
    std::sort(lhs_sorted.begin(), lhs_sorted.end());
    std::sort(rhs_sorted.begin(), rhs_sorted.end());
    return lhs_sorted == rhs_sorted;
}

static bool route_difference_is_near_tie(const std::vector<int32_t> & expected,
                                         const std::vector<int32_t> & actual,
                                         const std::vector<float> &   expected_scores,
                                         const std::vector<float> &   actual_scores,
                                         double &                     max_score_error,
                                         double &                     max_reference_gap) {
    require(expected.size() == actual.size(), "GLM expert route widths differ");
    require(expected_scores.size() == actual_scores.size(), "GLM expert score widths differ");

    max_score_error = 0.0;
    for (size_t i = 0; i < expected_scores.size(); ++i) {
        max_score_error =
            std::max(max_score_error, std::fabs(static_cast<double>(expected_scores[i]) - actual_scores[i]));
    }

    std::vector<int32_t> expected_only;
    std::vector<int32_t> actual_only;
    for (int32_t expert : expected) {
        if (std::find(actual.begin(), actual.end(), expert) == actual.end()) {
            expected_only.push_back(expert);
        }
    }
    for (int32_t expert : actual) {
        if (std::find(expected.begin(), expected.end(), expert) == expected.end()) {
            actual_only.push_back(expert);
        }
    }
    require(expected_only.size() == actual_only.size() && !expected_only.empty(),
            "GLM expert route difference is malformed");

    max_reference_gap = 0.0;
    for (int32_t selected : expected_only) {
        require(selected >= 0 && static_cast<size_t>(selected) < expected_scores.size(),
                "CPU GLM expert route is out of range");
        for (int32_t replacement : actual_only) {
            require(replacement >= 0 && static_cast<size_t>(replacement) < expected_scores.size(),
                    "Metal GLM expert route is out of range");
            max_reference_gap = std::max(max_reference_gap,
                                         static_cast<double>(expected_scores[selected]) - expected_scores[replacement]);
        }
    }

    return max_reference_gap <= 2.0 * max_score_error + 1e-6;
}

static bool validate_route_probe(const route_capture & expected, const route_capture & actual) {
    bool divergence_seen = false;
    for (size_t layer = 3; layer < k_layer_count; ++layer) {
        require(!expected.top_k[layer].empty() && !actual.top_k[layer].empty(),
                "GLM route probe is missing expert indices");
        require(!expected.scores[layer].empty() && !actual.scores[layer].empty(),
                "GLM route probe is missing expert scores");
        if (same_indices(expected.top_k[layer], actual.top_k[layer])) {
            continue;
        }

        if (!divergence_seen) {
            double max_score_error   = 0.0;
            double max_reference_gap = 0.0;
            require(route_difference_is_near_tie(expected.top_k[layer], actual.top_k[layer], expected.scores[layer],
                                                 actual.scores[layer], max_score_error, max_reference_gap),
                    "first CPU/Metal GLM expert route difference is not explained by a score tie");
            std::printf("GLM route probe: first difference at layer %zu, score gap %.3e, max score error %.3e\n", layer,
                        max_reference_gap, max_score_error);
        }
        divergence_seen = true;
    }
    return divergence_seen;
}

static void prefill(llama_context * context, batch_guard & batch) {
    std::array<llama_token, 16> prompt{};
    std::iota(prompt.begin(), prompt.end(), 1);
    decode_tokens(context, batch, prompt.data(), prompt.size(), 0, nullptr, nullptr);
}

struct reference_trace {
    int32_t                  n_vocab = 0;
    std::vector<llama_token> greedy_tokens;
    std::vector<float>       logits;
    route_capture            routes;
};

static reference_trace run_reference(const char * model_path, uint32_t context_length) {
    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(cpu_device != nullptr, "CPU device unavailable for GLM long parity");
    llama_model_ptr   model = load_model(model_path, cpu_device);
    decode_observer   observer;
    llama_context_ptr context = make_context(model.get(), context_length, observer, false);
    enable_cpu_reference(context.get());
    batch_guard batch;
    prefill(context.get(), batch);

    reference_trace trace;
    trace.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    trace.greedy_tokens.reserve(k_decode_steps);
    trace.logits.reserve(static_cast<size_t>(trace.n_vocab) * k_decode_steps);
    llama_token  current = 17;
    op_signature first_signature;
    const auto   started = std::chrono::steady_clock::now();
    for (int step = 0; step < k_decode_steps; ++step) {
        observer.capture_routes = step == k_route_probe_step;
        op_signature  signature;
        const float * logits    = decode_tokens(context.get(), batch, &current, 1, 16 + step, &observer, &signature);
        observer.capture_routes = false;
        if (step == 0) {
            first_signature = signature;
            require(signature.lightning_indexer == 0 && signature.selected_k == 0 && signature.selected_v == 0 &&
                        signature.dense_mask == 7,
                    "CPU reference did not retain the generic dense GLM-DSA graph");
        } else {
            require(same_shape(first_signature, signature), "CPU reference graph shape changed during decode");
        }
        current = greedy_token(logits, trace.n_vocab);
        trace.greedy_tokens.push_back(current);
        trace.logits.insert(trace.logits.end(), logits, logits + trace.n_vocab);
        if ((step + 1) % 64 == 0) {
            std::printf("GLM CPU reference: ctx=%u step=%d/%d\n", context_length, step + 1, k_decode_steps);
            std::fflush(stdout);
        }
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started);
    std::printf("GLM CPU reference: ctx=%u %.3f ms/step\n", context_length, elapsed.count() / k_decode_steps);
    trace.routes = std::move(observer.routes);
    return trace;
}

static void validate_native_signature(const op_signature & signature, const decode_observer & observer) {
    if (!observer.non_metal_ops.empty()) {
        for (const std::string & op : observer.non_metal_ops) {
            std::fprintf(stderr, "non-Metal GLM long-parity op: %s\n", op.c_str());
        }
    }
    require(signature.lightning_indexer == 4, "Metal decode did not execute four Full indexers");
    require(signature.selected_k == 7 && signature.selected_v == 7,
            "Metal decode did not execute seven compact K/V gathers");
    require(signature.dense_mask == 0, "Metal decode materialized a dense sparse-attention mask");
    require(signature.flash_attention == 7, "Metal decode did not execute seven flash-attention operations");
    require(signature.rope > 0, "Metal decode did not execute RoPE");
    require(signature.heavy > 0 && signature.heavy == signature.heavy_metal,
            "Metal decode assigned a heavy GLM-DSA operation outside Metal");
}

static void run_metal(const char * model_path, uint32_t context_length, const reference_trace & reference) {
    ggml_backend_dev_t metal_device = ggml_backend_dev_by_name("MTL0");
    require(metal_device != nullptr, "Metal device unavailable for GLM long parity");
    llama_model_ptr   model = load_model(model_path, metal_device);
    decode_observer   observer;
    llama_context_ptr context = make_context(model.get(), context_length, observer, true);
    batch_guard       batch;
    prefill(context.get(), batch);

    require(reference.greedy_tokens.size() == k_decode_steps, "GLM reference token trace is incomplete");
    require(reference.logits.size() == static_cast<size_t>(reference.n_vocab) * k_decode_steps,
            "GLM reference logit trace is incomplete");
    op_signature     first_signature;
    double           max_nmse             = 0.0;
    int              max_nmse_step        = 0;
    int              first_bad_nmse_step  = -1;
    int              first_bad_token_step = -1;
    uint64_t         rss_baseline         = 0;
    uint64_t         rss_peak             = 0;
    nmse_accumulator aggregate_nmse;
    const auto       started = std::chrono::steady_clock::now();
    for (int step = 0; step < k_decode_steps; ++step) {
        const llama_token input = step == 0 ? 17 : reference.greedy_tokens[step - 1];
        observer.capture_routes = step == k_route_probe_step;
        op_signature  signature;
        const float * logits    = decode_tokens(context.get(), batch, &input, 1, 16 + step, &observer, &signature);
        observer.capture_routes = false;
        validate_native_signature(signature, observer);
        if (step == 0) {
            first_signature = signature;
        } else {
            require(same_shape(first_signature, signature), "Metal GLM-DSA graph shape changed during decode");
        }

        const float * expected_logits = reference.logits.data() + static_cast<size_t>(step) * reference.n_vocab;
        const double  nmse            = logit_nmse(expected_logits, logits, reference.n_vocab);
        aggregate_nmse.add(expected_logits, logits, reference.n_vocab);
        if (nmse > max_nmse) {
            max_nmse      = nmse;
            max_nmse_step = step;
        }
        const llama_token actual = greedy_token(logits, reference.n_vocab);
        if (actual != reference.greedy_tokens[step] && first_bad_token_step < 0) {
            first_bad_token_step = step;
            std::fprintf(stderr, "GLM greedy mismatch: ctx=%u step=%d CPU=%d Metal=%d NMSE=%.3e\n", context_length,
                         step, reference.greedy_tokens[step], actual, nmse);
        }
        if (nmse > 2e-4 && first_bad_nmse_step < 0) {
            first_bad_nmse_step = step;
            std::fprintf(stderr, "GLM logit NMSE crossed tolerance: ctx=%u step=%d NMSE=%.3e\n", context_length, step,
                         nmse);
        }
        if (step + 1 == k_warmup_steps) {
            llama_synchronize(context.get());
            rss_baseline = resident_bytes();
            rss_peak     = rss_baseline;
        } else if (step + 1 > k_warmup_steps && (step + 1) % 16 == 0) {
            llama_synchronize(context.get());
            rss_peak = std::max(rss_peak, resident_bytes());
        }
        if ((step + 1) % 64 == 0) {
            std::printf("GLM Metal parity: ctx=%u step=%d/%d max-NMSE=%.3e\n", context_length, step + 1, k_decode_steps,
                        max_nmse);
            std::fflush(stdout);
        }
    }
    llama_synchronize(context.get());
    const auto     elapsed    = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started);
    const uint64_t rss_growth = rss_peak > rss_baseline ? rss_peak - rss_baseline : 0;
    const double   aggregate_error = aggregate_nmse.value();
    const bool     route_tie       = validate_route_probe(reference.routes, observer.routes);
    require(rss_baseline == 0 || rss_growth <= 32ULL * 1024 * 1024, "Metal GLM context grew after long-parity warmup");
    std::printf(
        "GLM long parity: ctx=%u tokens=%d aggregate-NMSE=%.3e max-NMSE=%.3e@%d Metal=%.3f ms/step "
        "RSS-growth=%.1f MiB\n",
        context_length, k_decode_steps, aggregate_error, max_nmse, max_nmse_step, elapsed.count() / k_decode_steps,
        rss_growth / (1024.0 * 1024.0));
    require(first_bad_token_step < 0, "CPU and Metal greedy token traces differ");
    require(aggregate_error <= 2e-4, "CPU and Metal aggregate logit NMSE exceeds tolerance");
    require(first_bad_nmse_step < 0 || (first_bad_nmse_step == k_route_probe_step && route_tie && max_nmse <= 5e-3),
            "CPU and Metal logit divergence is not limited to the characterized expert-route tie");
}

}  // namespace

void test_glm_dsa_long_greedy_parity(const char * model_path) {
    const reference_trace reference = run_reference(model_path, 2048);
    for (uint32_t context_length : std::array<uint32_t, 3>{ 2048, 32768, 131072 }) {
        run_metal(model_path, context_length, reference);
    }
}
