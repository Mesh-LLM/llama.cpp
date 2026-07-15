#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama-cpp.h"
#include "llama.h"
#include "test-glm-dsa-greedy.h"
#include "test-glm-dsa-moe.h"
#include "test-glm-dsa-stability.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static ggml_context_ptr make_context() {
    ggml_init_params params = {
        /*.mem_size   =*/8 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    require(ctx != nullptr, "failed to initialize ggml context");
    return ctx;
}

static ggml_backend_ptr make_cpu_backend() {
    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    require(backend != nullptr, "failed to initialize CPU backend");
    return backend;
}

static void set_f32(ggml_tensor * tensor, const std::vector<float> & values) {
    require(ggml_nelements(tensor) == static_cast<int64_t>(values.size()), "wrong f32 fixture size");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

static void set_f16(ggml_tensor * tensor, const std::vector<float> & values) {
    require(ggml_nelements(tensor) == static_cast<int64_t>(values.size()), "wrong f16 fixture size");
    std::vector<ggml_fp16_t> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(), ggml_fp32_to_fp16);
    ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size() * sizeof(ggml_fp16_t));
}

static void set_i32(ggml_tensor * tensor, const std::vector<int32_t> & values) {
    require(ggml_nelements(tensor) == static_cast<int64_t>(values.size()), "wrong i32 fixture size");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(int32_t));
}

static std::vector<float> get_f32(const ggml_tensor * tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

static std::vector<float> get_float_values(const ggml_tensor * tensor) {
    const size_t       count = ggml_nelements(tensor);
    std::vector<float> values(count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        return values;
    }

    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data(count);
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
        std::transform(data.begin(), data.end(), values.begin(), ggml_fp16_to_fp32);
        return values;
    }

    if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> data(count);
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(ggml_bf16_t));
        std::transform(data.begin(), data.end(), values.begin(), ggml_bf16_to_fp32);
        return values;
    }

    throw std::runtime_error("unsupported captured activation type");
}

static std::vector<int32_t> get_i32_values(const ggml_tensor * tensor) {
    require(tensor->type == GGML_TYPE_I32, "captured index tensor is not i32");
    std::vector<int32_t> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(int32_t));
    return values;
}

static void check_close(float actual, float expected, float tolerance, const char * message) {
    if (std::isinf(expected)) {
        require(std::isinf(actual) && std::signbit(actual) == std::signbit(expected), message);
        return;
    }
    require(std::fabs(actual - expected) <= tolerance, message);
}

static std::vector<float> reference_indexer_scores(const std::vector<float> & q,
                                                   const std::vector<float> & k,
                                                   const std::vector<float> & weights,
                                                   const std::vector<float> & mask,
                                                   int64_t                    head_size,
                                                   int64_t                    n_head,
                                                   int64_t                    n_kv,
                                                   int64_t                    n_query) {
    std::vector<float> scores(n_kv * n_query);
    const float        scale = 1.0f / std::sqrt(static_cast<float>(head_size * n_head));

    // Matches GlmMoeDsaIndexer.forward: weighted head-wise relu(q dot k), scaling, then the causal mask.
    for (int64_t iq = 0; iq < n_query; ++iq) {
        for (int64_t ik = 0; ik < n_kv; ++ik) {
            float score = 0.0f;
            for (int64_t ih = 0; ih < n_head; ++ih) {
                float dot = 0.0f;
                for (int64_t id = 0; id < head_size; ++id) {
                    dot += q[(iq * n_head + ih) * head_size + id] * k[ik * head_size + id];
                }
                score += std::max(dot, 0.0f) * weights[iq * n_head + ih] * scale;
            }
            scores[iq * n_kv + ik] = score + mask[iq * n_kv + ik];
        }
    }
    return scores;
}

static void test_indexer_scores_and_top_k() {
    constexpr int64_t head_size = 4;
    constexpr int64_t n_head    = 2;
    constexpr int64_t n_kv      = 6;
    constexpr int64_t n_query   = 2;
    constexpr int64_t n_top_k   = 3;

    const std::vector<float> q = {
        1.0f, -1.0f, 0.5f, 2.0f, -0.5f, 1.0f, 1.5f, -1.0f, 0.25f, 2.0f, -1.0f, 0.5f, 1.0f, 0.5f, -0.75f, 1.25f,
    };
    const std::vector<float> k = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 0.5f, 1.0f, -0.5f,
    };
    const std::vector<float> weights = { 0.7f, -0.2f, -0.3f, 0.9f };
    const float              neg_inf = -std::numeric_limits<float>::infinity();
    const std::vector<float> mask    = {
        0.0f, 0.0f, 0.0f, neg_inf, neg_inf, neg_inf, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    const std::vector<float> official_scores = {
        0.247487373f, -0.0707106781f, 0.0176776695f, neg_inf,      neg_inf,      neg_inf,
        0.291681547f, -0.0530330086f, 0.0f,          0.344714556f, 0.450780573f, 0.0f,
    };
    const std::vector<int32_t> official_top_k = { 0, 2, 1, 4, 3, 0 };

    ggml_context_ptr ctx      = make_context();
    ggml_tensor *    q_tensor = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_size, n_head, n_query, 1);
    ggml_tensor *    k_tensor = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_size, 1, n_kv, 1);
    ggml_tensor *    w_tensor = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_head, n_query, 1, 1);
    ggml_tensor *    m_tensor = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv, n_query, 1, 1);

    const float   scale          = 1.0f / std::sqrt(static_cast<float>(head_size * n_head));
    ggml_tensor * scaled_weights = ggml_scale(ctx.get(), w_tensor, scale);
    ggml_tensor * scores         = ggml_lightning_indexer(ctx.get(), q_tensor, k_tensor, scaled_weights, m_tensor);
    ggml_tensor * top_k          = ggml_top_k(ctx.get(), scores, n_top_k);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, top_k);

    ggml_backend_ptr        backend = make_cpu_backend();
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    require(buffer != nullptr, "failed to allocate indexer tensors");

    set_f32(q_tensor, q);
    set_f32(k_tensor, k);
    set_f32(w_tensor, weights);
    std::vector<ggml_fp16_t> mask_f16(mask.size());
    std::transform(mask.begin(), mask.end(), mask_f16.begin(), ggml_fp32_to_fp16);
    ggml_backend_tensor_set(m_tensor, mask_f16.data(), 0, mask_f16.size() * sizeof(ggml_fp16_t));

    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS,
            "indexer graph computation failed");

    const std::vector<float> expected_scores =
        reference_indexer_scores(q, k, weights, mask, head_size, n_head, n_kv, n_query);
    const std::vector<float> actual_scores = get_f32(scores);
    for (size_t i = 0; i < expected_scores.size(); ++i) {
        check_close(expected_scores[i], official_scores[i], 1e-7f,
                    "C++ reference score differs from the Transformers golden");
        check_close(actual_scores[i], official_scores[i], 1e-6f, "indexer score differs from the Transformers golden");
    }

    std::vector<int32_t> actual_top_k(ggml_nelements(top_k));
    ggml_backend_tensor_get(top_k, actual_top_k.data(), 0, actual_top_k.size() * sizeof(int32_t));
    // ggml_top_k need not rank output; K, V, and mask rows are gathered in the same order.
    for (int64_t iq = 0; iq < n_query; ++iq) {
        auto actual_begin   = actual_top_k.begin() + iq * n_top_k;
        auto expected_begin = official_top_k.begin() + iq * n_top_k;
        std::sort(actual_begin, actual_begin + n_top_k);
        std::vector<int32_t> expected(expected_begin, expected_begin + n_top_k);
        std::sort(expected.begin(), expected.end());
        require(std::equal(actual_begin, actual_begin + n_top_k, expected.begin()),
                "indexer top-k selection differs from reference");
    }
}

static void run_generic_indexer_mask_cast(ggml_backend_t backend, int64_t n_kv, int64_t n_query) {
    ggml_context_ptr ctx          = make_context();
    ggml_tensor *    scores       = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_kv, n_query);
    ggml_tensor *    mask         = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, n_kv, n_query);
    ggml_tensor *    mask_f32     = ggml_cast(ctx.get(), mask, scores->type);
    ggml_tensor *    masked_score = ggml_add(ctx.get(), scores, mask_f32);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(graph, masked_score);

    require(mask_f32->type == GGML_TYPE_F32, "generic indexer mask was not normalized to score precision");
    require(masked_score->src[0]->type == GGML_TYPE_F32 && masked_score->src[1]->type == GGML_TYPE_F32,
            "generic indexer add retained mixed input precision");
    require(ggml_backend_supports_op(backend, mask_f32), "backend does not support the generic indexer mask cast");
    require(ggml_backend_supports_op(backend, masked_score), "backend does not support the generic indexer mask add");

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    require(buffer != nullptr, "failed to allocate generic indexer mask tensors");

    const size_t             count = static_cast<size_t>(n_kv * n_query);
    std::vector<float>       scores_data(count);
    std::vector<float>       mask_data(count);
    std::vector<ggml_fp16_t> mask_f16(count);
    for (size_t i = 0; i < count; ++i) {
        scores_data[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f;
        mask_data[i]   = i % 7 == 0 ? -std::numeric_limits<float>::infinity() : 0.0f;
        mask_f16[i]    = ggml_fp32_to_fp16(mask_data[i]);
    }
    ggml_backend_tensor_set(scores, scores_data.data(), 0, scores_data.size() * sizeof(float));
    ggml_backend_tensor_set(mask, mask_f16.data(), 0, mask_f16.size() * sizeof(ggml_fp16_t));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "generic indexer mask graph computation failed");
    const std::vector<float> actual = get_f32(masked_score);
    for (size_t i = 0; i < count; ++i) {
        const float expected = scores_data[i] + mask_data[i];
        check_close(actual[i], expected, 0.0f, "generic indexer mask result differs");
    }
}

static void test_generic_indexer_mask_cast() {
    ggml_backend_ptr cpu = make_cpu_backend();
    run_generic_indexer_mask_cast(cpu.get(), 256, 19);
    run_generic_indexer_mask_cast(cpu.get(), 256, 2304);

    if (ggml_backend_dev_t metal_device = ggml_backend_dev_by_name("MTL0")) {
        ggml_backend_ptr metal(ggml_backend_dev_init(metal_device, nullptr));
        require(metal != nullptr, "failed to initialize Metal for generic indexer mask regression");
        run_generic_indexer_mask_cast(metal.get(), 256, 19);
        run_generic_indexer_mask_cast(metal.get(), 256, 2304);
    }
}

static void test_interleaved_rope() {
    constexpr int64_t                   n_dims    = 4;
    constexpr int64_t                   n_tokens  = 2;
    constexpr float                     freq_base = 10000.0f;
    const std::vector<float>            input     = { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.0f, -0.25f };
    const std::array<int32_t, n_tokens> positions = { 1, 3 };

    ggml_context_ptr ctx             = make_context();
    ggml_tensor *    input_tensor    = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_dims, 1, n_tokens);
    ggml_tensor *    position_tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_tokens);
    ggml_tensor *    output = ggml_rope_ext(ctx.get(), input_tensor, position_tensor, nullptr, n_dims,
                                            GGML_ROPE_TYPE_NORMAL, 128, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    ggml_backend_ptr        backend = make_cpu_backend();
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    require(buffer != nullptr, "failed to allocate RoPE tensors");
    set_f32(input_tensor, input);
    ggml_backend_tensor_set(position_tensor, positions.data(), 0, sizeof(positions));
    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "RoPE graph computation failed");

    const std::vector<float> actual = get_f32(output);
    for (int64_t it = 0; it < n_tokens; ++it) {
        for (int64_t ip = 0; ip < n_dims / 2; ++ip) {
            const float theta     = positions[it] * std::pow(freq_base, -2.0f * ip / n_dims);
            const float x0        = input[it * n_dims + 2 * ip];
            const float x1        = input[it * n_dims + 2 * ip + 1];
            const float expected0 = x0 * std::cos(theta) - x1 * std::sin(theta);
            const float expected1 = x0 * std::sin(theta) + x1 * std::cos(theta);
            check_close(actual[it * n_dims + 2 * ip], expected0, 1e-5f,
                        "GGML_ROPE_TYPE_NORMAL even component differs from reference");
            check_close(actual[it * n_dims + 2 * ip + 1], expected1, 1e-5f,
                        "GGML_ROPE_TYPE_NORMAL odd component differs from reference");
        }
    }
}

struct sparse_attention_outputs {
    std::vector<float> dense;
    std::vector<float> compact;
};

static std::vector<float> fixture_values(size_t count, int period) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(static_cast<int>(i % period) - period / 2) / period;
    }
    return values;
}

static sparse_attention_outputs run_sparse_attention(ggml_backend_t backend, bool require_native_support) {
    constexpr int64_t head_size_k = 576;
    constexpr int64_t head_size_v = 512;
    constexpr int64_t n_head      = 8;
    constexpr int64_t n_kv        = 512;
    constexpr int64_t n_top_k     = 8;
    constexpr int64_t n_stream    = 2;

    ggml_context_ptr ctx         = make_context();
    ggml_tensor *    q           = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_size_k, 1, n_head, n_stream);
    ggml_tensor *    k_cache     = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, head_size_k, 1, n_kv, n_stream);
    ggml_tensor *    v_cache     = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, head_size_v, 1, n_kv, n_stream);
    ggml_tensor *    causal_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv, 1, 1, n_stream);
    ggml_tensor *    dense_mask  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv, 1, 1, n_stream);
    ggml_tensor *    top_k       = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_I32, n_top_k, 1, 1, n_stream);

    ggml_tensor * k_full = ggml_permute(ctx.get(), k_cache, 0, 2, 1, 3);
    ggml_tensor * v_full = ggml_permute(ctx.get(), v_cache, 0, 2, 1, 3);
    ggml_tensor * dense  = ggml_flash_attn_ext(ctx.get(), q, k_full, v_full, dense_mask,
                                               1.0f / std::sqrt(static_cast<float>(head_size_k)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(dense, GGML_PREC_F32);

    ggml_tensor * top_k_rows = ggml_reshape_3d(ctx.get(), top_k, n_top_k, 1, n_stream);
    ggml_tensor * k_selected = ggml_get_rows(ctx.get(), k_full, top_k_rows);
    ggml_tensor * v_selected = ggml_get_rows(ctx.get(), v_full, top_k_rows);
    require(k_selected->ne[1] == n_top_k && v_selected->ne[1] == n_top_k,
            "compact attention did not reduce K/V rows to top-k");
    require(k_full->ne[1] == 64 * k_selected->ne[1] && v_full->ne[1] == 64 * v_selected->ne[1],
            "compact attention fixture did not exercise 64x selected-row scaling");

    k_selected                  = ggml_cast(ctx.get(), k_selected, GGML_TYPE_F16);
    v_selected                  = ggml_cast(ctx.get(), v_selected, GGML_TYPE_F16);
    ggml_tensor * mask_rows     = ggml_reshape_4d(ctx.get(), causal_mask, 1, n_kv, 1, n_stream);
    ggml_tensor * mask_selected = ggml_get_rows(ctx.get(), mask_rows, top_k_rows);
    mask_selected               = ggml_reshape_4d(ctx.get(), mask_selected, n_top_k, 1, 1, n_stream);
    mask_selected               = ggml_cast(ctx.get(), mask_selected, GGML_TYPE_F16);

    ggml_tensor * compact = ggml_flash_attn_ext(ctx.get(), q, k_selected, v_selected, mask_selected,
                                                1.0f / std::sqrt(static_cast<float>(head_size_k)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(compact, GGML_PREC_F32);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, dense);
    ggml_build_forward_expand(graph, compact);

    if (require_native_support) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            require(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)),
                    "compact sparse attention graph contains an op unsupported by Metal");
        }
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    require(buffer != nullptr, "failed to allocate sparse attention tensors");

    std::vector<int32_t> indices(n_top_k * n_stream);
    for (int64_t is = 0; is < n_stream; ++is) {
        for (int64_t ik = 0; ik < n_top_k; ++ik) {
            indices[is * n_top_k + ik] = is * 31 + ik * (n_kv / n_top_k);
        }
    }

    std::vector<float> causal_values(n_kv * n_stream, 0.0f);
    std::vector<float> dense_values(n_kv * n_stream, -std::numeric_limits<float>::infinity());
    for (int64_t is = 0; is < n_stream; ++is) {
        for (int64_t ik = 0; ik < n_top_k; ++ik) {
            dense_values[is * n_kv + indices[is * n_top_k + ik]] = 0.0f;
        }
    }

    set_f32(q, fixture_values(ggml_nelements(q), 29));
    set_f16(k_cache, fixture_values(ggml_nelements(k_cache), 31));
    set_f16(v_cache, fixture_values(ggml_nelements(v_cache), 37));
    set_f16(causal_mask, causal_values);
    set_f16(dense_mask, dense_values);
    set_i32(top_k, indices);

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "sparse attention graph computation failed");

    return { get_f32(dense), get_f32(compact) };
}

static void test_compact_sparse_attention() {
    ggml_backend_ptr               cpu         = make_cpu_backend();
    const sparse_attention_outputs cpu_outputs = run_sparse_attention(cpu.get(), false);
    require(cpu_outputs.dense.size() == cpu_outputs.compact.size(), "sparse attention output size differs");
    for (size_t i = 0; i < cpu_outputs.dense.size(); ++i) {
        check_close(cpu_outputs.compact[i], cpu_outputs.dense[i], 1e-5f,
                    "compact sparse attention differs from dense reference");
    }

    ggml_backend_dev_t metal_device = ggml_backend_dev_by_name("MTL0");
    if (!metal_device) {
        return;
    }

    ggml_backend_ptr metal(ggml_backend_dev_init(metal_device, nullptr));
    require(metal != nullptr, "failed to initialize Metal backend");
    const sparse_attention_outputs metal_outputs = run_sparse_attention(metal.get(), true);
    for (size_t i = 0; i < cpu_outputs.compact.size(); ++i) {
        check_close(metal_outputs.compact[i], cpu_outputs.compact[i], 5e-4f,
                    "Metal compact sparse attention differs from CPU");
    }
}

static constexpr size_t real_layer_count = 7;

struct real_step_capture {
    std::array<std::vector<int32_t>, real_layer_count> top_k;
    std::array<std::vector<float>, real_layer_count>   indexer_scores;
    std::array<std::vector<float>, real_layer_count>   hidden_states;
};

struct real_decode_observer {
    int     lightning_indexer_count = 0;
    int     k_selected_count        = 0;
    int     v_selected_count        = 0;
    int     dense_sparse_mask_count = 0;
    int     rope_count              = 0;
    int     flash_attention_count   = 0;
    int     heavy_op_count          = 0;
    int     heavy_op_metal_count    = 0;
    int64_t selected_rows           = 0;

    llama_context * context          = nullptr;
    bool            capture_values   = false;
    size_t          next_dense_layer = 0;

    std::vector<std::string>                        non_metal_heavy_ops;
    std::vector<real_step_capture>                  steps;
    std::unordered_map<const ggml_tensor *, size_t> dense_layer_by_tensor;

    void reset() {
        lightning_indexer_count = 0;
        k_selected_count        = 0;
        v_selected_count        = 0;
        dense_sparse_mask_count = 0;
        rope_count              = 0;
        flash_attention_count   = 0;
        heavy_op_count          = 0;
        heavy_op_metal_count    = 0;
        selected_rows           = 0;
        capture_values          = false;
        next_dense_layer        = 0;
        non_metal_heavy_ops.clear();
        steps.clear();
        dense_layer_by_tensor.clear();
    }

    void begin_decode_step() {
        capture_values   = true;
        next_dense_layer = 0;
        dense_layer_by_tensor.clear();
        steps.emplace_back();
    }
};

static int tensor_layer(const ggml_tensor * tensor, const char * prefix) {
    const size_t prefix_length = std::strlen(prefix);
    if (std::strncmp(tensor->name, prefix, prefix_length) != 0 || tensor->name[prefix_length] != '-') {
        return -1;
    }

    char *     end = nullptr;
    const long il  = std::strtol(tensor->name + prefix_length + 1, &end, 10);
    if (end == tensor->name + prefix_length + 1 || *end != '\0' || il < 0 || il >= (long) real_layer_count) {
        return -1;
    }
    return il;
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

static bool is_dense_sparse_mask(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_SET_ROWS && tensor->ne[0] == 1 && tensor->ne[1] > 1;
}

static bool is_k_selected(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_GET_ROWS && std::strstr(tensor->name, "k_selected");
}

static bool is_v_selected(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_GET_ROWS && std::strstr(tensor->name, "v_selected");
}

static bool is_heavy_glm_op(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_LIGHTNING_INDEXER || tensor->op == GGML_OP_ROPE ||
           tensor->op == GGML_OP_FLASH_ATTN_EXT || is_k_selected(tensor) || is_v_selected(tensor);
}

static bool is_final_indexer_score(const ggml_tensor * tensor) {
    return tensor_layer(tensor, "indexer_score") >= 0 &&
           (tensor->op == GGML_OP_LIGHTNING_INDEXER || tensor->op == GGML_OP_ADD);
}

static void observe_heavy_backend(real_decode_observer & observer, ggml_tensor * tensor) {
    if (!observer.context || !is_heavy_glm_op(tensor)) {
        return;
    }

    ++observer.heavy_op_count;
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(observer.context->get_sched(), tensor);
    if (is_metal_backend(backend)) {
        ++observer.heavy_op_metal_count;
        return;
    }

    const char * backend_name = backend ? ggml_backend_name(backend) : "unassigned";
    observer.non_metal_heavy_ops.emplace_back(std::string(tensor->name) + " (" + backend_name + ")");
}

static bool should_capture_real_tensor(const ggml_tensor * tensor) {
    return tensor_layer(tensor, "l_out") >= 0 || is_final_indexer_score(tensor) || is_k_selected(tensor) ||
           is_dense_sparse_mask(tensor);
}

static void capture_real_tensor(real_decode_observer & observer, ggml_tensor * tensor) {
    require(!observer.steps.empty(), "real GLM tensor capture has no active decode step");
    real_step_capture & capture = observer.steps.back();

    if (const int il = tensor_layer(tensor, "l_out"); il >= 0) {
        capture.hidden_states[il] = get_float_values(tensor);
        return;
    }

    if (const int il = tensor_layer(tensor, "indexer_score"); il >= 0 && is_final_indexer_score(tensor)) {
        capture.indexer_scores[il] = get_float_values(tensor);
        return;
    }

    if (is_k_selected(tensor)) {
        const int il = tensor_layer(tensor, "k_selected");
        require(il >= 0 && tensor->src[1] != nullptr, "selected K tensor has no layer or index source");
        capture.top_k[il] = get_i32_values(tensor->src[1]);
        return;
    }

    const auto dense_layer = observer.dense_layer_by_tensor.find(tensor);
    require(dense_layer != observer.dense_layer_by_tensor.end(), "dense sparse mask has no layer mapping");
    require(tensor->src[1] != nullptr, "dense sparse mask has no index source");
    capture.top_k[dense_layer->second] = get_i32_values(tensor->src[1]);
}

static bool observe_real_decode(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * observer = static_cast<real_decode_observer *>(user_data);
    if (ask) {
        if (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
            ++observer->lightning_indexer_count;
        } else if (is_k_selected(tensor)) {
            ++observer->k_selected_count;
            observer->selected_rows = tensor->ne[1];
        } else if (is_v_selected(tensor)) {
            ++observer->v_selected_count;
        } else if (is_dense_sparse_mask(tensor)) {
            ++observer->dense_sparse_mask_count;
            if (observer->capture_values) {
                require(observer->next_dense_layer < real_layer_count, "too many dense sparse masks in one decode");
                observer->dense_layer_by_tensor[tensor] = observer->next_dense_layer++;
            }
        }

        if (tensor->op == GGML_OP_ROPE) {
            ++observer->rope_count;
        } else if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            ++observer->flash_attention_count;
        }
        observe_heavy_backend(*observer, tensor);
        return observer->capture_values && should_capture_real_tensor(tensor);
    }

    capture_real_tensor(*observer, tensor);
    return true;
}

static bool silent_model_load_progress(float, void *) {
    return true;
}

struct real_execution_config {
    const char * name;
    bool         fused_indexer;
    bool         compact_decode;
};

static void append_real_logits(llama_context *                  context,
                               const llama_model *              model,
                               const std::vector<llama_token> & tokens,
                               int32_t                          position,
                               std::vector<float> &             logits,
                               real_decode_observer *           observer,
                               bool                             capture_values) {
    if (capture_values) {
        require(observer != nullptr, "real GLM decode capture requires an observer");
        observer->begin_decode_step();
    }

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], position + i, { 0 }, i + 1 == tokens.size());
    }

    require(llama_decode(context, batch) == 0, "real GLM layer decode failed");
    const float * batch_logits = llama_get_logits_ith(context, batch.n_tokens - 1);
    require(batch_logits != nullptr, "real GLM layer produced no logits");
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    logits.insert(logits.end(), batch_logits, batch_logits + n_vocab);
    llama_batch_free(batch);
}

struct real_model_result {
    std::vector<float>   logits;
    real_decode_observer prefill_observer;
    real_decode_observer decode_observer;
};

static real_model_result run_real_model(const char *                  model_path,
                                        ggml_backend_dev_t            device,
                                        const real_execution_config & config) {
    std::array<ggml_backend_dev_t, 2> devices      = { device, nullptr };
    llama_model_params                model_params = llama_model_default_params();
    model_params.devices                           = devices.data();
    model_params.n_gpu_layers                      = -1;
    model_params.split_mode                        = LLAMA_SPLIT_MODE_NONE;
    model_params.progress_callback                 = silent_model_load_progress;

    llama_model_ptr model(llama_model_load_from_file(model_path, model_params));
    require(model != nullptr, "failed to load real GLM layer fixture");

    real_decode_observer observer;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx                = 32;
    context_params.n_batch              = 32;
    context_params.n_ubatch             = 32;
    context_params.n_threads            = 8;
    context_params.n_threads_batch      = 8;
    context_params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.cb_eval              = observe_real_decode;
    context_params.cb_eval_user_data    = &observer;

    llama_context_ptr context(llama_init_from_model(model.get(), context_params));
    require(context != nullptr, "failed to create real GLM layer context");

    const char * device_name = ggml_backend_dev_name(device);
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU ||
        (device_name && std::strcmp(device_name, "BLAS") == 0)) {
        ggml_backend_sched_t scheduler = context->get_sched();
        for (int i = 0; i < ggml_backend_sched_get_n_backends(scheduler); ++i) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(scheduler, i);
            if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }

            using set_use_ref_fn           = void (*)(ggml_backend_t, bool);
            ggml_backend_reg_t reg         = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            auto               set_use_ref = reinterpret_cast<set_use_ref_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref"));
            require(set_use_ref != nullptr, "CPU backend has no reference-mode control");
            set_use_ref(backend, true);
        }
    }

    // Test-local controls let one loaded model exercise reference and native graph shapes.
    llama_cparams & test_cparams = const_cast<llama_cparams &>(context->get_cparams());
    test_cparams.fused_lid       = config.fused_indexer;
    test_cparams.auto_flid       = false;
    test_cparams.flash_attn      = config.compact_decode;
    test_cparams.auto_fa         = false;

    // Context initialization probes fused operations. Reset before inference.
    observer.context = context.get();
    observer.reset();

    std::vector<llama_token> tokens(18);
    std::iota(tokens.begin(), tokens.end(), 1);
    std::vector<float> logits;
    append_real_logits(context.get(), model.get(), std::vector<llama_token>(tokens.begin(), tokens.begin() + 16), 0,
                       logits, &observer, false);
    real_decode_observer prefill_observer = observer;
    prefill_observer.context              = nullptr;
    prefill_observer.dense_layer_by_tensor.clear();
    require(prefill_observer.k_selected_count == 0 && prefill_observer.v_selected_count == 0,
            "real GLM prefill unexpectedly used compact selected K/V rows");
    require(prefill_observer.dense_sparse_mask_count > 0,
            "real GLM prefill did not retain the dense sparse-attention fallback");

    observer.reset();
    append_real_logits(context.get(), model.get(), { tokens[16] }, 16, logits, &observer, true);
    const real_decode_observer first_decode_observer = observer;
    if (config.compact_decode) {
        require(first_decode_observer.k_selected_count == (int) real_layer_count &&
                    first_decode_observer.v_selected_count == (int) real_layer_count,
                "first real GLM decode did not use one compact selected K/V path per layer");
        require(first_decode_observer.selected_rows == 8, "first real GLM decode did not preserve the fixture top-k");
        require(first_decode_observer.dense_sparse_mask_count == 0,
                "first real GLM decode materialized a dense sparse-attention mask");
    } else {
        require(first_decode_observer.k_selected_count == 0 && first_decode_observer.v_selected_count == 0,
                "dense reference decode unexpectedly selected compact K/V rows");
        require(first_decode_observer.dense_sparse_mask_count == (int) real_layer_count,
                "dense reference decode did not materialize one sparse mask per layer");
    }

    append_real_logits(context.get(), model.get(), { tokens[17] }, 17, logits, &observer, true);
    require(observer.k_selected_count == 2 * first_decode_observer.k_selected_count &&
                observer.v_selected_count == 2 * first_decode_observer.v_selected_count,
            "repeated real GLM decode changed selected K/V graph shape");
    require(observer.lightning_indexer_count == 2 * first_decode_observer.lightning_indexer_count,
            "repeated real GLM decode changed Lightning Indexer placement");
    require(observer.dense_sparse_mask_count == 2 * first_decode_observer.dense_sparse_mask_count,
            "repeated real GLM decode changed dense sparse-mask graph shape");
    if (config.compact_decode) {
        require(observer.selected_rows == first_decode_observer.selected_rows,
                "repeated real GLM decode changed the selected-row count");
    }

    observer.context = nullptr;
    observer.dense_layer_by_tensor.clear();
    std::printf("real GLM mode %s: LID=%d selected-K/V=%d/%d dense-mask=%d Metal-heavy=%d/%d\n", config.name,
                observer.lightning_indexer_count, observer.k_selected_count, observer.v_selected_count,
                observer.dense_sparse_mask_count, observer.heavy_op_metal_count, observer.heavy_op_count);
    return { std::move(logits), std::move(prefill_observer), std::move(observer) };
}

static double normalized_mean_squared_error(const std::vector<float> & expected, const std::vector<float> & actual) {
    require(expected.size() == actual.size(), "real GLM logit vector sizes differ");
    double squared_error     = 0.0;
    double squared_reference = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
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

static double normalized_mean_squared_error(const std::vector<float> & expected,
                                            const std::vector<float> & actual,
                                            size_t                     begin,
                                            size_t                     end) {
    double squared_error     = 0.0;
    double squared_reference = 0.0;
    for (size_t i = begin; i < end; ++i) {
        const double difference = static_cast<double>(expected[i]) - actual[i];
        squared_error += difference * difference;
        squared_reference += static_cast<double>(expected[i]) * expected[i];
    }
    return squared_reference == 0.0 ? squared_error : squared_error / squared_reference;
}

static std::vector<int32_t> sorted_indices(std::vector<int32_t> values) {
    std::sort(values.begin(), values.end());
    return values;
}

static size_t full_indexer_source(size_t layer) {
    if (layer >= 3 && layer <= 5) {
        return 2;
    }
    return layer;
}

static void validate_real_captures(const real_model_result & result, const char * mode_name) {
    require(result.decode_observer.steps.size() == 2, "real GLM capture did not contain two decode steps");
    for (size_t step = 0; step < result.decode_observer.steps.size(); ++step) {
        const real_step_capture & capture = result.decode_observer.steps[step];
        for (size_t layer = 0; layer < real_layer_count; ++layer) {
            require(!capture.top_k[layer].empty(), "real GLM capture is missing a top-k set");
            require(!capture.hidden_states[layer].empty(), "real GLM capture is missing a hidden state");
        }
        for (const size_t full_layer : { 0U, 1U, 2U, 6U }) {
            require(!capture.indexer_scores[full_layer].empty(), "real GLM capture is missing Full indexer scores");
        }

        const std::vector<int32_t> shared = sorted_indices(capture.top_k[2]);
        for (size_t layer = 3; layer <= 5; ++layer) {
            require(sorted_indices(capture.top_k[layer]) == shared,
                    "Shared GLM-DSA layer did not reuse the preceding Full top-k set");
        }
    }
    std::printf("real GLM mode %s captured top-k, Full scores, and seven hidden states for two decode steps\n",
                mode_name);
}

static bool top_k_difference_is_score_tie(const real_step_capture &    reference,
                                          size_t                       layer,
                                          const std::vector<int32_t> & expected,
                                          const std::vector<int32_t> & actual) {
    const size_t source_layer = full_indexer_source(layer);
    const auto & scores       = reference.indexer_scores[source_layer];
    if (scores.empty() || expected.empty()) {
        return false;
    }

    float boundary = std::numeric_limits<float>::infinity();
    for (int32_t index : expected) {
        if (index < 0 || (size_t) index >= scores.size()) {
            return false;
        }
        boundary = std::min(boundary, scores[index]);
    }

    const float tolerance = 2e-5f * std::max(1.0f, std::fabs(boundary));
    for (int32_t index : expected) {
        if (!std::binary_search(actual.begin(), actual.end(), index) &&
            std::fabs(scores[index] - boundary) > tolerance) {
            return false;
        }
    }
    for (int32_t index : actual) {
        if (index < 0 || (size_t) index >= scores.size()) {
            return false;
        }
        if (!std::binary_search(expected.begin(), expected.end(), index) &&
            std::fabs(scores[index] - boundary) > tolerance) {
            return false;
        }
    }
    return true;
}

static void compare_real_captures(const real_model_result & reference,
                                  const real_model_result & actual,
                                  const char *              mode_name,
                                  double                    tolerance) {
    require(reference.decode_observer.steps.size() == actual.decode_observer.steps.size(),
            "real GLM capture step counts differ");

    double max_hidden_nmse  = 0.0;
    double max_score_nmse   = 0.0;
    size_t tie_count        = 0;
    size_t max_hidden_step  = 0;
    size_t max_hidden_layer = 0;
    size_t max_score_step   = 0;
    size_t max_score_layer  = 0;
    for (size_t step = 0; step < reference.decode_observer.steps.size(); ++step) {
        const auto & expected_step = reference.decode_observer.steps[step];
        const auto & actual_step   = actual.decode_observer.steps[step];
        for (size_t layer = 0; layer < real_layer_count; ++layer) {
            const std::vector<int32_t> expected_top_k = sorted_indices(expected_step.top_k[layer]);
            const std::vector<int32_t> actual_top_k   = sorted_indices(actual_step.top_k[layer]);
            if (expected_top_k != actual_top_k) {
                require(top_k_difference_is_score_tie(expected_step, layer, expected_top_k, actual_top_k),
                        "real GLM top-k sets differ without a boundary-score tie");
                ++tie_count;
            }

            const double hidden_nmse =
                normalized_mean_squared_error(expected_step.hidden_states[layer], actual_step.hidden_states[layer]);
            if (hidden_nmse > max_hidden_nmse) {
                max_hidden_nmse  = hidden_nmse;
                max_hidden_step  = step;
                max_hidden_layer = layer;
            }

            if (!expected_step.indexer_scores[layer].empty()) {
                require(!actual_step.indexer_scores[layer].empty(), "real GLM Full indexer score capture differs");
                const double score_nmse = normalized_mean_squared_error(expected_step.indexer_scores[layer],
                                                                        actual_step.indexer_scores[layer]);
                if (score_nmse > max_score_nmse) {
                    max_score_nmse  = score_nmse;
                    max_score_step  = step;
                    max_score_layer = layer;
                }
            }
        }
    }

    std::printf(
        "real GLM mode %s: max hidden NMSE %.3e (step %zu layer %zu), "
        "max indexer-score NMSE %.3e (step %zu layer %zu), top-k ties %zu\n",
        mode_name, max_hidden_nmse, max_hidden_step, max_hidden_layer, max_score_nmse, max_score_step, max_score_layer,
        tie_count);
    require(max_hidden_nmse <= tolerance, "real GLM hidden-state NMSE exceeds tolerance");
    require(max_score_nmse <= tolerance, "real GLM indexer-score NMSE exceeds tolerance");
}

static void require_native_metal_residency(const real_model_result & result) {
    const real_decode_observer & observer = result.decode_observer;
    if (!observer.non_metal_heavy_ops.empty()) {
        for (const std::string & op : observer.non_metal_heavy_ops) {
            std::fprintf(stderr, "non-Metal GLM-DSA heavy op: %s\n", op.c_str());
        }
    }
    require(observer.heavy_op_count > 0, "native Metal decode observed no heavy GLM-DSA operations");
    require(observer.heavy_op_count == observer.heavy_op_metal_count,
            "native Metal decode assigned a heavy GLM-DSA operation outside Metal");
    require(observer.lightning_indexer_count == 8,
            "native Metal decode did not execute four Full Lightning Indexers per step");
    require(observer.k_selected_count == 14 && observer.v_selected_count == 14,
            "native Metal decode did not execute seven compact K/V gathers per step");
    require(observer.flash_attention_count == 14,
            "native Metal decode did not execute seven flash-attention operations per step");
    require(observer.rope_count > 0, "native Metal decode did not execute RoPE operations");
}

static void compare_real_logits(const real_model_result & expected,
                                const real_model_result & actual,
                                const char *              device_name,
                                double                    tolerance) {
    const double error = normalized_mean_squared_error(expected.logits, actual.logits);
    std::printf("real GLM layer %s logit NMSE: %.3e\n", device_name, error);

    constexpr size_t n_steps = 3;
    require(expected.logits.size() % n_steps == 0, "real GLM logit fixture has an unexpected step count");
    const size_t n_vocab        = expected.logits.size() / n_steps;
    double       max_step_error = 0.0;
    bool         top_1_matches  = true;
    for (size_t step = 0; step < n_steps; ++step) {
        const size_t begin        = step * n_vocab;
        const size_t end          = begin + n_vocab;
        const double step_error   = normalized_mean_squared_error(expected.logits, actual.logits, begin, end);
        const auto   expected_top = std::max_element(expected.logits.begin() + begin, expected.logits.begin() + end);
        const auto   actual_top   = std::max_element(actual.logits.begin() + begin, actual.logits.begin() + end);
        std::printf("  step %zu: NMSE %.3e, top-1 %zu/%zu\n", step, step_error,
                    static_cast<size_t>(expected_top - expected.logits.begin()) - begin,
                    static_cast<size_t>(actual_top - actual.logits.begin()) - begin);
        max_step_error = std::max(max_step_error, step_error);
        top_1_matches &= expected_top - expected.logits.begin() == actual_top - actual.logits.begin();
    }
    require(error <= tolerance && max_step_error <= tolerance, "real GLM layer logits differ from CPU reference");
    require(top_1_matches, "real GLM layer top-1 logit differs from CPU reference");
}

static void test_real_layer_logits(const char * model_path) {
    static constexpr real_execution_config reference_config = {
        "CPU generic-indexer+dense-mask",
        false,
        false,
    };
    static constexpr real_execution_config metal_reference_config = {
        "Metal generic-indexer+dense-mask",
        false,
        false,
    };
    static constexpr real_execution_config metal_fused_dense_config = {
        "Metal fused-indexer+dense-mask",
        true,
        false,
    };
    static constexpr real_execution_config metal_generic_compact_config = {
        "Metal generic-indexer+compact-flash",
        false,
        true,
    };
    static constexpr real_execution_config metal_native_config = {
        "Metal fused-indexer+compact-flash",
        true,
        true,
    };

    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_name("CPU");
    require(cpu_device != nullptr, "CPU backend device is unavailable");
    const real_model_result cpu = run_real_model(model_path, cpu_device, reference_config);
    validate_real_captures(cpu, reference_config.name);
    require(cpu.decode_observer.lightning_indexer_count == 0,
            "CPU reference unexpectedly retained the fused Lightning Indexer");
    require(cpu.decode_observer.dense_sparse_mask_count == 14,
            "CPU reference did not build seven dense sparse masks per decode step");

    if (ggml_backend_dev_t blas_device = ggml_backend_dev_by_name("BLAS")) {
        const real_model_result blas = run_real_model(model_path, blas_device, reference_config);
        validate_real_captures(blas, "BLAS generic-indexer+dense-mask");
        compare_real_logits(cpu, blas, "BLAS fallback", 1e-7);
        compare_real_captures(cpu, blas, "BLAS fallback", 1e-7);
    }

    if (ggml_backend_dev_t metal_device = ggml_backend_dev_by_name("MTL0")) {
        const real_model_result metal_reference = run_real_model(model_path, metal_device, metal_reference_config);
        validate_real_captures(metal_reference, metal_reference_config.name);

        const real_model_result metal_fused_dense = run_real_model(model_path, metal_device, metal_fused_dense_config);
        validate_real_captures(metal_fused_dense, metal_fused_dense_config.name);
        compare_real_logits(metal_reference, metal_fused_dense, "Metal fused-indexer A/B", 2e-4);
        compare_real_captures(metal_reference, metal_fused_dense, "Metal fused-indexer A/B", 2e-4);

        const real_model_result metal_generic_compact =
            run_real_model(model_path, metal_device, metal_generic_compact_config);
        validate_real_captures(metal_generic_compact, metal_generic_compact_config.name);
        compare_real_logits(metal_reference, metal_generic_compact, "Metal compact-flash A/B", 2e-4);
        compare_real_captures(metal_reference, metal_generic_compact, "Metal compact-flash A/B", 2e-4);

        const real_model_result metal_native = run_real_model(model_path, metal_device, metal_native_config);
        validate_real_captures(metal_native, metal_native_config.name);
        compare_real_logits(metal_generic_compact, metal_native, "Metal fused-indexer compact A/B", 2e-4);
        compare_real_captures(metal_generic_compact, metal_native, "Metal fused-indexer compact A/B", 2e-4);
        require_native_metal_residency(metal_native);

        compare_real_captures(cpu, metal_reference, metal_reference_config.name, 2e-4);
        compare_real_logits(cpu, metal_reference, metal_reference_config.name, 2e-4);
        compare_real_captures(cpu, metal_native, "Metal native", 2e-4);
        compare_real_logits(cpu, metal_native, "Metal native", 2e-4);
    }
}

int main(int argc, char ** argv) {
    llama_backend_init();
    try {
        ggml_backend_load_all();
        test_indexer_scores_and_top_k();
        test_generic_indexer_mask_cast();
        test_interleaved_rope();
        test_compact_sparse_attention();
        test_glm_dsa_moe_precision();
        if (argc == 2 && std::strcmp(argv[1], "--stability") == 0) {
            test_glm_dsa_decode_stability();
        } else if (argc == 3 && std::strcmp(argv[1], "--long-parity") == 0) {
            test_glm_dsa_long_greedy_parity(argv[2]);
        } else if (argc == 3 && std::strcmp(argv[1], "--real-model") == 0) {
            test_real_layer_logits(argv[2]);
        } else if (argc != 1) {
            throw std::runtime_error(
                "usage: test-glm-dsa [--stability | --long-parity model.gguf | --real-model model.gguf]");
        }
        std::printf("GLM-DSA reference tests passed\n");
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GLM-DSA reference test failed: %s\n", error.what());
        llama_backend_free();
        return 1;
    }
}
