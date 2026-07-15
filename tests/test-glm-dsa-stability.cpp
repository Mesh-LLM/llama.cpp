#include "test-glm-dsa-stability.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#if defined(__APPLE__)
#    include <mach/mach.h>
#endif

namespace {

constexpr int64_t k_head_size_k       = 576;
constexpr int64_t k_head_size_v       = 512;
constexpr int64_t k_attention_heads   = 64;
constexpr int64_t k_indexer_head_size = 128;
constexpr int64_t k_indexer_heads     = 32;
constexpr int64_t k_indexer_top_k     = 2048;
constexpr int     k_warmup_steps      = 16;
constexpr int     k_decode_steps      = 256;

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

static std::vector<float> fixture_values(size_t count, int period) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(static_cast<int>(i % period) - period / 2) / period;
    }
    return values;
}

struct stability_graph {
    ggml_context_ptr        context;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph *           graph       = nullptr;
    ggml_tensor *           position    = nullptr;
    ggml_tensor *           lid_scores  = nullptr;
    ggml_tensor *           attention   = nullptr;
    size_t                  buffer_size = 0;
    int                     node_count  = 0;
};

static void set_f32(ggml_tensor * tensor, const std::vector<float> & values) {
    require(ggml_nelements(tensor) == static_cast<int64_t>(values.size()), "wrong stability f32 size");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

static stability_graph make_stability_graph(ggml_backend_t backend, int64_t context_length) {
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr context(ggml_init(params));
    require(context != nullptr, "failed to create GLM stability context");

    ggml_tensor * q_raw    = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, k_head_size_k, k_attention_heads, 1, 1);
    ggml_tensor * position = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 1);
    ggml_tensor * q_rope   = ggml_rope_ext(context.get(), q_raw, position, nullptr, 64, GGML_ROPE_TYPE_NORMAL, 131072,
                                           10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * q        = ggml_permute(context.get(), q_rope, 0, 2, 1, 3);

    ggml_tensor * indexer_q =
        ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, k_indexer_head_size, k_indexer_heads, 1, 1);
    ggml_tensor * indexer_k =
        ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, k_indexer_head_size, 1, context_length, 1);
    ggml_tensor * indexer_weights = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, k_indexer_heads, 1, 1, 1);
    ggml_tensor * indexer_mask    = ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, context_length, 1, 1, 1);
    ggml_tensor * lid_scores =
        ggml_lightning_indexer(context.get(), indexer_q, indexer_k, indexer_weights, indexer_mask);
    ggml_set_name(lid_scores, "stability_lid_scores");

    const int64_t selected_rows = std::min(context_length, k_indexer_top_k);
    ggml_tensor * top_k         = ggml_new_tensor_3d(context.get(), GGML_TYPE_I32, selected_rows, 1, 1);
    ggml_tensor * k_cache       = ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, k_head_size_k, 1, context_length, 1);
    ggml_tensor * v_cache       = ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, k_head_size_v, 1, context_length, 1);
    ggml_tensor * causal_mask   = ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, context_length, 1, 1, 1);

    ggml_tensor * k_full     = ggml_permute(context.get(), k_cache, 0, 2, 1, 3);
    ggml_tensor * v_full     = ggml_permute(context.get(), v_cache, 0, 2, 1, 3);
    ggml_tensor * k_selected = ggml_get_rows(context.get(), k_full, top_k);
    ggml_tensor * v_selected = ggml_get_rows(context.get(), v_full, top_k);
    ggml_set_name(k_selected, "stability_k_selected");
    ggml_set_name(v_selected, "stability_v_selected");
    require(k_selected->ne[1] == selected_rows && v_selected->ne[1] == selected_rows,
            "stability graph did not preserve compact selected rows");
    k_selected = ggml_cast(context.get(), k_selected, GGML_TYPE_F16);
    v_selected = ggml_cast(context.get(), v_selected, GGML_TYPE_F16);

    ggml_tensor * mask_rows     = ggml_reshape_4d(context.get(), causal_mask, 1, context_length, 1, 1);
    ggml_tensor * mask_selected = ggml_get_rows(context.get(), mask_rows, top_k);
    mask_selected               = ggml_reshape_4d(context.get(), mask_selected, selected_rows, 1, 1, 1);
    mask_selected               = ggml_cast(context.get(), mask_selected, GGML_TYPE_F16);

    ggml_tensor * attention = ggml_flash_attn_ext(context.get(), q, k_selected, v_selected, mask_selected,
                                                  1.0f / std::sqrt(static_cast<float>(k_head_size_k)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attention, GGML_PREC_F32);
    ggml_set_name(attention, "stability_flash_attention");

    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 128, false);
    ggml_build_forward_expand(graph, lid_scores);
    ggml_build_forward_expand(graph, attention);

    int lid_count        = 0;
    int gather_count     = 0;
    int rope_count       = 0;
    int flash_count      = 0;
    int dense_mask_count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        require(ggml_backend_supports_op(backend, node), "stability graph contains an op unsupported by Metal");
        lid_count += node->op == GGML_OP_LIGHTNING_INDEXER;
        gather_count += node->op == GGML_OP_GET_ROWS;
        rope_count += node->op == GGML_OP_ROPE;
        flash_count += node->op == GGML_OP_FLASH_ATTN_EXT;
        dense_mask_count += node->op == GGML_OP_SET_ROWS;
    }
    require(lid_count == 1 && gather_count == 3 && rope_count == 1 && flash_count == 1,
            "stability graph is missing a native GLM sparse-decode operation");
    require(dense_mask_count == 0, "stability graph materialized a dense sparse-attention mask");

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    require(buffer != nullptr, "failed to allocate GLM stability tensors");
    ggml_backend_tensor_memset(indexer_k, 0, 0, ggml_nbytes(indexer_k));
    ggml_backend_tensor_memset(indexer_mask, 0, 0, ggml_nbytes(indexer_mask));
    ggml_backend_tensor_memset(k_cache, 0, 0, ggml_nbytes(k_cache));
    ggml_backend_tensor_memset(v_cache, 0, 0, ggml_nbytes(v_cache));
    ggml_backend_tensor_memset(causal_mask, 0, 0, ggml_nbytes(causal_mask));
    set_f32(q_raw, fixture_values(ggml_nelements(q_raw), 31));
    set_f32(indexer_q, fixture_values(ggml_nelements(indexer_q), 29));
    set_f32(indexer_weights, fixture_values(ggml_nelements(indexer_weights), 17));

    std::vector<int32_t> indices(selected_rows);
    for (int64_t i = 0; i < selected_rows; ++i) {
        indices[i] = static_cast<int32_t>((i * context_length) / selected_rows);
    }
    ggml_backend_tensor_set(top_k, indices.data(), 0, indices.size() * sizeof(int32_t));

    stability_graph result;
    result.context     = std::move(context);
    result.buffer      = std::move(buffer);
    result.graph       = graph;
    result.position    = position;
    result.lid_scores  = lid_scores;
    result.attention   = attention;
    result.buffer_size = ggml_backend_buffer_get_size(result.buffer.get());
    result.node_count  = ggml_graph_n_nodes(graph);
    return result;
}

static void run_context(ggml_backend_t backend, int64_t context_length) {
    stability_graph fixture        = make_stability_graph(backend, context_length);
    const int32_t   first_position = static_cast<int32_t>(context_length - k_decode_steps);
    for (int step = 0; step < k_warmup_steps; ++step) {
        const int32_t position = first_position + step;
        ggml_backend_tensor_set(fixture.position, &position, 0, sizeof(position));
        require(ggml_backend_graph_compute(backend, fixture.graph) == GGML_STATUS_SUCCESS,
                "GLM stability warmup failed");
    }
    ggml_backend_synchronize(backend);

    const uint64_t rss_baseline = resident_bytes();
    uint64_t       rss_peak     = rss_baseline;
    const auto     started      = std::chrono::steady_clock::now();
    for (int step = 0; step < k_decode_steps; ++step) {
        const int32_t position = first_position + step;
        ggml_backend_tensor_set(fixture.position, &position, 0, sizeof(position));
        require(ggml_backend_graph_compute(backend, fixture.graph) == GGML_STATUS_SUCCESS,
                "GLM stability decode failed");
        if ((step + 1) % 16 == 0) {
            ggml_backend_synchronize(backend);
            rss_peak = std::max(rss_peak, resident_bytes());
        }
    }
    ggml_backend_synchronize(backend);
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started);

    require(ggml_graph_n_nodes(fixture.graph) == fixture.node_count, "GLM stability graph changed across decode");
    require(ggml_backend_buffer_get_size(fixture.buffer.get()) == fixture.buffer_size,
            "GLM stability buffer grew across decode");
    const uint64_t rss_growth = rss_peak > rss_baseline ? rss_peak - rss_baseline : 0;
    require(rss_baseline == 0 || rss_growth <= 32ULL * 1024 * 1024, "GLM stability RSS grew after Metal warmup");

    float lid_value       = 0.0f;
    float attention_value = 0.0f;
    ggml_backend_tensor_get(fixture.lid_scores, &lid_value, 0, sizeof(lid_value));
    ggml_backend_tensor_get(fixture.attention, &attention_value, 0, sizeof(attention_value));
    require(std::isfinite(lid_value) && std::isfinite(attention_value), "GLM stability output is not finite");

    std::printf(
        "GLM sparse stability: ctx=%lld top-k=%lld steps=%d nodes=%d buffer=%.1f MiB "
        "%.3f ms/step RSS-growth=%.1f MiB\n",
        static_cast<long long>(context_length), static_cast<long long>(std::min(context_length, k_indexer_top_k)),
        k_decode_steps, fixture.node_count, fixture.buffer_size / (1024.0 * 1024.0), elapsed.count() / k_decode_steps,
        rss_growth / (1024.0 * 1024.0));
}

}  // namespace

void test_glm_dsa_decode_stability() {
    ggml_backend_dev_t metal_device = ggml_backend_dev_by_name("MTL0");
    if (!metal_device) {
        return;
    }
    ggml_backend_ptr metal(ggml_backend_dev_init(metal_device, nullptr));
    require(metal != nullptr, "failed to initialize Metal stability backend");
    for (int64_t context_length : std::array<int64_t, 3>{ 2048, 32768, 131072 }) {
        run_context(metal.get(), context_length);
    }
}
