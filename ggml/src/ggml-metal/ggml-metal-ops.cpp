#include "ggml-metal-ops.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <cstring>

static ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

static bool ggml_metal_lightning_indexer_parallel_requested() {
    const char * value = getenv("LLAMA_GLM_DSA_PARALLEL_LIGHTNING_INDEXER");
    return value && atoi(value) != 0;
}

static bool ggml_metal_lightning_indexer_staged_q_requested() {
    const char * value = getenv("LLAMA_GLM_DSA_EXPERIMENTAL_LIGHTNING_INDEXER_STAGED_Q");
    return value && atoi(value) != 0;
}

static int ggml_metal_lightning_indexer_parallel_threads_requested() {
    const char * value = getenv("LLAMA_GLM_DSA_PARALLEL_LIGHTNING_INDEXER_THREADS");
    if (value == nullptr || value[0] == '\0') {
        return 64;
    }

    const int requested = atoi(value);
    switch (requested) {
        case 32:
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
            return requested;
        default:
            return 64;
    }
}

static bool ggml_metal_glm_dsa_dispatch_log_enabled() {
    const char * value = getenv("GGML_METAL_MOE_DISPATCH_LOG");
    if (value && atoi(value) != 0) {
        return true;
    }
    value = getenv("GGML_GLM_DSA_LOG_METAL_DISPATCH");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_topk_moe_fusion_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_TOPK_MOE_ROUTE_FUSION");
    if (value) {
        return atoi(value) != 0;
    }
    value = getenv("GGML_GLM_DSA_ENABLE_METAL_TOPK_MOE_FUSION");
    if (value) {
        return atoi(value) != 0;
    }
    value = getenv("GGML_METAL_DISABLE_TOPK_MOE_ROUTE_FUSION");
    if (value && atoi(value) != 0) {
        return false;
    }
    value = getenv("GGML_GLM_DSA_DISABLE_METAL_TOPK_MOE_FUSION");
    if (value && atoi(value) != 0) {
        return false;
    }
    return true;
}

static bool ggml_metal_glm_dsa_topk_moe_route_sg_reduce_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_TOPK_MOE_ROUTE_SG_REDUCE");
    if (value) {
        return atoi(value) != 0;
    }
    value = getenv("GGML_GLM_DSA_EXPERIMENTAL_TOPK_MOE_ROUTE_SG_REDUCE");
    if (value) {
        return atoi(value) != 0;
    }
    value = getenv("GGML_METAL_DISABLE_TOPK_MOE_ROUTE_SG_REDUCE");
    if (value && atoi(value) != 0) {
        return false;
    }
    value = getenv("GGML_GLM_DSA_DISABLE_TOPK_MOE_ROUTE_SG_REDUCE");
    if (value && atoi(value) != 0) {
        return false;
    }
    return true;
}

static bool ggml_metal_glm_dsa_topk_moe_route_glm_256_8_sg32_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_TOPK_MOE_ROUTE_SG32");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_decode_motif_reference_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_GLM_MOE_DECODE_MOTIF_REFERENCE");
    if (value) {
        return atoi(value) != 0;
    }
    value = getenv("GGML_GLM_DSA_EXPERIMENTAL_MOE_DECODE_MOTIF_REFERENCE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_private_scratch_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_PRIVATE_SCRATCH");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_two_phase_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_TWO_PHASE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_dual_lane_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_DUAL_LANE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_dual_lane_gate_slot4_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_DUAL_LANE_GATE_SLOT4");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_absorbed_qkv_phases_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_ABSORBED_QKV_PHASES");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_route_anchor_bypass_enabled() {
    const char * disabled = getenv("GGML_METAL_DISABLE_GLM_MOE_ROUTE_ANCHOR_BYPASS");
    if (disabled) {
        return atoi(disabled) == 0;
    }
    const char * value = getenv("GGML_METAL_ENABLE_GLM_MOE_ROUTE_ANCHOR_BYPASS");
    if (value) {
        return atoi(value) != 0;
    }
    return true;
}

static bool ggml_metal_glm_dsa_moe_swiglu_q3_down_fusion_enabled() {
    const char * disabled = getenv("GGML_METAL_DISABLE_GLM_MOE_SWIGLU_Q3_DOWN_FUSION");
    if (disabled) {
        return atoi(disabled) == 0;
    }
    const char * value = getenv("GGML_METAL_ENABLE_GLM_MOE_SWIGLU_Q3_DOWN_FUSION");
    if (value) {
        return atoi(value) != 0;
    }
    value = getenv("GGML_GLM_DSA_EXPERIMENTAL_MOE_SWIGLU_Q3_DOWN_FUSION");
    if (value) {
        return atoi(value) != 0;
    }
    return true;
}

static bool ggml_metal_glm_dsa_moe_swiglu_q2_down_fusion_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_SWIGLU_Q2_DOWN_FUSION");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_decode_skip_internal_barriers_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_DECODE_SKIP_INTERNAL_BARRIERS");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_decode_skip_route_barrier_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_DECODE_SKIP_ROUTE_BARRIER");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_decode_skip_gate_up_barrier_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_DECODE_SKIP_GATE_UP_BARRIER");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_decode_scoped_barriers_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_DECODE_SCOPED_BARRIERS");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_route_gate_up_fusion_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_ROUTE_GATE_UP_FUSION");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_sort_route_ids_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_SORT_ROUTE_IDS");
    return value && atoi(value) != 0;
}

static int ggml_metal_glm_dsa_moe_max_active_experts() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS");
    if (value == nullptr) {
        return 0;
    }

    const int requested = atoi(value);
    return requested >= 1 && requested <= 8 ? requested : 0;
}

static int ggml_metal_glm_dsa_moe_q2_weight_roofline_chunks() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_Q2_WEIGHT_ROOFLINE_CHUNKS");
    if (value == nullptr) {
        return 0;
    }

    const int requested = atoi(value);
    return requested >= 1 && requested <= 64 ? requested : 0;
}

static bool ggml_metal_glm_dsa_moe_q2_weight_roofline_bypass_route_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_Q2_WEIGHT_ROOFLINE_BYPASS_ROUTE");
    return value && atoi(value) != 0;
}

static int ggml_metal_glm_dsa_moe_q2_weight_roofline_block_bytes() {
    const int q2_k_block_bytes = int(ggml_type_size(GGML_TYPE_Q2_K));
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_MOE_Q2_WEIGHT_ROOFLINE_BLOCK_BYTES");
    if (value == nullptr) {
        return q2_k_block_bytes;
    }

    const int requested = atoi(value);
    return requested >= q2_k_block_bytes && requested <= 256 && requested % 4 == 0 ?
        requested : q2_k_block_bytes;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_Q2_GATE_UP_SWIGLU_FUSION");
    if (value) {
        return atoi(value) != 0;
    }
    value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q2_GATE_UP_SWIGLU");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_default_enabled() {
    const char * disabled = getenv("GGML_METAL_DISABLE_Q2_GATE_UP_SWIGLU_FUSION");
    if (disabled) {
        return atoi(disabled) == 0;
    }
    return false;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_vecscale_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_VECSCALE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot8_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT8");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot2_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT2");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot4_dual_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT4_DUAL");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot4_dual_r12_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT4_DUAL_R12");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot4_dual_r16_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT4_DUAL_R16");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot1_dual_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT1_DUAL");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot2_dual_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT2_DUAL");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot8_split_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT8_SPLIT");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_share_y_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SHARE_Y");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_vecscale_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_VECSCALE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_q8_act_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_Q8_ACT");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_prequant_q8_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_PREQUANT_Q8");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_inblock_repack_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_INBLOCK_REPACK");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_half_y_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_HALF_Y");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_rowtile_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_ROWTILE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_r16_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_R16");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_r12_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_R12");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_any_variant_enabled() {
    return
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot8_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot2_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot4_dual_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot4_dual_r12_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot4_dual_r16_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot1_dual_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot2_dual_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot8_split_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_share_y_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_vecscale_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_q8_act_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_prequant_q8_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_inblock_repack_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_half_y_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_rowtile_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_r16_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_r12_enabled();
}

static bool ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_default_enabled() {
    const char * disabled = getenv("GGML_METAL_DISABLE_Q2_GATE_UP_SWIGLU_PAIR_SG");
    if (disabled) {
        return atoi(disabled) == 0;
    }
    return true;
}

static bool ggml_metal_glm_dsa_weighted_swiglu_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_WEIGHTED_SWIGLU");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_weighted_reduce_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q2_DOWN_WEIGHTED_REDUCE_DIRECT");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_DOWN_SLOT_PARALLEL_REDUCE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_default_enabled() {
    const char * disabled = getenv("GGML_METAL_DISABLE_Q2_DOWN_SLOT_PARALLEL_REDUCE");
    if (disabled) {
        return atoi(disabled) == 0;
    }
    return true;
}

static bool ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r16_w1_enabled() {
    const char * disabled = getenv("GGML_METAL_DISABLE_Q2_DOWN_SLOT_PARALLEL_REDUCE");
    if (disabled) {
        return atoi(disabled) == 0;
    }
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_DOWN_SLOT_PARALLEL_REDUCE_R16_W1");
    if (value) {
        return atoi(value) != 0;
    }
    return true;
}

static bool ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r16_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_DOWN_SLOT_PARALLEL_REDUCE_R16");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r4_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_DOWN_SLOT_PARALLEL_REDUCE_R4");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_f16_act_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_DOWN_F16_ACT");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_shift_high_bits_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q2_DOWN_SHIFT_HIGH_BITS");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_vec_scale_enabled() {
    const char * disabled = getenv("GGML_METAL_DISABLE_Q2_DOWN_VEC_SCALE");
    if (disabled) {
        return atoi(disabled) == 0;
    }
    const char * value = getenv("GGML_METAL_ENABLE_Q2_DOWN_VEC_SCALE");
    if (value) {
        return atoi(value) != 0;
    }
    value = getenv("GGML_METAL_EXPERIMENTAL_Q2_DOWN_VEC_SCALE");
    if (value) {
        return atoi(value) != 0;
    }
    return true;
}

static bool ggml_metal_glm_dsa_q3_down_weighted_reduce_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_Q3_DOWN_WEIGHTED_REDUCE_FUSION");
    if (value) {
        return atoi(value) != 0;
    }
    value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q3_DOWN_WEIGHTED_REDUCE_DIRECT");
    return value && atoi(value) != 0;
}

static const char * ggml_metal_tensor_name(const ggml_tensor * tensor);

static bool ggml_metal_glm_dsa_q3_down_weighted_reduce_tensor_selected(
        const ggml_tensor * tensor) {
    const char * selected = getenv("GGML_METAL_Q3_DOWN_WEIGHTED_REDUCE_TENSOR");
    return selected == nullptr || selected[0] == '\0' ||
        strcmp(selected, ggml_metal_tensor_name(tensor)) == 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q3_DOWN_SLOT_PARALLEL_REDUCE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q3_DOWN_SLOT_PARALLEL_REDUCE_R8");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_Q3_DOWN_SLOT_PARALLEL_REDUCE_R8_NB8");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w0_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_Q3_DOWN_SLOT_PARALLEL_REDUCE_R8_NB8_W0");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r6_nb8_w0_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_Q3_DOWN_SLOT_PARALLEL_REDUCE_R6_NB8_W0");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r10_nb8_w0_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_Q3_DOWN_SLOT_PARALLEL_REDUCE_R10_NB8_W0");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_glm52_w0_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_Q3_DOWN_SLOT_PARALLEL_REDUCE_GLM52_W0");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_f16_act_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_Q3_DOWN_F16_ACT");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w0_default_enabled() {
    return false;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w1_default_enabled() {
    return false;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r12_nb8_w0_enabled() {
    const char * value = getenv("GGML_METAL_ENABLE_Q3_DOWN_SLOT_PARALLEL_REDUCE_R12_NB8_W0");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r16_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q3_DOWN_SLOT_PARALLEL_REDUCE_R16");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_slot_split2_reduce_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q3_DOWN_SLOT_SPLIT2_REDUCE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q3_down_atomic_accum_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q3_DOWN_ATOMIC_ACCUM");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_weighted_reduce_reference_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q2_DOWN_WEIGHTED_REDUCE_REFERENCE");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_weighted_reduce_noop_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q2_DOWN_WEIGHTED_REDUCE_NOOP");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_shared_expert_noop_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_SHARED_EXPERT_NOOP");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_routed_expert_noop_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_ROUTED_EXPERT_NOOP");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_indexer_projection_noop_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_INDEXER_PROJECTION_NOOP");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_attention_projection_noop_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_ATTN_PROJECTION_NOOP");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_lightning_indexer_noop_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_LIGHTNING_INDEXER_NOOP");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_selected_row_flash_noop_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_SELECTED_ROW_FLASH_NOOP");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_compact_multihead_flash_enabled(const ggml_tensor * op) {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_COMPACT_MULTIHEAD_FLASH");
    if (value == nullptr || atoi(value) == 0) {
        return false;
    }

    const char * tensor = getenv("GGML_METAL_EXPERIMENTAL_GLM_COMPACT_MULTIHEAD_FLASH_TENSOR");
    return tensor == nullptr || tensor[0] == '\0' || strstr(ggml_metal_tensor_name(op), tensor) != nullptr;
}

static bool ggml_metal_glm_compact_split_exact_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_COMPACT_SPLIT_EXACT");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_compact_sequential_v_diagnostic_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_COMPACT_SEQUENTIAL_V_DIAGNOSTIC");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_compact_legacy_scores_v_diagnostic_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_COMPACT_LEGACY_SCORES_V_DIAGNOSTIC");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_compact_dump_scores_diagnostic_enabled() {
    const char * value = getenv("GGML_METAL_EXPERIMENTAL_GLM_COMPACT_DUMP_SCORES_DIAGNOSTIC");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_moe_route_weights_slot0_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_MOE_ROUTE_WEIGHTS_SLOT0");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_q2_down_weighted_reduce_stock_slot0_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_Q2_DOWN_WEIGHTED_REDUCE_STOCK_SLOT0");
    return value && atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_selected_row_flash_enabled() {
    const char * value = getenv("GGML_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH");
    return value == nullptr || value[0] == '\0' || atoi(value) != 0;
}

static int ggml_metal_glm_dsa_sparse_attn_threads_requested() {
    const char * value = getenv("GGML_GLM_DSA_SPARSE_ATTN_THREADS");
    if (value == nullptr || value[0] == '\0') {
        return 256;
    }

    const int requested = atoi(value);
    switch (requested) {
        case 32:
        case 64:
        case 128:
        case 256:
            return requested;
        default:
            return 256;
    }
}

static int ggml_metal_glm_dsa_sparse_attn_threads_for_shape(int requested, int n_batch, int n_top_k) {
    // Large-top-k prefill shapes are sensitive to the 256-thread sparse-attn
    // kernel on Apple Metal and can leave rows unwritten. Keep decode and
    // small-top-k prefill on the requested path, but cap large prefill rows to
    // the shape that passes backend parity.
    if (n_batch > 1 && n_top_k >= 64) {
        return std::min(requested, 32);
    }
    if (n_top_k > 512) {
        return std::min(requested, 128);
    }
    return requested;
}

static bool ggml_metal_glm_dsa_sparse_attn_cache_topk_enabled() {
    const char * value = getenv("GGML_GLM_DSA_SPARSE_ATTN_CACHE_TOPK");
    return value && atoi(value) != 0;
}

static int ggml_metal_glm_dsa_sparse_attn_decode_group_heads_requested() {
    const char * value = getenv("GGML_GLM_DSA_SPARSE_ATTN_DECODE_GROUP_HEADS");
    if (value == nullptr || value[0] == '\0') {
        return 1;
    }

    const int requested = atoi(value);
    switch (requested) {
        case 2:
        case 4:
            return requested;
        default:
            return 1;
    }
}

static int ggml_metal_glm_dsa_mul_mm_id_min_tokens_requested() {
    const char * value = getenv("GGML_GLM_DSA_MUL_MM_ID_MIN_TOKENS");
    if (value == nullptr || value[0] == '\0') {
        return 32;
    }

    const int requested = atoi(value);
    return requested > 0 ? requested : 32;
}

static const char * ggml_metal_tensor_name(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->name[0] != '\0' ? tensor->name : "<unnamed>";
}

static void ggml_metal_log_topk_moe_route_encode_candidate(ggml_metal_op_t ctx, int idx, const char * reason);

struct ggml_metal_topk_moe_route_fusion {
    ggml_tensor * logits = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * clamp = nullptr;
    ggml_tensor * scale = nullptr;
    int n_fuse = 0;
};

struct ggml_metal_mul_mv_id_gate_up_swiglu_fusion {
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * glu = nullptr;
    ggml_tensor * cast = nullptr;
    ggml_tensor * weighted = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * q8 = nullptr;
    int n_fuse = 0;
};

struct ggml_metal_weighted_swiglu_fusion {
    ggml_tensor * glu = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * weighted = nullptr;
    int n_fuse = 0;
};

struct ggml_metal_mul_mv_id_weighted_reduce_fusion {
    ggml_tensor * down = nullptr;
    ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_up = nullptr;
    ggml_tensor * weighted_sum = nullptr;
    int shared_gate_offset = -1;
    int shared_up_offset = -1;
    int weighted_sum_offset = -1;
    int n_fuse = 0;
};

struct ggml_metal_glm_moe_private_bindings;

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_repeat(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_glu(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_sum(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_moe_weighted_sum(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_moe_mul_mat_id(ggml_metal_op_t ctx, int idx);
static int ggml_metal_op_weighted_swiglu(ggml_metal_op_t ctx, int idx);
static int ggml_metal_op_mul_mv_id_gate_up_swiglu(
        ggml_metal_op_t ctx,
        int idx,
        ggml_tensor * src1_override = nullptr);
static int ggml_metal_op_glm_moe_route_gate_up_swiglu(
        ggml_metal_op_t ctx,
        const struct ggml_metal_glm_moe_decode_motif_reference & motif,
        ggml_tensor * src1_override,
        const ggml_metal_glm_moe_private_bindings * private_bindings = nullptr);
static int ggml_metal_encode_moe_mul_mat_id(
        ggml_metal_op_t ctx,
        ggml_tensor * op,
        const ggml_metal_glm_moe_private_bindings * private_bindings);
static int ggml_metal_op_mul_mv_id_weighted_reduce(
        ggml_metal_op_t ctx,
        int idx,
        bool subgraph_owned = false);

struct ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_metal_buffer_id fusion_scratch,
        size_t fusion_scratch_size,
        ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion) {
        this->dev             = dev;
        this->lib             = ggml_metal_device_get_library(dev);
        this->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->mem_ranges      = ggml_mem_ranges_init(debug_graph);
        this->fusion_scratch  = fusion_scratch;
        this->fusion_scratch_size = fusion_scratch_size;
        this->idx_start       = idx_start;
        this->idx_end         = idx_end;
        this->use_fusion      = use_fusion;
        this->use_concurrency = use_concurrency;
        this->use_capture     = use_capture;
        this->debug_graph     = debug_graph;
        this->debug_fusion    = debug_fusion;
        this->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        for (int i = idx_start; i < idx_end; i++) {
            if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }

    }

    ~ggml_metal_op() {
        ggml_metal_encoder_end_encoding(this->enc);
        ggml_metal_encoder_free(this->enc);
        ggml_mem_ranges_free(this->mem_ranges);
    }

    int n_nodes() const {
        return idxs.size();
    }

    ggml_tensor * node(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_graph_node(gf, idxs[i]);
    }

    int graph_index(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return idxs[i];
    }

    int graph_node_count() const {
        return gf->n_nodes;
    }

    uint64_t graph_uid() const {
        return gf->uid;
    }

    int split_start() const {
        return idx_start;
    }

    int split_end() const {
        return idx_end;
    }

    ggml_tensor * graph_node(int i) const {
        assert(i >= 0 && i < gf->n_nodes);
        return ggml_graph_node(gf, i);
    }

    int32_t graph_node_use_count(int i) const {
        assert(i >= 0 && i < gf->n_nodes);
        return ggml_node_get_use_count(gf, i);
    }

    int filtered_count_for_graph_span(int i0, int graph_count) const {
        const int graph_start = graph_index(i0);
        const int graph_end = graph_start + graph_count;
        int count = 0;
        for (int i = i0; i < (int) idxs.size() && idxs[i] < graph_end; ++i) {
            ++count;
        }
        return count;
    }

    bool can_fuse_graph_subgraph(int graph_i0, const ggml_op * ops, int n_ops, const int * outputs, int n_outputs) const {
        assert(use_fusion);
        return ggml_can_fuse_subgraph(gf, graph_i0, n_ops, ops, outputs, n_outputs);
    }

    bool can_fuse(int i0, const ggml_op * ops, int n_ops) const {
        assert(use_fusion);
        assert(i0 >= 0 && i0 < n_nodes());

        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        return ggml_can_fuse_ext(gf, idxs.data() + i0, ops, n_ops);
    }

    bool can_fuse_subgraph(int i0, const ggml_op * ops, int n_ops, const int * output_offsets, int n_outputs) const {
        assert(use_fusion);
        assert(i0 >= 0 && i0 < n_nodes());

        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        int outputs[4];
        GGML_ASSERT(n_outputs <= 4);
        for (int i = 0; i < n_outputs; ++i) {
            outputs[i] = idxs[i0 + output_offsets[i]];
        }

        return ggml_can_fuse_subgraph_ext(gf, idxs.data() + i0, n_ops, ops, outputs, n_outputs);
    }

    bool can_fuse_filtered_subgraph(
            int i0,
            const ggml_op * ops,
            int n_ops,
            const int * output_offsets,
            int n_outputs) const {
        assert(use_fusion);
        assert(i0 >= 0 && i0 < n_nodes());
        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        const int graph_start = graph_index(i0);
        const int graph_end = graph_index(i0 + n_ops - 1) + 1;
        const auto is_output = [&](int offset) {
            for (int i = 0; i < n_outputs; ++i) {
                if (output_offsets[i] == offset) {
                    return true;
                }
            }
            return false;
        };
        const auto is_inside_graph_span = [&](const ggml_tensor * tensor) {
            for (int graph_idx = graph_start; graph_idx < graph_end; ++graph_idx) {
                if (graph_node(graph_idx) == tensor) {
                    return true;
                }
            }
            return false;
        };

        for (int rel = 0; rel < n_ops; ++rel) {
            const int graph_idx = graph_index(i0 + rel);
            const ggml_tensor * node = graph_node(graph_idx);
            if (node->op != ops[rel] || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                return false;
            }
            if (is_output(rel)) {
                continue;
            }
            if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
                return false;
            }

            int span_uses = 0;
            for (int consumer_idx = graph_start; consumer_idx < graph_end; ++consumer_idx) {
                const ggml_tensor * consumer = graph_node(consumer_idx);
                for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                    span_uses += consumer->src[src_idx] == node ? 1 : 0;
                }
            }
            if (span_uses != graph_node_use_count(graph_idx)) {
                return false;
            }

            for (const ggml_tensor * view_src = node->view_src;
                    view_src != nullptr;
                    view_src = view_src->view_src) {
                if (!is_inside_graph_span(view_src)) {
                    return false;
                }
            }
        }
        return true;
    }

    void set_fused_range_outputs(int output0, int output1 = -1, int output2 = -1) {
        fused_range_output_count = 0;
        const int outputs[] = { output0, output1, output2 };
        for (int output : outputs) {
            if (output >= 0) {
                fused_range_outputs[fused_range_output_count++] = output;
            }
        }
    }

    bool has_fused_range_outputs() const {
        return fused_range_output_count > 0;
    }

    bool tracks_fused_range_output(int offset) const {
        for (int i = 0; i < fused_range_output_count; ++i) {
            if (fused_range_outputs[i] == offset) {
                return true;
            }
        }
        return false;
    }

    void clear_fused_range_outputs() {
        fused_range_output_count = 0;
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;
    ggml_metal_buffer_id fusion_scratch;
    size_t               fusion_scratch_size;

    bool use_fusion;
    bool use_concurrency;
    bool use_capture;

    int debug_graph;
    int debug_fusion;

    int fused_range_outputs[3] = { -1, -1, -1 };
    int fused_range_output_count = 0;

private:
    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
};

static void ggml_metal_log_selected_row_flash_candidate(ggml_metal_op_t ctx, int idx) {
    if (!ggml_metal_glm_dsa_dispatch_log_enabled() || !ggml_metal_glm_dsa_selected_row_flash_enabled()) {
        return;
    }

    const ggml_tensor * get_rows = ctx->node(idx);
    if (get_rows == nullptr || get_rows->op != GGML_OP_GET_ROWS ||
            get_rows->name[0] == '\0' || std::strstr(get_rows->name, "dsa_compact_") == nullptr) {
        return;
    }

    const int graph_idx = ctx->graph_index(idx);
    const int use_count = ctx->graph_node_use_count(graph_idx);
    const int graph_end = std::min(ctx->split_end(), ctx->graph_node_count());

    const ggml_tensor * consumer = nullptr;
    int consumer_graph_idx = -1;
    int consumer_src_slot = -1;
    int consumer_count = 0;
    const ggml_tensor * view = nullptr;
    const ggml_tensor * flash = nullptr;
    int flash_graph_idx = -1;
    for (int graph_i = graph_idx + 1; graph_i < graph_end; ++graph_i) {
        const ggml_tensor * candidate = ctx->graph_node(graph_i);
        if (candidate->op == GGML_OP_VIEW && candidate->src[0] == get_rows && view == nullptr) {
            view = candidate;
        }
        if (candidate->op == GGML_OP_FLASH_ATTN_EXT &&
                candidate->src[1] == get_rows &&
                candidate->src[2] != nullptr &&
                (candidate->src[2] == get_rows ||
                 (candidate->src[2]->op == GGML_OP_VIEW && candidate->src[2]->src[0] == get_rows))) {
            flash = candidate;
            flash_graph_idx = graph_i;
        }
        for (int src_i = 0; src_i < GGML_MAX_SRC; ++src_i) {
            if (candidate->src[src_i] == get_rows) {
                ++consumer_count;
                if (consumer == nullptr) {
                    consumer = candidate;
                    consumer_graph_idx = graph_i;
                    consumer_src_slot = src_i;
                }
            }
        }
    }

    const bool is_flash = consumer != nullptr && consumer->op == GGML_OP_FLASH_ATTN_EXT;
    const bool is_k_or_v = is_flash && (consumer_src_slot == 1 || consumer_src_slot == 2);
    const bool packed_kv_view = flash != nullptr && flash->src[1] == get_rows &&
            flash->src[2] != nullptr && flash->src[2]->op == GGML_OP_VIEW && flash->src[2]->src[0] == get_rows;
    const char * reason = packed_kv_view ? "accepted_packed_kv_view" :
            consumer == nullptr ? "consumer_not_found" :
            !is_flash ? "consumer_not_flash_attn_ext" :
            !is_k_or_v ? "consumer_not_kv" :
            consumer_count != use_count ? "consumer_count_mismatch" :
            "accepted_graph_pattern";

    GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=selected_row_flash_candidate tensor=%s reason=%s graph_uid=%llu split_start=%d split_end=%d filtered_idx=%d filtered_nodes=%d graph_idx=%d use_count=%d get_rows_uses=%d consumer_count=%d consumer_graph_idx=%d consumer_op=%s consumer_tensor=%s consumer_src_slot=%d next_tensor=%s flash_graph_idx=%d generic=0 view=%d src_type=%s rows_type=%s dst_type=%s rows=%lld width=%lld consumer_kv=%lld consumer_heads=%lld consumer_stream=%lld kv=%lld heads=%lld stream=%lld grid_x=1 grid_y=1 grid_z=1 threads_x=1\n",
            ggml_metal_tensor_name(get_rows),
            reason,
            (unsigned long long) ctx->graph_uid(),
            ctx->split_start(),
            ctx->split_end(),
            idx,
            ctx->n_nodes(),
            graph_idx,
            use_count,
            use_count,
            consumer_count,
            consumer_graph_idx,
            consumer ? ggml_op_name(consumer->op) : "none",
            ggml_metal_tensor_name(consumer),
            consumer_src_slot,
            ggml_metal_tensor_name(flash),
            flash_graph_idx,
            view != nullptr,
            get_rows->src[0] ? ggml_type_name(get_rows->src[0]->type) : "none",
            get_rows->src[1] ? ggml_type_name(get_rows->src[1]->type) : "none",
            ggml_type_name(get_rows->type),
            (long long) get_rows->ne[1],
            (long long) get_rows->ne[0],
            consumer ? (long long) consumer->ne[1] : -1LL,
            consumer ? (long long) consumer->ne[2] : -1LL,
            consumer ? (long long) consumer->ne[3] : -1LL,
            flash && flash->src[1] ? (long long) flash->src[1]->ne[1] : 0LL,
            flash && flash->src[0] ? (long long) flash->src[0]->ne[2] : 0LL,
            flash && flash->src[0] ? (long long) flash->src[0]->ne[3] : 0LL);
}

static bool ggml_metal_selected_row_flash_shape_ok(const ggml_tensor * flash) {
    if (flash == nullptr || flash->op != GGML_OP_FLASH_ATTN_EXT ||
            !ggml_metal_glm_dsa_selected_row_flash_enabled()) {
        return false;
    }

    ggml_tensor * rows = flash->src[1];
    ggml_tensor * view = flash->src[2];
    if (flash->src[0] == nullptr || rows == nullptr || view == nullptr ||
            rows->op != GGML_OP_GET_ROWS || view->op != GGML_OP_VIEW || view->src[0] != rows ||
            rows->src[0] == nullptr || rows->src[1] == nullptr) {
        return false;
    }

    const float max_bias = ggml_get_op_params_f32(flash, 1);
    const float logit_softcap = ggml_get_op_params_f32(flash, 2);
    if (flash->src[3] != nullptr || flash->src[4] != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    return flash->src[0]->type == GGML_TYPE_F32 &&
        rows->src[0]->type == GGML_TYPE_F16 &&
        rows->src[1]->type == GGML_TYPE_I32 &&
        rows->type == GGML_TYPE_F16 &&
        view->type == GGML_TYPE_F16 &&
        flash->type == GGML_TYPE_F32 &&
        rows->ne[0] == rows->src[0]->ne[0] &&
        view->ne[0] > 0 &&
        view->ne[0] <= rows->src[0]->ne[0] &&
        rows->ne[1] == rows->src[1]->ne[0] &&
        rows->src[1]->ne[0] <= 4096 &&
        flash->src[0]->ne[0] == rows->src[0]->ne[0] &&
        flash->ne[0] == view->ne[0];
}

static ggml_tensor * ggml_metal_find_selected_row_flash_consumer(ggml_metal_op_t ctx, int idx, ggml_tensor * rows) {
    if (!ctx->use_fusion || rows == nullptr || !ggml_metal_glm_dsa_selected_row_flash_enabled()) {
        return nullptr;
    }

    const int graph_idx = ctx->graph_index(idx);
    const int graph_end = std::min(ctx->split_end(), ctx->graph_node_count());
    for (int graph_i = graph_idx + 1; graph_i < graph_end; ++graph_i) {
        ggml_tensor * candidate = ctx->graph_node(graph_i);
        if (candidate->op == GGML_OP_FLASH_ATTN_EXT &&
                candidate->src[1] == rows &&
                candidate->src[2] != nullptr &&
                candidate->src[2]->op == GGML_OP_VIEW &&
                candidate->src[2]->src[0] == rows &&
                ggml_metal_selected_row_flash_shape_ok(candidate)) {
            return candidate;
        }
    }

    return nullptr;
}

static bool ggml_metal_selected_row_flash_can_defer_compact_k_rows(ggml_metal_op_t ctx, int idx, const ggml_tensor * rows) {
    if (rows == nullptr || rows->op != GGML_OP_GET_ROWS || !ggml_metal_glm_dsa_selected_row_flash_enabled()) {
        return false;
    }

    if (rows->name[0] == '\0' || std::strstr(rows->name, "dsa_compact_k_topk_rows") == nullptr) {
        return false;
    }

    if (rows->src[0] == nullptr || rows->src[1] == nullptr) {
        return false;
    }

    const int graph_idx = ctx->graph_index(idx);
    const int32_t use_count = ctx->graph_node_use_count(graph_idx);

    return use_count == 2 &&
        rows->src[0]->type == GGML_TYPE_F16 &&
        rows->src[1]->type == GGML_TYPE_I32 &&
        rows->type == GGML_TYPE_F16 &&
        rows->ne[0] == rows->src[0]->ne[0] &&
        rows->ne[1] == rows->src[1]->ne[0] &&
        rows->src[1]->ne[0] <= 4096;
}

static bool ggml_metal_selected_row_flash_vec_shape_ok(const ggml_tensor * op) {
    if (!ggml_metal_selected_row_flash_shape_ok(op)) {
        return false;
    }

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * rows = op->src[1];
    const ggml_tensor * view = op->src[2];

    return q->ne[0] == 576 &&
        view->ne[0] == 512 &&
        q->ne[1] == 1 &&
        rows->src[0]->ne[2] == 1 &&
        view->ne[2] == 1 &&
        rows->src[0]->ne[3] == 1 &&
        view->ne[3] == 1;
}

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_metal_buffer_id fusion_scratch,
        size_t fusion_scratch_size,
        ggml_cgraph * gf,
        int idx_start,
        int idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int debug_graph,
        int debug_fusion) {
    ggml_metal_op_t res = new ggml_metal_op(
        dev,
        cmd_buf,
        fusion_scratch,
        fusion_scratch_size,
        gf,
        idx_start,
        idx_end,
        use_fusion,
        use_concurrency,
        use_capture,
        debug_graph,
        debug_fusion);

    return res;
}

void ggml_metal_op_free(ggml_metal_op_t ctx) {
    delete ctx;
}

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}

static bool ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    if (!ctx->mem_ranges) {
        return true;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static void ggml_metal_op_internal_phase_barrier(ggml_metal_op_t ctx) {
    ggml_metal_encoder_memory_barrier(ctx->enc);
    if (ctx->mem_ranges) {
        ggml_mem_ranges_reset(ctx->mem_ranges);
    }
}

static bool ggml_metal_op_concurrency_reset_tensors(
        ggml_metal_op_t    ctx,
        const ggml_tensor * tensor0,
        const ggml_tensor * tensor1,
        const ggml_tensor * tensor2) {
    if (!ctx->mem_ranges) {
        return true;
    }

    ggml_metal_encoder_memory_barrier_buffer(ctx->enc, ggml_metal_get_buffer_id(tensor0));
    ggml_metal_encoder_memory_barrier_buffer(ctx->enc, ggml_metal_get_buffer_id(tensor1));
    ggml_metal_encoder_memory_barrier_buffer(ctx->enc, ggml_metal_get_buffer_id(tensor2));
    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return ggml_mem_ranges_add(ctx->mem_ranges, node);
}

static bool ggml_metal_tensor_name_contains(const ggml_tensor * tensor, const char * needle) {
    return tensor != nullptr && std::strstr(ggml_metal_tensor_name(tensor), needle) != nullptr;
}

static bool ggml_metal_tensor_is_shared_expert_gate(const ggml_tensor * tensor) {
    return ggml_metal_tensor_name_contains(tensor, "ffn_shexp_gate") ||
        ggml_metal_tensor_name_contains(tensor, "ffn_gate");
}

static bool ggml_metal_tensor_is_shared_expert_up(const ggml_tensor * tensor) {
    return ggml_metal_tensor_name_contains(tensor, "ffn_shexp_up") ||
        ggml_metal_tensor_name_contains(tensor, "ffn_up");
}

static bool ggml_metal_tensor_is_glm_dsa_qk_moe_expert(const ggml_tensor * tensor) {
    return tensor != nullptr && (tensor->type == GGML_TYPE_Q2_K || tensor->type == GGML_TYPE_Q3_K);
}

static void ggml_metal_log_glm_dsa_moe_motif_candidate(ggml_metal_op_t ctx, int idx) {
    if (!ggml_metal_glm_dsa_dispatch_log_enabled() || idx >= ctx->n_nodes()) {
        return;
    }

    int route_weights_idx = -1;
    int gate_idx = -1;
    int up_idx = -1;
    int glu_idx = -1;
    int down_idx = -1;
    int shared_gate_idx = -1;
    int shared_up_idx = -1;
    int weighted_sum_idx = -1;

    for (int j = idx; j < ctx->n_nodes() && j < idx + 18; ++j) {
        ggml_tensor * node = ctx->node(j);
        if (route_weights_idx < 0 &&
                node->op == GGML_OP_MOE_ROUTE_WEIGHTS &&
                ggml_metal_tensor_name_contains(node, "ffn_moe_route_weights")) {
            route_weights_idx = j;
        } else if (gate_idx < 0 &&
                node->op == GGML_OP_MUL_MAT_ID &&
                ggml_metal_tensor_name_contains(node, "ffn_moe_gate")) {
            gate_idx = j;
        } else if (up_idx < 0 &&
                node->op == GGML_OP_MUL_MAT_ID &&
                ggml_metal_tensor_name_contains(node, "ffn_moe_up")) {
            up_idx = j;
        } else if (glu_idx < 0 &&
                node->op == GGML_OP_GLU &&
                ggml_metal_tensor_name_contains(node, "ffn_moe_swiglu")) {
            glu_idx = j;
        } else if (down_idx < 0 &&
                node->op == GGML_OP_MUL_MAT_ID &&
                ggml_metal_tensor_name_contains(node, "ffn_moe_down")) {
            down_idx = j;
        } else if (down_idx >= 0 && shared_gate_idx < 0 &&
                node->op == GGML_OP_MUL_MAT &&
                ggml_metal_tensor_is_shared_expert_gate(node)) {
            shared_gate_idx = j;
        } else if (down_idx >= 0 && shared_up_idx < 0 &&
                node->op == GGML_OP_MUL_MAT &&
                ggml_metal_tensor_is_shared_expert_up(node)) {
            shared_up_idx = j;
        } else if (weighted_sum_idx < 0 &&
                node->op == GGML_OP_MOE_WEIGHTED_SUM &&
                ggml_metal_tensor_name_contains(node, "ffn_moe_out")) {
            weighted_sum_idx = j;
            break;
        }
    }

    if (route_weights_idx < 0 || gate_idx < 0 || up_idx < 0 || glu_idx < 0 || down_idx < 0 || weighted_sum_idx < 0) {
        return;
    }

    ggml_tensor * route_weights = ctx->node(route_weights_idx);
    ggml_tensor * gate = ctx->node(gate_idx);
    ggml_tensor * up = ctx->node(up_idx);
    ggml_tensor * down = ctx->node(down_idx);
    ggml_tensor * shared_gate = shared_gate_idx >= 0 ? ctx->node(shared_gate_idx) : nullptr;
    ggml_tensor * shared_up = shared_up_idx >= 0 ? ctx->node(shared_up_idx) : nullptr;
    ggml_tensor * weighted_sum = ctx->node(weighted_sum_idx);

    const bool natural_order =
        route_weights_idx < gate_idx &&
        gate_idx < up_idx &&
        up_idx < glu_idx &&
        glu_idx < down_idx &&
        down_idx < weighted_sum_idx;

    const bool backend_candidate =
        natural_order &&
        ggml_metal_tensor_is_glm_dsa_qk_moe_expert(gate->src[0]) &&
        ggml_metal_tensor_is_glm_dsa_qk_moe_expert(up->src[0]) &&
        ggml_metal_tensor_is_glm_dsa_qk_moe_expert(down->src[0]) &&
        gate->src[1] != nullptr && gate->src[1]->type == GGML_TYPE_F32 &&
        up->src[1] != nullptr && up->src[1]->type == GGML_TYPE_F32 &&
        down->src[1] != nullptr && down->src[1]->type == GGML_TYPE_F32 &&
        gate->src[2] != nullptr && gate->src[2]->type == GGML_TYPE_I32 &&
        up->src[2] != nullptr && up->src[2]->type == GGML_TYPE_I32 &&
        down->src[2] != nullptr && down->src[2]->type == GGML_TYPE_I32 &&
        gate->type == GGML_TYPE_F32 &&
        up->type == GGML_TYPE_F32 &&
        down->type == GGML_TYPE_F32 &&
        weighted_sum->type == GGML_TYPE_F32;

    bool subgraph_fusable = false;
    if (ctx->use_fusion && down_idx + 3 < ctx->n_nodes() && down_idx + 3 == weighted_sum_idx &&
            shared_gate_idx == down_idx + 1 && shared_up_idx == down_idx + 2) {
        const ggml_op ops[] = {
            GGML_OP_MUL_MAT_ID,
            GGML_OP_MUL_MAT,
            GGML_OP_MUL_MAT,
            GGML_OP_MOE_WEIGHTED_SUM,
        };
        const int outputs[] = { 1, 2, 3 };
        subgraph_fusable = ctx->can_fuse_subgraph(down_idx, ops, 4, outputs, 3);
    }

    const int filtered_gap = weighted_sum_idx - down_idx;
    const int graph_gap = ctx->graph_index(weighted_sum_idx) - ctx->graph_index(down_idx);

    GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=glm_dsa_moe_motif_candidate tensor=%s route_weights=%s shared_gate=%s shared_up=%s weighted_sum=%s reason=full_motif natural_order=%d backend_candidate=%d subgraph_fusable=%d motif_nodes=%d fusion_outputs=%d filtered_gap=%d graph_gap=%d weighted_sum_gap=%d weighted_sum_graph_gap=%d src0_type=%s src1_type=%s ids_type=%s dst_type=%s experts=%lld used_experts=%lld tokens=%lld grid_x=1 grid_y=1 grid_z=1 threads_x=1\n",
            ggml_metal_tensor_name(down),
            ggml_metal_tensor_name(route_weights),
            ggml_metal_tensor_name(shared_gate),
            ggml_metal_tensor_name(shared_up),
            ggml_metal_tensor_name(weighted_sum),
            natural_order ? 1 : 0,
            backend_candidate ? 1 : 0,
            subgraph_fusable ? 1 : 0,
            4,
            3,
            filtered_gap,
            graph_gap,
            filtered_gap,
            graph_gap,
            ggml_type_name(down->src[0]->type),
            ggml_type_name(down->src[1]->type),
            ggml_type_name(down->src[2]->type),
            ggml_type_name(down->type),
            (long long) down->src[0]->ne[2],
            (long long) down->src[2]->ne[0],
            (long long) down->src[2]->ne[1]);
}

static void ggml_metal_log_topk_moe_route_encode_candidate(ggml_metal_op_t ctx, int idx, const char * reason) {
    if (!ggml_metal_glm_dsa_dispatch_log_enabled() || idx >= ctx->n_nodes()) {
        return;
    }

    const ggml_tensor * node = ctx->node(idx);
    if (node->name[0] == '\0' || std::strstr(node->name, "ffn_moe_probs") == nullptr) {
        return;
    }

    char ops[512] = {};
    size_t offset = 0;
    for (int j = idx; j < ctx->n_nodes() && j < idx + 14 && offset < sizeof(ops); ++j) {
        const ggml_tensor * t = ctx->node(j);
        const int written = snprintf(
                ops + offset,
                sizeof(ops) - offset,
                "%s%s/%s",
                j == idx ? "" : ",",
                ggml_op_name(t->op),
                ggml_metal_tensor_name(t));
        if (written < 0) {
            break;
        }
        offset += (size_t) written;
    }

    GGML_LOG_INFO(
            "ggml_metal: moe_dispatch op=topk_moe_route_encode tensor=%s candidate=%s reason=%s graph_uid=%llu split_start=%d split_end=%d filtered_nodes=%d graph_nodes=%d graph_idx=%d grid_x=1 grid_y=1 grid_z=1 threads_x=1\n",
            ggml_metal_tensor_name(node),
            ops,
            reason,
            (unsigned long long) ctx->graph_uid(),
            ctx->split_start(),
            ctx->split_end(),
            ctx->n_nodes(),
            ctx->graph_node_count(),
            ctx->graph_index(idx));
    ggml_metal_log_glm_dsa_moe_motif_candidate(ctx, idx);
}

static bool ggml_metal_match_topk_moe_route_fusion(
        ggml_metal_op_t ctx,
        int idx,
        ggml_metal_topk_moe_route_fusion & fusion) {
    if (!ctx->use_fusion || !ggml_metal_glm_dsa_topk_moe_fusion_enabled()) {
        ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "disabled_or_short");
        return false;
    }

    const int graph_idx = ctx->graph_index(idx);
    if (graph_idx + 11 > ctx->graph_node_count()) {
        ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "disabled_or_short");
        return false;
    }

    ggml_tensor * sigmoid = ctx->graph_node(graph_idx);
    if (sigmoid->op != GGML_OP_UNARY || ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID ||
            sigmoid->src[0] == nullptr || sigmoid->src[0]->type != GGML_TYPE_F32) {
        ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "not_sigmoid");
        return false;
    }

    if (idx + 4 <= ctx->n_nodes()) {
        ggml_tensor * local_sigmoid = ctx->node(idx);
        ggml_tensor * biased_probs  = ctx->node(idx + 1);
        ggml_tensor * ids           = ctx->node(idx + 2);
        ggml_tensor * weights       = ctx->node(idx + 3);

        const bool compact_sequence =
            local_sigmoid == sigmoid &&
            biased_probs->op == GGML_OP_ADD && biased_probs->src[0] == sigmoid &&
            biased_probs->src[1] != nullptr && biased_probs->src[1]->type == GGML_TYPE_F32 &&
            ids->op == GGML_OP_TOP_K && ids->src[0] == biased_probs && ids->type == GGML_TYPE_I32 &&
            weights->op == GGML_OP_MOE_ROUTE_WEIGHTS && weights->src[0] != nullptr && weights->src[1] == ids &&
            weights->src[0]->op == GGML_OP_RESHAPE && weights->src[0]->src[0] == sigmoid;

        if (compact_sequence) {
            const bool compact_shape =
                ids->ne[0] > 0 && ids->ne[0] <= 16 && ids->ne[1] == sigmoid->ne[1] &&
                sigmoid->ne[0] > 0 && sigmoid->ne[0] <= 256 && sigmoid->ne[1] > 0 &&
                weights->type == GGML_TYPE_F32 && weights->ne[0] == 1 &&
                weights->ne[1] == ids->ne[0] && weights->ne[2] == ids->ne[1];
            if (!compact_shape) {
                ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "shape");
                return false;
            }

            const bool compact_safe =
                (sigmoid->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                (biased_probs->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                ctx->graph_node_use_count(ctx->graph_index(idx + 1)) == 1;
            if (!compact_safe) {
                ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "can_fuse");
                return false;
            }

            ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "fused");
            fusion.logits  = sigmoid->src[0];
            fusion.ids     = ids;
            fusion.weights = weights;
            fusion.bias    = biased_probs->src[1];
            fusion.clamp   = weights;
            fusion.scale   = nullptr;
            fusion.n_fuse  = 4;
            return true;
        }

    }

    if (graph_idx + 6 <= ctx->graph_node_count()) {
        ggml_tensor * reshape_probs = ctx->graph_node(graph_idx + 1);
        ggml_tensor * biased_probs  = ctx->graph_node(graph_idx + 2);
        ggml_tensor * argsort       = ctx->graph_node(graph_idx + 3);
        ggml_tensor * ids           = ctx->graph_node(graph_idx + 4);
        ggml_tensor * weights       = ctx->graph_node(graph_idx + 5);

        const bool argsort_route_weights_graph_sequence =
            reshape_probs->op == GGML_OP_RESHAPE && reshape_probs->src[0] == sigmoid &&
            biased_probs->op == GGML_OP_ADD && biased_probs->src[0] == sigmoid &&
            biased_probs->src[1] != nullptr && biased_probs->src[1]->type == GGML_TYPE_F32 &&
            argsort->op == GGML_OP_ARGSORT && argsort->src[0] == biased_probs &&
            ids->op == GGML_OP_VIEW && ids->src[0] == argsort && ids->type == GGML_TYPE_I32 &&
            weights->op == GGML_OP_MOE_ROUTE_WEIGHTS && weights->src[0] == reshape_probs && weights->src[1] == ids;

        if (argsort_route_weights_graph_sequence) {
            const bool argsort_route_weights_shape =
                ids->ne[0] > 0 && ids->ne[0] <= 16 && ids->ne[1] == sigmoid->ne[1] &&
                sigmoid->ne[0] > 0 && sigmoid->ne[0] <= 256 && sigmoid->ne[1] > 0 &&
                weights->type == GGML_TYPE_F32 && weights->ne[0] == 1 &&
                weights->ne[1] == ids->ne[0] && weights->ne[2] == ids->ne[1];
            if (!argsort_route_weights_shape) {
                ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "shape");
                return false;
            }

            const ggml_op ops[] = {
                GGML_OP_UNARY,
                GGML_OP_RESHAPE,
                GGML_OP_ADD,
                GGML_OP_ARGSORT,
                GGML_OP_VIEW,
                GGML_OP_MOE_ROUTE_WEIGHTS,
            };
            const int outputs[] = { graph_idx + 4, graph_idx + 5 };
            if (!ctx->can_fuse_graph_subgraph(graph_idx, ops, 6, outputs, 2)) {
                ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "can_fuse");
                return false;
            }

            ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "fused");
            fusion.logits  = sigmoid->src[0];
            fusion.ids     = ids;
            fusion.weights = weights;
            fusion.bias    = biased_probs->src[1];
            fusion.clamp   = weights;
            fusion.scale   = nullptr;
            fusion.n_fuse  = ctx->filtered_count_for_graph_span(idx, 6);
            return true;
        }
    }

    ggml_tensor * reshape_probs = ctx->graph_node(graph_idx + 1);
    ggml_tensor * biased_probs  = ctx->graph_node(graph_idx + 2);
    ggml_tensor * argsort       = ctx->graph_node(graph_idx + 3);
    ggml_tensor * ids           = ctx->graph_node(graph_idx + 4);
    ggml_tensor * get_rows      = ctx->graph_node(graph_idx + 5);
    ggml_tensor * weights_2d    = ctx->graph_node(graph_idx + 6);
    ggml_tensor * weights_sum   = ctx->graph_node(graph_idx + 7);
    ggml_tensor * clamp         = ctx->graph_node(graph_idx + 8);
    ggml_tensor * div           = ctx->graph_node(graph_idx + 9);
    ggml_tensor * weights_3d    = ctx->graph_node(graph_idx + 10);

    if (reshape_probs->op != GGML_OP_RESHAPE || reshape_probs->src[0] != sigmoid ||
            biased_probs->op != GGML_OP_ADD || biased_probs->src[0] != sigmoid ||
            biased_probs->src[1] == nullptr || biased_probs->src[1]->type != GGML_TYPE_F32 ||
            argsort->op != GGML_OP_ARGSORT || argsort->src[0] != biased_probs ||
            ids->op != GGML_OP_VIEW || ids->src[0] != argsort || ids->type != GGML_TYPE_I32 ||
            get_rows->op != GGML_OP_GET_ROWS || get_rows->src[0] != reshape_probs || get_rows->src[1] != ids ||
            weights_2d->op != GGML_OP_RESHAPE || weights_2d->src[0] != get_rows ||
            weights_sum->op != GGML_OP_SUM_ROWS || weights_sum->src[0] != weights_2d ||
            clamp->op != GGML_OP_CLAMP || clamp->src[0] != weights_sum ||
            div->op != GGML_OP_DIV || div->src[0] != weights_2d || div->src[1] != clamp ||
            weights_3d->op != GGML_OP_RESHAPE || weights_3d->src[0] != div) {
        ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "shape_or_sequence");
        return false;
    }

    ggml_tensor * weights = weights_3d;
    int graph_n_fuse = 11;
    ggml_tensor * scale = nullptr;
    if (graph_idx + 11 < ctx->graph_node_count()) {
        ggml_tensor * maybe_scale = ctx->graph_node(graph_idx + 11);
        if (maybe_scale->op == GGML_OP_SCALE && maybe_scale->src[0] == weights_3d) {
            weights = maybe_scale;
            scale = maybe_scale;
            graph_n_fuse = 12;
        }
    }

    if (ids->ne[0] <= 0 || ids->ne[0] > 16 || ids->ne[1] != sigmoid->ne[1] ||
            sigmoid->ne[0] > 256 || sigmoid->ne[1] <= 0 ||
            weights->type != GGML_TYPE_F32 || weights->ne[0] != 1 || weights->ne[1] != ids->ne[0] ||
            weights->ne[2] != ids->ne[1]) {
        ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "shape");
        return false;
    }

    const ggml_op ops_with_scale[] = {
        GGML_OP_UNARY,
        GGML_OP_RESHAPE,
        GGML_OP_ADD,
        GGML_OP_ARGSORT,
        GGML_OP_VIEW,
        GGML_OP_GET_ROWS,
        GGML_OP_RESHAPE,
        GGML_OP_SUM_ROWS,
        GGML_OP_CLAMP,
        GGML_OP_DIV,
        GGML_OP_RESHAPE,
        GGML_OP_SCALE,
    };
    const ggml_op ops_without_scale[] = {
        GGML_OP_UNARY,
        GGML_OP_RESHAPE,
        GGML_OP_ADD,
        GGML_OP_ARGSORT,
        GGML_OP_VIEW,
        GGML_OP_GET_ROWS,
        GGML_OP_RESHAPE,
        GGML_OP_SUM_ROWS,
        GGML_OP_CLAMP,
        GGML_OP_DIV,
        GGML_OP_RESHAPE,
    };
    const int outputs[] = { graph_idx + 4, graph_idx + graph_n_fuse - 1 };
    if (!ctx->can_fuse_graph_subgraph(
                graph_idx,
                scale ? ops_with_scale : ops_without_scale,
                graph_n_fuse,
                outputs,
                2)) {
        ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "can_fuse");
        return false;
    }

    ggml_metal_log_topk_moe_route_encode_candidate(ctx, idx, "fused");
    fusion.logits  = sigmoid->src[0];
    fusion.ids     = ids;
    fusion.weights = weights;
    fusion.bias    = biased_probs->src[1];
    fusion.clamp   = clamp;
    fusion.scale   = scale;
    fusion.n_fuse  = ctx->filtered_count_for_graph_span(idx, graph_n_fuse);
    return true;
}

static int ggml_metal_op_topk_moe_route_fused(ggml_metal_op_t ctx, int idx) {
    ggml_metal_topk_moe_route_fusion fusion;
    if (!ggml_metal_match_topk_moe_route_fusion(ctx, idx, fusion)) {
        return 0;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(uint64_t, nb_logits, fusion.logits,  nb);
    GGML_TENSOR_LOCALS(uint64_t, nb_bias,   fusion.bias,    nb);
    GGML_TENSOR_LOCALS(uint64_t, nb_ids,    fusion.ids,     nb);
    GGML_TENSOR_LOCALS(uint64_t, nb_weights,fusion.weights, nb);

    const bool route_weights_op = fusion.weights->op == GGML_OP_MOE_ROUTE_WEIGHTS;
    const float scale = fusion.scale ? ggml_get_op_params_f32(fusion.scale, 0) :
        (route_weights_op ? ggml_get_op_params_f32(fusion.weights, 1) : 1.0f);
    const float clamp_min = ggml_get_op_params_f32(fusion.clamp, 0);
    const int32_t norm = route_weights_op ? ggml_get_op_params_i32(fusion.weights, 2) != 0 : 1;

    ggml_metal_kargs_topk_moe_route args = {
        /*.n_expert    =*/ (int32_t) fusion.logits->ne[0],
        /*.n_tokens    =*/ (int32_t) fusion.logits->ne[1],
        /*.top_k       =*/ (int32_t) fusion.ids->ne[0],
        /*.has_bias    =*/ fusion.bias ? 1 : 0,
        /*.norm        =*/ norm,
        /*._pad0       =*/ ggml_metal_glm_dsa_moe_sort_route_ids_enabled() ? 1 : 0,
        /*._pad1       =*/ ggml_metal_glm_dsa_moe_max_active_experts(),
        /*._pad2       =*/ 0,
        /*.scale       =*/ scale,
        /*.clamp_min   =*/ clamp_min,
        /*.logits_nb0  =*/ nb_logits0,
        /*.logits_nb1  =*/ nb_logits1,
        /*.bias_nb0    =*/ nb_bias0,
        /*.ids_nb0     =*/ nb_ids0,
        /*.ids_nb1     =*/ nb_ids1,
        /*.weights_nb1 =*/ nb_weights1,
        /*.weights_nb2 =*/ nb_weights2,
    };

    const bool glm_256_8_sg32 =
        ggml_metal_glm_dsa_topk_moe_route_glm_256_8_sg32_enabled() &&
        fusion.logits->ne[0] == 256 &&
        fusion.ids->ne[0] == 8;
    const bool sg_reduce = !glm_256_8_sg32 && ggml_metal_glm_dsa_topk_moe_route_sg_reduce_enabled();
    auto pipeline = glm_256_8_sg32 ?
        ggml_metal_library_get_pipeline_topk_moe_route_glm_256_8_sg32(lib) :
        (sg_reduce ?
            ggml_metal_library_get_pipeline_topk_moe_route_sg_reduce(lib) :
            ggml_metal_library_get_pipeline_topk_moe_route(lib));

    ggml_metal_op_concurrency_reset(ctx);

    int ida = 0;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                         ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(fusion.logits),      ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(fusion.bias),        ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(fusion.ids),         ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(fusion.weights),     ida++);

    const int nth = glm_256_8_sg32 ? 32 : 256;
    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml_metal: moe_dispatch op=topk_moe_route_fused kernel=%s tensor=%s logits=%s ids=%s weights=%s graph_uid=%llu split_start=%d split_end=%d experts=%d tokens=%d top_k=%d sort_ids=%d fused_nodes=%d scale=%g grid_x=%d grid_y=1 grid_z=1 threads_x=%d\n",
            glm_256_8_sg32 ? "glm_256_8_sg32" : (sg_reduce ? "simdgroup_reduce" : "parallel_reduce"),
            ggml_metal_tensor_name(ctx->node(idx)),
            ggml_metal_tensor_name(fusion.logits),
            ggml_metal_tensor_name(fusion.ids),
            ggml_metal_tensor_name(fusion.weights),
            (unsigned long long) ctx->graph_uid(),
            ctx->split_start(),
            ctx->split_end(),
            args.n_expert,
            args.n_tokens,
            args.top_k,
            args._pad0,
            fusion.n_fuse,
            (double) args.scale,
            args.n_tokens,
            nth);
    }
    ggml_metal_encoder_dispatch_threadgroups(enc, args.n_tokens, 1, 1, nth, 1, 1);

    return fusion.n_fuse;
}

struct ggml_metal_glm_moe_decode_motif_reference {
    ggml_metal_topk_moe_route_fusion route;
    ggml_tensor * ids = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * glu = nullptr;
    ggml_tensor * weighted_down_input = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * out = nullptr;
    ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_up = nullptr;
    ggml_tensor * shared_glu = nullptr;
    ggml_tensor * shared_down = nullptr;
    ggml_tensor * final_out = nullptr;
    int shared_gate_offset = -1;
    int shared_up_offset = -1;
    int shared_glu_offset = -1;
    int shared_down_offset = -1;
    int final_out_offset = -1;
    int out_offset = -1;
    ggml_tensor * route_anchor_src = nullptr;
    int route_n_fuse = 0;
    int n_fuse = 0;
    bool has_route_anchor = false;
    bool has_weighted_down = false;
    bool has_shared_expert_tail = false;
    bool has_native_down = false;
    bool final_only_fusable = false;
};

struct ggml_metal_glm_moe_private_bindings {
    ggml_metal_buffer_id ids;
    ggml_metal_buffer_id weights;
    ggml_metal_buffer_id activation;
};

struct ggml_metal_routed_moe_decode_contract {
    const char * family = "generic";
    const ggml_tensor * ids = nullptr;
    const ggml_tensor * weights = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * up = nullptr;
    const ggml_tensor * glu = nullptr;
    const ggml_tensor * down = nullptr;
    const ggml_tensor * out = nullptr;
    const ggml_tensor * gate_w = nullptr;
    const ggml_tensor * up_w = nullptr;
    const ggml_tensor * down_w = nullptr;
    const ggml_tensor * cur = nullptr;
    int route_n_fuse = 0;
    int n_fuse = 0;
    bool has_route_anchor = false;
    bool has_weighted_down = false;
    bool has_shared_expert_tail = false;
    long long n_expert = 0;
    long long top_k = 0;
    long long n_tokens = 0;
    long long n_embd = 0;
    long long n_ff = 0;
    long long out_embd = 0;
    uint64_t selected_gate_weight_bytes = 0;
    uint64_t selected_up_weight_bytes = 0;
    uint64_t selected_down_weight_bytes = 0;
    uint64_t selected_weight_bytes = 0;
    uint64_t unfused_intermediate_bytes = 0;
    uint64_t current_fused_intermediate_bytes = 0;
    bool gate_up_pair_sg_shape = false;
    bool q2_weighted_reduce_shape = false;
    bool q3_weighted_reduce_shape = false;
};

static bool ggml_metal_make_routed_moe_decode_contract(
        const char * family,
        const ggml_metal_glm_moe_decode_motif_reference & motif,
        ggml_metal_routed_moe_decode_contract & contract) {
    const ggml_tensor * gate_w = motif.gate != nullptr ? motif.gate->src[0] : nullptr;
    const ggml_tensor * up_w   = motif.up   != nullptr ? motif.up->src[0]   : nullptr;
    const ggml_tensor * down_w = motif.down != nullptr ? motif.down->src[0] : nullptr;
    const ggml_tensor * cur    = motif.gate != nullptr ? motif.gate->src[1] : nullptr;
    if (gate_w == nullptr || up_w == nullptr || down_w == nullptr ||
            cur == nullptr || motif.ids == nullptr || motif.weights == nullptr ||
            motif.gate == nullptr || motif.up == nullptr || motif.glu == nullptr ||
            motif.down == nullptr || motif.out == nullptr) {
        return false;
    }

    contract.family = family;
    contract.ids = motif.ids;
    contract.weights = motif.weights;
    contract.gate = motif.gate;
    contract.up = motif.up;
    contract.glu = motif.glu;
    contract.down = motif.down;
    contract.out = motif.out;
    contract.gate_w = gate_w;
    contract.up_w = up_w;
    contract.down_w = down_w;
    contract.cur = cur;
    contract.route_n_fuse = motif.route_n_fuse;
    contract.n_fuse = motif.n_fuse;
    contract.has_route_anchor = motif.has_route_anchor;
    contract.has_weighted_down = motif.has_weighted_down;
    contract.has_shared_expert_tail = motif.has_shared_expert_tail;
    contract.top_k = motif.ids->ne[0];
    contract.n_tokens = motif.ids->ne[1];
    contract.n_expert = gate_w->ne[2];
    contract.n_embd = cur->ne[0];
    contract.n_ff = motif.glu->ne[0];
    contract.out_embd = motif.out->ne[0];
    contract.selected_gate_weight_bytes =
        uint64_t(contract.top_k) * uint64_t(contract.n_ff) *
        ggml_row_size(gate_w->type, contract.n_embd);
    contract.selected_up_weight_bytes =
        uint64_t(contract.top_k) * uint64_t(contract.n_ff) *
        ggml_row_size(up_w->type, contract.n_embd);
    contract.selected_down_weight_bytes =
        uint64_t(contract.top_k) * uint64_t(contract.out_embd) *
        ggml_row_size(down_w->type, contract.n_ff);
    contract.selected_weight_bytes =
        contract.selected_gate_weight_bytes +
        contract.selected_up_weight_bytes +
        contract.selected_down_weight_bytes;
    const uint64_t slot_activation_bytes =
        uint64_t(contract.top_k) * uint64_t(contract.n_tokens) *
        uint64_t(contract.n_ff) * sizeof(float);
    const uint64_t out_activation_bytes =
        uint64_t(contract.n_tokens) * uint64_t(contract.out_embd) * sizeof(float);
    contract.unfused_intermediate_bytes =
        3 * slot_activation_bytes + out_activation_bytes;
    contract.current_fused_intermediate_bytes =
        slot_activation_bytes + out_activation_bytes;
    contract.gate_up_pair_sg_shape =
        gate_w->type == GGML_TYPE_Q2_K &&
        up_w->type == GGML_TYPE_Q2_K &&
        contract.n_embd == 6144 &&
        contract.n_ff == 2048 &&
        contract.top_k == 8 &&
        contract.n_tokens == 1;
    contract.q2_weighted_reduce_shape =
        down_w->type == GGML_TYPE_Q2_K &&
        down_w->ne[0] == contract.n_ff &&
        down_w->ne[1] == contract.out_embd &&
        contract.top_k == 8 &&
        contract.n_tokens == 1;
    contract.q3_weighted_reduce_shape =
        down_w->type == GGML_TYPE_Q3_K &&
        down_w->ne[0] == contract.n_ff &&
        down_w->ne[1] == contract.out_embd &&
        contract.top_k == 8 &&
        contract.n_tokens == 1;
    return true;
}

static bool ggml_metal_glm_moe_can_scan_q2_selected_weights(
        const ggml_metal_glm_moe_decode_motif_reference & motif,
        int chunks_per_expert,
        int storage_block_bytes) {
    ggml_metal_routed_moe_decode_contract contract;
    if (chunks_per_expert <= 0 ||
            !ggml_metal_make_routed_moe_decode_contract("glm_dsa", motif, contract)) {
        return false;
    }

    const uint64_t vector_bytes = 4*sizeof(uint32_t);
    const uint64_t q2_k_block_bytes = ggml_type_size(GGML_TYPE_Q2_K);
    const auto stored_bytes = [storage_block_bytes, q2_k_block_bytes](uint64_t bytes, ggml_type type) {
        if (type != GGML_TYPE_Q2_K) {
            return bytes;
        }
        GGML_ASSERT(bytes % q2_k_block_bytes == 0);
        return bytes/q2_k_block_bytes*uint64_t(storage_block_bytes);
    };
    const uint64_t gate_expert_bytes = stored_bytes(
        contract.selected_gate_weight_bytes/uint64_t(contract.top_k), contract.gate_w->type);
    const uint64_t up_expert_bytes = stored_bytes(
        contract.selected_up_weight_bytes/uint64_t(contract.top_k), contract.up_w->type);
    const uint64_t down_expert_bytes = stored_bytes(
        contract.selected_down_weight_bytes/uint64_t(contract.top_k), contract.down_w->type);

    return contract.gate_up_pair_sg_shape &&
        (contract.q2_weighted_reduce_shape || contract.q3_weighted_reduce_shape) &&
        storage_block_bytes >= int(q2_k_block_bytes) &&
        storage_block_bytes <= 256 &&
        storage_block_bytes % 4 == 0 &&
        (storage_block_bytes == int(q2_k_block_bytes) || contract.n_expert >= 3*contract.top_k + 2) &&
        ggml_is_contiguous(contract.gate_w) &&
        ggml_is_contiguous(contract.up_w) &&
        ggml_is_contiguous(contract.down_w) &&
        gate_expert_bytes % vector_bytes == 0 &&
        up_expert_bytes % vector_bytes == 0 &&
        down_expert_bytes % vector_bytes == 0 &&
        ggml_nelements(contract.out) >= contract.top_k*chunks_per_expert;
}

static int ggml_metal_op_glm_moe_q2_selected_weight_scan(
        ggml_metal_op_t ctx,
        const ggml_metal_glm_moe_decode_motif_reference & motif,
        int chunks_per_expert,
        int storage_block_bytes) {
    GGML_ASSERT(ggml_metal_glm_moe_can_scan_q2_selected_weights(
        motif, chunks_per_expert, storage_block_bytes));

    ggml_metal_routed_moe_decode_contract contract;
    GGML_ASSERT(ggml_metal_make_routed_moe_decode_contract("glm_dsa", motif, contract));

    const uint64_t top_k = uint64_t(contract.top_k);
    const uint64_t q2_k_block_bytes = ggml_type_size(GGML_TYPE_Q2_K);
    const auto stored_bytes = [storage_block_bytes, q2_k_block_bytes](uint64_t bytes, ggml_type type) {
        if (type != GGML_TYPE_Q2_K) {
            return bytes;
        }
        GGML_ASSERT(bytes % q2_k_block_bytes == 0);
        return bytes/q2_k_block_bytes*uint64_t(storage_block_bytes);
    };
    ggml_metal_kargs_glm_moe_q2_weight_scan args = {
        /*.gate_expert_bytes  =*/ stored_bytes(contract.selected_gate_weight_bytes/top_k, contract.gate_w->type),
        /*.gate_expert_stride =*/ contract.gate_w->nb[2],
        /*.up_expert_bytes    =*/ stored_bytes(contract.selected_up_weight_bytes/top_k, contract.up_w->type),
        /*.up_expert_stride   =*/ contract.up_w->nb[2],
        /*.down_expert_bytes  =*/ stored_bytes(contract.selected_down_weight_bytes/top_k, contract.down_w->type),
        /*.down_expert_stride =*/ contract.down_w->nb[2],
        /*.ids_nb1            =*/ contract.ids->nb[1],
        /*.top_k              =*/ int32_t(contract.top_k),
        /*.n_tokens           =*/ int32_t(contract.n_tokens),
        /*.n_experts          =*/ int32_t(contract.n_expert),
        /*.chunks_per_expert  =*/ chunks_per_expert,
        /*.storage_block_bytes=*/ storage_block_bytes,
    };

    auto pipeline = ggml_metal_library_get_pipeline_glm_moe_q2_selected_weight_scan(ctx->lib);
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(contract.gate_w), 1);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(contract.up_w), 2);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(contract.down_w), 3);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(contract.ids), 4);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(contract.out), 5);
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(
        ctx->enc,
        chunks_per_expert,
        int(contract.top_k),
        int(contract.n_tokens),
        256,
        1,
        1);
    ggml_metal_op_concurrency_reset(ctx);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml_metal: moe_dispatch op=glm_moe_q2_selected_weight_roofline tensor=%s selected_weight_bytes=%llu storage_block_bytes=%d chunks_per_expert=%d threadgroups=%lld threads_per_threadgroup=256\n",
            ggml_metal_tensor_name(contract.out),
            (unsigned long long) (stored_bytes(contract.selected_gate_weight_bytes, contract.gate_w->type) +
                                  stored_bytes(contract.selected_up_weight_bytes, contract.up_w->type) +
                                  stored_bytes(contract.selected_down_weight_bytes, contract.down_w->type)),
            storage_block_bytes,
            chunks_per_expert,
            contract.top_k*contract.n_tokens*chunks_per_expert);
    }

    return motif.n_fuse;
}

static void ggml_metal_log_routed_moe_decode_motif_contract(
        ggml_metal_op_t ctx,
        const ggml_metal_glm_moe_decode_motif_reference & motif) {
    if (!ggml_metal_glm_dsa_dispatch_log_enabled()) {
        return;
    }

    ggml_metal_routed_moe_decode_contract contract;
    if (!ggml_metal_make_routed_moe_decode_contract("glm_dsa", motif, contract)) {
        return;
    }

    const double weight_to_current_intermediate =
        contract.current_fused_intermediate_bytes == 0 ? 0.0 :
            double(contract.selected_weight_bytes) / double(contract.current_fused_intermediate_bytes);

    GGML_LOG_INFO(
            "ggml_metal: moe_dispatch op=routed_moe_decode_motif_contract family=%s tensor=%s ids=%s weights=%s gate=%s up=%s glu=%s down=%s out=%s graph_uid=%llu route_nodes=%d motif_nodes=%d route_anchor=%d weighted_down=%d shared_expert_tail=%d gate_type=%s up_type=%s down_type=%s cur_type=%s out_type=%s experts=%lld top_k=%lld tokens=%lld embd=%lld ff=%lld out_embd=%lld selected_weight_bytes=%llu selected_gate_weight_bytes=%llu selected_up_weight_bytes=%llu selected_down_weight_bytes=%llu unfused_intermediate_bytes=%llu current_fused_intermediate_bytes=%llu weight_to_current_intermediate=%.2f gate_up_pair_sg_shape=%d q2_weighted_reduce_shape=%d q3_weighted_reduce_shape=%d split_start=%d split_end=%d\n",
            contract.family,
            ggml_metal_tensor_name(contract.out),
            ggml_metal_tensor_name(contract.ids),
            ggml_metal_tensor_name(contract.weights),
            ggml_metal_tensor_name(contract.gate),
            ggml_metal_tensor_name(contract.up),
            ggml_metal_tensor_name(contract.glu),
            ggml_metal_tensor_name(contract.down),
            ggml_metal_tensor_name(contract.out),
            (unsigned long long) ctx->graph_uid(),
            contract.route_n_fuse,
            contract.n_fuse,
            contract.has_route_anchor ? 1 : 0,
            contract.has_weighted_down ? 1 : 0,
            contract.has_shared_expert_tail ? 1 : 0,
            ggml_type_name(contract.gate_w->type),
            ggml_type_name(contract.up_w->type),
            ggml_type_name(contract.down_w->type),
            ggml_type_name(contract.cur->type),
            ggml_type_name(contract.out->type),
            contract.n_expert,
            contract.top_k,
            contract.n_tokens,
            contract.n_embd,
            contract.n_ff,
            contract.out_embd,
            (unsigned long long) contract.selected_weight_bytes,
            (unsigned long long) contract.selected_gate_weight_bytes,
            (unsigned long long) contract.selected_up_weight_bytes,
            (unsigned long long) contract.selected_down_weight_bytes,
            (unsigned long long) contract.unfused_intermediate_bytes,
            (unsigned long long) contract.current_fused_intermediate_bytes,
            weight_to_current_intermediate,
            contract.gate_up_pair_sg_shape ? 1 : 0,
            contract.q2_weighted_reduce_shape ? 1 : 0,
            contract.q3_weighted_reduce_shape ? 1 : 0,
            ctx->split_start(),
            ctx->split_end());
}

static bool ggml_metal_match_glm_moe_decode_motif_reference(
        ggml_metal_op_t ctx,
        int idx,
        ggml_metal_glm_moe_decode_motif_reference & motif) {
    const bool enabled =
        ggml_metal_glm_dsa_moe_decode_motif_reference_enabled() ||
        ggml_metal_glm_dsa_moe_private_scratch_enabled() ||
        ggml_metal_glm_dsa_moe_two_phase_enabled() ||
        ggml_metal_glm_dsa_moe_dual_lane_enabled();
    if (!ctx->use_fusion || !enabled) {
        return false;
    }

    ggml_metal_topk_moe_route_fusion route;
    ggml_tensor * route_weights = ctx->node(idx);
    if (route_weights->op == GGML_OP_MOE_ROUTE_WEIGHTS &&
            route_weights->src[0] != nullptr &&
            route_weights->src[1] != nullptr &&
            route_weights->src[0]->type == GGML_TYPE_F32 &&
            route_weights->src[1]->type == GGML_TYPE_I32 &&
            route_weights->type == GGML_TYPE_F32 &&
            route_weights->ne[0] == 1 &&
            route_weights->ne[1] == route_weights->src[1]->ne[0] &&
            route_weights->ne[2] == route_weights->src[1]->ne[1]) {
        route.ids = route_weights->src[1];
        route.weights = route_weights;
        route.n_fuse = 1;
    } else if (!ggml_metal_match_topk_moe_route_fusion(ctx, idx, route)) {
        return false;
    }

    int cursor = idx + route.n_fuse;
    if (cursor >= ctx->n_nodes()) {
        return false;
    }

    bool has_route_anchor = false;
    if (cursor + 4 < ctx->n_nodes() &&
            ctx->node(cursor + 0)->op == GGML_OP_SUM &&
            ctx->node(cursor + 1)->op == GGML_OP_SCALE &&
            ctx->node(cursor + 2)->op == GGML_OP_REPEAT &&
            ctx->node(cursor + 3)->op == GGML_OP_ADD &&
            ggml_metal_tensor_name_contains(ctx->node(cursor + 0), "ffn_moe_route_anchor_sum") &&
            ggml_metal_tensor_name_contains(ctx->node(cursor + 3), "ffn_moe_cur_route_anchored")) {
        ggml_tensor * anchor_sum = ctx->node(cursor + 0);
        ggml_tensor * anchor_zero = ctx->node(cursor + 1);
        ggml_tensor * anchor_repeat = ctx->node(cursor + 2);
        ggml_tensor * anchor_cur = ctx->node(cursor + 3);
        if (anchor_sum->src[0] != route.weights ||
                anchor_zero->src[0] != anchor_sum ||
                anchor_repeat->src[0] != anchor_zero ||
                (anchor_cur->src[0] != anchor_repeat && anchor_cur->src[1] != anchor_repeat)) {
            return false;
        }
        motif.route_anchor_src = anchor_cur->src[0] == anchor_repeat ? anchor_cur->src[1] : anchor_cur->src[0];
        has_route_anchor = true;
        cursor += 4;
    }

    if (cursor + 5 > ctx->n_nodes()) {
        return false;
    }

    ggml_tensor * gate = ctx->node(cursor + 0);
    ggml_tensor * up   = ctx->node(cursor + 1);
    ggml_tensor * glu  = ctx->node(cursor + 2);
    if (gate->op != GGML_OP_MUL_MAT_ID ||
            up->op != GGML_OP_MUL_MAT_ID ||
            glu->op != GGML_OP_GLU ||
            !ggml_metal_tensor_name_contains(gate, "ffn_moe_gate") ||
            !ggml_metal_tensor_name_contains(up, "ffn_moe_up") ||
            !ggml_metal_tensor_name_contains(glu, "ffn_moe_swiglu") ||
            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
            ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }

    if (gate->src[1] != up->src[1] ||
            gate->src[2] != route.ids ||
            up->src[2] != route.ids ||
            gate->type != GGML_TYPE_F32 ||
            up->type != GGML_TYPE_F32 ||
            glu->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(gate, up) ||
            !ggml_are_same_shape(gate, glu)) {
        return false;
    }

    if (glu->src[0] != gate || glu->src[1] != up) {
        return false;
    }

    cursor += 3;

    bool has_weighted_down = false;
    ggml_tensor * down_input = glu;
    if (cursor < ctx->n_nodes() &&
            ctx->node(cursor)->op == GGML_OP_MUL &&
            ggml_metal_tensor_name_contains(ctx->node(cursor), "ffn_moe_down_weighted_input")) {
        ggml_tensor * weighted = ctx->node(cursor);
        const bool weighted_sources_ok =
            (weighted->src[0] == glu && weighted->src[1] == route.weights) ||
            (weighted->src[0] == route.weights && weighted->src[1] == glu);
        if (!weighted_sources_ok || weighted->type != GGML_TYPE_F32) {
            return false;
        }
        has_weighted_down = true;
        down_input = weighted;
        cursor++;
    }
    if (cursor < ctx->n_nodes() &&
            ctx->node(cursor)->op == GGML_OP_CPY &&
            ctx->node(cursor)->src[0] == glu &&
            ctx->node(cursor)->type == GGML_TYPE_F16 &&
            ggml_metal_tensor_name_contains(ctx->node(cursor), "ffn_moe_swiglu_f16")) {
        down_input = ctx->node(cursor);
        cursor++;
    }

    if (ggml_metal_glm_dsa_moe_private_scratch_enabled()) {
        ggml_tensor * native_down = nullptr;
        ggml_tensor * shared_gate = nullptr;
        ggml_tensor * shared_up = nullptr;
        int native_down_offset = -1;
        int shared_gate_offset = -1;
        int shared_up_offset = -1;

        for (int rel = 0; rel < 4 && cursor + rel < ctx->n_nodes(); ++rel) {
            ggml_tensor * candidate = ctx->node(cursor + rel);
            if (candidate->op == GGML_OP_MOE_MUL_MAT_ID) {
                native_down = candidate;
                native_down_offset = rel;
            } else if (candidate->op == GGML_OP_MUL_MAT &&
                    ggml_metal_tensor_is_shared_expert_gate(candidate)) {
                shared_gate = candidate;
                shared_gate_offset = rel;
            } else if (candidate->op == GGML_OP_MUL_MAT &&
                    ggml_metal_tensor_is_shared_expert_up(candidate)) {
                shared_up = candidate;
                shared_up_offset = rel;
            }
        }

        if (native_down == nullptr || shared_gate == nullptr || shared_up == nullptr ||
                native_down->src[0] == nullptr || native_down->src[1] == nullptr ||
                native_down->src[2] == nullptr || native_down->src[3] == nullptr) {
            return false;
        }

        const bool native_shape_ok =
            !has_weighted_down &&
            native_down->src[0]->type == GGML_TYPE_Q3_K &&
            native_down->src[1] == glu &&
            native_down->src[2] == route.ids &&
            native_down->src[3] == route.weights &&
            native_down->type == GGML_TYPE_F32 &&
            native_down->src[0]->ne[0] == 2048 &&
            native_down->src[0]->ne[1] == 6144 &&
            native_down->src[0]->ne[2] == 256 &&
            native_down->src[1]->type == GGML_TYPE_F32 &&
            native_down->src[1]->ne[0] == 2048 &&
            native_down->src[1]->ne[1] == 8 &&
            native_down->src[1]->ne[2] == 1 &&
            native_down->src[2]->type == GGML_TYPE_I32 &&
            native_down->src[2]->ne[0] == 8 &&
            native_down->src[2]->ne[1] == 1 &&
            native_down->src[3]->type == GGML_TYPE_F32 &&
            native_down->ne[0] == 6144 &&
            native_down->ne[1] == 1 &&
            shared_gate->type == GGML_TYPE_F32 &&
            shared_up->type == GGML_TYPE_F32;
        if (!native_shape_ok) {
            return false;
        }

        const int native_offset = cursor + native_down_offset - idx;
        const int native_shared_gate_offset = cursor + shared_gate_offset - idx;
        const int native_shared_up_offset = cursor + shared_up_offset - idx;
        const int n_fuse = std::max({
            native_offset,
            native_shared_gate_offset,
            native_shared_up_offset,
        }) + 1;

        std::vector<ggml_op> ops;
        ops.reserve(n_fuse);
        for (int rel = 0; rel < n_fuse; ++rel) {
            ops.push_back(ctx->node(idx + rel)->op);
        }
        const int outputs[] = {
            native_offset,
            native_shared_gate_offset,
            native_shared_up_offset,
        };
        if (!ctx->can_fuse_subgraph(idx, ops.data(), n_fuse, outputs, 3)) {
            return false;
        }

        motif.route = route;
        motif.ids = route.ids;
        motif.weights = route.weights;
        motif.gate = gate;
        motif.up = up;
        motif.glu = glu;
        motif.down = native_down;
        motif.out = native_down;
        motif.shared_gate = shared_gate;
        motif.shared_up = shared_up;
        motif.shared_gate_offset = native_shared_gate_offset;
        motif.shared_up_offset = native_shared_up_offset;
        motif.out_offset = native_offset;
        motif.route_anchor_src = has_route_anchor ? motif.route_anchor_src : gate->src[1];
        motif.route_n_fuse = route.n_fuse;
        motif.n_fuse = n_fuse;
        motif.has_route_anchor = has_route_anchor;
        motif.has_weighted_down = false;
        motif.has_shared_expert_tail = true;
        motif.has_native_down = true;
        return true;
    }

    if (cursor + 2 > ctx->n_nodes()) {
        return false;
    }

    ggml_tensor * down = ctx->node(cursor + 0);
    ggml_tensor * out  = nullptr;
    ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_up = nullptr;
    int out_offset = -1;
    int shared_gate_offset = -1;
    int shared_up_offset = -1;

    for (int rel = 1; rel < 4 && cursor + rel < ctx->n_nodes(); ++rel) {
        ggml_tensor * candidate = ctx->node(cursor + rel);
        if (candidate->op == GGML_OP_MOE_WEIGHTED_SUM &&
                candidate->src[0] == down &&
                ggml_metal_tensor_name_contains(candidate, "ffn_moe_out")) {
            out = candidate;
            out_offset = rel;
        } else if (candidate->op == GGML_OP_MUL_MAT &&
                ggml_metal_tensor_is_shared_expert_gate(candidate)) {
            shared_gate = candidate;
            shared_gate_offset = rel;
        } else if (candidate->op == GGML_OP_MUL_MAT &&
                ggml_metal_tensor_is_shared_expert_up(candidate)) {
            shared_up = candidate;
            shared_up_offset = rel;
        }
    }

    if ((shared_gate == nullptr) != (shared_up == nullptr)) {
        return false;
    }
    const bool has_shared_expert_tail = shared_gate != nullptr;
    const int tail_n_fuse = std::max({
        out_offset,
        shared_gate_offset,
        shared_up_offset,
    }) + 1;
    if (down->op != GGML_OP_MUL_MAT_ID ||
            out == nullptr ||
            !ggml_metal_tensor_name_contains(down, "ffn_moe_down") ||
            !ggml_metal_tensor_name_contains(out, "ffn_moe_out") ||
            down->src[1] != down_input ||
            down->src[2] != route.ids ||
            out->src[0] != down ||
            out->src[1] != route.weights ||
            down->type != GGML_TYPE_F32 ||
            out->type != GGML_TYPE_F32 ||
            (has_shared_expert_tail && (shared_gate->type != GGML_TYPE_F32 || shared_up->type != GGML_TYPE_F32))) {
        return false;
    }

    const bool expert_types_ok =
        ggml_metal_tensor_is_glm_dsa_qk_moe_expert(gate->src[0]) &&
        ggml_metal_tensor_is_glm_dsa_qk_moe_expert(up->src[0]) &&
        ggml_metal_tensor_is_glm_dsa_qk_moe_expert(down->src[0]);
    const bool shape_ok =
        expert_types_ok &&
        gate->src[0]->ne[2] == down->src[0]->ne[2] &&
        gate->src[2]->ne[0] == down->src[2]->ne[0] &&
        gate->src[2]->ne[1] == down->src[2]->ne[1] &&
        out->ne[0] == down->ne[0] &&
        out->ne[1] == down->ne[2];
    if (!shape_ok) {
        return false;
    }

    motif.route = route;
    motif.ids = route.ids;
    motif.weights = route.weights;
    motif.gate = gate;
    motif.up = up;
    motif.glu = glu;
    motif.weighted_down_input = has_weighted_down ? down_input : nullptr;
    motif.down = down;
    motif.out = out;
    motif.shared_gate = shared_gate;
    motif.shared_up = shared_up;
    motif.shared_gate_offset = has_shared_expert_tail ? cursor + shared_gate_offset - idx : -1;
    motif.shared_up_offset = has_shared_expert_tail ? cursor + shared_up_offset - idx : -1;
    motif.out_offset = cursor + out_offset - idx;
    if (!has_route_anchor) {
        motif.route_anchor_src = gate->src[1];
    }
    motif.route_n_fuse = route.n_fuse;
    motif.n_fuse = cursor + tail_n_fuse - idx;
    motif.has_route_anchor = has_route_anchor;
    motif.has_weighted_down = has_weighted_down;
    motif.has_shared_expert_tail = has_shared_expert_tail;
    return true;
}

static bool ggml_metal_extend_glm_moe_two_phase(
        ggml_metal_op_t ctx,
        int idx,
        ggml_metal_glm_moe_decode_motif_reference & motif) {
    if ((!ggml_metal_glm_dsa_moe_two_phase_enabled() &&
         !ggml_metal_glm_dsa_moe_dual_lane_enabled()) ||
            !motif.has_shared_expert_tail || motif.has_native_down ||
            motif.shared_gate == nullptr || motif.shared_up == nullptr ||
            motif.route_anchor_src == nullptr ||
            idx + motif.n_fuse + 3 > ctx->n_nodes()) {
        return false;
    }

    const int shared_glu_offset = motif.n_fuse;
    const int shared_down_offset = shared_glu_offset + 1;
    const int final_out_offset = shared_down_offset + 1;
    ggml_tensor * shared_glu = ctx->node(idx + shared_glu_offset);
    ggml_tensor * shared_down = ctx->node(idx + shared_down_offset);
    ggml_tensor * final_out = ctx->node(idx + final_out_offset);

    const bool shared_pair_ok =
        (shared_glu->src[0] == motif.shared_gate && shared_glu->src[1] == motif.shared_up) ||
        (shared_glu->src[0] == motif.shared_up && shared_glu->src[1] == motif.shared_gate);
    const bool final_pair_ok =
        (final_out->src[0] == motif.out && final_out->src[1] == shared_down) ||
        (final_out->src[0] == shared_down && final_out->src[1] == motif.out);
    const bool shape_ok =
        motif.gate->src[0]->type == GGML_TYPE_Q2_K &&
        motif.up->src[0]->type == GGML_TYPE_Q2_K &&
        motif.down->src[0]->type == GGML_TYPE_Q3_K &&
        motif.shared_gate->src[0]->type == GGML_TYPE_Q4_K &&
        motif.shared_up->src[0]->type == GGML_TYPE_Q4_K &&
        shared_down->src[0]->type == GGML_TYPE_Q4_K &&
        motif.route_anchor_src->type == GGML_TYPE_F32 &&
        motif.route_anchor_src->ne[0] == 6144 &&
        motif.route_anchor_src->ne[1] == 1 &&
        motif.ids->ne[0] == 8 && motif.ids->ne[1] == 1 &&
        motif.glu->ne[0] == 2048 && motif.glu->ne[1] == 8 && motif.glu->ne[2] == 1 &&
        shared_glu->ne[0] == 2048 && shared_glu->ne[1] == 1 &&
        shared_down->ne[0] == 6144 && shared_down->ne[1] == 1 &&
        final_out->ne[0] == 6144 && final_out->ne[1] == 1;
    if (shared_glu->op != GGML_OP_GLU ||
            ggml_get_glu_op(shared_glu) != GGML_GLU_OP_SWIGLU ||
            ggml_get_op_params_i32(shared_glu, 1) != 0 ||
            !shared_pair_ok ||
            shared_down->op != GGML_OP_MUL_MAT ||
            shared_down->src[1] != shared_glu ||
            final_out->op != GGML_OP_ADD ||
            !final_pair_ok ||
            !shape_ok) {
        return false;
    }

    motif.shared_glu = shared_glu;
    motif.shared_down = shared_down;
    motif.final_out = final_out;
    motif.shared_glu_offset = shared_glu_offset;
    motif.shared_down_offset = shared_down_offset;
    motif.final_out_offset = final_out_offset;
    motif.n_fuse += 3;

    std::vector<ggml_op> ops;
    ops.reserve(motif.n_fuse);
    for (int rel = 0; rel < motif.n_fuse; ++rel) {
        ops.push_back(ctx->node(idx + rel)->op);
    }
    const int final_only_output[] = { motif.final_out_offset };
    motif.final_only_fusable =
        ctx->can_fuse_subgraph(idx, ops.data(), motif.n_fuse, final_only_output, 1) ||
        ctx->can_fuse_filtered_subgraph(idx, ops.data(), motif.n_fuse, final_only_output, 1);
    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml_metal: moe_dispatch op=glm_moe_dual_lane_contract tensor=%s fused_nodes=%d final_only_fusable=%d routed_use_count=%d final_use_count=%d\n",
            ggml_metal_tensor_name(motif.final_out),
            motif.n_fuse,
            motif.final_only_fusable ? 1 : 0,
            ctx->graph_node_use_count(ctx->graph_index(idx + motif.out_offset)),
            ctx->graph_node_use_count(ctx->graph_index(idx + motif.final_out_offset)));
    }
    return true;
}

static int ggml_metal_op_reference_dispatch_existing_node(ggml_metal_op_t ctx, int idx) {
    switch (ctx->node(idx)->op) {
        case GGML_OP_SUM:
            return ggml_metal_op_sum(ctx, idx);
        case GGML_OP_SCALE:
        case GGML_OP_UNARY:
        case GGML_OP_CLAMP:
            return ggml_metal_op_unary(ctx, idx);
        case GGML_OP_REPEAT:
            return ggml_metal_op_repeat(ctx, idx);
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
            return ggml_metal_op_bin(ctx, idx);
        case GGML_OP_MUL_MAT_ID:
            return ggml_metal_op_mul_mat_id(ctx, idx);
        case GGML_OP_MUL_MAT:
            return ggml_metal_op_mul_mat(ctx, idx);
        case GGML_OP_GLU:
            return ggml_metal_op_glu(ctx, idx);
        case GGML_OP_MOE_WEIGHTED_SUM:
            return ggml_metal_op_moe_weighted_sum(ctx, idx);
        default:
            return 0;
    }
}

static const ggml_tensor * ggml_metal_view_root(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static bool ggml_metal_pair_is(
        const ggml_tensor * pair,
        const ggml_tensor * lhs,
        const ggml_tensor * rhs) {
    return pair != nullptr &&
        ((ggml_metal_view_root(pair->src[0]) == ggml_metal_view_root(lhs) &&
          ggml_metal_view_root(pair->src[1]) == ggml_metal_view_root(rhs)) ||
         (ggml_metal_view_root(pair->src[0]) == ggml_metal_view_root(rhs) &&
          ggml_metal_view_root(pair->src[1]) == ggml_metal_view_root(lhs)));
}

static int ggml_metal_op_glm_absorbed_q(ggml_metal_op_t ctx, int idx) {
    constexpr int n_fuse = 5;
    if (!ctx->use_fusion || !ggml_metal_glm_dsa_absorbed_qkv_phases_enabled() ||
            idx + n_fuse > ctx->n_nodes()) {
        return 0;
    }

    ggml_tensor * q      = ctx->node(idx + 0);
    ggml_tensor * q_abs  = ctx->node(idx + 1);
    ggml_tensor * q_rope = ctx->node(idx + 2);
    ggml_tensor * kv     = ctx->node(idx + 3);
    ggml_tensor * q_pack = ctx->node(idx + 4);
    const int rope_mode = ((const int32_t *) q_rope->op_params)[2];

    const bool ops_ok =
        q->op == GGML_OP_MUL_MAT && q_abs->op == GGML_OP_MUL_MAT &&
        q_rope->op == GGML_OP_ROPE && kv->op == GGML_OP_MUL_MAT &&
        q_pack->op == GGML_OP_CONCAT;
    const bool q_shape_ok =
        q->type == GGML_TYPE_F32 && ggml_is_contiguous(q) &&
        q->ne[0] == 16384 && q->ne[1] == 1 &&
        q->src[0] != nullptr && q->src[0]->type == GGML_TYPE_Q8_0 &&
        q->src[0]->ne[0] == 2048 && q->src[0]->ne[1] == 16384 &&
        q->src[1] != nullptr && q->src[1]->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(q->src[1]) && q->src[1]->ne[0] == 2048;
    const bool q_abs_shape_ok =
        q_abs->type == GGML_TYPE_F32 && ggml_is_contiguous(q_abs) &&
        q_abs->ne[0] == 512 && q_abs->ne[1] == 1 && q_abs->ne[2] == 64 &&
        q_abs->src[0] != nullptr && q_abs->src[0]->type == GGML_TYPE_Q4_0 &&
        q_abs->src[0]->ne[0] == 192 && q_abs->src[0]->ne[1] == 512 &&
        q_abs->src[0]->ne[2] == 64 && q_abs->src[1] != nullptr;
    const bool rope_shape_ok =
        q_rope->type == GGML_TYPE_F32 && ggml_is_contiguous(q_rope) &&
        q_rope->ne[0] == 64 && q_rope->ne[1] == 64 &&
        q_rope->src[1] != nullptr && q_rope->src[1]->type == GGML_TYPE_I32 &&
        q_rope->src[1]->ne[0] == 1 && q_rope->src[2] == nullptr &&
        (rope_mode == GGML_ROPE_TYPE_NORMAL || rope_mode == GGML_ROPE_TYPE_NEOX);
    const bool kv_shape_ok =
        kv->type == GGML_TYPE_F32 && ggml_is_contiguous(kv) &&
        kv->ne[0] == 576 && kv->ne[1] == 1;
    const bool pack_shape_ok =
        q_pack->type == GGML_TYPE_F32 && ggml_is_contiguous(q_pack) &&
        q_pack->ne[0] == 576 && q_pack->ne[1] == 64;
    const bool dependencies =
        ggml_metal_view_root(q_abs->src[1]) == q &&
        ggml_metal_view_root(q_rope->src[0]) == q &&
        ggml_metal_view_root(q_pack->src[0]) == q_abs &&
        ggml_metal_view_root(q_pack->src[1]) == q_rope &&
        ggml_metal_tensor_name_contains(q, "q-") &&
        ggml_metal_tensor_name_contains(q->src[0], "attn_q_b.weight") &&
        ggml_metal_tensor_name_contains(q_abs, "q_nope_absorbed") &&
        ggml_metal_tensor_name_contains(q_abs->src[0], "attn_k_b.weight") &&
        ggml_metal_tensor_name_contains(q_rope, "q_pe") &&
        ggml_metal_tensor_name_contains(kv, "kv_cmpr_pe") &&
        ggml_metal_tensor_name_contains(q_pack, "Qcur");
    if (!ops_ok || !q_shape_ok || !q_abs_shape_ok || !rope_shape_ok ||
            !kv_shape_ok || !pack_shape_ok || !dependencies) {
        return 0;
    }

    ggml_metal_kargs_glm_absorbed_q fused_args = {
        /*.q_b_nb1    =*/ q->src[0]->nb[1],
        /*.wk_b_nb1   =*/ q_abs->src[0]->nb[1],
        /*.wk_b_nb2   =*/ q_abs->src[0]->nb[2],
        /*.q_rank     =*/ 2048,
        /*.q_head_dim =*/ 256,
        /*.q_nope_dim =*/ 192,
        /*.q_abs_dim  =*/ 512,
        /*.rope_dim   =*/ 64,
        /*.n_head     =*/ 64,
        /*.rope_mode  =*/ rope_mode,
        /*._pad0      =*/ 0,
    };

    const int n_past = ((const int32_t *) q_rope->op_params)[0];
    const int n_dims = ((const int32_t *) q_rope->op_params)[1];
    const int n_ctx_orig = ((const int32_t *) q_rope->op_params)[4];
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    memcpy(&freq_base,   (const int32_t *) q_rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) q_rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) q_rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) q_rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) q_rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) q_rope->op_params + 10, sizeof(float));
    ggml_metal_kargs_rope rope_args = {
        /*.ne00 =*/ 64, /*.ne01 =*/ 1, /*.ne02 =*/ 1, /*.ne03 =*/ 1,
        /*.nb00 =*/ sizeof(float), /*.nb01 =*/ 64*sizeof(float),
        /*.nb02 =*/ 64*sizeof(float), /*.nb03 =*/ 64*sizeof(float),
        /*.ne0  =*/ 64, /*.ne1  =*/ 1, /*.ne2  =*/ 1, /*.ne3  =*/ 1,
        /*.nb0  =*/ sizeof(float), /*.nb1  =*/ 64*sizeof(float),
        /*.nb2  =*/ 64*sizeof(float), /*.nb3  =*/ 64*sizeof(float),
        /*.n_past =*/ n_past,
        /*.n_dims =*/ n_dims,
        /*.n_ctx_orig =*/ n_ctx_orig,
        /*.freq_base =*/ freq_base,
        /*.freq_scale =*/ freq_scale,
        /*.ext_factor =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast =*/ beta_fast,
        /*.beta_slow =*/ beta_slow,
        /*.sect_0 =*/ ((const int32_t *) q_rope->op_params)[11],
        /*.sect_1 =*/ ((const int32_t *) q_rope->op_params)[12],
        /*.sect_2 =*/ ((const int32_t *) q_rope->op_params)[13],
        /*.sect_3 =*/ ((const int32_t *) q_rope->op_params)[14],
        /*.src2 =*/ false,
    };

    const char * kernel = "kernel_glm_absorbed_q_q8_q4";
    auto pipeline = ggml_metal_library_get_pipeline(ctx->lib, kernel);
    if (!pipeline.pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(ctx->lib, kernel, kernel, nullptr);
    }
    int ida = 0;
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &fused_args, sizeof(fused_args), ida++);
    ggml_metal_encoder_set_bytes(ctx->enc, &rope_args, sizeof(rope_args), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(q->src[0]), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(q->src[1]), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(q_abs->src[0]), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(q_rope->src[1]), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(q->src[1]), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(q_pack), ida++);
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, 320*sizeof(float), 0);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, 64, 1, 1, 1024, 1, 1);

    const int kv_encoded = ggml_metal_op_mul_mat(ctx, idx + 3);
    GGML_ASSERT(kv_encoded == 1);
    ctx->set_fused_range_outputs(3, 4);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=glm_absorbed_q_fused kernel=%s tensor=%s fused_nodes=%d dispatch_groups=2 grid_x=64 grid_y=1 grid_z=1 threads_x=1024\n",
            kernel,
            ggml_metal_tensor_name(q_pack),
            n_fuse);
    }
    return n_fuse;
}

static bool ggml_metal_glm_moe_can_fuse_swiglu_q3_down(const ggml_metal_glm_moe_decode_motif_reference & motif) {
    if (!ggml_metal_glm_dsa_moe_swiglu_q3_down_fusion_enabled()) {
        return false;
    }
    if (motif.has_weighted_down ||
            motif.gate == nullptr || motif.up == nullptr ||
            motif.glu == nullptr || motif.down == nullptr || motif.out == nullptr ||
            motif.ids == nullptr || motif.weights == nullptr) {
        return false;
    }
    return motif.gate->type == GGML_TYPE_F32 &&
        motif.up->type == GGML_TYPE_F32 &&
        motif.glu->type == GGML_TYPE_F32 &&
        motif.down->type == GGML_TYPE_F32 &&
        motif.out->type == GGML_TYPE_F32 &&
        motif.down->src[0]->type == GGML_TYPE_Q3_K &&
        motif.down->src[1] == motif.glu &&
        motif.down->src[2] == motif.ids &&
        motif.out->src[0] == motif.down &&
        motif.out->src[1] == motif.weights &&
        motif.ids->ne[0] == 8 &&
        motif.ids->ne[1] == 1 &&
        motif.down->src[0]->ne[0] == 2048 &&
        motif.down->src[0]->ne[1] == 6144 &&
        motif.down->src[0]->ne[2] == 256 &&
        motif.out->ne[0] == 6144 &&
        motif.out->ne[1] == 1 &&
        motif.gate->ne[0] == motif.down->src[0]->ne[0] &&
        motif.up->ne[0] == motif.down->src[0]->ne[0] &&
        motif.gate->ne[1] == motif.ids->ne[0] &&
        motif.up->ne[1] == motif.ids->ne[0] &&
        motif.gate->ne[2] == motif.ids->ne[1] &&
        motif.up->ne[2] == motif.ids->ne[1] &&
        motif.glu->ne[0] == motif.gate->ne[0] &&
        motif.glu->ne[1] == motif.gate->ne[1] &&
        motif.glu->ne[2] == motif.gate->ne[2];
}

static int ggml_metal_op_glm_moe_swiglu_q3_down_weighted(
        ggml_metal_op_t ctx,
        const ggml_metal_glm_moe_decode_motif_reference & motif) {
    GGML_ASSERT(ggml_metal_glm_moe_can_fuse_swiglu_q3_down(motif));

    ggml_tensor * down = motif.down;
    ggml_tensor * out = motif.out;

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, down->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, down->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, motif.glu,     ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, motif.glu,     nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, motif.ids,     ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, motif.ids,     nb);
    GGML_TENSOR_LOCALS(uint64_t, nb3, motif.weights, nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  out,           ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  out,           nb);

    auto pipeline = ggml_metal_library_get_pipeline_glm_moe_swiglu_q3_down_weighted(lib, out);
    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;

    ggml_metal_kargs_mul_mv_id args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb1  =*/ nb1,
        /*.nr0  =*/ nr0,
    };
    ggml_metal_kargs_mul_mv_id_weighted_reduce_extra extra = {
        /*.weights_nb1 =*/ nb31,
        /*.weights_nb2 =*/ nb32,
        /*.dst_nb0 =*/ nb0,
        /*.dst_nb1 =*/ nb1,
        /*.already_weighted =*/ ggml_get_op_params_i32(out, 0) != 0 ? 1 : 0,
        /*._pad0 =*/ 0,
    };
    ggml_metal_kargs_glm_moe_swiglu_q3_down swiglu_args = {
        /*.gate_nb1 =*/ motif.gate->nb[1],
        /*.gate_nb2 =*/ motif.gate->nb[2],
        /*.up_nb1 =*/ motif.up->nb[1],
        /*.up_nb2 =*/ motif.up->nb[2],
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.gate),   2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.up),     3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(out),          4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.ids),    5);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.weights),6);
    ggml_metal_encoder_set_bytes(enc, &extra, sizeof(extra), 7);
    ggml_metal_encoder_set_bytes(enc, &swiglu_args, sizeof(swiglu_args), 8);

    const int grid_x = (ne01 + nr0 - 1)/nr0;
    const int grid_y = ne21;
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, 1, 32, nsg, 1);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        const ggml_metal_buffer_id bid_gate = ggml_metal_get_buffer_id(motif.gate);
        const ggml_metal_buffer_id bid_up   = ggml_metal_get_buffer_id(motif.up);
        const ggml_metal_buffer_id bid_out  = ggml_metal_get_buffer_id(out);
        const size_t gate_begin = bid_gate.offs;
        const size_t gate_end   = gate_begin + ggml_nbytes(motif.gate);
        const size_t up_begin   = bid_up.offs;
        const size_t up_end     = up_begin + ggml_nbytes(motif.up);
        const size_t out_begin  = bid_out.offs;
        const size_t out_end    = out_begin + ggml_nbytes(out);
        const bool gate_out_overlap =
            bid_gate.metal == bid_out.metal && gate_begin < out_end && out_begin < gate_end;
        const bool up_out_overlap =
            bid_up.metal == bid_out.metal && up_begin < out_end && out_begin < up_end;
        GGML_LOG_INFO(
                "ggml_metal: moe_dispatch op=glm_moe_swiglu_q3_down_weighted tensor=%s gate=%s up=%s down=%s weights=%s gate_out_overlap=%d up_out_overlap=%d gate_offs=%llu gate_nbytes=%llu up_offs=%llu up_nbytes=%llu out_offs=%llu out_nbytes=%llu nr0=%d nsg=%d grid_x=%d grid_y=%d threads_x=%d threads_y=%d\n",
                ggml_metal_tensor_name(out),
                ggml_metal_tensor_name(motif.gate),
                ggml_metal_tensor_name(motif.up),
                ggml_metal_tensor_name(down),
                ggml_metal_tensor_name(motif.weights),
                gate_out_overlap ? 1 : 0,
                up_out_overlap ? 1 : 0,
                (unsigned long long) gate_begin,
                (unsigned long long) ggml_nbytes(motif.gate),
                (unsigned long long) up_begin,
                (unsigned long long) ggml_nbytes(motif.up),
                (unsigned long long) out_begin,
                (unsigned long long) ggml_nbytes(out),
                nr0,
                nsg,
                grid_x,
                grid_y,
                32,
                nsg);
    }

    return 3;
}

static bool ggml_metal_glm_moe_can_fuse_swiglu_q2_down(const ggml_metal_glm_moe_decode_motif_reference & motif) {
    if (!ggml_metal_glm_dsa_moe_swiglu_q2_down_fusion_enabled()) {
        return false;
    }
    if (motif.has_weighted_down ||
            motif.gate == nullptr || motif.up == nullptr ||
            motif.glu == nullptr || motif.down == nullptr || motif.out == nullptr ||
            motif.ids == nullptr || motif.weights == nullptr) {
        return false;
    }
    return motif.gate->type == GGML_TYPE_F32 &&
        motif.up->type == GGML_TYPE_F32 &&
        motif.glu->type == GGML_TYPE_F32 &&
        motif.down->type == GGML_TYPE_F32 &&
        motif.out->type == GGML_TYPE_F32 &&
        motif.down->src[0]->type == GGML_TYPE_Q2_K &&
        motif.down->src[1] == motif.glu &&
        motif.down->src[2] == motif.ids &&
        motif.out->src[0] == motif.down &&
        motif.out->src[1] == motif.weights &&
        ggml_get_op_params_i32(motif.out, 0) == 0 &&
        motif.ids->ne[0] == 8 &&
        motif.ids->ne[1] == 1 &&
        motif.down->src[0]->ne[0] == 2048 &&
        motif.down->src[0]->ne[1] == 6144 &&
        motif.down->src[0]->ne[2] == 256 &&
        motif.out->ne[0] == 6144 &&
        motif.out->ne[1] == 1 &&
        motif.gate->ne[0] == motif.down->src[0]->ne[0] &&
        motif.up->ne[0] == motif.down->src[0]->ne[0] &&
        motif.gate->ne[1] == motif.ids->ne[0] &&
        motif.up->ne[1] == motif.ids->ne[0] &&
        motif.gate->ne[2] == motif.ids->ne[1] &&
        motif.up->ne[2] == motif.ids->ne[1] &&
        motif.glu->ne[0] == motif.gate->ne[0] &&
        motif.glu->ne[1] == motif.gate->ne[1] &&
        motif.glu->ne[2] == motif.gate->ne[2];
}

static int ggml_metal_op_glm_moe_swiglu_q2_down_weighted(
        ggml_metal_op_t ctx,
        const ggml_metal_glm_moe_decode_motif_reference & motif) {
    GGML_ASSERT(ggml_metal_glm_moe_can_fuse_swiglu_q2_down(motif));

    ggml_tensor * down = motif.down;
    ggml_tensor * out = motif.out;

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, down->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, down->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, motif.glu,     ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, motif.glu,     nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, motif.ids,     ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, motif.ids,     nb);
    GGML_TENSOR_LOCALS(uint64_t, nb3, motif.weights, nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  out,           ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  out,           nb);

    auto pipeline = ggml_metal_library_get_pipeline_glm_moe_swiglu_q2_down_weighted(lib, out);
    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;

    ggml_metal_kargs_mul_mv_id args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb1  =*/ nb1,
        /*.nr0  =*/ nr0,
    };
    ggml_metal_kargs_mul_mv_id_weighted_reduce_extra extra = {
        /*.weights_nb1 =*/ nb31,
        /*.weights_nb2 =*/ nb32,
        /*.dst_nb0 =*/ nb0,
        /*.dst_nb1 =*/ nb1,
        /*.already_weighted =*/ ggml_get_op_params_i32(out, 0) != 0 ? 1 : 0,
        /*._pad0 =*/ 0,
    };
    ggml_metal_kargs_glm_moe_swiglu_q3_down swiglu_args = {
        /*.gate_nb1 =*/ motif.gate->nb[1],
        /*.gate_nb2 =*/ motif.gate->nb[2],
        /*.up_nb1 =*/ motif.up->nb[1],
        /*.up_nb2 =*/ motif.up->nb[2],
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.gate),   2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.up),     3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(out),          4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.ids),    5);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.weights),6);
    ggml_metal_encoder_set_bytes(enc, &extra, sizeof(extra), 7);
    ggml_metal_encoder_set_bytes(enc, &swiglu_args, sizeof(swiglu_args), 8);

    const int grid_x = (ne01 + nr0 - 1)/nr0;
    const int grid_y = ne21;
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, 1, 32, nsg, 1);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
                "ggml_metal: moe_dispatch op=glm_moe_swiglu_q2_down_weighted tensor=%s gate=%s up=%s down=%s weights=%s nr0=%d nsg=%d grid_x=%d grid_y=%d threads_x=%d threads_y=%d\n",
                ggml_metal_tensor_name(out),
                ggml_metal_tensor_name(motif.gate),
                ggml_metal_tensor_name(motif.up),
                ggml_metal_tensor_name(down),
                ggml_metal_tensor_name(motif.weights),
                nr0,
                nsg,
                grid_x,
                grid_y,
                32,
                nsg);
    }

    return 3;
}

static bool ggml_metal_glm_moe_route_gate_up_swiglu_shape_ok(
        const ggml_metal_glm_moe_decode_motif_reference & motif,
        const ggml_tensor * src1) {
    if (motif.route.logits == nullptr ||
            motif.route.ids == nullptr ||
            motif.route.weights == nullptr ||
            motif.route.clamp == nullptr ||
            motif.gate == nullptr ||
            motif.up == nullptr ||
            motif.glu == nullptr ||
            src1 == nullptr) {
        return false;
    }

    const ggml_tensor * dst = motif.weighted_down_input != nullptr ? motif.weighted_down_input : motif.glu;
    return motif.route.logits->type == GGML_TYPE_F32 &&
        motif.route.logits->ne[0] == 256 &&
        motif.route.logits->ne[1] == 1 &&
        motif.route.ids->type == GGML_TYPE_I32 &&
        motif.route.ids->ne[0] == 8 &&
        motif.route.ids->ne[1] == 1 &&
        motif.route.weights->type == GGML_TYPE_F32 &&
        motif.route.weights->ne[1] == 8 &&
        motif.route.weights->ne[2] == 1 &&
        motif.up->src[0]->type == GGML_TYPE_Q2_K &&
        motif.gate->src[0]->type == GGML_TYPE_Q2_K &&
        motif.up->src[1] == motif.gate->src[1] &&
        motif.up->src[2] == motif.route.ids &&
        motif.gate->src[2] == motif.route.ids &&
        src1->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(src1, motif.up->src[1]) &&
        motif.up->type == GGML_TYPE_F32 &&
        motif.gate->type == GGML_TYPE_F32 &&
        motif.glu->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        motif.up->src[0]->ne[0] >= 2048 &&
        motif.up->src[0]->ne[1] >= 1024 &&
        motif.up->src[0]->ne[2] == 256 &&
        motif.gate->src[0]->ne[0] >= 2048 &&
        motif.gate->src[0]->ne[1] >= 1024 &&
        motif.gate->src[0]->ne[2] == 256 &&
        motif.gate->src[0]->ne[0] == motif.up->src[0]->ne[0] &&
        motif.gate->src[0]->ne[1] == motif.up->src[0]->ne[1] &&
        motif.glu->ne[0] == motif.up->src[0]->ne[1] &&
        motif.glu->ne[1] == 8 &&
        motif.glu->ne[2] == 1 &&
        dst->ne[0] == motif.glu->ne[0] &&
        dst->ne[1] == motif.glu->ne[1] &&
        dst->ne[2] == motif.glu->ne[2];
}

static bool ggml_metal_glm_moe_can_fuse_route_gate_up_swiglu(
        const ggml_metal_glm_moe_decode_motif_reference & motif,
        const ggml_tensor * src1) {
    return ggml_metal_glm_dsa_moe_route_gate_up_fusion_enabled() &&
        ggml_metal_glm_moe_route_gate_up_swiglu_shape_ok(motif, src1);
}

static int ggml_metal_op_glm_moe_route_gate_up_swiglu(
        ggml_metal_op_t ctx,
        const ggml_metal_glm_moe_decode_motif_reference & motif,
        ggml_tensor * src1_override,
        const ggml_metal_glm_moe_private_bindings * private_bindings) {
    ggml_tensor * src1 = src1_override != nullptr ? src1_override : motif.up->src[1];
    const bool private_path = private_bindings != nullptr;
    const bool can_fuse = private_path ?
        ggml_metal_glm_moe_route_gate_up_swiglu_shape_ok(motif, src1) :
        ggml_metal_glm_moe_can_fuse_route_gate_up_swiglu(motif, src1);
    if (!can_fuse) {
        return 0;
    }

    ggml_tensor * dst = motif.weighted_down_input != nullptr ? motif.weighted_down_input : motif.glu;
    ggml_metal_topk_moe_route_fusion route = motif.route;

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(uint64_t, nb_logits,  route.logits,  nb);
    GGML_TENSOR_LOCALS(uint64_t, nb_bias,    route.bias,    nb);
    GGML_TENSOR_LOCALS(uint64_t, nb_ids,     route.ids,     nb);
    GGML_TENSOR_LOCALS(uint64_t, nb_weights, route.weights, nb);
    GGML_TENSOR_LOCALS( int32_t, ne0,        motif.up->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0,        motif.up->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1,        src1,             ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1,        src1,             nb);
    GGML_TENSOR_LOCALS( int32_t, ne2,        route.ids,        ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2,        route.ids,        nb);
    GGML_TENSOR_LOCALS( int32_t, ne,         dst,              ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,         dst,              nb);

    const bool route_weights_op = route.weights->op == GGML_OP_MOE_ROUTE_WEIGHTS;
    const float scale = route.scale ? ggml_get_op_params_f32(route.scale, 0) :
        (route_weights_op ? ggml_get_op_params_f32(route.weights, 1) : 1.0f);
    const float clamp_min = ggml_get_op_params_f32(route.clamp, 0);
    const int32_t norm = route_weights_op ? ggml_get_op_params_i32(route.weights, 2) != 0 : 1;

    ggml_metal_kargs_topk_moe_route route_args = {
        /*.n_expert    =*/ (int32_t) route.logits->ne[0],
        /*.n_tokens    =*/ (int32_t) route.logits->ne[1],
        /*.top_k       =*/ (int32_t) route.ids->ne[0],
        /*.has_bias    =*/ route.bias ? 1 : 0,
        /*.norm        =*/ norm,
        /*._pad0       =*/ ggml_metal_glm_dsa_moe_sort_route_ids_enabled() ? 1 : 0,
        /*._pad1       =*/ ggml_metal_glm_dsa_moe_max_active_experts(),
        /*._pad2       =*/ 0,
        /*.scale       =*/ scale,
        /*.clamp_min   =*/ clamp_min,
        /*.logits_nb0  =*/ nb_logits0,
        /*.logits_nb1  =*/ nb_logits1,
        /*.bias_nb0    =*/ nb_bias0,
        /*.ids_nb0     =*/ nb_ids0,
        /*.ids_nb1     =*/ nb_ids1,
        /*.weights_nb1 =*/ nb_weights1,
        /*.weights_nb2 =*/ nb_weights2,
    };
    ggml_metal_kargs_mul_mv_id_gate_up_swiglu gate_args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb1  =*/ nb1,
        /*.nr0  =*/ 8,
        /*.weights_nb1 =*/ nb_weights1,
        /*.weights_nb2 =*/ nb_weights2,
        /*.weighted =*/ motif.weighted_down_input != nullptr ? 1 : 0,
        /*._pad0 =*/ ggml_metal_glm_dsa_moe_max_active_experts(),
    };

    auto pipeline = ggml_metal_library_get_pipeline_glm_moe_route_q2_gate_up_swiglu_pair_sg_slot8(lib, dst);
    ggml_metal_op_concurrency_reset(ctx);

    const ggml_metal_buffer_id ids_buffer = private_path ?
        private_bindings->ids : ggml_metal_get_buffer_id(route.ids);
    const ggml_metal_buffer_id weights_buffer = private_path ?
        private_bindings->weights : ggml_metal_get_buffer_id(route.weights);
    const ggml_metal_buffer_id activation_buffer = private_path ?
        private_bindings->activation : ggml_metal_get_buffer_id(dst);

    int ida = 0;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes (enc, &route_args, sizeof(route_args), ida++);
    ggml_metal_encoder_set_bytes (enc, &gate_args,  sizeof(gate_args),  ida++);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(route.logits), ida++);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(route.bias ? route.bias : route.logits), ida++);
    ggml_metal_encoder_set_buffer(enc, ids_buffer, ida++);
    ggml_metal_encoder_set_buffer(enc, weights_buffer, ida++);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.up->src[0]), ida++);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(motif.gate->src[0]), ida++);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src1), ida++);
    ggml_metal_encoder_set_buffer(enc, activation_buffer, ida++);

    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const int grid_x = (ne01 + nr0 - 1)/nr0;
    const int grid_z = ne21;
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, 1, grid_z, 32, nsg, 1);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
                "ggml_metal: moe_dispatch op=glm_moe_route_q2_gate_up_swiglu tensor=%s logits=%s ids=%s weights=%s gate=%s up=%s src1=%s src1_override=%d weighted=%d private_scratch=%d experts=%d top_k=%d sort_ids=%d tokens=%d embd=%d ff=%d grid_x=%d grid_z=%d threads_x=32 threads_y=%d\n",
                ggml_metal_tensor_name(dst),
                ggml_metal_tensor_name(route.logits),
                ggml_metal_tensor_name(route.ids),
                ggml_metal_tensor_name(route.weights),
                ggml_metal_tensor_name(motif.gate),
                ggml_metal_tensor_name(motif.up),
                ggml_metal_tensor_name(src1),
                src1_override != nullptr ? 1 : 0,
                motif.weighted_down_input != nullptr ? 1 : 0,
                private_path ? 1 : 0,
                route_args.n_expert,
                route_args.top_k,
                route_args._pad0,
                route_args.n_tokens,
                ne00,
                ne01,
                grid_x,
                grid_z,
                nsg);
    }

    return motif.weighted_down_input != nullptr ? 4 : 3;
}

static bool ggml_metal_glm_moe_weights_gate_up_swiglu_shape_ok(
        const ggml_metal_glm_moe_decode_motif_reference & motif,
        const ggml_tensor * src1) {
    const ggml_tensor * route_weights = motif.route.weights;
    if (route_weights == nullptr || route_weights->op != GGML_OP_MOE_ROUTE_WEIGHTS ||
            route_weights->src[0] == nullptr || route_weights->src[1] == nullptr ||
            motif.route.ids == nullptr || motif.gate == nullptr || motif.up == nullptr ||
            motif.glu == nullptr || src1 == nullptr) {
        return false;
    }

    return route_weights->src[0]->type == GGML_TYPE_F32 &&
        route_weights->src[1] == motif.route.ids &&
        motif.route.ids->type == GGML_TYPE_I32 &&
        motif.route.ids->ne[0] == 8 &&
        motif.route.ids->ne[1] == 1 &&
        route_weights->type == GGML_TYPE_F32 &&
        route_weights->ne[0] == 1 &&
        route_weights->ne[1] == 8 &&
        route_weights->ne[2] == 1 &&
        motif.up->src[0]->type == GGML_TYPE_Q2_K &&
        motif.gate->src[0]->type == GGML_TYPE_Q2_K &&
        motif.up->src[1] == motif.gate->src[1] &&
        motif.up->src[2] == motif.route.ids &&
        motif.gate->src[2] == motif.route.ids &&
        src1->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(src1, motif.up->src[1]) &&
        motif.up->type == GGML_TYPE_F32 &&
        motif.gate->type == GGML_TYPE_F32 &&
        motif.glu->type == GGML_TYPE_F32 &&
        motif.up->src[0]->ne[0] == 6144 &&
        motif.up->src[0]->ne[1] == 2048 &&
        motif.up->src[0]->ne[2] == 256 &&
        motif.gate->src[0]->ne[0] == 6144 &&
        motif.gate->src[0]->ne[1] == 2048 &&
        motif.gate->src[0]->ne[2] == 256 &&
        motif.glu->ne[0] == 2048 &&
        motif.glu->ne[1] == 8 &&
        motif.glu->ne[2] == 1;
}

static int ggml_metal_op_glm_moe_weights_gate_up_swiglu(
        ggml_metal_op_t ctx,
        const ggml_metal_glm_moe_decode_motif_reference & motif,
        ggml_tensor * src1,
        const ggml_metal_glm_moe_private_bindings & bindings) {
    if (!ggml_metal_glm_moe_weights_gate_up_swiglu_shape_ok(motif, src1)) {
        return 0;
    }

    ggml_tensor * route_weights = motif.route.weights;
    ggml_tensor * probs = route_weights->src[0];
    ggml_tensor * ids = motif.route.ids;

    GGML_TENSOR_LOCALS( int32_t, ne_route_probs, probs, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb_route_probs, probs, nb);
    GGML_TENSOR_LOCALS( int32_t, ne_route_ids, ids, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb_route_ids, ids, nb);
    GGML_TENSOR_LOCALS(uint64_t, nb_route_weights, route_weights, nb);
    GGML_TENSOR_LOCALS( int32_t, ne0, motif.up->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, motif.up->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, src1, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, src1, nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, ids, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, ids, nb);
    GGML_TENSOR_LOCALS( int32_t, ne, motif.glu, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, motif.glu, nb);

    ggml_metal_kargs_moe_route_weights route_args = {
        /*.n_expert      =*/ ne_route_probs1,
        /*.n_tokens      =*/ ne_route_ids1,
        /*.n_expert_used =*/ ne_route_ids0,
        /*.norm          =*/ ggml_get_op_params_i32(route_weights, 2),
        /*.clamp_min     =*/ ggml_get_op_params_f32(route_weights, 0),
        /*.scale         =*/ ggml_get_op_params_f32(route_weights, 1),
        /*._pad0         =*/ ggml_metal_glm_dsa_moe_route_weights_slot0_enabled() ? 1 : 0,
        /*._pad1         =*/ 0,
        /*.probs_nb1     =*/ nb_route_probs1,
        /*.probs_nb2     =*/ nb_route_probs2,
        /*.ids_nb0       =*/ nb_route_ids0,
        /*.ids_nb1       =*/ nb_route_ids1,
        /*.dst_nb1       =*/ nb_route_weights1,
        /*.dst_nb2       =*/ nb_route_weights2,
    };
    ggml_metal_kargs_mul_mv_id_gate_up_swiglu gate_args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb1  =*/ nb1,
        /*.nr0  =*/ 8,
        /*.weights_nb1 =*/ nb_route_weights1,
        /*.weights_nb2 =*/ nb_route_weights2,
        /*.weighted =*/ 0,
        /*._pad0 =*/ ggml_metal_glm_dsa_moe_max_active_experts(),
    };

    auto pipeline = ggml_metal_library_get_pipeline_glm_moe_weights_q2_gate_up_swiglu_pair_sg_slot1(
        ctx->lib, motif.glu);
    ggml_metal_op_concurrency_reset(ctx);

    int ida = 0;
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &route_args, sizeof(route_args), ida++);
    ggml_metal_encoder_set_bytes(ctx->enc, &gate_args, sizeof(gate_args), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(probs), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, bindings.ids, ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, bindings.weights, ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.up->src[0]), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.gate->src[0]), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(src1), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, bindings.activation, ida++);

    const int grid_x = (ne01 + pipeline.nr0 - 1)/pipeline.nr0;
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, pipeline.smem, 0);
    const int grid_z = ne21*ne20;
    ggml_metal_encoder_dispatch_threadgroups(
        ctx->enc, grid_x, 1, grid_z, 32, pipeline.nsg, 1);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml_metal: moe_dispatch op=glm_moe_weights_q2_gate_up_swiglu tensor=%s probs=%s ids=%s weights=%s private_scratch=1 experts=%d top_k=%d tokens=%d embd=%d ff=%d grid_x=%d grid_z=%d threads_x=32 threads_y=%d\n",
            ggml_metal_tensor_name(motif.glu),
            ggml_metal_tensor_name(probs),
            ggml_metal_tensor_name(ids),
            ggml_metal_tensor_name(route_weights),
            route_args.n_expert,
            route_args.n_expert_used,
            route_args.n_tokens,
            ne00,
            ne01,
            grid_x,
            grid_z,
            pipeline.nsg);
    }

    return motif.route_n_fuse + 3;
}

static size_t ggml_metal_fusion_scratch_align(size_t offset) {
    constexpr size_t alignment = 256;
    return (offset + alignment - 1) & ~(alignment - 1);
}

static bool ggml_metal_make_glm_moe_private_bindings(
        ggml_metal_op_t ctx,
        const ggml_metal_glm_moe_decode_motif_reference & motif,
        ggml_metal_glm_moe_private_bindings & bindings) {
    if (ctx->fusion_scratch.metal == nullptr || ctx->fusion_scratch_size == 0 ||
            motif.ids == nullptr || motif.weights == nullptr || motif.glu == nullptr) {
        return false;
    }

    size_t cursor = 0;
    bindings.ids = ctx->fusion_scratch;
    bindings.ids.offs += cursor;
    cursor = ggml_metal_fusion_scratch_align(cursor + ggml_nbytes(motif.ids));

    bindings.weights = ctx->fusion_scratch;
    bindings.weights.offs += cursor;
    cursor = ggml_metal_fusion_scratch_align(cursor + ggml_nbytes(motif.weights));

    bindings.activation = ctx->fusion_scratch;
    bindings.activation.offs += cursor;
    cursor = ggml_metal_fusion_scratch_align(cursor + ggml_nbytes(motif.glu));

    return cursor <= ctx->fusion_scratch_size;
}

static int ggml_metal_op_glm_moe_private_scratch_decode(
        ggml_metal_op_t ctx,
        int idx,
        const ggml_metal_glm_moe_decode_motif_reference & motif) {
    const bool fused_topk_route =
        ggml_metal_glm_moe_route_gate_up_swiglu_shape_ok(motif, motif.route_anchor_src);
    const bool native_route_weights =
        ggml_metal_glm_moe_weights_gate_up_swiglu_shape_ok(motif, motif.route_anchor_src);
    if (!ggml_metal_glm_dsa_moe_private_scratch_enabled() ||
            !motif.has_native_down || !motif.has_shared_expert_tail ||
            motif.out_offset < 0 || motif.shared_gate_offset < 0 || motif.shared_up_offset < 0 ||
            motif.route_anchor_src == nullptr ||
            (!fused_topk_route && !native_route_weights)) {
        return 0;
    }

    ggml_metal_glm_moe_private_bindings bindings;
    if (!ggml_metal_make_glm_moe_private_bindings(ctx, motif, bindings)) {
        return 0;
    }

    if (native_route_weights) {
        // TOP_K is outside this owned span, so retain its graph allocation as
        // the exact selected-ID source while weights and activation stay private.
        bindings.ids = ggml_metal_get_buffer_id(motif.ids);
    }

    const int route_gate_up_n = fused_topk_route ?
        ggml_metal_op_glm_moe_route_gate_up_swiglu(
            ctx, motif, motif.route_anchor_src, &bindings) :
        ggml_metal_op_glm_moe_weights_gate_up_swiglu(
            ctx, motif, motif.route_anchor_src, bindings);
    if (route_gate_up_n <= 0) {
        return 0;
    }

    const int shared_gate_n = ggml_metal_op_reference_dispatch_existing_node(
        ctx, idx + motif.shared_gate_offset);
    const int shared_up_n = ggml_metal_op_reference_dispatch_existing_node(
        ctx, idx + motif.shared_up_offset);
    GGML_ASSERT(shared_gate_n == 1);
    GGML_ASSERT(shared_up_n == 1);

    // Only the private routed resource is a dependency of the Q3 terminal.
    // The two shared-expert projections remain eligible to overlap it.
    ggml_metal_encoder_memory_barrier_buffer(ctx->enc, bindings.activation);

    const int down_n = ggml_metal_encode_moe_mul_mat_id(ctx, motif.down, &bindings);
    GGML_ASSERT(down_n == 1);

    ctx->set_fused_range_outputs(
        motif.out_offset,
        motif.shared_gate_offset,
        motif.shared_up_offset);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml_metal: moe_dispatch op=glm_moe_private_scratch_decode tensor=%s fused_nodes=%d route_nodes=%d out_offset=%d shared_gate_offset=%d shared_up_offset=%d scratch_bytes=%llu activation_bytes=%llu dispatch_groups=4 graph_uid=%llu split_start=%d split_end=%d\n",
            ggml_metal_tensor_name(motif.out),
            motif.n_fuse,
            motif.route_n_fuse,
            motif.out_offset,
            motif.shared_gate_offset,
            motif.shared_up_offset,
            (unsigned long long) ctx->fusion_scratch_size,
            (unsigned long long) ggml_nbytes(motif.glu),
            (unsigned long long) ctx->graph_uid(),
            ctx->split_start(),
            ctx->split_end());
    }

    return motif.n_fuse;
}

static void ggml_metal_encode_glm_moe_shared_mul_mv(
        ggml_metal_op_t ctx,
        ggml_tensor * op,
        ggml_metal_buffer_id src,
        ggml_metal_buffer_id dst) {
    GGML_ASSERT(op->op == GGML_OP_MUL_MAT);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->ne[1] == 1);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;
    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(ctx->lib, op);
    const int nr0 = pipeline.nr0;
    const int nr1 = pipeline.nr1;
    const int nsg = pipeline.nsg;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
    };

    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(ctx->enc, src, 2);
    ggml_metal_encoder_set_buffer(ctx->enc, dst, 3);
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(
        ctx->enc,
        (ne01 + nr0*nsg - 1)/(nr0*nsg),
        (ne11 + nr1 - 1)/nr1,
        ne12*ne13,
        32,
        nsg,
        1);
}

static int ggml_metal_op_glm_moe_two_phase(
        ggml_metal_op_t ctx,
        int idx,
        const ggml_metal_glm_moe_decode_motif_reference & motif) {
    const bool dual_lane = ggml_metal_glm_dsa_moe_dual_lane_enabled();
    if ((!ggml_metal_glm_dsa_moe_two_phase_enabled() && !dual_lane) ||
            (dual_lane && !motif.final_only_fusable) ||
            motif.shared_glu == nullptr || motif.shared_down == nullptr || motif.final_out == nullptr) {
        return 0;
    }

    ggml_metal_buffer_id shared_gate_scratch = { nullptr, 0 };
    ggml_metal_buffer_id shared_up_scratch = { nullptr, 0 };
    ggml_metal_buffer_id shared_activation_scratch = { nullptr, 0 };
    if (dual_lane) {
        constexpr size_t shared_projection_bytes = 2048*sizeof(float);
        size_t scratch_cursor = 0;
        shared_gate_scratch = ctx->fusion_scratch;
        shared_gate_scratch.offs += scratch_cursor;
        scratch_cursor = ggml_metal_fusion_scratch_align(scratch_cursor + shared_projection_bytes);
        shared_up_scratch = ctx->fusion_scratch;
        shared_up_scratch.offs += scratch_cursor;
        scratch_cursor = ggml_metal_fusion_scratch_align(scratch_cursor + shared_projection_bytes);
        shared_activation_scratch = ctx->fusion_scratch;
        shared_activation_scratch.offs += scratch_cursor;
        scratch_cursor = ggml_metal_fusion_scratch_align(scratch_cursor + shared_projection_bytes);
        if (ctx->fusion_scratch.metal == nullptr || scratch_cursor > ctx->fusion_scratch_size) {
            return 0;
        }
    }

    const int route_n = motif.route.weights->op == GGML_OP_MOE_ROUTE_WEIGHTS && motif.route.n_fuse == 1 ?
        ggml_metal_op_moe_route_weights(ctx, idx) :
        ggml_metal_op_topk_moe_route_fused(ctx, idx);
    GGML_ASSERT(route_n == motif.route_n_fuse);
    ggml_metal_op_concurrency_reset(ctx);

    ggml_tensor * routed_up_w = motif.up->src[0];
    ggml_tensor * routed_gate_w = motif.gate->src[0];
    ggml_tensor * routed_down_w = motif.down->src[0];
    ggml_tensor * shared_gate = dual_lane ? motif.shared_glu->src[0] : motif.shared_gate;
    ggml_tensor * shared_up = dual_lane ? motif.shared_glu->src[1] : motif.shared_up;
    ggml_tensor * shared_up_w = shared_up->src[0];
    ggml_tensor * shared_gate_w = shared_gate->src[0];
    ggml_tensor * shared_down_w = motif.shared_down->src[0];
    ggml_tensor * cur = motif.route_anchor_src;

    ggml_metal_kargs_mul_mv_id_gate_up_swiglu gate_args = {
        /*.nei0 =*/ (int32_t) motif.ids->ne[0],
        /*.nei1 =*/ (int32_t) motif.ids->ne[1],
        /*.nbi1 =*/ motif.ids->nb[1],
        /*.ne00 =*/ (int32_t) routed_up_w->ne[0],
        /*.ne01 =*/ (int32_t) routed_up_w->ne[1],
        /*.ne02 =*/ (int32_t) routed_up_w->ne[2],
        /*.nb00 =*/ routed_up_w->nb[0],
        /*.nb01 =*/ routed_up_w->nb[1],
        /*.nb02 =*/ routed_up_w->nb[2],
        /*.ne10 =*/ (int32_t) cur->ne[0],
        /*.ne11 =*/ (int32_t) cur->ne[1],
        /*.ne12 =*/ (int32_t) cur->ne[2],
        /*.ne13 =*/ (int32_t) cur->ne[3],
        /*.nb10 =*/ cur->nb[0],
        /*.nb11 =*/ cur->nb[1],
        /*.nb12 =*/ cur->nb[2],
        /*.ne0  =*/ (int32_t) motif.glu->ne[0],
        /*.ne1  =*/ (int32_t) motif.glu->ne[1],
        /*.nb1  =*/ motif.glu->nb[1],
        /*.nr0  =*/ 8,
        /*.weights_nb1 =*/ motif.weights->nb[1],
        /*.weights_nb2 =*/ motif.weights->nb[2],
        /*.weighted =*/ 0,
        /*._pad0 =*/ 0,
    };
    ggml_metal_kargs_glm_moe_two_phase phase_args = {
        /*.n_embd =*/ 6144,
        /*.n_ff =*/ 2048,
        /*.n_out =*/ 6144,
        /*.routed_gate_groups =*/ 2048,
        /*.shared_gate_nb1 =*/ shared_gate_w->nb[1],
        /*.shared_up_nb1 =*/ shared_up_w->nb[1],
        /*.shared_down_nb1 =*/ shared_down_w->nb[1],
    };

    const bool gate_slot4 = dual_lane && ggml_metal_glm_dsa_moe_dual_lane_gate_slot4_enabled();
    const char * gate_kernel = gate_slot4 ?
        "kernel_glm_moe_dual_lane_gate_slot4" : "kernel_glm_moe_two_phase_gate";
    auto gate_pipeline = ggml_metal_library_get_pipeline(ctx->lib, gate_kernel);
    if (!gate_pipeline.pipeline) {
        gate_pipeline = ggml_metal_library_compile_pipeline(ctx->lib, gate_kernel, gate_kernel, nullptr);
    }
    int ida = 0;
    ggml_metal_encoder_set_pipeline(ctx->enc, gate_pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &gate_args, sizeof(gate_args), ida++);
    ggml_metal_encoder_set_bytes(ctx->enc, &phase_args, sizeof(phase_args), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(routed_up_w), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(routed_gate_w), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(shared_up_w), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(shared_gate_w), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(cur), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.glu), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.shared_glu), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.ids), ida++);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.weights), ida++);
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, 8*8*sizeof(float), 0);
    const int32_t gate_groups = gate_slot4 ?
        ((phase_args.n_ff + gate_args.nr0 - 1)/gate_args.nr0)*((gate_args.nei0 + 3)/4) :
        (dual_lane ? phase_args.routed_gate_groups : phase_args.routed_gate_groups + phase_args.n_ff);
    const int32_t gate_nsg = gate_slot4 ? 4 : 1;
    GGML_ASSERT(32*gate_nsg <= ggml_metal_pipeline_max_theads_per_threadgroup(gate_pipeline));
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, gate_groups, 1, 1, 32, gate_nsg, 1);

    if (dual_lane) {
        ggml_metal_encode_glm_moe_shared_mul_mv(
            ctx,
            shared_gate,
            ggml_metal_get_buffer_id(shared_gate->src[1]),
            shared_gate_scratch);
        ggml_metal_encode_glm_moe_shared_mul_mv(
            ctx,
            shared_up,
            ggml_metal_get_buffer_id(shared_up->src[1]),
            shared_up_scratch);
    }
    ggml_metal_op_concurrency_reset(ctx);

    if (dual_lane) {
        const char * shared_glu_kernel = "kernel_glm_moe_dual_lane_swiglu";
        auto shared_glu_pipeline = ggml_metal_library_get_pipeline(ctx->lib, shared_glu_kernel);
        if (!shared_glu_pipeline.pipeline) {
            shared_glu_pipeline = ggml_metal_library_compile_pipeline(
                ctx->lib, shared_glu_kernel, shared_glu_kernel, nullptr);
        }
        ida = 0;
        ggml_metal_encoder_set_pipeline(ctx->enc, shared_glu_pipeline);
        ggml_metal_encoder_set_bytes(ctx->enc, &phase_args, sizeof(phase_args), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, shared_gate_scratch, ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, shared_up_scratch, ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, shared_activation_scratch, ida++);
        ggml_metal_encoder_dispatch_threadgroups(
            ctx->enc, (phase_args.n_ff + 255)/256, 1, 1, 256, 1, 1);
        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_mul_mv_id down_args = {
        /*.nei0 =*/ (int32_t) motif.ids->ne[0],
        /*.nei1 =*/ (int32_t) motif.ids->ne[1],
        /*.nbi1 =*/ motif.ids->nb[1],
        /*.ne00 =*/ (int32_t) routed_down_w->ne[0],
        /*.ne01 =*/ (int32_t) routed_down_w->ne[1],
        /*.ne02 =*/ (int32_t) routed_down_w->ne[2],
        /*.nb00 =*/ routed_down_w->nb[0],
        /*.nb01 =*/ routed_down_w->nb[1],
        /*.nb02 =*/ routed_down_w->nb[2],
        /*.ne10 =*/ (int32_t) motif.glu->ne[0],
        /*.ne11 =*/ (int32_t) motif.glu->ne[1],
        /*.ne12 =*/ (int32_t) motif.glu->ne[2],
        /*.ne13 =*/ (int32_t) motif.glu->ne[3],
        /*.nb10 =*/ motif.glu->nb[0],
        /*.nb11 =*/ motif.glu->nb[1],
        /*.nb12 =*/ motif.glu->nb[2],
        /*.ne0  =*/ (int32_t) motif.final_out->ne[0],
        /*.ne1  =*/ (int32_t) motif.final_out->ne[1],
        /*.nb1  =*/ motif.final_out->nb[1],
        /*.nr0  =*/ 8,
    };
    ggml_metal_kargs_mul_mv_id_weighted_reduce_extra down_extra = {
        /*.weights_nb1 =*/ motif.weights->nb[1],
        /*.weights_nb2 =*/ motif.weights->nb[2],
        /*.dst_nb0 =*/ motif.final_out->nb[0],
        /*.dst_nb1 =*/ motif.final_out->nb[1],
        /*.already_weighted =*/ 0,
        /*._pad0 =*/ 0,
    };

    if (dual_lane) {
        auto routed_down_pipeline =
            ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r8_nb8_w0(
                ctx->lib, motif.out);
        ggml_metal_encoder_set_pipeline(ctx->enc, routed_down_pipeline);
        ggml_metal_encoder_set_bytes(ctx->enc, &down_args, sizeof(down_args), 0);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(routed_down_w), 1);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.glu), 2);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.out), 3);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.ids), 4);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.weights), 5);
        ggml_metal_encoder_set_bytes(ctx->enc, &down_extra, sizeof(down_extra), 6);
        ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, routed_down_pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(
            ctx->enc, (phase_args.n_out + 7)/8, 1, 1, 32, 8, 1);

        ggml_metal_encode_glm_moe_shared_mul_mv(
            ctx,
            motif.shared_down,
            shared_activation_scratch,
            ggml_metal_get_buffer_id(motif.shared_down));
        ggml_metal_op_concurrency_reset(ctx);

        const char * final_add_kernel = "kernel_glm_moe_dual_lane_add";
        auto final_add_pipeline = ggml_metal_library_get_pipeline(ctx->lib, final_add_kernel);
        if (!final_add_pipeline.pipeline) {
            final_add_pipeline = ggml_metal_library_compile_pipeline(
                ctx->lib, final_add_kernel, final_add_kernel, nullptr);
        }
        ida = 0;
        ggml_metal_encoder_set_pipeline(ctx->enc, final_add_pipeline);
        ggml_metal_encoder_set_bytes(ctx->enc, &phase_args, sizeof(phase_args), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.out), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.shared_down), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.final_out), ida++);
        ggml_metal_encoder_dispatch_threadgroups(
            ctx->enc, (phase_args.n_out + 255)/256, 1, 1, 256, 1, 1);
        ctx->set_fused_range_outputs(motif.final_out_offset);
    } else {
        const char * down_kernel = "kernel_glm_moe_two_phase_down";
        auto down_pipeline = ggml_metal_library_get_pipeline(ctx->lib, down_kernel);
        if (!down_pipeline.pipeline) {
            down_pipeline = ggml_metal_library_compile_pipeline(
                ctx->lib, down_kernel, down_kernel, nullptr);
        }
        ida = 0;
        ggml_metal_encoder_set_pipeline(ctx->enc, down_pipeline);
        ggml_metal_encoder_set_bytes(ctx->enc, &down_args, sizeof(down_args), ida++);
        ggml_metal_encoder_set_bytes(ctx->enc, &down_extra, sizeof(down_extra), ida++);
        ggml_metal_encoder_set_bytes(ctx->enc, &phase_args, sizeof(phase_args), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(routed_down_w), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.glu), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(shared_down_w), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.shared_glu), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.out), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.final_out), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.ids), ida++);
        ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(motif.weights), ida++);
        ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, 8*8*sizeof(float), 0);
        ggml_metal_encoder_dispatch_threadgroups(
            ctx->enc, (phase_args.n_out + 7)/8, 1, 1, 32, 8, 1);
        ctx->set_fused_range_outputs(motif.out_offset, motif.final_out_offset);
    }
    ggml_metal_op_concurrency_reset(ctx);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml_metal: moe_dispatch op=%s tensor=%s fused_nodes=%d route_nodes=%d gate_groups=%d down_groups=%d dispatch_groups=%d final_only=%d\n",
            dual_lane ? "glm_moe_dual_lane" : "glm_moe_two_phase",
            ggml_metal_tensor_name(motif.final_out),
            motif.n_fuse,
            motif.route_n_fuse,
            gate_groups,
            (phase_args.n_out + 7)/8,
            dual_lane ? 8 : 3,
            dual_lane ? 1 : 0);
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=%s kernel=%s tensor=%s fused_nodes=%d dispatch_groups=%d grid_x=%d grid_y=1 grid_z=1 threads_x=32\n",
            dual_lane ? "glm_moe_dual_lane" : "glm_moe_two_phase",
            dual_lane ? "dual_lane_q2_q3_q4" : "two_phase_q2_q3_q4",
            ggml_metal_tensor_name(motif.final_out),
            motif.n_fuse,
            dual_lane ? 8 : 3,
            gate_groups);
    }

    return motif.n_fuse;
}

static int ggml_metal_op_glm_moe_decode_motif_reference(ggml_metal_op_t ctx, int idx) {
    ggml_metal_glm_moe_decode_motif_reference motif;
    if (!ggml_metal_match_glm_moe_decode_motif_reference(ctx, idx, motif)) {
        return 0;
    }
    if (ggml_metal_extend_glm_moe_two_phase(ctx, idx, motif)) {
        const int two_phase_n = ggml_metal_op_glm_moe_two_phase(ctx, idx, motif);
        if (two_phase_n > 0) {
            return two_phase_n;
        }
    }
    if (!ggml_metal_glm_dsa_moe_decode_motif_reference_enabled() &&
            !ggml_metal_glm_dsa_moe_private_scratch_enabled()) {
        return 0;
    }
    ggml_metal_log_routed_moe_decode_motif_contract(ctx, motif);

    const int private_scratch_n = ggml_metal_op_glm_moe_private_scratch_decode(ctx, idx, motif);
    if (private_scratch_n > 0) {
        return private_scratch_n;
    }
    if (motif.has_native_down) {
        return 0;
    }

    const bool skip_internal_barriers = ggml_metal_glm_dsa_moe_decode_skip_internal_barriers_enabled();
    const bool skip_route_barrier =
        skip_internal_barriers || ggml_metal_glm_dsa_moe_decode_skip_route_barrier_enabled();
    const bool skip_gate_up_barrier =
        skip_internal_barriers || ggml_metal_glm_dsa_moe_decode_skip_gate_up_barrier_enabled();
    auto motif_barrier = [&]() {
        if (!skip_internal_barriers) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    };
    auto route_barrier = [&]() {
        if (!skip_route_barrier) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    };
    auto gate_up_barrier = [&]() {
        if (!skip_gate_up_barrier) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    };
    auto route_gate_up_barrier = [&]() {
        if (skip_gate_up_barrier) {
            return;
        }

        if (ggml_metal_glm_dsa_moe_decode_scoped_barriers_enabled()) {
            const ggml_tensor * route_gate_up_dst =
                motif.weighted_down_input != nullptr ? motif.weighted_down_input : motif.glu;
            ggml_metal_op_concurrency_reset_tensors(ctx, route_gate_up_dst, motif.ids, motif.weights);
            return;
        }

        ggml_metal_op_concurrency_reset(ctx);
    };
    bool route_dispatched = false;
    int motif_dispatch_groups = 0;
    int motif_reference_nodes = 0;
    int motif_route_dispatches = 0;
    int motif_route_gate_up_dispatches = 0;
    int motif_gate_up_dispatches = 0;
    int motif_down_out_dispatches = 0;
    int motif_fused_tail_dispatches = 0;
    auto dispatch_route = [&]() {
        if (!route_dispatched) {
            const int route_n_fuse = motif.route.weights->op == GGML_OP_MOE_ROUTE_WEIGHTS &&
                    motif.route.n_fuse == 1 ?
                ggml_metal_op_moe_route_weights(ctx, idx) :
                ggml_metal_op_topk_moe_route_fused(ctx, idx);
            GGML_ASSERT(route_n_fuse == motif.route_n_fuse);
            route_dispatched = true;
            ++motif_dispatch_groups;
            ++motif_route_dispatches;
            route_barrier();
        }
    };

    const int q2_weight_roofline_chunks = ggml_metal_glm_dsa_moe_q2_weight_roofline_chunks();
    const int q2_weight_roofline_block_bytes = ggml_metal_glm_dsa_moe_q2_weight_roofline_block_bytes();
    if (ggml_metal_glm_moe_can_scan_q2_selected_weights(
            motif, q2_weight_roofline_chunks, q2_weight_roofline_block_bytes)) {
        if (!ggml_metal_glm_dsa_moe_q2_weight_roofline_bypass_route_enabled()) {
            dispatch_route();
        }
        return ggml_metal_op_glm_moe_q2_selected_weight_scan(
            ctx, motif, q2_weight_roofline_chunks, q2_weight_roofline_block_bytes);
    }

    int rel = motif.route_n_fuse;
    bool skipped_route_anchor = false;

    const bool fused_q3_tail_enabled = ggml_metal_glm_moe_can_fuse_swiglu_q3_down(motif);
    const bool fused_q2_tail_enabled = ggml_metal_glm_moe_can_fuse_swiglu_q2_down(motif);
    const bool fused_tail_enabled = fused_q3_tail_enabled || fused_q2_tail_enabled;
    if (fused_tail_enabled) {
        dispatch_route();
        if (motif.has_route_anchor) {
            for (int i = 0; i < 4; ++i) {
                const int n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
                GGML_ASSERT(n == 1);
                rel += n;
                ++motif_dispatch_groups;
                ++motif_reference_nodes;
                motif_barrier();
            }
        }

        const int gate_n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
        GGML_ASSERT(gate_n == 1);
        rel += gate_n;
        ++motif_dispatch_groups;
        ++motif_reference_nodes;

        const int up_n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
        GGML_ASSERT(up_n == 1);
        rel += up_n;
        ++motif_dispatch_groups;
        ++motif_reference_nodes;
        gate_up_barrier();

        if (motif.has_shared_expert_tail) {
            GGML_ASSERT(motif.shared_gate_offset >= 0);
            GGML_ASSERT(motif.shared_up_offset >= 0);
            const int shared_gate_n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + motif.shared_gate_offset);
            const int shared_up_n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + motif.shared_up_offset);
            GGML_ASSERT(shared_gate_n == 1);
            GGML_ASSERT(shared_up_n == 1);
            motif_dispatch_groups += 2;
            motif_reference_nodes += 2;
        }

        const int fused_tail = fused_q2_tail_enabled ?
            ggml_metal_op_glm_moe_swiglu_q2_down_weighted(ctx, motif) :
            ggml_metal_op_glm_moe_swiglu_q3_down_weighted(ctx, motif);
        GGML_ASSERT(fused_tail == 3);
        ++motif_dispatch_groups;
        ++motif_fused_tail_dispatches;
        rel = motif.n_fuse;
        GGML_ASSERT(rel == motif.n_fuse);
        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                    "ggml_metal: moe_dispatch op=glm_moe_decode_motif_reference_tail tensor=%s mode=%s fused_tail=%d\n",
                    ggml_metal_tensor_name(motif.out),
                    fused_q2_tail_enabled ? "swiglu_q2_down_weighted" : "swiglu_q3_down_weighted",
                    fused_tail);
        }
    } else {
        bool consumed_weighted_down = false;
        bool consumed_route_gate_up = false;
        ggml_tensor * gate_up_src1_override = nullptr;
        if (ggml_metal_glm_dsa_moe_route_anchor_bypass_enabled() &&
                motif.has_route_anchor && motif.route_anchor_src != nullptr && motif.route_anchor_src->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(motif.route_anchor_src, motif.gate->src[1])) {
            gate_up_src1_override = motif.route_anchor_src;
            rel += 4;
            skipped_route_anchor = true;
        }

        if (!motif.has_route_anchor || skipped_route_anchor) {
            const int route_gate_up_glu_n =
                ggml_metal_op_glm_moe_route_gate_up_swiglu(ctx, motif, gate_up_src1_override);
            if (route_gate_up_glu_n > 0) {
                rel += route_gate_up_glu_n;
                consumed_route_gate_up = true;
                consumed_weighted_down = motif.has_weighted_down && route_gate_up_glu_n == 4;
                ++motif_dispatch_groups;
                ++motif_route_gate_up_dispatches;
                route_gate_up_barrier();
            }
        }

        if (!consumed_route_gate_up) {
            dispatch_route();
            if (skipped_route_anchor) {
                rel = motif.route_n_fuse + 4;
            }
        }

        if (!consumed_route_gate_up && motif.has_route_anchor && !skipped_route_anchor) {
            for (int i = 0; i < 4; ++i) {
                const int n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
                GGML_ASSERT(n == 1);
                rel += n;
                ++motif_dispatch_groups;
                ++motif_reference_nodes;
                motif_barrier();
            }
        }

        if (!consumed_route_gate_up) {
            const int gate_up_glu_n = ggml_metal_op_mul_mv_id_gate_up_swiglu(
                    ctx,
                    idx + rel,
                    gate_up_src1_override);
            if (gate_up_glu_n > 0) {
                rel += gate_up_glu_n;
                consumed_weighted_down = motif.has_weighted_down && gate_up_glu_n == 4;
                ++motif_dispatch_groups;
                ++motif_gate_up_dispatches;
            } else {
                if (skipped_route_anchor) {
                    rel = motif.route_n_fuse;
                    skipped_route_anchor = false;
                    for (int i = 0; i < 4; ++i) {
                        const int n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
                        GGML_ASSERT(n == 1);
                        rel += n;
                        ++motif_dispatch_groups;
                        ++motif_reference_nodes;
                        motif_barrier();
                    }
                }
                const int gate_n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
                GGML_ASSERT(gate_n == 1);
                rel += gate_n;
                ++motif_dispatch_groups;
                ++motif_reference_nodes;

                const int up_n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
                GGML_ASSERT(up_n == 1);
                rel += up_n;
                ++motif_dispatch_groups;
                ++motif_reference_nodes;
                gate_up_barrier();

                const int glu_n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
                GGML_ASSERT(glu_n == 1);
                rel += glu_n;
                ++motif_dispatch_groups;
                ++motif_reference_nodes;
            }
            gate_up_barrier();
        }

        if (motif.has_weighted_down && !consumed_weighted_down) {
            const int weighted_n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
            GGML_ASSERT(weighted_n == 1);
            rel += weighted_n;
            ++motif_dispatch_groups;
            ++motif_reference_nodes;
            motif_barrier();
        }

        const int down_out_n = ggml_metal_op_mul_mv_id_weighted_reduce(ctx, idx + rel, true);
        if (down_out_n > 0) {
            rel += down_out_n;
            ++motif_dispatch_groups;
            ++motif_down_out_dispatches;
        } else {
            while (rel < motif.n_fuse) {
                const int n = ggml_metal_op_reference_dispatch_existing_node(ctx, idx + rel);
                GGML_ASSERT(n == 1);
                rel += n;
                ++motif_dispatch_groups;
                ++motif_reference_nodes;
                motif_barrier();
            }
        }
        GGML_ASSERT(rel == motif.n_fuse);
    }

    if (motif.has_shared_expert_tail) {
        ggml_metal_op_concurrency_reset(ctx);
    }

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
                "ggml_metal: moe_dispatch op=glm_moe_decode_motif_reference tensor=%s last_node=%s fused_nodes=%d route_nodes=%d route_anchor=%d route_anchor_bypassed=%d weighted_down=%d shared_expert_tail=%d dispatch_groups=%d reference_nodes=%d route_dispatches=%d route_gate_up_dispatches=%d gate_up_dispatches=%d down_out_dispatches=%d fused_tail_dispatches=%d graph_uid=%llu split_start=%d split_end=%d\n",
                ggml_metal_tensor_name(motif.out),
                ggml_metal_tensor_name(ctx->node(idx + motif.n_fuse - 1)),
                motif.n_fuse,
                motif.route_n_fuse,
                motif.has_route_anchor ? 1 : 0,
                skipped_route_anchor ? 1 : 0,
                motif.has_weighted_down ? 1 : 0,
                motif.has_shared_expert_tail ? 1 : 0,
                motif_dispatch_groups,
                motif_reference_nodes,
                motif_route_dispatches,
                motif_route_gate_up_dispatches,
                motif_gate_up_dispatches,
                motif_down_out_dispatches,
                motif_fused_tail_dispatches,
                (unsigned long long) ctx->graph_uid(),
                ctx->split_start(),
                ctx->split_end());
    }

    return motif.n_fuse;
}

static bool ggml_metal_match_mul_mv_id_gate_up_swiglu(
        ggml_metal_op_t ctx,
        int idx,
        ggml_metal_mul_mv_id_gate_up_swiglu_fusion & fusion) {
    const bool explicit_enabled =
        ggml_metal_glm_dsa_q2_gate_up_swiglu_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_vecscale_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_any_variant_enabled();
    const bool default_enabled = ggml_metal_glm_dsa_q2_gate_up_swiglu_default_enabled();
    const bool enabled = explicit_enabled || default_enabled || ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_default_enabled();
    if (!ctx->use_fusion || !enabled || idx + 3 > ctx->n_nodes()) {
        return false;
    }

    ggml_tensor * first  = ctx->node(idx);
    ggml_tensor * second = ctx->node(idx + 1);
    ggml_tensor * glu    = ctx->node(idx + 2);
    ggml_tensor * next = idx + 3 < ctx->n_nodes() ? ctx->node(idx + 3) : nullptr;
    if (first->op != GGML_OP_MUL_MAT_ID || second->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU) {
        return false;
    }

    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }

    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up   = glu->src[1];
    const bool same_pair = (gate == first && up == second) || (gate == second && up == first);
    if (!same_pair) {
        return false;
    }

    if (up->src[0] == nullptr || gate->src[0] == nullptr ||
            up->src[1] == nullptr || gate->src[1] == nullptr ||
            up->src[2] == nullptr || gate->src[2] == nullptr) {
        return false;
    }

    ggml_tensor * weights = nullptr;
    ggml_tensor * cast = nullptr;
    ggml_tensor * weighted = next;
    if (next != nullptr && next->op == GGML_OP_CPY && next->src[0] == glu && next->type == GGML_TYPE_F16 &&
            ggml_is_contiguous_1(next)) {
        cast = next;
        weighted = nullptr;
    }
    if (weighted != nullptr && weighted->op == GGML_OP_MUL &&
            ggml_metal_tensor_name_contains(weighted, "ffn_moe_down_weighted_input")) {
        if (weighted->src[0] == glu) {
            weights = weighted->src[1];
        } else if (weighted->src[1] == glu) {
            weights = weighted->src[0];
        }
    }

    const bool weighted_shape_ok =
        weights == nullptr ||
        (weighted != nullptr &&
         weighted->type == GGML_TYPE_F32 &&
         weights->type == GGML_TYPE_F32 &&
         weighted->ne[0] == glu->ne[0] &&
         weighted->ne[1] == glu->ne[1] &&
         weighted->ne[2] == glu->ne[2] &&
         weights->ne[0] == 1 &&
         weights->ne[1] == glu->ne[1] &&
         weights->ne[2] == glu->ne[2] &&
         ggml_is_contiguous_1(weighted));

    const bool cast_shape_ok =
        cast == nullptr ||
        (cast->ne[0] == glu->ne[0] &&
         cast->ne[1] == glu->ne[1] &&
         cast->ne[2] == glu->ne[2] &&
         cast->ne[3] == glu->ne[3]);

    const bool shape_ok =
        up->src[0]->type == GGML_TYPE_Q2_K &&
        gate->src[0]->type == GGML_TYPE_Q2_K &&
        up->src[1]->type == GGML_TYPE_F32 &&
        gate->src[1] == up->src[1] &&
        gate->src[2] == up->src[2] &&
        up->type == GGML_TYPE_F32 &&
        gate->type == GGML_TYPE_F32 &&
        glu->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(up->src[0], gate->src[0]) &&
        ggml_are_same_stride(up->src[0], gate->src[0]) &&
        ggml_are_same_shape(up, gate) &&
        ggml_are_same_shape(up, glu) &&
        up->src[0]->ne[0] >= 2048 &&
        up->src[0]->ne[1] >= 1024 &&
        up->src[2]->ne[0] == 8 &&
        up->src[2]->ne[1] >= 1 &&
        up->src[2]->ne[1] <= 32 &&
        up->src[0]->ne[0] % ggml_blck_size(up->src[0]->type) == 0 &&
        up->src[0]->ne[1] > 0 &&
        up->src[0]->ne[2] >= up->src[2]->ne[0] &&
        cast_shape_ok &&
        weighted_shape_ok;
    if (!shape_ok) {
        return false;
    }
    const bool default_pair_sg_shape =
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_default_enabled() &&
        up->src[0]->ne[0] == 6144 &&
        up->src[0]->ne[1] == 2048 &&
        up->src[2]->ne[0] == 8 &&
        up->src[2]->ne[1] == 1;
    const bool default_plain_shape =
        default_enabled &&
        up->src[0]->ne[0] == 6144 &&
        up->src[0]->ne[1] == 2048 &&
        up->src[2]->ne[0] == 8 &&
        up->src[2]->ne[1] == 1;
    if (!explicit_enabled && !default_pair_sg_shape && !default_plain_shape) {
        return false;
    }

    if (weights != nullptr) {
        const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU, GGML_OP_MUL };
        const int outputs[] = { 3 };
        if (!ctx->can_fuse_subgraph(idx, ops, 4, outputs, 1)) {
            return false;
        }
    } else if (cast != nullptr) {
        const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU, GGML_OP_CPY };
        const int outputs[] = { 3 };
        if (!ctx->can_fuse_subgraph(idx, ops, 4, outputs, 1)) {
            return false;
        }
    } else {
        const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
        const int outputs[] = { 2 };
        if (!ctx->can_fuse_subgraph(idx, ops, 3, outputs, 1)) {
            return false;
        }
    }

    fusion.up = up;
    fusion.gate = gate;
    fusion.glu = glu;
    fusion.cast = cast;
    fusion.weighted = weights != nullptr ? weighted : nullptr;
    fusion.weights = weights;
    fusion.q8 = up->src[3] != nullptr &&
            up->src[3] == gate->src[3] &&
            up->src[3]->type == GGML_TYPE_Q8_0 &&
            ggml_are_same_shape(up->src[3], up->src[1]) ? up->src[3] : nullptr;
    fusion.n_fuse = (weights != nullptr || cast != nullptr) ? 4 : 3;
    return true;
}

static int ggml_metal_op_mul_mv_id_gate_up_swiglu(
        ggml_metal_op_t ctx,
        int idx,
        ggml_tensor * src1_override) {
    ggml_metal_mul_mv_id_gate_up_swiglu_fusion fusion;
    if (!ggml_metal_match_mul_mv_id_gate_up_swiglu(ctx, idx, fusion)) {
        return 0;
    }

    ggml_tensor * up = fusion.up;
    ggml_tensor * gate = fusion.gate;
    ggml_tensor * glu = fusion.glu;
    ggml_tensor * dst = fusion.weighted != nullptr ? fusion.weighted : (fusion.cast != nullptr ? fusion.cast : fusion.glu);
    ggml_tensor * weights = fusion.weights;
    ggml_tensor * q8 = fusion.q8;
    ggml_tensor * src1 = src1_override != nullptr ? src1_override : up->src[1];
    if (src1_override != nullptr &&
            (src1_override->type != GGML_TYPE_F32 || !ggml_are_same_shape(src1_override, up->src[1]))) {
        return 0;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    if (ggml_metal_glm_routed_expert_noop_enabled() && dst->type == GGML_TYPE_F32) {
        auto pipeline = ggml_metal_library_get_pipeline_zero_f32(lib);
        const int nth = 256;
        const int64_t n_tg = (ggml_nelements(dst) + nth - 1)/nth;
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(dst), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, nth, 1, 1);
        ggml_metal_op_concurrency_reset(ctx);
        return fusion.n_fuse;
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, up->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, up->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, src1,       ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, src1,       nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, up->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, up->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  dst,        ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  dst,        nb);
    const uint64_t weights_nb1 = weights != nullptr ? weights->nb[1] : 0;
    const uint64_t weights_nb2 = weights != nullptr ? weights->nb[2] : 0;

    const bool pair_sg_default_shape =
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_default_enabled() &&
        ne00 == 6144 &&
        ne01 == 2048 &&
        ne20 == 8 &&
        ne21 == 1;
    const bool pair_sg =
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_enabled() ||
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_any_variant_enabled() ||
        pair_sg_default_shape;
    if (dst->type == GGML_TYPE_F16 && !pair_sg) {
        return 0;
    }
    const bool pair_sg_slot8 = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot8_enabled();
    const bool pair_sg_slot2 = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot2_enabled();
    const bool pair_sg_slot4_dual_default = pair_sg_default_shape && src1_override != nullptr;
    const bool pair_sg_slot4_dual = pair_sg && dst->type == GGML_TYPE_F32 &&
        (ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot4_dual_enabled() ||
         pair_sg_slot4_dual_default);
    const bool pair_sg_slot4_dual_r12 = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot4_dual_r12_enabled();
    const bool pair_sg_slot4_dual_r16 = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot4_dual_r16_enabled();
    const bool pair_sg_slot1_dual = pair_sg && dst->type == GGML_TYPE_F32 &&
        (ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot1_dual_enabled() ||
         (pair_sg_default_shape && !pair_sg_slot4_dual_default));
    const bool pair_sg_slot2_dual = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot2_dual_enabled();
    const bool pair_sg_slot8_split = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_slot8_split_enabled();
    const bool pair_sg_share_y = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_share_y_enabled() &&
        ne00 == 6144 &&
        ne01 == 2048 &&
        ne20 == 8 &&
        ne21 == 1 &&
        src1->type == GGML_TYPE_F32;
    const bool pair_sg_vecscale = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_vecscale_enabled() &&
        ne00 == 6144 &&
        ne01 == 2048 &&
        ne20 == 8 &&
        ne21 == 1 &&
        src1->type == GGML_TYPE_F32;
    const bool pair_sg_q8_act = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_q8_act_enabled() &&
        ne00 == 6144 &&
        ne01 == 2048 &&
        ne20 == 8 &&
        ne21 == 1 &&
        src1->type == GGML_TYPE_F32;
    const bool pair_sg_prequant_q8 = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_prequant_q8_enabled() &&
        q8 != nullptr &&
        ne00 == 6144 &&
        ne01 == 2048 &&
        ne20 == 8 &&
        ne21 == 1 &&
        src1->type == GGML_TYPE_F32;
    const bool pair_sg_inblock_q2 = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_inblock_repack_enabled() &&
        ne00 == 6144 &&
        ne01 == 2048 &&
        ne20 == 8 &&
        ne21 == 1 &&
        src1->type == GGML_TYPE_F32;
    const bool pair_sg_half_y = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_half_y_enabled() &&
        ne00 == 6144 &&
        ne01 == 2048 &&
        ne20 == 8 &&
        ne21 == 1 &&
        src1->type == GGML_TYPE_F32;
    const bool pair_sg_rowtile = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_rowtile_enabled() &&
        ne00 == 6144 &&
        ne01 == 2048 &&
        ne20 == 8 &&
        ne21 == 1 &&
        src1->type == GGML_TYPE_F32;
    const bool pair_sg_r16 = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_r16_enabled();
    const bool pair_sg_r12 = pair_sg && dst->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_pair_sg_r12_enabled();
    const bool plain_vecscale = !pair_sg &&
        ggml_metal_glm_dsa_q2_gate_up_swiglu_vecscale_enabled() &&
        dst->type == GGML_TYPE_F32 &&
        src1->type == GGML_TYPE_F32;
    ggml_metal_pipeline_with_params pipeline = {};
    pipeline = pair_sg_inblock_q2 ?
        ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot1_dual_inblock_q2(lib, glu) :
        (pair_sg_prequant_q8 ?
        ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot1_dual_prequant_q8(lib, glu) :
        (pair_sg ?
        (pair_sg_r12 ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_r12(lib, dst) :
        (pair_sg_r16 ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_r16(lib, dst) :
        (pair_sg_slot2 ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot2(lib, glu) :
        (pair_sg_slot1_dual ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot1_dual(lib, glu) :
        (pair_sg_slot2_dual ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot2_dual(lib, glu) :
        (pair_sg_slot4_dual_r12 ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot4_dual_r12(lib, glu) :
        (pair_sg_slot4_dual_r16 ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot4_dual_r16(lib, glu) :
        (pair_sg_slot4_dual ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot4_dual(lib, glu) :
        (pair_sg_slot8_split ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot8_split(lib, glu) :
        (pair_sg_slot8 ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_slot8(lib, glu) :
        (pair_sg_share_y ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_share_y(lib, glu) :
        (pair_sg_q8_act ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_q8_act(lib, glu) :
        (pair_sg_half_y ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_half_y(lib, glu) :
        (pair_sg_rowtile ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_rowtile(lib, glu) :
        (pair_sg_vecscale ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_vecscale(lib, glu) :
        (dst->type == GGML_TYPE_F16 ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg_f16(lib, dst) :
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_pair_sg(lib, glu))))))))))))))))) :
        (plain_vecscale ?
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu_vecscale(lib, glu) :
            ggml_metal_library_get_pipeline_mul_mv_id_gate_up_swiglu(lib, glu))));
    const int nr0 = pipeline.nr0;
    const int nr1 = pipeline.nr1;
    const int nsg = pipeline.nsg;

    ggml_metal_kargs_mul_mv_id_gate_up_swiglu args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb1  =*/ nb1,
        /*.nr0  =*/ nr0,
        /*.weights_nb1 =*/ weights_nb1,
        /*.weights_nb2 =*/ weights_nb2,
        /*.weighted =*/ weights != nullptr ? 1 : 0,
        /*._pad0 =*/ ggml_metal_glm_dsa_moe_max_active_experts(),
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(up->src[0]),   1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(gate->src[0]), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src1),         3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(dst),          4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(up->src[2]),   5);
    ggml_metal_encoder_set_buffer(enc, weights != nullptr ?
            ggml_metal_get_buffer_id(weights) : ggml_metal_get_buffer_id(up->src[2]), 6);
    ggml_metal_encoder_set_buffer(enc, q8 != nullptr ?
            ggml_metal_get_buffer_id(q8) : ggml_metal_get_buffer_id(src1), 7);

    int slots_per_threadgroup = 1;
    if (pair_sg) {
        if (pair_sg_inblock_q2 || pair_sg_prequant_q8) {
            slots_per_threadgroup = 1;
        } else if (pair_sg_r12 || pair_sg_r16 || pair_sg_slot4_dual ||
                   pair_sg_slot4_dual_r12 || pair_sg_slot4_dual_r16 ||
                   pair_sg_share_y || pair_sg_vecscale || pair_sg_q8_act ||
                   pair_sg_half_y || pair_sg_rowtile) {
            slots_per_threadgroup = 4;
        } else if (pair_sg_slot8 || pair_sg_slot8_split) {
            slots_per_threadgroup = 8;
        } else if (pair_sg_slot2 || pair_sg_slot2_dual) {
            slots_per_threadgroup = 2;
        } else if (pair_sg_slot1_dual) {
            slots_per_threadgroup = 1;
        } else {
            slots_per_threadgroup = 4;
        }
    }
    const int64_t ne123 = pair_sg ?
        ((ne20 + slots_per_threadgroup - 1)/slots_per_threadgroup)*ne21 :
        ne20*ne21;
    const int grid_x = pair_sg ? (ne01 + nr0 - 1)/nr0 : (ne01 + nr0*nsg - 1)/(nr0*nsg);
    const int grid_y = (1 + nr1 - 1)/nr1;
    const int grid_z = ne123;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        const char * gate_up_op =
            pair_sg_r12 ? "mul_mv_id_q2_gate_up_swiglu_r12" :
            pair_sg_r16 ? "mul_mv_id_q2_gate_up_swiglu_r16" :
            pair_sg_slot2 ? "mul_mv_id_q2_gate_up_swiglu_slot2" :
            pair_sg_slot1_dual ? "mul_mv_id_q2_gate_up_swiglu_slot1_dual" :
            pair_sg_slot2_dual ? "mul_mv_id_q2_gate_up_swiglu_slot2_dual" :
            pair_sg_slot4_dual_r12 ? "mul_mv_id_q2_gate_up_swiglu_slot4_dual_r12" :
            pair_sg_slot4_dual_r16 ? "mul_mv_id_q2_gate_up_swiglu_slot4_dual_r16" :
            pair_sg_slot4_dual ? "mul_mv_id_q2_gate_up_swiglu_slot4_dual" :
            pair_sg_slot8_split ? "mul_mv_id_q2_gate_up_swiglu_slot8_split" :
            pair_sg_slot8 ? "mul_mv_id_q2_gate_up_swiglu_slot8" :
            pair_sg_share_y ? "mul_mv_id_q2_gate_up_swiglu_share_y" :
            pair_sg_inblock_q2 ? "mul_mv_id_q2_gate_up_swiglu_inblock_q2" :
            pair_sg_q8_act ? "mul_mv_id_q2_gate_up_swiglu_q8_act" :
            pair_sg_prequant_q8 ? "mul_mv_id_q2_gate_up_swiglu_prequant_q8" :
            pair_sg_half_y ? "mul_mv_id_q2_gate_up_swiglu_half_y" :
            pair_sg_rowtile ? "mul_mv_id_q2_gate_up_swiglu_rowtile" :
            (pair_sg_vecscale || plain_vecscale) ? "mul_mv_id_q2_gate_up_swiglu_vecscale" :
            pair_sg ? "mul_mv_id_q2_gate_up_swiglu_pair_sg" :
            "mul_mv_id_q2_gate_up_swiglu";
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=%s tensor=%s up=%s gate=%s weights=%s src1=%s src1_override=%d src0_type=%s src1_type=%s ids_type=%s dst_type=%s ne00=%d ne01=%d experts=%d used_experts=%d tokens=%d weighted=%d nr0=%d nr1=%d nsg=%d grid_x=%d grid_y=%d grid_z=%d threads_x=%d threads_y=%d fused_nodes=%d\n",
            gate_up_op,
            ggml_metal_tensor_name(dst),
            ggml_metal_tensor_name(up),
            ggml_metal_tensor_name(gate),
            ggml_metal_tensor_name(weights),
            ggml_metal_tensor_name(src1),
            src1_override != nullptr ? 1 : 0,
            ggml_type_name(up->src[0]->type),
            ggml_type_name(src1->type),
            ggml_type_name(up->src[2]->type),
            ggml_type_name(dst->type),
            ne00,
            ne01,
            ne02,
            ne20,
            ne21,
            weights != nullptr ? 1 : 0,
            nr0,
            nr1,
            nsg,
            grid_x,
            grid_y,
            grid_z,
            32,
            nsg,
            fusion.n_fuse);
    }

    // The generic concurrency tracker only sees the first node in this fused
    // span. Publish the fused destination before a later node consumes it.
    ggml_metal_op_concurrency_reset(ctx);

    return fusion.n_fuse;
}

static bool ggml_metal_match_mul_mv_id_weighted_reduce(
        ggml_metal_op_t ctx,
        int idx,
        ggml_metal_mul_mv_id_weighted_reduce_fusion & fusion,
        bool subgraph_owned) {
    const bool enabled =
        ggml_metal_glm_dsa_q2_down_weighted_reduce_enabled() ||
        ggml_metal_glm_dsa_q2_down_f16_act_enabled() ||
        ggml_metal_glm_dsa_q2_down_vec_scale_enabled() ||
        ggml_metal_glm_dsa_q3_down_weighted_reduce_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w0_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r6_nb8_w0_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r10_nb8_w0_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_glm52_w0_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w0_default_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w1_default_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r12_nb8_w0_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r16_enabled() ||
        ggml_metal_glm_dsa_q3_down_slot_split2_reduce_enabled() ||
        ggml_metal_glm_dsa_q3_down_atomic_accum_enabled() ||
        ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_enabled() ||
        ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_default_enabled() ||
        ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r4_enabled() ||
        ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r16_enabled() ||
        ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r16_w1_enabled();
    if (!ctx->use_fusion || !enabled || idx + 2 > ctx->n_nodes()) {
        if (enabled && ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                "ggml: glm_dsa_q2_down_weighted_reduce_reject idx=%d reason=precheck use_fusion=%d n_nodes=%d\n",
                idx,
                ctx->use_fusion ? 1 : 0,
                ctx->n_nodes());
        }
        return false;
    }

    ggml_tensor * down = ctx->node(idx);
    ggml_tensor * weighted_sum = nullptr;
    ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_up = nullptr;
    int weighted_sum_offset = -1;
    int shared_gate_offset = -1;
    int shared_up_offset = -1;

    for (int rel = 1; rel < 4 && idx + rel < ctx->n_nodes(); ++rel) {
        ggml_tensor * candidate = ctx->node(idx + rel);
        if (candidate->op == GGML_OP_MOE_WEIGHTED_SUM && candidate->src[0] == down &&
                ggml_metal_tensor_name_contains(candidate, "ffn_moe_out")) {
            weighted_sum = candidate;
            weighted_sum_offset = rel;
        } else if (candidate->op == GGML_OP_MUL_MAT && ggml_metal_tensor_is_shared_expert_gate(candidate)) {
            shared_gate = candidate;
            shared_gate_offset = rel;
        } else if (candidate->op == GGML_OP_MUL_MAT && ggml_metal_tensor_is_shared_expert_up(candidate)) {
            shared_up = candidate;
            shared_up_offset = rel;
        }
    }

    auto reject = [&](const char * reason) {
        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                "ggml: glm_dsa_q2_down_weighted_reduce_reject idx=%d reason=%s node0=%s:%s shared_gate=%s:%s shared_up=%s:%s weighted_sum=%s:%s\n",
                idx,
                reason,
                ggml_metal_tensor_name(down),
                ggml_op_name(down->op),
                ggml_metal_tensor_name(shared_gate),
                shared_gate ? ggml_op_name(shared_gate->op) : "none",
                ggml_metal_tensor_name(shared_up),
                shared_up ? ggml_op_name(shared_up->op) : "none",
                ggml_metal_tensor_name(weighted_sum),
                weighted_sum ? ggml_op_name(weighted_sum->op) : "none");
        }
        return false;
    };

    if (down->op != GGML_OP_MUL_MAT_ID || weighted_sum == nullptr) {
        return reject("op_window");
    }
    if ((shared_gate == nullptr) != (shared_up == nullptr)) {
        return reject("partial_shared_window");
    }
    if (weighted_sum->src[0] != down || weighted_sum->src[1] == nullptr) {
        return reject("weighted_sum_sources");
    }
    if (!ggml_metal_tensor_name_contains(down, "ffn_moe_down") ||
            !ggml_metal_tensor_name_contains(weighted_sum, "ffn_moe_out")) {
        return reject("tensor_names");
    }
    if (shared_gate != nullptr &&
            (!ggml_metal_tensor_is_shared_expert_gate(shared_gate) ||
             !ggml_metal_tensor_is_shared_expert_up(shared_up))) {
        return reject("tensor_names");
    }
    if (down->src[0] == nullptr || down->src[1] == nullptr || down->src[2] == nullptr) {
        return reject("down_sources");
    }

    const ggml_type down_type = down->src[0]->type;
    const bool down_type_ok = down_type == GGML_TYPE_Q2_K || down_type == GGML_TYPE_Q3_K;
    const bool q2_default_shape =
        down_type == GGML_TYPE_Q2_K &&
        ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_default_enabled() &&
        down->src[0]->ne[0] == 2048 &&
        down->src[2]->ne[0] == 8 &&
        ggml_get_op_params_i32(weighted_sum, 0) == 0;
    if (down_type == GGML_TYPE_Q2_K &&
            !ggml_metal_glm_dsa_q2_down_weighted_reduce_enabled() &&
            !ggml_metal_glm_dsa_q2_down_f16_act_enabled() &&
            !ggml_metal_glm_dsa_q2_down_vec_scale_enabled() &&
            !ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_enabled() &&
            !q2_default_shape &&
            !ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r4_enabled() &&
            !ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r16_enabled() &&
            !ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r16_w1_enabled()) {
        return reject("q2_direct_disabled");
    }
    if (down_type == GGML_TYPE_Q2_K &&
            ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_enabled() &&
            down->src[0]->ne[0] != 2048) {
        return reject("q2_slot_parallel_shape");
    }
    const bool shared_shape_ok =
        shared_gate == nullptr ||
        (shared_gate->type == GGML_TYPE_F32 && shared_up->type == GGML_TYPE_F32);
    const bool glm_down_shape =
        down->src[0]->ne[0] >= 1024 &&
        down->src[0]->ne[1] >= 2048;
    const bool src1_type_ok =
        down->src[1]->type == GGML_TYPE_F32 ||
        (down_type == GGML_TYPE_Q2_K &&
         down->src[1]->type == GGML_TYPE_F16 &&
         ggml_metal_glm_dsa_q2_down_f16_act_enabled()) ||
        (down_type == GGML_TYPE_Q3_K &&
         down->src[1]->type == GGML_TYPE_F16 &&
         ggml_metal_glm_dsa_q3_down_f16_act_enabled());
    const bool shape_ok =
        down_type_ok &&
        glm_down_shape &&
        src1_type_ok &&
        down->src[2]->type == GGML_TYPE_I32 &&
        weighted_sum->src[1]->type == GGML_TYPE_F32 &&
        down->type == GGML_TYPE_F32 &&
        shared_shape_ok &&
        weighted_sum->type == GGML_TYPE_F32 &&
        down->src[0]->ne[0] % ggml_blck_size(down_type) == 0 &&
        down->src[0]->ne[0] == down->src[1]->ne[0] &&
        down->src[0]->ne[1] == weighted_sum->ne[0] &&
        down->src[0]->ne[2] >= down->src[2]->ne[0] &&
        down->src[2]->ne[0] == 8 &&
        down->src[1]->ne[1] == 8 &&
        down->src[1]->ne[2] == down->src[2]->ne[1] &&
        weighted_sum->src[1]->ne[0] == 1 &&
        weighted_sum->src[1]->ne[1] == down->src[2]->ne[0] &&
        weighted_sum->src[1]->ne[2] == down->src[2]->ne[1] &&
        weighted_sum->ne[0] == down->ne[0] &&
        weighted_sum->ne[1] == down->ne[2];
    if (!shape_ok) {
        return reject("shape");
    }
    const bool q3_default_w0_shape =
        down_type == GGML_TYPE_Q3_K &&
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w0_default_enabled() &&
        down->src[0]->ne[0] == 2048 &&
        down->src[2]->ne[1] == 1 &&
        ggml_get_op_params_i32(weighted_sum, 0) == 0;
    const bool q3_default_w1_shape =
        down_type == GGML_TYPE_Q3_K &&
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w1_default_enabled() &&
        down->src[0]->ne[0] == 2048 &&
        down->src[2]->ne[1] == 1 &&
        ggml_get_op_params_i32(weighted_sum, 0) != 0;
    if (down_type == GGML_TYPE_Q3_K &&
            !ggml_metal_glm_dsa_q3_down_weighted_reduce_tensor_selected(weighted_sum)) {
        return reject("q3_tensor");
    }
    if (down_type == GGML_TYPE_Q3_K &&
            !q3_default_w0_shape &&
            !q3_default_w1_shape &&
            !ggml_metal_glm_dsa_q3_down_weighted_reduce_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w0_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r6_nb8_w0_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r10_nb8_w0_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_glm52_w0_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r12_nb8_w0_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r16_enabled() &&
            !ggml_metal_glm_dsa_q3_down_slot_split2_reduce_enabled() &&
            !ggml_metal_glm_dsa_q3_down_atomic_accum_enabled()) {
        return reject("q3_direct_disabled");
    }

    const int n_fuse = std::max({
        weighted_sum_offset,
        shared_gate_offset,
        shared_up_offset,
    }) + 1;
    ggml_op ops[4];
    for (int rel = 0; rel < n_fuse; ++rel) {
        ops[rel] = ctx->node(idx + rel)->op;
    }
    int outputs[3];
    int n_outputs = 0;
    outputs[n_outputs++] = weighted_sum_offset;
    if (shared_gate != nullptr) {
        outputs[n_outputs++] = shared_gate_offset;
        outputs[n_outputs++] = shared_up_offset;
    }
    if (!subgraph_owned && !ctx->can_fuse_subgraph(idx, ops, n_fuse, outputs, n_outputs)) {
        return reject("can_fuse_subgraph");
    }

    fusion.down = down;
    fusion.shared_gate = shared_gate;
    fusion.shared_up = shared_up;
    fusion.weighted_sum = weighted_sum;
    fusion.shared_gate_offset = shared_gate_offset;
    fusion.shared_up_offset = shared_up_offset;
    fusion.weighted_sum_offset = weighted_sum_offset;
    fusion.n_fuse = n_fuse;
    return true;
}

static int ggml_metal_op_mul_mv_id_weighted_reduce(
        ggml_metal_op_t ctx,
        int idx,
        bool subgraph_owned) {
    ggml_metal_mul_mv_id_weighted_reduce_fusion fusion;
    if (!ggml_metal_match_mul_mv_id_weighted_reduce(ctx, idx, fusion, subgraph_owned)) {
        return 0;
    }
    if (ggml_metal_glm_dsa_q2_down_weighted_reduce_noop_enabled()) {
        return 0;
    }

    ggml_tensor * down = fusion.down;
    ggml_tensor * weighted_sum = fusion.weighted_sum;

    if (ggml_metal_glm_dsa_q2_down_weighted_reduce_reference_enabled()) {
        for (int rel = 0; rel < fusion.n_fuse; ++rel) {
            int encoded = 0;
            if (rel == 0) {
                encoded = ggml_metal_op_mul_mat_id(ctx, idx);
            } else if (rel == fusion.shared_gate_offset || rel == fusion.shared_up_offset) {
                encoded = ggml_metal_op_mul_mat(ctx, idx + rel);
            } else if (rel == fusion.weighted_sum_offset) {
                encoded = ggml_metal_op_moe_weighted_sum(ctx, idx + rel);
            }
            GGML_ASSERT(encoded == 1);
            ggml_metal_op_concurrency_reset(ctx);
        }
        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                "ggml: glm_dsa_metal_dispatch op=mul_mv_id_q2_down_weighted_reduce_reference tensor=%s down=%s weights=%s src0_type=%s dst_type=%s grid_x=1 grid_y=1 grid_z=1 threads_x=1 fused_nodes=%d\n",
                ggml_metal_tensor_name(weighted_sum),
                ggml_metal_tensor_name(down),
                ggml_metal_tensor_name(weighted_sum->src[1]),
                ggml_type_name(down->src[0]->type),
                ggml_type_name(weighted_sum->type),
                fusion.n_fuse);
        }
        return fusion.n_fuse;
    }

    auto encode_shared = [&]() {
        if (fusion.shared_gate == nullptr) {
            return;
        }
        const int shared_gate_fuse = ggml_metal_op_mul_mat(ctx, idx + fusion.shared_gate_offset);
        const int shared_up_fuse = ggml_metal_op_mul_mat(ctx, idx + fusion.shared_up_offset);
        GGML_ASSERT(shared_gate_fuse == 1);
        GGML_ASSERT(shared_up_fuse == 1);
        ggml_metal_op_concurrency_reset(ctx);
    };

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, down->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, down->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, down->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, down->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, down->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, down->src[2], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb3, weighted_sum->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ned, down, ne);
    GGML_TENSOR_LOCALS(uint64_t, nbd, down, nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  weighted_sum, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  weighted_sum, nb);

    const bool preserve_routed_input = fusion.shared_gate != nullptr;
    const uint64_t scratch_nb0 = ggml_type_size(down->src[1]->type);
    const uint64_t scratch_nb1 = scratch_nb0*ne10;
    const uint64_t scratch_nb2 = scratch_nb1*ne11;
    const uint64_t scratch_nb3 = scratch_nb2*ne12;
    ggml_metal_buffer_id fusion_scratch = ggml_metal_get_buffer_id(down);
    fusion_scratch.offs += ggml_nbytes(down);
    fusion_scratch.offs += ggml_metal_op_mul_mat_id_extra_tpe(down);
    fusion_scratch.offs += ggml_metal_op_mul_mat_id_extra_ids(down);
    ggml_metal_buffer_id routed_input = ggml_metal_get_buffer_id(down->src[1]);
    if (preserve_routed_input) {
        GGML_ASSERT(ggml_blck_size(down->src[1]->type) == 1);
        GGML_ASSERT(scratch_nb3*ne13 <= ggml_metal_op_mul_mat_id_extra_src1_scratch(down));

        routed_input = fusion_scratch;

        auto copy_pipeline = ggml_metal_library_get_pipeline_cpy(
            lib, down->src[1]->type, down->src[1]->type);
        ggml_metal_kargs_cpy copy_args = {
            /*.nk0  =*/ ne10,
            /*.ne00 =*/ ne10,
            /*.ne01 =*/ ne11,
            /*.ne02 =*/ ne12,
            /*.ne03 =*/ ne13,
            /*.nb00 =*/ nb10,
            /*.nb01 =*/ nb11,
            /*.nb02 =*/ nb12,
            /*.nb03 =*/ nb13,
            /*.ne0  =*/ ne10,
            /*.ne1  =*/ ne11,
            /*.ne2  =*/ ne12,
            /*.ne3  =*/ ne13,
            /*.nb0  =*/ scratch_nb0,
            /*.nb1  =*/ scratch_nb1,
            /*.nb2  =*/ scratch_nb2,
            /*.nb3  =*/ scratch_nb3,
        };
        const int nth = std::min<int>(
            ggml_metal_pipeline_max_theads_per_threadgroup(copy_pipeline), ne10);
        const int nw0 = (ne10 + nth - 1)/nth;

        ggml_metal_encoder_set_pipeline(enc, copy_pipeline);
        ggml_metal_encoder_set_bytes(enc, &copy_args, sizeof(copy_args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[1]), 1);
        ggml_metal_encoder_set_buffer(enc, routed_input, 2);
        ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne11, ne12, ne13, nth, 1, 1);
    }

    // The shared experts may reuse the routed activation's allocation after
    // the original down node. Preserve that activation first, then keep the
    // copy and both shared matvecs in the same concurrent Metal span.
    encode_shared();

    // The fused kernel consumes selected-expert IDs through the down node even
    // when it replaces the following weighted reduction. Make the producer's
    // writes visible before this manually encoded cross-node read.
    ggml_metal_encoder_memory_barrier_buffer(
        enc, ggml_metal_get_buffer_id(down->src[2]));

    const bool use_output_scratch =
        down->src[0]->type == GGML_TYPE_Q3_K && !preserve_routed_input;
    const ggml_metal_buffer_id weighted_reduce_dst = use_output_scratch ?
        fusion_scratch : ggml_metal_get_buffer_id(weighted_sum);

    if (ggml_metal_glm_routed_expert_noop_enabled() && weighted_sum->type == GGML_TYPE_F32) {
        auto pipeline = ggml_metal_library_get_pipeline_zero_f32(lib);
        const int nth = 256;
        const int64_t n_tg = (ggml_nelements(weighted_sum) + nth - 1)/nth;
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer(enc, weighted_reduce_dst, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, nth, 1, 1);
        ggml_metal_op_concurrency_reset(ctx);
        return fusion.n_fuse;
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id_weighted_reduce(lib, weighted_sum);
    const int nr0 = pipeline.nr0;
    const int nr1 = pipeline.nr1;
    const int nsg = pipeline.nsg;

    ggml_metal_kargs_mul_mv_id args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ preserve_routed_input ? scratch_nb0 : nb10,
        /*.nb11 =*/ preserve_routed_input ? scratch_nb1 : nb11,
        /*.nb12 =*/ preserve_routed_input ? scratch_nb2 : nb12,
        /*.ne0  =*/ ned0,
        /*.ne1  =*/ ned1,
        /*.nb1  =*/ nbd1,
        /*.nr0  =*/ nr0,
    };
    ggml_metal_kargs_mul_mv_id_weighted_reduce_extra extra = {
        /*.weights_nb1 =*/ nb31,
        /*.weights_nb2 =*/ nb32,
        /*.dst_nb0 =*/ use_output_scratch ? sizeof(float) : nb0,
        /*.dst_nb1 =*/ use_output_scratch ? sizeof(float)*ne0 : nb1,
        /*.already_weighted =*/ ggml_get_op_params_i32(weighted_sum, 0) != 0 ? 1 : 0,
        /*._pad0 =*/ ggml_metal_glm_dsa_q2_down_weighted_reduce_stock_slot0_enabled() ? 1 : 0,
    };
    auto publish_weighted_output = [&]() {
        ggml_metal_op_concurrency_reset(ctx);
        if (!use_output_scratch) {
            return;
        }

        GGML_ASSERT(sizeof(float)*ggml_nelements(weighted_sum) <=
            ggml_metal_op_mul_mat_id_extra_src1_scratch(down));
        auto copy_pipeline = ggml_metal_library_get_pipeline_cpy(
            lib, GGML_TYPE_F32, GGML_TYPE_F32);
        ggml_metal_kargs_cpy copy_args = {
            /*.nk0  =*/ ne0,
            /*.ne00 =*/ ne0,
            /*.ne01 =*/ ne1,
            /*.ne02 =*/ ne2,
            /*.ne03 =*/ ne3,
            /*.nb00 =*/ sizeof(float),
            /*.nb01 =*/ sizeof(float)*ne0,
            /*.nb02 =*/ sizeof(float)*ne0*ne1,
            /*.nb03 =*/ sizeof(float)*ne0*ne1*ne2,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };
        const int nth = std::min<int>(
            ggml_metal_pipeline_max_theads_per_threadgroup(copy_pipeline), ne0);
        const int nw0 = (ne0 + nth - 1)/nth;

        ggml_metal_encoder_set_pipeline(enc, copy_pipeline);
        ggml_metal_encoder_set_bytes(enc, &copy_args, sizeof(copy_args), 0);
        ggml_metal_encoder_set_buffer(enc, fusion_scratch, 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weighted_sum), 2);
        ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne1, ne2, ne3, nth, 1, 1);
        ggml_metal_op_concurrency_reset(ctx);
    };
    const bool q3_default_w0_shape =
        down->src[0]->type == GGML_TYPE_Q3_K &&
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w0_default_enabled() &&
        ne00 == 2048 &&
        ne21 == 1 &&
        extra.already_weighted == 0;
    const bool q3_default_w1_shape =
        down->src[0]->type == GGML_TYPE_Q3_K &&
        ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w1_default_enabled() &&
        ne00 == 2048 &&
        ne21 == 1 &&
        extra.already_weighted != 0;
    const bool q3_f16_w0_shape =
        down->src[0]->type == GGML_TYPE_Q3_K &&
        down->src[1]->type == GGML_TYPE_F16 &&
        ggml_metal_glm_dsa_q3_down_f16_act_enabled() &&
        ne00 == 2048 &&
        extra.already_weighted == 0;
    const bool q2_slot_parallel_requested = ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_enabled();
    const bool q2_slot_parallel_r4 = ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r4_enabled();
    const bool q2_slot_parallel_r16 = ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r16_enabled();
    const bool q2_slot_parallel_r16_w1 =
        ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_r16_w1_enabled() &&
        ne00 == 2048 &&
        ne20 == 8 &&
        extra.already_weighted != 0;
    const bool q2_slot_parallel_default_shape =
        ggml_metal_glm_dsa_q2_down_slot_parallel_reduce_default_enabled() &&
        ne00 == 2048 &&
        ne20 == 8 &&
        extra.already_weighted == 0;
    const bool q2_f16_act_shape =
        down->src[0]->type == GGML_TYPE_Q2_K &&
        down->src[1]->type == GGML_TYPE_F16 &&
        ggml_metal_glm_dsa_q2_down_f16_act_enabled() &&
        ne00 == 2048 &&
        ne20 == 8 &&
        extra.already_weighted == 0;
    const bool q2_shift_high_bits_shape =
        down->src[0]->type == GGML_TYPE_Q2_K &&
        ggml_metal_glm_dsa_q2_down_shift_high_bits_enabled() &&
        (down->src[1]->type == GGML_TYPE_F32 || down->src[1]->type == GGML_TYPE_F16) &&
        ne00 == 2048 &&
        ne20 == 8 &&
        extra.already_weighted == 0;
    const bool q2_vec_scale_shape =
        down->src[0]->type == GGML_TYPE_Q2_K &&
        down->src[1]->type == GGML_TYPE_F32 &&
        ggml_metal_glm_dsa_q2_down_vec_scale_enabled() &&
        ne00 == 2048 &&
        ne20 == 8 &&
        extra.already_weighted == 0;
    if (down->src[0]->type == GGML_TYPE_Q2_K &&
            (q2_shift_high_bits_shape || q2_f16_act_shape || q2_vec_scale_shape || q2_slot_parallel_requested || q2_slot_parallel_default_shape || q2_slot_parallel_r4 || q2_slot_parallel_r16 || q2_slot_parallel_r16_w1) &&
            ne00 == 2048 &&
            ne20 == 8) {
        ggml_metal_pipeline_with_params pipeline_slot_parallel;
        const char * q2_slot_parallel_op = "mul_mv_id_q2_down_slot_parallel_reduce_r8";
        if (q2_shift_high_bits_shape && down->src[1]->type == GGML_TYPE_F16) {
            pipeline_slot_parallel = ggml_metal_library_get_pipeline_mul_mv_id_q2_weighted_reduce_slots_sg_r8_nb8_f16_shifted(lib, weighted_sum);
            q2_slot_parallel_op = "mul_mv_id_q2_down_slot_parallel_reduce_r8_f16_shifted";
        } else if (q2_shift_high_bits_shape) {
            pipeline_slot_parallel = ggml_metal_library_get_pipeline_mul_mv_id_q2_weighted_reduce_slots_sg_r8_nb8_shifted(lib, weighted_sum);
            q2_slot_parallel_op = "mul_mv_id_q2_down_slot_parallel_reduce_r8_shifted";
        } else if (q2_f16_act_shape) {
            pipeline_slot_parallel = ggml_metal_library_get_pipeline_mul_mv_id_q2_weighted_reduce_slots_sg_r8_nb8_f16(lib, weighted_sum);
            q2_slot_parallel_op = "mul_mv_id_q2_down_slot_parallel_reduce_r8_f16";
        } else if (q2_slot_parallel_r4 && q2_vec_scale_shape) {
            pipeline_slot_parallel = ggml_metal_library_get_pipeline_mul_mv_id_q2_weighted_reduce_slots_sg_r4_nb8_vecscale(lib, weighted_sum);
            q2_slot_parallel_op = "mul_mv_id_q2_down_slot_parallel_reduce_r4_vecscale";
        } else if (q2_vec_scale_shape) {
            pipeline_slot_parallel = ggml_metal_library_get_pipeline_mul_mv_id_q2_weighted_reduce_slots_sg_r8_nb8_vecscale(lib, weighted_sum);
            q2_slot_parallel_op = "mul_mv_id_q2_down_slot_parallel_reduce_r8_vecscale";
        } else if (q2_slot_parallel_r4) {
            pipeline_slot_parallel = ggml_metal_library_get_pipeline_mul_mv_id_q2_weighted_reduce_slots_sg_r4_nb8(lib, weighted_sum);
            q2_slot_parallel_op = "mul_mv_id_q2_down_slot_parallel_reduce_r4";
        } else if (q2_slot_parallel_r16 || q2_slot_parallel_r16_w1) {
            pipeline_slot_parallel = ggml_metal_library_get_pipeline_mul_mv_id_q2_weighted_reduce_slots_sg_r16_nb8(lib, weighted_sum);
            q2_slot_parallel_op = "mul_mv_id_q2_down_slot_parallel_reduce_r16";
        } else {
            pipeline_slot_parallel = ggml_metal_library_get_pipeline_mul_mv_id_q2_weighted_reduce_slots_sg_r8_nb8(lib, weighted_sum);
        }
        const int nr0_slot_parallel = pipeline_slot_parallel.nr0;
        const int nsg_slot_parallel = pipeline_slot_parallel.nsg;
        const int grid_x_slot_parallel = (ne01 + nr0_slot_parallel - 1)/nr0_slot_parallel;
        const int grid_y_slot_parallel = ne21;

        ggml_metal_encoder_set_pipeline(enc, pipeline_slot_parallel);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_bytes(enc, &extra, sizeof(extra), 6);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[0]),         1);
        ggml_metal_encoder_set_buffer(enc, routed_input,                                   2);
        ggml_metal_encoder_set_buffer(enc, weighted_reduce_dst,                            3);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[2]),         4);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weighted_sum->src[1]), 5);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline_slot_parallel.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x_slot_parallel, grid_y_slot_parallel, 1, 32, nsg_slot_parallel, 1);

        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                    "ggml: glm_dsa_metal_dispatch op=%s tensor=%s down=%s src1=%s ids=%s weights=%s already_weighted=%d nr0=%d nsg=%d grid_x=%d grid_y=%d grid_z=1 threads_x=%d threads_y=%d fused_nodes=%d\n",
                    q2_slot_parallel_op,
                    ggml_metal_tensor_name(weighted_sum),
                    ggml_metal_tensor_name(down),
                    ggml_metal_tensor_name(down->src[1]),
                    ggml_metal_tensor_name(down->src[2]),
                    ggml_metal_tensor_name(weighted_sum->src[1]),
                    extra.already_weighted,
                    nr0_slot_parallel,
                    nsg_slot_parallel,
                    grid_x_slot_parallel,
                    grid_y_slot_parallel,
                    32,
                    nsg_slot_parallel,
                    fusion.n_fuse);
        }

        // The fused result replaces both MUL_MAT_ID and MOE_WEIGHTED_SUM, but
        // the generic tracker only registered the first node in that span.
        ggml_metal_op_concurrency_reset(ctx);

        return fusion.n_fuse;
    }

    if (down->src[0]->type == GGML_TYPE_Q3_K &&
            (ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_enabled() ||
             ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_enabled() ||
             ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_enabled() ||
             ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w0_enabled() ||
             ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r6_nb8_w0_enabled() ||
             ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r10_nb8_w0_enabled() ||
             ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_glm52_w0_enabled() ||
             q3_default_w0_shape ||
             q3_default_w1_shape ||
             q3_f16_w0_shape ||
             ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r12_nb8_w0_enabled() ||
             ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r16_enabled() ||
             ggml_metal_glm_dsa_q3_down_slot_split2_reduce_enabled())) {
        auto pipeline_slot_parallel = ggml_metal_glm_dsa_q3_down_slot_split2_reduce_enabled() ?
            ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_split2(lib, weighted_sum) :
            (ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r16_enabled() ?
            ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r16(lib, weighted_sum) :
            (ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_glm52_w0_enabled() && ne00 == 2048 && ne01 == 6144 && ne20 == 8 && ne21 == 1 && extra.already_weighted == 0 ?
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_glm52_w0(lib, weighted_sum) :
            (ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r10_nb8_w0_enabled() && ne00 == 2048 && extra.already_weighted == 0 ?
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r10_nb8_w0(lib, weighted_sum) :
            (ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r12_nb8_w0_enabled() && ne00 == 2048 && extra.already_weighted == 0 ?
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r12_nb8_w0(lib, weighted_sum) :
            (ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r6_nb8_w0_enabled() && ne00 == 2048 && extra.already_weighted == 0 ?
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r6_nb8_w0(lib, weighted_sum) :
            (q3_f16_w0_shape ?
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r8_nb8_w0_f16(lib, weighted_sum) :
            ((ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_w0_enabled() || q3_default_w0_shape) && ne00 == 2048 && extra.already_weighted == 0 ?
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r8_nb8_w0(lib, weighted_sum) :
            (q3_default_w1_shape && ne00 == 2048 && extra.already_weighted != 0 ?
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r8_nb8(lib, weighted_sum) :
            (ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_nb8_enabled() && ne00 == 2048 ?
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r8_nb8(lib, weighted_sum) :
            (ggml_metal_glm_dsa_q3_down_slot_parallel_reduce_r8_enabled() ?
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r8(lib, weighted_sum) :
                ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg(lib, weighted_sum)))))))))));
        const int nr0_slot_parallel = pipeline_slot_parallel.nr0;
        const int nsg_slot_parallel = pipeline_slot_parallel.nsg;
        const int grid_x_slot_parallel = (ne01 + nr0_slot_parallel - 1)/nr0_slot_parallel;
        const int grid_y_slot_parallel = ne21;

        ggml_metal_encoder_set_pipeline(enc, pipeline_slot_parallel);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_bytes(enc, &extra, sizeof(extra), 6);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[0]),         1);
        ggml_metal_encoder_set_buffer(enc, routed_input,                                   2);
        ggml_metal_encoder_set_buffer(enc, weighted_reduce_dst,                            3);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[2]),         4);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weighted_sum->src[1]), 5);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline_slot_parallel.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x_slot_parallel, grid_y_slot_parallel, 1, 32, nsg_slot_parallel, 1);

        publish_weighted_output();

        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                    "ggml: glm_dsa_metal_dispatch op=mul_mv_id_q3_down_slot_parallel_reduce tensor=%s down=%s src1=%s ids=%s weights=%s already_weighted=%d nr0=%d nsg=%d grid_x=%d grid_y=%d grid_z=1 threads_x=%d threads_y=%d fused_nodes=%d\n",
                    ggml_metal_tensor_name(weighted_sum),
                    ggml_metal_tensor_name(down),
                    ggml_metal_tensor_name(down->src[1]),
                    ggml_metal_tensor_name(down->src[2]),
                    ggml_metal_tensor_name(weighted_sum->src[1]),
                    extra.already_weighted,
                    nr0_slot_parallel,
                    nsg_slot_parallel,
                    grid_x_slot_parallel,
                    grid_y_slot_parallel,
                    32,
                    nsg_slot_parallel,
                    fusion.n_fuse);
        }

        return fusion.n_fuse;
    }

    if (down->src[0]->type == GGML_TYPE_Q3_K && ggml_metal_glm_dsa_q3_down_atomic_accum_enabled()) {
        {
            auto zero_pipeline = ggml_metal_library_get_pipeline_zero_f32(lib);
            const int64_t n_zero = ggml_nelements(weighted_sum);
            const int nth = 256;
            const int64_t n_tg = (n_zero + nth - 1) / nth;

            ggml_metal_encoder_set_pipeline(enc, zero_pipeline);
            ggml_metal_encoder_set_buffer(enc, weighted_reduce_dst, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        auto pipeline_atomic = ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_accum_atomic(lib, weighted_sum);
        const int nr0_atomic = pipeline_atomic.nr0;
        const int nsg_atomic = pipeline_atomic.nsg;
        const int grid_x_atomic = (ne01 + nr0_atomic*nsg_atomic - 1)/(nr0_atomic*nsg_atomic);
        const int grid_y_atomic = ne21;

        ggml_metal_encoder_set_pipeline(enc, pipeline_atomic);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_kargs_mul_mv_id_weighted_reduce_extra atomic_extra = extra;
        atomic_extra._pad0 = 0;
        ggml_metal_encoder_set_bytes(enc, &atomic_extra, sizeof(atomic_extra), 6);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[0]),         1);
        ggml_metal_encoder_set_buffer(enc, routed_input,                                   2);
        ggml_metal_encoder_set_buffer(enc, weighted_reduce_dst,                            3);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[2]),         4);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weighted_sum->src[1]), 5);

        const int grid_z_atomic = ne20;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline_atomic.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x_atomic, grid_y_atomic, grid_z_atomic, 32, nsg_atomic, 1);

        publish_weighted_output();

        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                    "ggml: glm_dsa_metal_dispatch op=mul_mv_id_q3_down_atomic_accum tensor=%s down=%s src1=%s ids=%s weights=%s already_weighted=%d nr0=%d nsg=%d grid_x=%d grid_y=%d grid_z=%d threads_x=%d threads_y=%d fused_nodes=%d\n",
                    ggml_metal_tensor_name(weighted_sum),
                    ggml_metal_tensor_name(down),
                    ggml_metal_tensor_name(down->src[1]),
                    ggml_metal_tensor_name(down->src[2]),
                    ggml_metal_tensor_name(weighted_sum->src[1]),
                    extra.already_weighted,
                    nr0_atomic,
                    nsg_atomic,
                    grid_x_atomic,
                    grid_y_atomic,
                    grid_z_atomic,
                    32,
                    nsg_atomic,
                    fusion.n_fuse);
        }

        return fusion.n_fuse;
    }

    ggml_metal_op_concurrency_reset(ctx);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_bytes(enc, &extra, sizeof(extra), 6);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[0]),         1);
    ggml_metal_encoder_set_buffer(enc, routed_input,                                   2);
    ggml_metal_encoder_set_buffer(enc, weighted_reduce_dst,                            3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down->src[2]),         4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weighted_sum->src[1]), 5);

    const int grid_x = (ne01 + nr0*nsg - 1)/(nr0*nsg);
    const int grid_y = ne21;
    const int grid_z = 1;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);
    publish_weighted_output();

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        const ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(down->src[1]);
        const ggml_metal_buffer_id bid_ids  = ggml_metal_get_buffer_id(down->src[2]);
        const ggml_metal_buffer_id bid_w    = ggml_metal_get_buffer_id(weighted_sum->src[1]);
        const ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(weighted_sum);
        const size_t src1_begin = bid_src1.offs;
        const size_t src1_end   = src1_begin + ggml_nbytes(down->src[1]);
        const size_t ids_begin  = bid_ids.offs;
        const size_t ids_end    = ids_begin + ggml_nbytes(down->src[2]);
        const size_t w_begin    = bid_w.offs;
        const size_t w_end      = w_begin + ggml_nbytes(weighted_sum->src[1]);
        const size_t dst_begin  = bid_dst.offs;
        const size_t dst_end    = dst_begin + ggml_nbytes(weighted_sum);
        const bool src1_dst_overlap =
            bid_src1.metal == bid_dst.metal && src1_begin < dst_end && dst_begin < src1_end;
        const bool ids_dst_overlap =
            bid_ids.metal == bid_dst.metal && ids_begin < dst_end && dst_begin < ids_end;
        const bool weights_dst_overlap =
            bid_w.metal == bid_dst.metal && w_begin < dst_end && dst_begin < w_end;
        GGML_LOG_INFO(
                "ggml: glm_dsa_metal_dispatch op=mul_mv_id_down_weighted_reduce tensor=%s down=%s src1=%s ids=%s weights=%s src0_type=%s src1_type=%s ids_type=%s dst_type=%s ne00=%d ne01=%d experts=%d ne10=%d ne11=%d ne12=%d nb10=%llu nb11=%llu nb12=%llu used_experts=%d tokens=%d ids_nb1=%llu weights_nb1=%llu weights_nb2=%llu already_weighted=%d dst_ne0=%d dst_ne1=%d dst_nb0=%llu dst_nb1=%llu src1_dst_overlap=%d ids_dst_overlap=%d weights_dst_overlap=%d src1_offs=%llu src1_nbytes=%llu ids_offs=%llu ids_nbytes=%llu weights_offs=%llu weights_nbytes=%llu dst_offs=%llu dst_nbytes=%llu same_src1_buffer=%d same_ids_buffer=%d same_weights_buffer=%d nr0=%d nr1=%d nsg=%d grid_x=%d grid_y=%d grid_z=%d threads_x=%d threads_y=%d fused_nodes=%d\n",
            ggml_metal_tensor_name(weighted_sum),
            ggml_metal_tensor_name(down),
            ggml_metal_tensor_name(down->src[1]),
            ggml_metal_tensor_name(down->src[2]),
            ggml_metal_tensor_name(weighted_sum->src[1]),
            ggml_type_name(down->src[0]->type),
            ggml_type_name(down->src[1]->type),
            ggml_type_name(down->src[2]->type),
            ggml_type_name(weighted_sum->type),
            ne00,
            ne01,
            ne02,
            ne10,
            ne11,
            ne12,
            (unsigned long long) nb10,
            (unsigned long long) nb11,
            (unsigned long long) nb12,
            ne20,
            ne21,
            (unsigned long long) nb21,
            (unsigned long long) nb31,
            (unsigned long long) nb32,
            extra.already_weighted,
            ne0,
            ne1,
            (unsigned long long) nb0,
            (unsigned long long) nb1,
            src1_dst_overlap ? 1 : 0,
            ids_dst_overlap ? 1 : 0,
            weights_dst_overlap ? 1 : 0,
            (unsigned long long) src1_begin,
            (unsigned long long) ggml_nbytes(down->src[1]),
            (unsigned long long) ids_begin,
            (unsigned long long) ggml_nbytes(down->src[2]),
            (unsigned long long) w_begin,
            (unsigned long long) ggml_nbytes(weighted_sum->src[1]),
            (unsigned long long) dst_begin,
            (unsigned long long) ggml_nbytes(weighted_sum),
            bid_src1.metal == bid_dst.metal ? 1 : 0,
            bid_ids.metal == bid_dst.metal ? 1 : 0,
            bid_w.metal == bid_dst.metal ? 1 : 0,
            nr0,
            nr1,
            nsg,
            grid_x,
            grid_y,
            grid_z,
            32,
            nsg,
            fusion.n_fuse);
    }

    // Internal barriers clear the scheduler's tracked ranges before this
    // fused output is encoded, so publish it before later graph nodes read it.
    ggml_metal_op_concurrency_reset(ctx);

    return fusion.n_fuse;
}

static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {
    struct ggml_tensor * node = ctx->node(idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    if (ggml_is_empty(node)) {
        return 1;
    }

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            {
                // noop -> next node
                if (ctx->debug_graph > 0) {
                    GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(node->op), "(noop)");
                }
            } return 1;
        default:
            {
            } break;
    }

    if (!ggml_metal_device_supports_op(ctx->dev, node)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(node));
        GGML_ABORT("unsupported op");
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 1;
    }

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        const bool is_concurrent = ggml_metal_op_concurrency_check(ctx, node);

        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        if (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %-12s %s\n", __func__, idx, ggml_op_name(node->op), ggml_get_name(node), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, node->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, node->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, node->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, node->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne2, node->src[2], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb2, node->src[2], nb);
            GGML_TENSOR_LOCALS( int64_t, ne3, node->src[3], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb3, node->src[3], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  node,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  node,         nb);

            if (node->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(node->src[0]), node->src[0]->name);
            }
            if (node->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(node->src[1]), node->src[1]->name);
            }
            if (node->src[2]) {
                GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        ggml_is_contiguous(node->src[2]), node->src[2]->name);
            }
            if (node->src[3]) {
                GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        ggml_is_contiguous(node->src[3]), node->src[3]->name);
            }
            if (node) {
                GGML_LOG_DEBUG("%s: node  - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(node->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        node->name);
            }
        }
    }

    if (node->op == GGML_OP_UNARY) {
        n_fuse = ggml_metal_op_glm_moe_decode_motif_reference(ctx, idx);
        if (n_fuse > 0) {
            goto done;
        }
        n_fuse = ggml_metal_op_topk_moe_route_fused(ctx, idx);
        if (n_fuse > 0) {
            goto done;
        }
        n_fuse = 1;
    }

    switch (node->op) {
        case GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } break;
        case GGML_OP_ADD:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } break;
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } break;
        case GGML_OP_ADD_ID:
            {
                n_fuse = ggml_metal_op_add_id(ctx, idx);
            } break;
        case GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } break;
        case GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_unary(ctx, idx);
            } break;
        case GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_weighted_swiglu(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_glu(ctx, idx);
                }
            } break;
        case GGML_OP_SUM:
            {
                n_fuse = ggml_metal_op_sum(ctx, idx);
            } break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            {
                n_fuse = ggml_metal_op_sum_rows(ctx, idx);
            } break;
        case GGML_OP_CUMSUM:
            {
                n_fuse = ggml_metal_op_cumsum(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_soft_max(ctx, idx);
            } break;
        case GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                n_fuse = ggml_metal_op_ssm_scan(ctx, idx);
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                n_fuse = ggml_metal_op_gated_delta_net(ctx, idx);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_fuse = ggml_metal_op_lightning_indexer(ctx, idx);
            } break;
        case GGML_OP_DSA_SPARSE_MASK:
            {
                n_fuse = ggml_metal_op_dsa_sparse_mask(ctx, idx);
            } break;
        case GGML_OP_DSA_SPARSE_ATTN:
            {
                n_fuse = ggml_metal_op_dsa_sparse_attn(ctx, idx);
            } break;
        case GGML_OP_DSA_TOP1_ATTN:
            {
                n_fuse = ggml_metal_op_dsa_top1_attn(ctx, idx);
            } break;
        case GGML_OP_MOE_ROUTE_WEIGHTS:
            {
                n_fuse = ggml_metal_op_glm_moe_decode_motif_reference(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_moe_route_weights(ctx, idx);
                }
            } break;
        case GGML_OP_MOE_WEIGHTED_SUM:
            {
                n_fuse = ggml_metal_op_moe_weighted_sum(ctx, idx);
            } break;
        case GGML_OP_MOE_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_moe_mul_mat_id(ctx, idx);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                n_fuse = ggml_metal_op_solve_tri(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_glm_absorbed_q(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_mul_mat(ctx, idx);
                }
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mv_id_weighted_reduce(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_mul_mv_id_gate_up_swiglu(ctx, idx);
                }
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
                }
            } break;
        case GGML_OP_GET_ROWS:
            {
                n_fuse = ggml_metal_op_get_rows(ctx, idx);
            } break;
        case GGML_OP_SET_ROWS:
            {
                n_fuse = ggml_metal_op_set_rows(ctx, idx);
            } break;
        case GGML_OP_DIAG:
            {
                n_fuse = ggml_metal_op_diag(ctx, idx);
            } break;
        case GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                n_fuse = ggml_metal_op_group_norm(ctx, idx);
            } break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } break;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } break;
        case GGML_OP_IM2COL:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } break;
        case GGML_OP_CONV_2D:
            {
                n_fuse = ggml_metal_op_conv_2d(ctx, idx);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                n_fuse = ggml_metal_op_conv_2d_dw(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = ggml_metal_op_conv_transpose_2d(ctx, idx);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                n_fuse = ggml_metal_op_col2im_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_3D:
            {
                n_fuse = ggml_metal_op_conv_3d(ctx, idx);
            } break;
        case GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } break;
        case GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } break;
        case GGML_OP_ROLL:
            {
                n_fuse = ggml_metal_op_roll(ctx, idx);
            } break;
        case GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } break;
        case GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_argsort(ctx, idx);
            } break;
        case GGML_OP_TOP_K:
            {
                n_fuse = ggml_metal_op_top_k(ctx, idx);
            } break;
        case GGML_OP_TRI:
            {
                n_fuse = ggml_metal_op_tri(ctx, idx);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } break;
        case GGML_OP_SET:
            {
                n_fuse = ggml_metal_op_set(ctx, idx);
            } break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } break;
        case GGML_OP_POOL_1D:
            {
                n_fuse = ggml_metal_op_pool_1d(ctx, idx);
            } break;
        case GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } break;
        case GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                n_fuse = ggml_metal_op_opt_step_adamw(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_SGD:
            {
                n_fuse = ggml_metal_op_opt_step_sgd(ctx, idx);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                n_fuse = ggml_metal_op_count_equal(ctx, idx);
            } break;
        default:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(node->op));
                GGML_ABORT("fatal error");
            }
    }

done:
    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // A backend-owned fusion can keep internal graph values in a private
    // resource. In that case only publish the graph-visible outputs to the
    // ordinary arena dependency tracker.
    const bool track_selected_outputs = ctx->has_fused_range_outputs();
    for (int i = 0; i < n_fuse; ++i) {
        if (track_selected_outputs && !ctx->tracks_fused_range_output(i)) {
            continue;
        }
        if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }
    ctx->clear_fused_range_outputs();

    return n_fuse;
}

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx) {
    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_push(ctx->enc, ggml_op_desc(ctx->node(idx)));
    }

    int res = ggml_metal_op_encode_impl(ctx, idx);
    if (idx + res > ctx->n_nodes()) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_pop(ctx->enc);
    }

    return res;
}

int ggml_metal_op_concat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
    };

    auto pipeline = ggml_metal_library_get_pipeline_concat(lib, op->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    int nth = std::min(256, ne0);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;
    if (nth < 256) {
        nrptg = std::min((256 + nth - 1) / nth, ne1);
        if (nrptg * nth > 256) {
            nrptg = 256 / nth;
        }
    }

    const int nw0 = (ne1 + nrptg - 1) / nrptg;

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0, ne2, ne3, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_repeat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_acc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    auto pipeline = ggml_metal_library_get_pipeline_bin_one(lib, GGML_OP_ADD);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    int nth = 1;

    while (2*nth < args.ne0 && nth < nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_unary args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.slope =*/ 0.0,
        /*.scale =*/ 0.0,
        /*.bias  =*/ 0.0,
        /*.val   =*/ 0.0,
        /*.min   =*/ 0.0,
        /*.max   =*/ 0.0,
    };

    if (op->op == GGML_OP_LEAKY_RELU) {
        args.slope = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_SCALE) {
        args.scale = ggml_get_op_params_f32(op, 0);
        args.bias  = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_FILL) {
        args.val = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_CLAMP) {
        args.min = ggml_get_op_params_f32(op, 0);
        args.max = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
        args.slope = ggml_get_op_params_f32(op, 1); // alpha_n
        args.scale = ggml_get_op_params_f32(op, 2); // alpha_p
        args.bias  = ggml_get_op_params_f32(op, 3); // beta
        args.val   = ggml_get_op_params_f32(op, 4); // eps
    }

    auto pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    if (pipeline.cnt) {
        const int n = pipeline.c4 ? ggml_nelements(op)/4 : ggml_nelements(op);

        ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        const int nth = MIN(args.ne00, nth_max);
        const int nk0 = (args.ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}

static bool ggml_metal_match_weighted_swiglu(
        ggml_metal_op_t ctx,
        int idx,
        ggml_metal_weighted_swiglu_fusion & fusion) {
    if (!ctx->use_fusion || !ggml_metal_glm_dsa_weighted_swiglu_enabled() || idx + 2 > ctx->n_nodes()) {
        return false;
    }

    ggml_tensor * glu = ctx->node(idx);
    ggml_tensor * weighted = ctx->node(idx + 1);
    if (glu->op != GGML_OP_GLU || weighted->op != GGML_OP_MUL) {
        return false;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    if (!ggml_metal_tensor_name_contains(glu, "ffn_moe_swiglu") ||
            !ggml_metal_tensor_name_contains(weighted, "ffn_moe_down_weighted_input")) {
        return false;
    }

    ggml_tensor * weights = nullptr;
    if (weighted->src[0] == glu) {
        weights = weighted->src[1];
    } else if (weighted->src[1] == glu) {
        weights = weighted->src[0];
    } else {
        return false;
    }

    const bool shape_ok =
        glu->src[0] != nullptr &&
        glu->src[1] != nullptr &&
        weights != nullptr &&
        glu->type == GGML_TYPE_F32 &&
        glu->src[0]->type == GGML_TYPE_F32 &&
        glu->src[1]->type == GGML_TYPE_F32 &&
        weighted->type == GGML_TYPE_F32 &&
        weights->type == GGML_TYPE_F32 &&
        weighted->ne[0] == glu->ne[0] &&
        weighted->ne[1] == glu->ne[1] &&
        weighted->ne[2] == glu->ne[2] &&
        weights->ne[0] == 1 &&
        weights->ne[1] == glu->ne[1] &&
        weights->ne[2] == glu->ne[2] &&
        ggml_is_contiguous_1(glu->src[0]) &&
        ggml_is_contiguous_1(glu->src[1]) &&
        ggml_is_contiguous_1(weighted);
    if (!shape_ok) {
        return false;
    }

    const ggml_op ops[] = { GGML_OP_GLU, GGML_OP_MUL };
    const int outputs[] = { 1 };
    if (!ctx->can_fuse_subgraph(idx, ops, 2, outputs, 1)) {
        return false;
    }

    fusion.glu = glu;
    fusion.weights = weights;
    fusion.weighted = weighted;
    fusion.n_fuse = 2;
    return true;
}

static int ggml_metal_op_weighted_swiglu(ggml_metal_op_t ctx, int idx) {
    ggml_metal_weighted_swiglu_fusion fusion;
    if (!ggml_metal_match_weighted_swiglu(ctx, idx, fusion)) {
        return 0;
    }

    ggml_tensor * glu = fusion.glu;
    ggml_tensor * weights = fusion.weights;
    ggml_tensor * weighted = fusion.weighted;

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, glu->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, glu->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, glu->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, glu->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb2, weights, nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  weighted, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  weighted, nb);

    GGML_ASSERT(ggml_are_same_shape(glu->src[0], glu->src[1]));

    auto pipeline = ggml_metal_library_get_pipeline_glu_weighted(lib, weighted);

    ggml_metal_kargs_glu_weighted args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ ne10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ 0,
        /*.i10  =*/ 0,
        /*.weights_nb1 =*/ nb21,
        /*.weights_nb2 =*/ nb22,
        /*.dst_nb1 =*/ nb1,
    };

    const int64_t nrows = ggml_nrows(weighted);
    const int32_t nth = std::min(
        ggml_metal_pipeline_max_theads_per_threadgroup(pipeline),
        pipeline.c4 ? std::max(1, ne00/4) : ne00/2);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(glu->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(glu->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights),     3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weighted),    4);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=weighted_swiglu tensor=%s glu=%s weights=%s src0_type=%s weights_type=%s dst_type=%s ne0=%d rows=%lld slots=%d tokens=%d c4=%d grid_x=%lld grid_y=1 grid_z=1 threads_x=%d fused_nodes=%d\n",
            ggml_metal_tensor_name(weighted),
            ggml_metal_tensor_name(glu),
            ggml_metal_tensor_name(weights),
            ggml_type_name(glu->src[0]->type),
            ggml_type_name(weights->type),
            ggml_type_name(weighted->type),
            ne0,
            (long long) nrows,
            ne1,
            ne2,
            pipeline.c4 ? 1 : 0,
            (long long) nrows,
            nth,
            fusion.n_fuse);
    }

    return fusion.n_fuse;
}

int ggml_metal_op_glu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    if (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    auto pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    const int32_t swp = ggml_get_op_params_i32(op, 1);
    const float alpha = ggml_get_op_params_f32(op, 2);
    const float limit = ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    const int64_t nrows = ggml_nrows(op->src[0]);

    const int32_t nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00/2);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    if (ggml_metal_glm_dsa_dispatch_log_enabled() &&
            ggml_metal_tensor_name_contains(op, "ffn_moe_swiglu")) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=glu kernel=%s tensor=%s gate=%s up=%s src0_type=%s src1_type=%s dst_type=%s ne0=%d rows=%lld slots=%d tokens=%d grid_x=%lld grid_y=1 grid_z=1 threads_x=%d\n",
            ggml_glu_op_name(ggml_get_glu_op(op)),
            ggml_metal_tensor_name(op),
            ggml_metal_tensor_name(op->src[0]),
            ggml_metal_tensor_name(op->src[1]),
            ggml_type_name(op->src[0]->type),
            op->src[1] ? ggml_type_name(op->src[1]->type) : ggml_type_name(op->src[0]->type),
            ggml_type_name(op->type),
            ne0,
            (long long) nrows,
            ne1,
            ne2,
            (long long) nrows,
            nth);
    }

    return 1;
}

int ggml_metal_op_sum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op  = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const uint64_t n = (uint64_t) ggml_nelements(op->src[0]);

    ggml_metal_kargs_sum args = {
        /*.np =*/ n,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum(lib, op);

    int nth = 32; // SIMD width

    while (nth < (int) n && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) n);

    const int nsg = (nth + 31) / 32;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, nsg * sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum_rows(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < args.ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) args.ne00);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_cumsum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline_blk = ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    const int64_t net0 = (ne00 + nth - 1) / nth;
    const int64_t net1 = ne01;
    const int64_t net2 = ne02;
    const int64_t net3 = ne03;

    const uint64_t nbt0 = sizeof(float);
    const uint64_t nbt1 = net0*nbt0;
    const uint64_t nbt2 = net1*nbt1;
    const uint64_t nbt3 = net2*nbt2;

    const size_t smem = GGML_PAD(32*sizeof(float), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    {
        ggml_metal_kargs_cumsum_blk args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.net0 =*/ net0,
            /*.net1 =*/ net1,
            /*.net2 =*/ net2,
            /*.net3 =*/ net3,
            /*.nbt0 =*/ nbt0,
            /*.nbt1 =*/ nbt1,
            /*.nbt2 =*/ nbt2,
            /*.nbt3 =*/ nbt3,
            /*.outb =*/ ne00 > nth,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    if (ne00 > nth) {
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_kargs_cumsum_blk args = {
                /*.ne00 =*/ net0,
                /*.ne01 =*/ net1,
                /*.ne02 =*/ net2,
                /*.ne03 =*/ net3,
                /*.nb00 =*/ nbt0,
                /*.nb01 =*/ nbt1,
                /*.nb02 =*/ nbt2,
                /*.nb03 =*/ nbt3,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
                /*.outb =*/ false,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline_add = ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            ggml_metal_kargs_cumsum_add args = {
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.ne03 =*/ ne03,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.nb03 =*/ nb03,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    return 1;
}

static int32_t ggml_metal_glm_dsa_packed_gather_rows_per_tg(
        const ggml_metal_device_props * props_dev) {
    const char * value = getenv("LLAMA_GLM_DSA_EXPERIMENTAL_PACKED_GATHER_ROWS_PER_TG");
    if (value == nullptr || value[0] == '\0') {
        return props_dev->device_id == GGML_METAL_DEVICE_M3_ULTRA ? 16 : 1;
    }

    switch (atoi(value)) {
        case 2:
        case 4:
        case 8:
        case 16:
        case 32:
        case 64:
            return atoi(value);
        default:
            return 1;
    }
}

static int32_t ggml_metal_glm_dsa_packed_gather_threads_per_row(
        const ggml_metal_device_props * props_dev) {
    const char * value = getenv("LLAMA_GLM_DSA_EXPERIMENTAL_PACKED_GATHER_THREADS_PER_ROW");
    if (value == nullptr || value[0] == '\0') {
        return props_dev->device_id == GGML_METAL_DEVICE_M3_ULTRA ? 32 : 64;
    }

    switch (atoi(value)) {
        case 16:
        case 32:
        case 64:
            return atoi(value);
        default:
            return 64;
    }
}

int ggml_metal_op_get_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_log_selected_row_flash_candidate(ctx, idx);
    if (ggml_metal_find_selected_row_flash_consumer(ctx, idx, op) != nullptr) {
        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                "ggml: glm_dsa_metal_dispatch op=selected_row_flash_skip tensor=%s reason=deferred_to_flash grid_x=1 grid_y=1 grid_z=1 threads_x=1\n",
                ggml_metal_tensor_name(op));
        }
        return 1;
    }
    if (ggml_metal_selected_row_flash_can_defer_compact_k_rows(ctx, idx, op)) {
        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                "ggml: glm_dsa_metal_dispatch op=selected_row_flash_skip tensor=%s reason=deferred_compact_k_contract grid_x=1 grid_y=1 grid_z=1 threads_x=1\n",
                ggml_metal_tensor_name(op));
        }
        return 1;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const bool f16_to_f16 = op->src[0]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16;

    ggml_metal_kargs_get_rows args = {
        /*.ne00t =*/ ggml_is_quantized(op->src[0]->type) ? ne00/16 : ne00,
        /*.ne00  =*/ ne00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    const ggml_metal_buffer_id src0_buffer = ggml_metal_get_buffer_id(op->src[0]);
    const ggml_metal_buffer_id src1_buffer = ggml_metal_get_buffer_id(op->src[1]);
    const ggml_metal_buffer_id dst_buffer  = ggml_metal_get_buffer_id(op);

    const bool use_f16_vec4 = f16_to_f16 &&
            args.ne00 >= 4 &&
            args.nb01%8 == 0 &&
            args.nb1%8 == 0 &&
            src0_buffer.offs%8 == 0 &&
            dst_buffer.offs%8 == 0;
    if (use_f16_vec4) {
        args.ne00t = (args.ne00 + 3)/4;
    }

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
    const int32_t requested_rows_per_tg = ggml_metal_glm_dsa_packed_gather_rows_per_tg(props_dev);
    const int32_t requested_threads_per_row = ggml_metal_glm_dsa_packed_gather_threads_per_row(props_dev);
    const bool use_packed_rows = use_f16_vec4 &&
        requested_rows_per_tg > 1 &&
        requested_rows_per_tg*requested_threads_per_row <= 1024 &&
        args.ne00 == 576 &&
        args.ne00%4 == 0 &&
        args.ne10 >= 256 &&
        ne11 == 1 &&
        ne12 == 1;

    auto pipeline = use_packed_rows ?
        ggml_metal_library_get_pipeline_get_rows_packed_f16(lib) :
        ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type, op->type, use_f16_vec4);

    const int nth = use_packed_rows ? requested_threads_per_row :
        std::min(args.ne00t, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int rows_per_tg = use_packed_rows ? requested_rows_per_tg : 1;
    GGML_ASSERT(nth*rows_per_tg <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const int nw0 = (args.ne00t + nth - 1)/nth;
    const int grid_x = use_packed_rows ? (ne10 + rows_per_tg - 1)/rows_per_tg : nw0*ne10;
    const int grid_y = ne11;
    const int grid_z = ne12;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, src0_buffer, 1);
    ggml_metal_encoder_set_buffer  (enc, src1_buffer, 2);
    ggml_metal_encoder_set_buffer  (enc, dst_buffer,  3);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=get_rows kernel=%s tensor=%s src_type=%s top_k_type=%s dst_type=%s rows=%lld rows_per_tg=%d grid_x=%d grid_y=%d grid_z=%d threads_x=%d threads_y=%d\n",
            use_packed_rows ? "packed_rows" :
                (use_f16_vec4 ? "typed_vec4" : op->src[0]->type == op->type ? "typed" : "promote"),
            ggml_metal_tensor_name(op),
            ggml_type_name(op->src[0]->type),
            ggml_type_name(op->src[1]->type),
            ggml_type_name(op->type),
            (long long) ne10,
            rows_per_tg,
            grid_x,
            grid_y,
            grid_z,
            nth,
            rows_per_tg);
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, nth, rows_per_tg, 1);

    return 1;
}

int ggml_metal_op_set_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_set_rows(lib, op->src[1]->type, op->type);

    const int32_t nk0 = ne0/ggml_blck_size(op->type);

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_set_rows args = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_diag(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t,  ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_diag args = {
        /*.ne00 =*/ne00,
        /*.ne01 =*/ne01,
        /*.ne02 =*/ne02,
        /*.ne03 =*/ne03,
        /*.nb00 =*/nb00,
        /*.nb01 =*/nb01,
        /*.nb02 =*/nb02,
        /*.nb03 =*/nb03,
        /*.ne0  =*/ne0,
        /*.ne1  =*/ne1,
        /*.ne2  =*/ne2,
        /*.ne3  =*/ne3,
        /*.nb0  =*/nb0,
        /*.nb1  =*/nb1,
        /*.nb2  =*/nb2,
        /*.nb3  =*/nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_diag(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, 32, 1, 1);

    return 1;
}

int ggml_metal_op_soft_max(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    const bool use_batched = (ne1 > 1);

    if (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        int BATCH_SIZE;
        if      (ne1 > 128) BATCH_SIZE = 256;
        else if (ne1 > 64 ) BATCH_SIZE = 128;
        else if (ne1 > 32 ) BATCH_SIZE = 64;
        else if (ne1 > 16 ) BATCH_SIZE = 32;
        else if (ne1 > 8  ) BATCH_SIZE = 16;
        else if (ne1 > 4  ) BATCH_SIZE = 8;
        else                BATCH_SIZE = 2;

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const int n_token_batches = (ne1 + BATCH_SIZE - 1) / BATCH_SIZE;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);
    }

    return 1;
}

int ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_ssm_scan(lib, op);

    GGML_ASSERT(d_state <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, d_inner, n_head, n_seqs, d_state, 1, 1);

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    auto pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

int ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;


    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    int ida = 0;

    ggml_metal_kargs_gated_delta_net args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20,
        /*.ne21 =*/ ne21,
        /*.ne22 =*/ ne22,
        /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20,
        /*.nb21 =*/ nb21,
        /*.nb22 =*/ nb22,
        /*.nb23 =*/ nb23,
        /*.ns02 =*/ (int32_t) (nb02/sizeof(float)),
        /*.ns12 =*/ (int32_t) (nb12/sizeof(float)),
        /*.ns22 =*/ (int32_t) (nb22/sizeof(float)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++); // state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/nsg, op->src[2]->ne[1], op->src[2]->ne[3], 32, nsg, 1);

    return 1;
}

int ggml_metal_op_solve_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, 32, nsg, 1);

    return 1;
}

int ggml_metal_op_set(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[1]->type, op->type);

    GGML_ASSERT(ne10 % ggml_blck_size(op->src[1]->type) == 0);

    int64_t nk0 = ne10;
    if (ggml_is_quantized(op->src[1]->type)) {
        nk0 = ne10/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne10/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne11, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[1]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb10,
        /*.nb01 =*/ nb11,
        /*.nb02 =*/ nb12,
        /*.nb03 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ ggml_element_size(op),
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    bid_dst.offs += offs;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne11 + nrptg - 1)/nrptg, ne12, ne13, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

    GGML_ASSERT(ne00 % ggml_blck_size(op->src[0]->type) == 0);

    int64_t nk0 = ne00;
    if (ggml_is_quantized(op->src[0]->type)) {
        nk0 = ne00/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne00/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne01, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_pool_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t s0 = opts[2];
    const int32_t p0 = opts[3];

    const int64_t IW = op->src[0]->ne[0];
    const int64_t OW = op->ne[0];

    const int64_t np = ggml_nelements(op);

    ggml_metal_kargs_pool_1d args_pool_1d = {
        /* .k0 = */  k0,
        /* .s0 = */  s0,
        /* .p0 = */  p0,
        /* .IW = */  IW,
        /* .OW = */  OW,
        /* .np = */  np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_1d, sizeof(args_pool_1d),  0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}


int ggml_metal_op_pool_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, sizeof(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const char * block_ceiling = getenv("GGML_METAL_EXPERIMENTAL_GLM_DECODE_BLOCK_BYTE_CEILING");
    if (block_ceiling != nullptr && atoi(block_ceiling) != 0 &&
            strcmp(ggml_metal_tensor_name(op->src[0]), "glm_decode_block_byte_ceiling_weights") == 0) {
        const char * kernel = "kernel_glm_decode_block_phase_scan";
        auto pipeline = ggml_metal_library_get_pipeline(lib, kernel);
        if (!pipeline.pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, kernel, kernel, nullptr);
        }
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        struct scan_args {
            uint64_t offset;
            uint64_t nbytes;
        };
        constexpr uint64_t chunk_bytes = 64*1024;
        constexpr uint64_t layer_bytes = 238569728;
        constexpr uint64_t phase_bytes[4] = {
            17129472,
            35651584,
            49580032,
            136208640,
        };
        constexpr uint64_t phase_prefix[4] = {
            0,
            17129472,
            52781056,
            102361088,
        };

        const int mode = atoi(block_ceiling);
        if (mode == 2) {
            scan_args args = { 0, 8*layer_bytes };
            ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 2);
            ggml_metal_encoder_dispatch_threadgroups(
                    enc, (args.nbytes + chunk_bytes - 1)/chunk_bytes, 1, 1, 32, 4, 1);
        } else {
            for (uint64_t layer = 0; layer < 8; ++layer) {
                for (uint64_t phase = 0; phase < 4; ++phase) {
                    scan_args args = {
                        layer*layer_bytes + phase_prefix[phase],
                        phase_bytes[phase],
                    };
                    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 2);
                    ggml_metal_encoder_dispatch_threadgroups(
                            enc, (args.nbytes + chunk_bytes - 1)/chunk_bytes, 1, 1, 32, 4, 1);
                }
            }
        }
        return 1;
    }

    const bool shared_expert_weight =
        ggml_metal_tensor_name_contains(op->src[0], "ffn_gate_shexp") ||
        ggml_metal_tensor_name_contains(op->src[0], "ffn_up_shexp") ||
        ggml_metal_tensor_name_contains(op->src[0], "ffn_down_shexp");
    const bool routed_expert_weight =
        ggml_metal_tensor_name_contains(op->src[0], "ffn_gate_exps") ||
        ggml_metal_tensor_name_contains(op->src[0], "ffn_up_exps") ||
        ggml_metal_tensor_name_contains(op->src[0], "ffn_down_exps");
    const bool attention_projection_weight =
        ggml_metal_tensor_name_contains(op->src[0], "attn_kv_a_mqa") ||
        ggml_metal_tensor_name_contains(op->src[0], "attn_q_a") ||
        ggml_metal_tensor_name_contains(op->src[0], "attn_q_b") ||
        ggml_metal_tensor_name_contains(op->src[0], "attn_k_b") ||
        ggml_metal_tensor_name_contains(op->src[0], "attn_v_b") ||
        ggml_metal_tensor_name_contains(op->src[0], "attn_output");
    const bool indexer_projection_weight =
        ggml_metal_tensor_name_contains(op->src[0], "indexer.attn_q_b") ||
        ggml_metal_tensor_name_contains(op->src[0], "indexer.attn_k") ||
        ggml_metal_tensor_name_contains(op->src[0], "indexer.proj");
    const bool zero_projection =
        (ggml_metal_glm_shared_expert_noop_enabled() && shared_expert_weight) ||
        (ggml_metal_glm_routed_expert_noop_enabled() && routed_expert_weight) ||
        (ggml_metal_glm_attention_projection_noop_enabled() && attention_projection_weight) ||
        (ggml_metal_glm_indexer_projection_noop_enabled() && indexer_projection_weight);
    if (zero_projection) {
        auto pipeline = ggml_metal_library_get_pipeline_zero_f32(lib);
        const int nth = 256;
        const int64_t n_tg = (ggml_nelements(op) + nth - 1)/nth;
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, nth, 1, 1);
        return 1;
    }

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    if (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_BF16 ||
           op->src[0]->type == GGML_TYPE_Q1_0 ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be something cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                GGML_ABORT("unsupported ne11");
        };

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } else if (
        !ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
        props_dev->has_simdgroup_mm && ne00 >= 64 && ne11 > ne11_mm_min) {
        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = pipeline.smem;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + nr1 - 1) / nr1), ((ne01 + nr0 - 1) / nr0), ne12 * ne13, 32, nsg, 1);
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nr0  =*/ nr0,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        }
    }

    return 1;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert

    return ggml_type_size(GGML_TYPE_I32)*ne02;
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    return ggml_type_size(GGML_TYPE_I32)*ne02*ne21;
}

size_t ggml_metal_op_mul_mat_id_extra_src1_scratch(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    if (!ggml_metal_tensor_name_contains(op, "ffn_moe_down") ||
            (op->src[0]->type != GGML_TYPE_Q2_K && op->src[0]->type != GGML_TYPE_Q3_K) ||
            (op->src[1]->type != GGML_TYPE_F32 && op->src[1]->type != GGML_TYPE_F16) ||
            op->src[2]->ne[0] != 8) {
        return 0;
    }

    return ggml_nbytes(op->src[1]);
}

int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    // ne20 = n_used_experts
    // ne21 = n_rows (batch size)
    const int ne21_mm_id_min = ggml_metal_glm_dsa_mul_mm_id_min_tokens_requested();

    if (props_dev->has_simdgroup_mm && ne00 >= 64 && (ne21 >= ne21_mm_id_min)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne02,
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne21  =*/ ne21, // n_tokens
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);

            const size_t smem = pipeline.smem;

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            const int grid_x = (ne21 + 31)/32;
            const int grid_y = (ne01 + 63)/64;
            const int grid_z = ne02;
            ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 128, 1, 1);

            if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
                GGML_LOG_INFO(
                    "ggml: glm_dsa_metal_dispatch op=mul_mat_id kernel=mul_mm_id tensor=%s src0_type=%s src1_type=%s ids_type=%s dst_type=%s ne00=%d ne01=%d experts=%d used_experts=%d tokens=%d min_tokens=%d grid_x=%d grid_y=%d grid_z=%d threads_x=%d threads_y=%d\n",
                    ggml_metal_tensor_name(op),
                    ggml_type_name(op->src[0]->type),
                    ggml_type_name(op->src[1]->type),
                    ggml_type_name(op->src[2]->type),
                    ggml_type_name(op->type),
                    ne00,
                    ne01,
                    ne02,
                    ne20,
                    ne21,
                    ne21_mm_id_min,
                    grid_x,
                    grid_y,
                    grid_z,
                    128,
                    1);
            }
        }
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        int grid_x = 0;
        int grid_y = 0;
        const int grid_z = ne123;
        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            grid_x = (ne01 + nr0 - 1)/(nr0);
            grid_y = (_ne1 + nr1 - 1)/nr1;
            ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);
        } else {
            grid_x = (ne01 + nr0*nsg - 1)/(nr0*nsg);
            grid_y = (_ne1 + nr1 - 1)/nr1;
            ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);
        }

        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            const bool weighted_slots = ggml_get_op_params_i32(op, 0) != 0;
            GGML_LOG_INFO(
                "ggml: glm_dsa_metal_dispatch op=mul_mat_id kernel=mul_mv_id tensor=%s src0_type=%s src1_type=%s ids_type=%s dst_type=%s ne00=%d ne01=%d experts=%d used_experts=%d tokens=%d min_tokens=%d nr0=%d nr1=%d nsg=%d grid_x=%d grid_y=%d grid_z=%d threads_x=%d threads_y=%d\n",
                ggml_metal_tensor_name(op),
                ggml_type_name(op->src[0]->type),
                ggml_type_name(op->src[1]->type),
                ggml_type_name(op->src[2]->type),
                ggml_type_name(op->type),
                ne00,
                ne01,
                ne02,
                ne20,
                ne21,
                ne21_mm_id_min,
                nr0,
                nr1,
                nsg,
                grid_x,
                grid_y,
                grid_z,
                32,
                nsg);
            if (weighted_slots) {
                GGML_LOG_INFO(
                    "ggml: glm_dsa_metal_dispatch op=mul_mv_id_weighted_slots kernel=%s tensor=%s src0_type=%s src1_type=%s ids_type=%s dst_type=%s ne00=%d ne01=%d experts=%d used_experts=%d tokens=%d grid_x=%d grid_y=%d grid_z=%d threads_x=%d threads_y=%d\n",
                    ggml_type_name(op->src[0]->type),
                    ggml_metal_tensor_name(op),
                    ggml_type_name(op->src[0]->type),
                    ggml_type_name(op->src[1]->type),
                    ggml_type_name(op->src[2]->type),
                    ggml_type_name(op->type),
                    ne00,
                    ne01,
                    ne02,
                    ne20,
                    ne21,
                    grid_x,
                    grid_y,
                    grid_z,
                    32,
                    nsg);
            }
        }
    }

    return 1;
}

int ggml_metal_op_add_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

static int32_t ggml_metal_glm_dsa_compact_flash_nwg_requested() {
    const char * value = getenv("LLAMA_GLM_DSA_COMPACT_FLASH_NWG");
    if (value == nullptr || value[0] == '\0') {
        return 4;
    }

    const int requested = atoi(value);
    switch (requested) {
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
        case 32:
            return requested;
        default:
            return 4;
    }
}

static int32_t ggml_metal_glm_dsa_selected_row_flash_nwg_requested() {
    const char * value = getenv("LLAMA_GLM_DSA_SELECTED_ROW_FLASH_NWG");
    if (value == nullptr || value[0] == '\0') {
        return 16;
    }

    const int requested = atoi(value);
    switch (requested) {
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
        case 32:
            return requested;
        default:
            return 16;
    }
}

static int32_t ggml_metal_glm_dsa_selected_row_flash_heads_per_tg_requested() {
    const char * value = getenv("LLAMA_GLM_DSA_SELECTED_ROW_FLASH_HEADS_PER_TG");
    if (value == nullptr || value[0] == '\0') {
        return 1;
    }

    const int requested = atoi(value);
    switch (requested) {
        case 1:
            return requested;
        default:
            return 1;
    }
}

static bool ggml_metal_glm_dsa_selected_row_flash_tiled_enabled() {
    const char * value = getenv("LLAMA_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH_TILED");
    return value == nullptr || value[0] == '\0' || atoi(value) != 0;
}

static bool ggml_metal_glm_dsa_selected_row_flash_tiled_selected(const ggml_tensor * op) {
    if (!ggml_metal_glm_dsa_selected_row_flash_tiled_enabled()) {
        return false;
    }

    const char * tensor = getenv("LLAMA_GLM_DSA_SELECTED_ROW_FLASH_TILED_TENSOR");
    return tensor == nullptr || tensor[0] == '\0' || strcmp(tensor, ggml_metal_tensor_name(op)) == 0;
}

static bool ggml_metal_glm_dsa_selected_row_flash_tiled_shape(const ggml_tensor * op) {
    if (!ggml_metal_selected_row_flash_vec_shape_ok(op)) {
        return false;
    }

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * rows = op->src[1];
    const ggml_tensor * view = op->src[2];
    return q->ne[1] == 1 && q->ne[2] >= 8 && q->ne[2] % 8 == 0 &&
        rows->src[0]->ne[2] == 1 && view->ne[2] == 1;
}

static int32_t ggml_metal_glm_dsa_selected_row_flash_tiled_nwg(const ggml_tensor * op) {
    const char * value = getenv("LLAMA_GLM_DSA_SELECTED_ROW_FLASH_NWG");
    if (value != nullptr && value[0] != '\0') {
        return ggml_metal_glm_dsa_selected_row_flash_nwg_requested();
    }

    const int64_t tiles = (op->src[1]->src[1]->ne[0] + 63)/64;
    if (tiles >= 32) {
        return 32;
    }
    if (tiles >= 16) {
        return 16;
    }
    if (tiles >= 8) {
        return 8;
    }
    if (tiles >= 4) {
        return 4;
    }
    if (tiles >= 2) {
        return 2;
    }
    return 1;
}

static int32_t ggml_metal_glm_dsa_selected_row_flash_tiled_nwg_for_top_k(int64_t top_k) {
    const char * value = getenv("LLAMA_GLM_DSA_SELECTED_ROW_FLASH_NWG");
    if (value != nullptr && value[0] != '\0') {
        return ggml_metal_glm_dsa_selected_row_flash_nwg_requested();
    }

    const int64_t tiles = (top_k + 63)/64;
    if (tiles >= 32) {
        return 32;
    }
    if (tiles >= 16) {
        return 16;
    }
    if (tiles >= 8) {
        return 8;
    }
    if (tiles >= 4) {
        return 4;
    }
    if (tiles >= 2) {
        return 2;
    }
    return 1;
}

static bool ggml_metal_glm_dsa_compact_flash_shape(
        const ggml_tensor * op,
        bool                has_mask,
        bool                has_sinks,
        bool                has_bias,
        bool                has_scap,
        bool                has_kvpad) {
    return !has_mask &&
            !has_sinks &&
            !has_bias &&
            !has_scap &&
            !has_kvpad &&
            op->src[1]->type == GGML_TYPE_F16 &&
            op->src[2]->type == GGML_TYPE_F16 &&
            op->src[0]->ne[0] == 576 &&
            op->src[2]->ne[0] == 512 &&
            op->src[0]->ne[1] == 1 &&
            op->src[1]->ne[1] == 2048;
}

static int32_t ggml_metal_flash_attn_ext_vec_nwg(
        const ggml_tensor * op,
        bool                has_mask,
        bool                has_sinks,
        bool                has_bias,
        bool                has_scap,
        bool                has_kvpad) {
    if (ggml_metal_selected_row_flash_vec_shape_ok(op)) {
        if (ggml_metal_glm_dsa_selected_row_flash_tiled_selected(op) &&
                ggml_metal_glm_dsa_selected_row_flash_tiled_shape(op)) {
            return ggml_metal_glm_dsa_selected_row_flash_tiled_nwg(op);
        }
        return ggml_metal_glm_dsa_selected_row_flash_nwg_requested();
    }

    if (ggml_metal_glm_dsa_compact_flash_shape(op, has_mask, has_sinks, has_bias, has_scap, has_kvpad)) {
        const char * nwg8_tensor = getenv("GGML_METAL_EXPERIMENTAL_GLM_COMPACT_NWG8_TENSOR");
        if (nwg8_tensor != nullptr && nwg8_tensor[0] != '\0' &&
                strcmp(nwg8_tensor, ggml_metal_tensor_name(op)) == 0) {
            return 8;
        }
        return ggml_metal_glm_dsa_compact_flash_nwg_requested();
    }

    return 32;
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (false) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } else {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (!has_mask) {
        return res;
    }

    const bool is_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    const int nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    const int ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_size(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (true) {
        float max_bias;
        float logit_softcap;

        memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
        memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

        const bool has_mask  = op->src[3] != nullptr;
        const bool has_sinks = op->src[4] != nullptr;
        const bool has_bias  = max_bias != 0.0f;
        const bool has_scap  = logit_softcap != 0.0f;
        const bool has_kvpad = op->src[1]->ne[1] % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const int64_t nwg = ggml_metal_flash_attn_ext_vec_nwg(
                op, has_mask, has_sinks, has_bias, has_scap, has_kvpad);
        if (nwg == 1) {
            return res;
        }

        const int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_size(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));

        if (ggml_metal_glm_compact_split_exact_enabled() &&
                !has_mask && !has_sinks && !has_bias && !has_scap && !has_kvpad &&
                op->src[1]->type == GGML_TYPE_F16 &&
                op->src[2]->type == GGML_TYPE_F16 &&
                op->src[0]->ne[0] == 576 &&
                op->src[0]->ne[1] == 1 &&
                op->src[0]->ne[2] == 64 &&
                op->src[0]->ne[3] == 1 &&
                op->src[1]->ne[1] == 2048 &&
                op->src[1]->ne[2] == 1 &&
                op->src[2]->ne[0] == 512 &&
                op->src[2]->ne[2] == 1) {
            constexpr size_t chunk_rows = OP_FLASH_ATTN_EXT_VEC_NCPSG;
            const size_t heads = size_t(op->src[0]->ne[2])*size_t(op->src[0]->ne[3]);
            const size_t rows = size_t(op->src[1]->ne[1]);
            const size_t chunks = (rows + chunk_rows - 1)/chunk_rows;
            const size_t score_values = heads*rows;
            const size_t chunk_ms_values = heads*chunks;
            const size_t chunk_v_values = heads*chunks*size_t(op->src[2]->ne[0]);
            res += ggml_type_size(GGML_TYPE_F32)*
                (score_values + chunk_ms_values + chunk_v_values);
        }
    }

    return res;
}

size_t ggml_metal_op_dsa_sparse_attn_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSA_SPARSE_ATTN);

    if (op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr || op->src[4] == nullptr ||
            op->type != GGML_TYPE_F32 ||
            op->src[0]->type != GGML_TYPE_F32 ||
            op->src[1]->type != GGML_TYPE_F16 ||
            op->src[2]->type != GGML_TYPE_F16 ||
            op->src[4]->type != GGML_TYPE_I32 ||
            op->src[0]->ne[0] != 576 || op->ne[0] != 512 ||
            op->src[4]->ne[0] <= 0 || op->src[4]->ne[0] > 4096) {
        return 0;
    }

    constexpr size_t max_nwg = 32;
    const size_t nrows = size_t(op->ne[1])*size_t(op->ne[2])*size_t(op->ne[3]);
    return sizeof(float)*nrows*max_nwg*size_t(op->ne[0] + 2);
}

static int ggml_metal_op_selected_row_flash_vec(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);
    if (!ggml_metal_selected_row_flash_vec_shape_ok(op)) {
        return 0;
    }

    ggml_tensor * q = op->src[0];
    ggml_tensor * rows = op->src[1];
    ggml_tensor * view = op->src[2];
    ggml_tensor * packed_kv = rows->src[0];
    ggml_tensor * top_k = rows->src[1];

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    if (ggml_metal_glm_selected_row_flash_noop_enabled()) {
        auto pipeline = ggml_metal_library_get_pipeline_zero_f32(lib);
        const int nth = 256;
        const int64_t n_tg = (ggml_nelements(op) + nth - 1)/nth;
        ggml_metal_op_concurrency_reset(ctx);
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, nth, 1, 1);
        return 1;
    }

    ggml_metal_kargs_selected_row_flash args = {
        /*.ne00 =*/ (int32_t) q->ne[0],
        /*.ne01 =*/ (int32_t) q->ne[1],
        /*.ne02 =*/ (int32_t) q->ne[2],
        /*.ne03 =*/ (int32_t) q->ne[3],
        /*.nb00 =*/ q->nb[0],
        /*.nb01 =*/ q->nb[1],
        /*.nb02 =*/ q->nb[2],
        /*.nb03 =*/ q->nb[3],
        /*.ne10 =*/ (int32_t) packed_kv->ne[0],
        /*.ne11 =*/ (int32_t) packed_kv->ne[1],
        /*.ne12 =*/ (int32_t) packed_kv->ne[2],
        /*.ne13 =*/ (int32_t) packed_kv->ne[3],
        /*.nb10 =*/ packed_kv->nb[0],
        /*.nb11 =*/ packed_kv->nb[1],
        /*.nb12 =*/ packed_kv->nb[2],
        /*.nb13 =*/ packed_kv->nb[3],
        /*.ne20 =*/ (int32_t) view->ne[0],
        /*.ne21 =*/ (int32_t) view->ne[1],
        /*.ne22 =*/ (int32_t) view->ne[2],
        /*.ne23 =*/ (int32_t) view->ne[3],
        /*.nb20 =*/ view->nb[0],
        /*.nb21 =*/ view->nb[1],
        /*.nb22 =*/ view->nb[2],
        /*.nb23 =*/ view->nb[3],
        /*.ne30 =*/ 0,
        /*.ne31 =*/ 0,
        /*.ne32 =*/ 0,
        /*.ne33 =*/ 0,
        /*.nb30 =*/ 0,
        /*.nb31 =*/ 0,
        /*.nb32 =*/ 0,
        /*.nb33 =*/ 0,
        /*.ne40 =*/ (int32_t) top_k->ne[0],
        /*.ne41 =*/ (int32_t) top_k->ne[1],
        /*.ne42 =*/ (int32_t) top_k->ne[2],
        /*.ne43 =*/ (int32_t) top_k->ne[3],
        /*.nb40 =*/ top_k->nb[0],
        /*.nb41 =*/ top_k->nb[1],
        /*.nb42 =*/ top_k->nb[2],
        /*.nb43 =*/ top_k->nb[3],
        /*.ne0  =*/ (int32_t) op->ne[0],
        /*.ne1  =*/ (int32_t) op->ne[1],
        /*.ne2  =*/ (int32_t) op->ne[2],
        /*.ne3  =*/ (int32_t) op->ne[3],
        /*.nb0  =*/ op->nb[0],
        /*.nb1  =*/ op->nb[1],
        /*.nb2  =*/ op->nb[2],
        /*.nb3  =*/ op->nb[3],
        /*.scale =*/ ggml_get_op_params_f32(op, 0),
    };

    const bool use_tiled =
        ggml_metal_glm_dsa_selected_row_flash_tiled_selected(op) &&
        ggml_metal_glm_dsa_selected_row_flash_tiled_shape(op);
    const int32_t requested_nsg =
        ggml_metal_glm_dsa_selected_row_flash_heads_per_tg_requested();
    const bool use_pair = !use_tiled && requested_nsg == 2 && args.ne02 % 2 == 0;
    const int32_t nsg = use_tiled ? 4 : (use_pair ? 2 : 1);
    // Selected-row flash fuses the compact gather into attention. On the
    // GLM-5.2 top_k=768 shape it wants the generic flash width, not the
    // lower all-KV compact-flash default.
    const int32_t nwg = ggml_metal_flash_attn_ext_vec_nwg(op, false, false, false, false, false);

    auto pipeline = use_tiled ?
        ggml_metal_library_get_pipeline_selected_row_flash_tiled(lib, nwg) :
        (use_pair ?
            ggml_metal_library_get_pipeline_selected_row_flash_pair(lib, nwg) :
            ggml_metal_library_get_pipeline_selected_row_flash_vec(lib, 1, nwg));
    const int32_t dispatch_nsg = use_tiled ? 4 : 1;
    GGML_ASSERT(dispatch_nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);
    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    ggml_metal_buffer_id bid_v = ggml_metal_get_buffer_id(packed_kv);
    bid_v.offs += view->view_offs;

#define SELECTED_ROW_FLASH_VEC_SMEM(nsg_) (GGML_PAD(((GGML_PAD(args.ne00, 128) + 4*OP_FLASH_ATTN_EXT_VEC_NCPSG + 2*GGML_PAD(args.ne20, 128))*(nsg_))*(sizeof(float)/2), 16))
#define SELECTED_ROW_FLASH_TILED_SMEM (GGML_PAD((8*576 + 2*8*512 + 2*8*64 + 4*4*16*8)*sizeof(uint16_t), 16))
    const size_t smem = use_tiled ? SELECTED_ROW_FLASH_TILED_SMEM : SELECTED_ROW_FLASH_VEC_SMEM(nsg);
#undef SELECTED_ROW_FLASH_TILED_SMEM
#undef SELECTED_ROW_FLASH_VEC_SMEM

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
    GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

    ggml_metal_op_concurrency_reset(ctx);
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                    0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(q),            1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(packed_kv),    2);
    ggml_metal_encoder_set_buffer  (enc, bid_v,                                  3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(top_k),        4);

    const int grid_x = use_tiled ? (args.ne02 + 7)/8 :
        (args.ne01 + OP_FLASH_ATTN_EXT_VEC_NQPSG - 1)/OP_FLASH_ATTN_EXT_VEC_NQPSG;
    const int grid_y = use_tiled ? args.ne01 : (args.ne02 + nsg - 1)/nsg;
    const int grid_z = args.ne03*nwg;

    if (nwg == 1) {
        GGML_ASSERT(ggml_metal_op_flash_attn_ext_extra_tmp(op) == 0);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 5);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(top_k), 6); // unused mask
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, dispatch_nsg, 1);
    } else {
        GGML_ASSERT(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);
        ggml_metal_encoder_set_buffer(enc, bid_tmp, 5);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(top_k), 6); // unused mask
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, dispatch_nsg, 1);

        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
            (int32_t) (args.ne1*args.ne2*args.ne3),
        };

        auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, args.ne20, nwg);
        ggml_metal_encoder_set_pipeline(enc, pipeline0);
        ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

        ggml_metal_encoder_dispatch_threadgroups(enc, args0.nrows, 1, 1, 32*nwg, 1, 1);
    }

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=selected_row_flash kernel=%s tensor=%s q_type=%s k_type=%s top_k_type=%s dst_type=%s q_width=%lld v_width=%lld batch=%lld heads=%lld stream=%lld kv=%lld top_k=%lld nwg=%d smem=%zu grid_x=%d grid_y=%d grid_z=%d threads_x=32 threads_y=%d\n",
            use_tiled ? "indirect_tiled" : (use_pair ? "gather_pair" : "gather_vec"),
            ggml_metal_tensor_name(op),
            ggml_type_name(q->type),
            ggml_type_name(packed_kv->type),
            ggml_type_name(top_k->type),
            ggml_type_name(op->type),
            (long long) args.ne00,
            (long long) args.ne20,
            (long long) args.ne01,
            (long long) args.ne02,
            (long long) args.ne03,
            (long long) args.ne11,
            (long long) args.ne40,
            nwg,
            smem,
            grid_x,
            grid_y,
            grid_z,
            dispatch_nsg);
    }

    return 1;
}

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (int selected_row_flash = ggml_metal_op_selected_row_flash_vec(ctx, idx)) {
        return selected_row_flash;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_src3 = has_mask  ? ggml_metal_get_buffer_id(op->src[3]) : bid_src0;
    ggml_metal_buffer_id bid_src4 = has_sinks ? ggml_metal_get_buffer_id(op->src[4]) : bid_src0;

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    if (!ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11,
                /*.nb12    =*/nb12,
                /*.nb13    =*/nb13,
                /*.nb21    =*/nb21,
                /*.nb22    =*/nb22,
                /*.nb23    =*/nb23,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        const int is_q = ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // 2*(2*ncpsg)
        // ncpsg soft_max values + ncpsg mask values
        //
        // 16*32*(nsg)
        // the shared memory needed for the simdgroups to load the KV cache
        // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
        //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(ne00 + 2*GGML_PAD(ne20, 64) + 2*(2*ncpsg)) + is_q*(16*32*(nsg)))*(sizeof(float)/2), 16))

        //int64_t nsgmax = 4;
        //
        //if (is_q) {
        //    nsgmax = 2;
        //    while (true) {
        //        const size_t smem = FATTN_SMEM(nsgmax);
        //        if (smem > props_dev->max_theadgroup_memory_size) {
        //            break;
        //        }
        //        nsgmax *= 2;
        //    }
        //    nsgmax /= 2;
        //}

        // simdgroups per threadgroup (a.k.a. warps)
        //nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;
        int32_t nsg = ne00 >= 512 ? 8 : 4;

        const size_t smem = FATTN_SMEM(nsg);

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  8);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const int grid_x = (ne01 + nqptg - 1)/nqptg;
        const int grid_y = ne02;
        const int grid_z = ne03;
        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                "ggml: glm_dsa_metal_dispatch op=flash_attn_ext kernel=tile tensor=%s q_type=%s k_type=%s v_type=%s mask_type=%s dst_type=%s q_width=%lld v_width=%lld batch=%lld heads=%lld stream=%lld kv=%lld grid_x=%d grid_y=%d grid_z=%d threads_x=32 threads_y=%d\n",
                ggml_metal_tensor_name(op),
                ggml_type_name(op->src[0]->type),
                ggml_type_name(op->src[1]->type),
                ggml_type_name(op->src[2]->type),
                has_mask ? ggml_type_name(op->src[3]->type) : "none",
                ggml_type_name(op->type),
                (long long) ne00,
                (long long) ne20,
                (long long) ne01,
                (long long) ne02,
                (long long) ne03,
                (long long) ne11,
                grid_x,
                grid_y,
                grid_z,
                nsg);
        }
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);
#undef FATTN_SMEM
    } else {
        // half4x4 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_VEC_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        const int nhptg = 1;                           // heads per threadgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 1  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11,
                /*.nb12    =*/nb12,
                /*.nb13    =*/nb13,
                /*.nb21    =*/nb21,
                /*.nb22    =*/nb22,
                /*.nb23    =*/nb23,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        GGML_ASSERT(ne10 >= ne20);

        // ne00 + 2*ncpsg*(nsg)
        // for each query, we load it as f16 in shared memory (ne00)
        // and store the soft_max values and the mask
        //
        // ne20*(nsg)
        // each simdgroup has a full f32 head vector in shared mem to accumulate results
        //
#define FATTN_SMEM(nsg) (GGML_PAD(((GGML_PAD(ne00, 128) + 4*ncpsg + 2*GGML_PAD(ne20, 128))*(nsg))*(sizeof(float)/2), 16))

        int64_t nsg = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (false) {
            // for small KV caches, we could launch a single workgroup and write the results directly to dst/
            // however, this does not lead to significant improvement, so disabled
            nwg = 1;
            nsg = 4;
        } else {
            nwg = 32;
            nsg = 1;
            while (2*nwg*nsg*ncpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }
        if (!has_mask &&
                !has_sinks &&
                !has_bias &&
                !has_scap &&
                !has_kvpad &&
                op->src[1]->type == GGML_TYPE_F16 &&
                op->src[2]->type == GGML_TYPE_F16 &&
                ne00 == 576 &&
                ne20 == 512 &&
                ne01 == 1 &&
                ne11 == 2048) {
            nwg = ggml_metal_flash_attn_ext_vec_nwg(
                    op, has_mask, has_sinks, has_bias, has_scap, has_kvpad);
            nsg = 1;
        }

        ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        const char * split_exact_tensor =
            getenv("GGML_METAL_EXPERIMENTAL_GLM_COMPACT_SPLIT_EXACT_TENSOR");
        const bool split_exact_tensor_selected =
            split_exact_tensor == nullptr || split_exact_tensor[0] == '\0' ||
            strcmp(split_exact_tensor, ggml_metal_tensor_name(op)) == 0;
        const bool glm_compact_split_exact =
            ggml_metal_glm_compact_split_exact_enabled() &&
            split_exact_tensor_selected &&
            !has_mask &&
            !has_sinks &&
            !has_bias &&
            !has_scap &&
            !has_kvpad &&
            op->src[1]->type == GGML_TYPE_F16 &&
            op->src[2]->type == GGML_TYPE_F16 &&
            ne00 == 576 &&
            ne20 == 512 &&
            ne01 == 1 &&
            ne02 == 64 &&
            ne03 == 1 &&
            ne11 == 2048 &&
            ne12 == 1 &&
            ne22 == 1 &&
            nwg == 4;
        if (glm_compact_split_exact) {
            constexpr int32_t exact_nwg = 4;
            constexpr size_t chunk_rows = OP_FLASH_ATTN_EXT_VEC_NCPSG;
            const size_t nrows = size_t(ne1)*size_t(ne2)*size_t(ne3);
            const size_t chunk_count = (size_t(ne11) + chunk_rows - 1)/chunk_rows;
            const size_t score_bytes = sizeof(float)*size_t(ne02)*size_t(ne03)*size_t(ne11);
            const size_t chunk_ms_bytes = sizeof(float)*size_t(ne02)*size_t(ne03)*chunk_count;
            const size_t chunk_v_bytes = sizeof(float)*size_t(ne02)*size_t(ne03)*chunk_count*size_t(ne20);
            const size_t partial_bytes = sizeof(float)*nrows*exact_nwg*size_t(ne20 + 2);
            GGML_ASSERT(chunk_count == size_t(ne11)/OP_FLASH_ATTN_EXT_VEC_NCPSG);
            GGML_ASSERT(ggml_metal_op_flash_attn_ext_extra_tmp(op) >=
                    score_bytes + chunk_ms_bytes + chunk_v_bytes + partial_bytes);

            ggml_metal_buffer_id bid_scores = bid_tmp;
            ggml_metal_buffer_id bid_chunk_ms = bid_scores;
            bid_chunk_ms.offs += score_bytes;
            ggml_metal_buffer_id bid_chunk_v = bid_chunk_ms;
            bid_chunk_v.offs += chunk_ms_bytes;
            ggml_metal_buffer_id bid_partials = bid_chunk_v;
            bid_partials.offs += chunk_v_bytes;
            ggml_metal_buffer_id bid_reduce_partials = bid_partials;

            auto qk_pipeline = ggml_metal_library_get_pipeline_glm_compact_qk_scores(lib);
            constexpr size_t qk_smem = GGML_PAD(576, 128)*sizeof(uint16_t);
            GGML_ASSERT(32 <= ggml_metal_pipeline_max_theads_per_threadgroup(qk_pipeline));
            GGML_ASSERT(qk_smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, qk_pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_scores, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 4);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, qk_smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(
                    enc, (ne11 + OP_FLASH_ATTN_EXT_VEC_NCPSG - 1)/OP_FLASH_ATTN_EXT_VEC_NCPSG,
                    ne02, ne03, 32, 1, 1);

            ggml_metal_op_internal_phase_barrier(ctx);

            if (ggml_metal_glm_compact_dump_scores_diagnostic_enabled()) {
                return 1;
            }

            const bool legacy_scores_v =
                ggml_metal_glm_compact_legacy_scores_v_diagnostic_enabled();
            if (legacy_scores_v) {
                bid_reduce_partials = bid_scores;
                bid_reduce_partials.offs += score_bytes;
                auto legacy_v_pipeline = ggml_metal_library_get_pipeline_glm_compact_scores_v(lib);
                constexpr size_t legacy_v_smem = (OP_FLASH_ATTN_EXT_VEC_NCPSG + 512)*sizeof(float);
                GGML_ASSERT(32 <= ggml_metal_pipeline_max_theads_per_threadgroup(legacy_v_pipeline));
                GGML_ASSERT(legacy_v_smem <= props_dev->max_theadgroup_memory_size);

                ggml_metal_encoder_set_pipeline(enc, legacy_v_pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_scores, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_reduce_partials, 3);
                ggml_metal_encoder_set_threadgroup_memory_size(enc, legacy_v_smem, 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03*exact_nwg, 32, 1, 1);
            } else {
                auto prefix_pipeline = ggml_metal_library_get_pipeline_glm_compact_softmax_prefix(lib);
                GGML_ASSERT(32 <= ggml_metal_pipeline_max_theads_per_threadgroup(prefix_pipeline));

                ggml_metal_encoder_set_pipeline(enc, prefix_pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_scores, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_chunk_ms, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_partials, 3);
                ggml_metal_encoder_set_threadgroup_memory_size(
                        enc, 2*OP_FLASH_ATTN_EXT_VEC_NCPSG*sizeof(float), 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, exact_nwg, ne02, ne03, 32, 1, 1);

                ggml_metal_op_internal_phase_barrier(ctx);
            }

            if (!legacy_scores_v && ggml_metal_glm_compact_sequential_v_diagnostic_enabled()) {
                auto sequential_v_pipeline =
                    ggml_metal_library_get_pipeline_glm_compact_probs_v_sequential(lib);
                GGML_ASSERT(32 <= ggml_metal_pipeline_max_theads_per_threadgroup(sequential_v_pipeline));

                ggml_metal_encoder_set_pipeline(enc, sequential_v_pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_scores, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_chunk_ms, 3);
                ggml_metal_encoder_set_buffer  (enc, bid_partials, 4);
                ggml_metal_encoder_set_threadgroup_memory_size(enc, 512*sizeof(float), 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, exact_nwg, ne02, ne03, 32, 1, 1);
            } else if (!legacy_scores_v) {
                auto chunk_v_pipeline = ggml_metal_library_get_pipeline_glm_compact_chunk_v(lib);
                GGML_ASSERT(32 <= ggml_metal_pipeline_max_theads_per_threadgroup(chunk_v_pipeline));

                ggml_metal_encoder_set_pipeline(enc, chunk_v_pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_scores, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_chunk_v, 3);
                ggml_metal_encoder_dispatch_threadgroups(enc, chunk_count, ne02, ne03, 32, 1, 1);

                ggml_metal_op_internal_phase_barrier(ctx);

                auto fold_pipeline = ggml_metal_library_get_pipeline_glm_compact_chunk_fold(lib);
                GGML_ASSERT(32 <= ggml_metal_pipeline_max_theads_per_threadgroup(fold_pipeline));

                ggml_metal_encoder_set_pipeline(enc, fold_pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_chunk_ms, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_chunk_v, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_partials, 3);
                ggml_metal_encoder_set_threadgroup_memory_size(enc, 512*sizeof(float), 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, exact_nwg, ne02, ne03, 32, 1, 1);
            }

            ggml_metal_op_internal_phase_barrier(ctx);

            const int32_t nrows_i32 = ne1*ne2*ne3;
            ggml_metal_kargs_flash_attn_ext_vec_reduce reduce_args = {
                nrows_i32,
            };
            auto reduce_pipeline =
                ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, exact_nwg);
            ggml_metal_encoder_set_pipeline(enc, reduce_pipeline);
            ggml_metal_encoder_set_bytes   (enc, &reduce_args, sizeof(reduce_args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_reduce_partials, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);
            ggml_metal_encoder_dispatch_threadgroups(enc, nrows_i32, 1, 1, 32*exact_nwg, 1, 1);

            if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
                GGML_LOG_INFO(
                    "ggml: glm_dsa_metal_dispatch op=flash_attn_ext kernel=glm_compact_chunk_exact tensor=%s q_type=%s k_type=%s v_type=%s mask_type=none dst_type=%s q_width=%lld v_width=%lld batch=%lld heads=%lld stream=%lld kv=%lld chunks=%zu score_bytes=%zu chunk_v_bytes=%zu reduction_strategy=scale_%.9g q_nb01=%llu q_nb02=%llu q_nb03=%llu k_nb11=%llu k_nb12=%llu k_nb13=%llu v_nb21=%llu v_nb22=%llu v_nb23=%llu out_ne1=%lld out_ne2=%lld out_ne3=%lld grid_x=%zu grid_y=%lld grid_z=%lld threads_x=32 threads_y=1 nwg=%d\n",
                    ggml_metal_tensor_name(op),
                    ggml_type_name(op->src[0]->type),
                    ggml_type_name(op->src[1]->type),
                    ggml_type_name(op->src[2]->type),
                    ggml_type_name(op->type),
                    (long long) ne00,
                    (long long) ne20,
                    (long long) ne01,
                    (long long) ne02,
                    (long long) ne03,
                    (long long) ne11,
                    chunk_count,
                    score_bytes,
                    chunk_v_bytes,
                    (double) scale,
                    (unsigned long long) nb01,
                    (unsigned long long) nb02,
                    (unsigned long long) nb03,
                    (unsigned long long) nb11,
                    (unsigned long long) nb12,
                    (unsigned long long) nb13,
                    (unsigned long long) nb21,
                    (unsigned long long) nb22,
                    (unsigned long long) nb23,
                    (long long) ne1,
                    (long long) ne2,
                    (long long) ne3,
                    chunk_count,
                    (long long) ne02,
                    (long long) ne03,
                    exact_nwg);
            }

            return 1;
        }

        const bool glm_compact_multihead =
            ggml_metal_glm_compact_multihead_flash_enabled(op) &&
            !has_mask &&
            !has_sinks &&
            !has_bias &&
            !has_scap &&
            !has_kvpad &&
            op->src[1]->type == GGML_TYPE_F16 &&
            op->src[2]->type == GGML_TYPE_F16 &&
            ne00 == 576 &&
            ne20 == 512 &&
            ne01 == 1 &&
            ne02 >= 8 &&
            ne02%8 == 0 &&
            ne11 >= 64 &&
            ne11%64 == 0 &&
            ne12 == 1 &&
            ne22 == 1;
        if (glm_compact_multihead) {
            auto glm_pipeline = ggml_metal_library_get_pipeline_glm_compact_multihead_flash(lib, nwg);
            GGML_ASSERT(2*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(glm_pipeline));

            constexpr size_t glm_smem =
                GGML_PAD(576*sizeof(uint16_t) + (32 + 4 + 512)*sizeof(float), 16);
            GGML_ASSERT(glm_smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, glm_pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
            ggml_metal_encoder_set_buffer  (enc, nwg == 1 ? bid_dst : bid_tmp, 4);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, glm_smem, 0);

            const int grid_x = ne01;
            const int grid_y = ne02;
            const int grid_z = ne03*nwg;
            ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, 2, 1);

            if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
                GGML_LOG_INFO(
                    "ggml: glm_dsa_metal_dispatch op=flash_attn_ext kernel=glm_compact_multihead tensor=%s q_type=%s k_type=%s v_type=%s dst_type=%s q_width=%lld v_width=%lld batch=%lld heads=%lld stream=%lld kv=%lld grid_x=%d grid_y=%d grid_z=%d threads_x=32 threads_y=4 nwg=%d smem=%zu\n",
                    ggml_metal_tensor_name(op),
                    ggml_type_name(op->src[0]->type),
                    ggml_type_name(op->src[1]->type),
                    ggml_type_name(op->src[2]->type),
                    ggml_type_name(op->type),
                    (long long) ne00,
                    (long long) ne20,
                    (long long) ne01,
                    (long long) ne02,
                    (long long) ne03,
                    (long long) ne11,
                    grid_x,
                    grid_y,
                    grid_z,
                    nwg,
                    glm_smem);
            }

            if (nwg > 1) {
                GGML_ASSERT(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);
                ggml_metal_op_concurrency_reset(ctx);

                const int32_t nrows = ne1*ne2*ne3;
                ggml_metal_kargs_flash_attn_ext_vec_reduce reduce_args = {
                    nrows,
                };
                auto reduce_pipeline =
                    ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml_metal_encoder_set_pipeline(enc, reduce_pipeline);
                ggml_metal_encoder_set_bytes   (enc, &reduce_args, sizeof(reduce_args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);
                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }

            return 1;
        }

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, nwg);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);

        const size_t smem = FATTN_SMEM(nsg);

        //printf("smem: %zu, max: %zu, nsg = %d, nsgmax = %d\n", smem, props_dev->max_theadgroup_memory_size, (int) nsg, (int) nsgmax);
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) == 0);

            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            const int grid_x = (ne01 + nqptg - 1)/nqptg;
            const int grid_y = (ne02 + nhptg - 1)/nhptg;
            const int grid_z = ne03*nwg;
            if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
                GGML_LOG_INFO(
                    "ggml: glm_dsa_metal_dispatch op=flash_attn_ext kernel=vec tensor=%s q_type=%s k_type=%s v_type=%s mask_type=%s dst_type=%s q_width=%lld v_width=%lld batch=%lld heads=%lld stream=%lld kv=%lld grid_x=%d grid_y=%d grid_z=%d threads_x=32 threads_y=%lld nwg=%d\n",
                    ggml_metal_tensor_name(op),
                    ggml_type_name(op->src[0]->type),
                    ggml_type_name(op->src[1]->type),
                    ggml_type_name(op->src[2]->type),
                    has_mask ? ggml_type_name(op->src[3]->type) : "none",
                    ggml_type_name(op->type),
                    (long long) ne00,
                    (long long) ne20,
                    (long long) ne01,
                    (long long) ne02,
                    (long long) ne03,
                    (long long) ne11,
                    grid_x,
                    grid_y,
                    grid_z,
                    nsg,
                    nwg);
            }
            ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);
        } else {
            // sanity checks
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            // write the results from each workgroup into a temp buffer
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            const int grid_x = (ne01 + nqptg - 1)/nqptg;
            const int grid_y = (ne02 + nhptg - 1)/nhptg;
            const int grid_z = ne03*nwg;
            if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
                GGML_LOG_INFO(
                    "ggml: glm_dsa_metal_dispatch op=flash_attn_ext kernel=vec tensor=%s q_type=%s k_type=%s v_type=%s mask_type=%s dst_type=%s q_width=%lld v_width=%lld batch=%lld heads=%lld stream=%lld kv=%lld grid_x=%d grid_y=%d grid_z=%d threads_x=32 threads_y=%lld nwg=%d\n",
                    ggml_metal_tensor_name(op),
                    ggml_type_name(op->src[0]->type),
                    ggml_type_name(op->src[1]->type),
                    ggml_type_name(op->src[2]->type),
                    has_mask ? ggml_type_name(op->src[3]->type) : "none",
                    ggml_type_name(op->type),
                    (long long) ne00,
                    (long long) ne20,
                    (long long) ne01,
                    (long long) ne02,
                    (long long) ne03,
                    (long long) ne11,
                    grid_x,
                    grid_y,
                    grid_z,
                    nsg,
                    nwg);
            }
            ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                const int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml_metal_encoder_set_pipeline(enc, pipeline0);
                ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
#undef FATTN_SMEM
    }

    return 1;
}

int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion) {
        fops[0] = GGML_OP_ADD;
        fops[1] = GGML_OP_ADD;
        fops[2] = GGML_OP_ADD;
        fops[3] = GGML_OP_ADD;
        fops[4] = GGML_OP_ADD;
        fops[5] = GGML_OP_ADD;
        fops[6] = GGML_OP_ADD;
        fops[7] = GGML_OP_ADD;

        // note: in metal, we sometimes encode the graph in parallel so we have to avoid fusing ops
        //       across splits. idx_end indicates the last node in the current split
        for (n_fuse = 0; n_fuse <= 6; ++n_fuse) {
            if (!ctx->can_fuse(idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            ggml_tensor * f0 = ctx->node(idx + n_fuse);
            ggml_tensor * f1 = ctx->node(idx + n_fuse + 1);

            if (f0 != f1->src[0]) {
                break;
            }

            // b[0] === b[1] === ...
            if (!ggml_are_same_layout(f0->src[1], f1->src[1])) {
                break;
            }

            // only fuse ops if src1 is in the same Metal buffer
            ggml_metal_buffer_id bid_fuse = ggml_metal_get_buffer_id(f1->src[1]);
            if (bid_fuse.metal != bid_src1.metal) {
                break;
            }

            //ctx->fuse_cnt[ops[n_fuse + 1]->op]++;

            args.o1[n_fuse + 1] = bid_fuse.offs;
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    struct ggml_metal_pipeline_with_params pipeline;

    pipeline = ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne10 = ne10/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    int grid_x = 0;
    int grid_y = 0;
    int grid_z = 0;
    int nth = 0;
    if (pipeline.cnt) {
        grid_x = args.ne0;
        grid_y = ggml_nrows(op);
        grid_z = 1;
        nth = 1;
        ggml_metal_encoder_dispatch_threadgroups(enc, args.ne0, ggml_nrows(op), 1, 1, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        nth = 1;

        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        grid_x = ne01;
        grid_y = ne02;
        grid_z = ne03;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    if (ggml_metal_glm_dsa_dispatch_log_enabled() &&
            ggml_metal_tensor_name_contains(op, "ffn_moe_down_weighted_input")) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=bin kernel=%s tensor=%s src0=%s src1=%s src0_type=%s src1_type=%s dst_type=%s ne0=%d rows=%lld n_fuse=%d c4=%d cnt=%d grid_x=%d grid_y=%d grid_z=%d threads_x=%d\n",
            ggml_op_name(op->op),
            ggml_metal_tensor_name(op),
            ggml_metal_tensor_name(op->src[0]),
            ggml_metal_tensor_name(op->src[1]),
            ggml_type_name(op->src[0]->type),
            ggml_type_name(op->src[1]->type),
            ggml_type_name(op->type),
            ne0,
            (long long) ggml_nrows(op),
            n_fuse,
            pipeline.c4 ? 1 : 0,
            pipeline.cnt ? 1 : 0,
            grid_x,
            grid_y,
            grid_z,
            nth);
    }

    return n_fuse;
}

int ggml_metal_op_l2_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    if (ggml_metal_glm_lightning_indexer_noop_enabled()) {
        auto pipeline = ggml_metal_library_get_pipeline_zero_f32(lib);
        const int nth = 256;
        const int64_t n_tg = (ggml_nelements(op) + nth - 1)/nth;
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, nth, 1, 1);
        return 1;
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_l2_norm args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.eps   =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_t =*/ ne00 % 4 == 0 ? ne00/4 : ne00,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b)
    // d[2] = add(d[1], c)
    if (use_fusion) {
        fops[0] = op->op;
        fops[1] = GGML_OP_MUL;
        fops[2] = GGML_OP_ADD;

        for (n_fuse = 0; n_fuse <= 1; ++n_fuse) {
            if (!ctx->can_fuse(idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            ggml_tensor * f0 = ctx->node(idx + n_fuse);
            ggml_tensor * f1 = ctx->node(idx + n_fuse + 1);

            if (f0 != f1->src[0]) {
                break;
            }

            if (f1->src[1]->ne[0] != op->ne[0]) {
                break;
            }

            if (!ggml_is_contiguous_rows(f1->src[1])) {
                break;
            }

            if (f1->type != GGML_TYPE_F32) {
                break;
            }

            //ctx->fuse_cnt[f1->op]++;

            bid_fuse[n_fuse] = ggml_metal_get_buffer_id(f1->src[1]);

            args.nef1[n_fuse + 1] = f1->src[1]->ne[1];
            args.nef2[n_fuse + 1] = f1->src[1]->ne[2];
            args.nef3[n_fuse + 1] = f1->src[1]->ne[3];

            args.nbf1[n_fuse + 1] = f1->src[1]->nb[1];
            args.nbf2[n_fuse + 1] = f1->src[1]->nb[2];
            args.nbf3[n_fuse + 1] = f1->src[1]->nb[3];
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            if (n_fuse == 2) {
                GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, ggml_op_name(op->op));
            }
            if (n_fuse == 3) {
                GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, ggml_op_name(op->op));
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    int nth = 32; // SIMD width

    while (nth < args.ne00_t && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, args.ne00_t);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0,    1);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    const int nth = std::min(1024, ne00);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != nullptr,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rope(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    auto pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    if (KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        const uint64_t ntptg0 = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/(KH*KW), N);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
    } else {
        const uint64_t n_threads = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), N);
        const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, quotient * CHW, OH, OW, n_threads, 1, 1);
    }

    return 1;
}

int ggml_metal_op_conv_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.IC   =*/ ne02,
        /*.OC   =*/ ne03,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const uint64_t n_out = ggml_nelements(op);

    uint64_t tg = (n_out + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<int>::max());

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_2d_dw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d_dw args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.C    =*/ ne12,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne13,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    const bool use_tiled = (nb12 < nb10);

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d_dw(lib, op, use_tiled);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const int32_t OW = ne0;
    const int32_t OH = ne1;
    const int32_t C  = ne12;
    const int32_t N  = ne13;

    const int tg_x = use_tiled ? (C + nth - 1) / nth : (OW + nth - 1) / nth;
    const int tg_y = OH;
    const int tg_z = use_tiled ? OW * N : C * N;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_3d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // 1. Extract standard dimensions and byte strides
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // 2. Extract hyperparams from op_params
    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t s2 = ((const int32_t *)(op->op_params))[2];
    const int32_t p0 = ((const int32_t *)(op->op_params))[3];
    const int32_t p1 = ((const int32_t *)(op->op_params))[4];
    const int32_t p2 = ((const int32_t *)(op->op_params))[5];
    const int32_t d0 = ((const int32_t *)(op->op_params))[6];
    const int32_t d1 = ((const int32_t *)(op->op_params))[7];
    const int32_t d2 = ((const int32_t *)(op->op_params))[8];
    const int32_t IC = ((const int32_t *)(op->op_params))[9];
    const int32_t N  = ((const int32_t *)(op->op_params))[10];
    const int32_t OC = ((const int32_t *)(op->op_params))[11];

    // 3. Build the parameter struct using the macro-generated variables
    ggml_metal_kargs_conv_3d args = {
        /*.IW =*/ (int32_t)op->src[1]->ne[0],
        /*.IH =*/ (int32_t)op->src[1]->ne[1],
        /*.ID =*/ (int32_t)op->src[1]->ne[2],
        /*.OW =*/ (int32_t)op->ne[0],
        /*.OH =*/ (int32_t)op->ne[1],
        /*.OD =*/ (int32_t)op->ne[2],
        /*.KW =*/ (int32_t)op->src[0]->ne[0],
        /*.KH =*/ (int32_t)op->src[0]->ne[1],
        /*.KD =*/ (int32_t)op->src[0]->ne[2],
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2,
        IC, N, OC,
        nb00, nb01, nb02, nb03, // Weight strides
        nb10, nb11, nb12, nb13, // Input strides
        nb0,  nb1,  nb2,  nb3   // Output strides
    };

    // 4. Fetch the JIT pipeline
    auto pipeline = ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    int nth0 = 32; // Standard SIMD width for Apple Silicon
    int nth1 = 1;
    int nth2 = 1;

    int64_t spatial_volume = args.OW * args.OH * args.OD;

    int ntg0 = (spatial_volume + nth0 - 1) / nth0;
    int ntg1 = args.OC;
    int ntg2 = args.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    return 1;
}

int ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_col2im_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t OC = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];

    const int32_t K_OC  = (int32_t) op->src[0]->ne[0];
    const int32_t T_in  = (int32_t) op->src[0]->ne[1];
    const int32_t K     = K_OC / OC;
    const int32_t T_out = (int32_t) op->ne[0];

    ggml_metal_kargs_col2im_1d args = {
        /*.T_in  =*/ T_in,
        /*.T_out =*/ T_out,
        /*.OC    =*/ OC,
        /*.K     =*/ K,
        /*.K_OC  =*/ K_OC,
        /*.s0    =*/ s0,
        /*.p0    =*/ p0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_col2im_1d(lib, op);

    const int total = T_out * OC;
    const int nth   = 256;
    const int ntg   = (total + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_transpose_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[2];
    const int32_t IH = op->src[1]->ne[1];
    const int32_t IW = op->src[1]->ne[0];

    const int32_t KH = op->src[0]->ne[1];
    const int32_t KW = op->src[0]->ne[0];

    const int32_t OW = op->ne[0];
    const int32_t OH = op->ne[1];
    const int32_t OC = op->ne[2];

    ggml_metal_kargs_conv_transpose_2d args = {
        /*.IC  =*/ IC,
        /*.IH  =*/ IH,
        /*.IW  =*/ IW,
        /*.KH  =*/ KH,
        /*.KW  =*/ KW,
        /*.OC  =*/ OC,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    const size_t smem = GGML_PAD(KW * KH * sizeof(float), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC, KW, KH, 1);

    return 1;
}

int ggml_metal_op_upscale(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float sf0 = (float)ne0/op->src[0]->ne[0];
    float sf1 = (float)ne1/op->src[0]->ne[1];
    float sf2 = (float)ne2/op->src[0]->ne[2];
    float sf3 = (float)ne3/op->src[0]->ne[3];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);

    float poffs = 0.5f;

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        poffs = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ggml_metal_kargs_upscale args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.sf0   =*/ sf0,
        /*.sf1   =*/ sf1,
        /*.sf2   =*/ sf2,
        /*.sf3   =*/ sf3,
        /*.poffs =*/ poffs,
    };

    auto pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_roll(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t s2 = ggml_get_op_params_i32(op, 2);
    const int32_t s3 = ggml_get_op_params_i32(op, 3);

    ggml_metal_kargs_roll args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.s2   =*/ s2,
        /*.s3   =*/ s3
    };

    auto pipeline = ggml_metal_library_get_pipeline_roll(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    const int nth_max = MIN(64, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int nth = MIN(args.ne0, nth_max);
    const int nk0 = (args.ne0 + 1024 - 1)/1024; // note: 1024 is hardcoded in the kernel!

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_arange(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    auto pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    auto pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    auto pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    int nth = 32; // SIMD width
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argsort(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const int npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ nth,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    int len = nth;

    while (len < ne00) {
        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ ne00,
            /*.len   =*/ len,
        };

        // merges per row
        const int nm = (ne00 + 2*len - 1) / (2*len);

        const int nth = std::min(512, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge));

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

int ggml_metal_op_lightning_indexer(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const bool parallel_requested = ggml_metal_lightning_indexer_parallel_requested();
    bool parallel = parallel_requested && ne01 <= 1024;
    const size_t staged_q_bytes = size_t(ne00)*size_t(ne01)*sizeof(float);
    const bool staged_q_requested = ggml_metal_lightning_indexer_staged_q_requested();
    const bool staged_q = staged_q_requested && !parallel &&
        op->src[0]->type == GGML_TYPE_F32 &&
        (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32) &&
        staged_q_bytes <= 32*1024;
    const int requested_parallel_threads = ggml_metal_lightning_indexer_parallel_threads_requested();
    auto pipeline = ggml_metal_library_get_pipeline_lightning_indexer(lib, op, parallel, staged_q);
    int nth = std::min(parallel ? requested_parallel_threads : 64, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    if (parallel && ne01 > nth) {
        parallel = false;
        pipeline = ggml_metal_library_get_pipeline_lightning_indexer(lib, op, parallel, false);
        nth = std::min(64, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    }

    static bool logged = false;
    if (!logged) {
        logged = true;
        const char * parallel_env = getenv("LLAMA_GLM_DSA_PARALLEL_LIGHTNING_INDEXER");
        GGML_LOG_INFO(
            "%s: env=%s requested=%d staged_q_requested=%d staged_q_selected=%d staged_q_bytes=%zu q_ne=(%lld,%lld,%lld,%lld) k_type=%s selected=%d nth=%d\n",
            __func__,
            parallel_env ? parallel_env : "<unset>",
            parallel_requested,
            staged_q_requested,
            staged_q,
            staged_q_bytes,
            (long long) ne00,
            (long long) ne01,
            (long long) ne02,
            (long long) ne03,
            ggml_type_name(op->src[1]->type),
            parallel,
            nth);
    }

    ggml_metal_kargs_lightning_indexer args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne10        =*/ ne10,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb10        =*/ nb10,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.ne20        =*/ ne20,
        /*.ne21        =*/ ne21,
        /*.ne22        =*/ ne22,
        /*.ne23        =*/ ne23,
        /*.nb20        =*/ nb20,
        /*.nb21        =*/ nb21,
        /*.nb22        =*/ nb22,
        /*.nb23        =*/ nb23,
        /*.ne30        =*/ ne30,
        /*.ne31        =*/ ne31,
        /*.ne32        =*/ ne32,
        /*.ne33        =*/ ne33,
        /*.nb30        =*/ nb30,
        /*.nb31        =*/ nb31,
        /*.nb32        =*/ nb32,
        /*.nb33        =*/ nb33,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // weights
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // mask
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst
    if (staged_q) {
        ggml_metal_encoder_set_threadgroup_memory_size(enc, staged_q_bytes, 0);
    }

    const int grid_x = parallel ? ne0 : (ne0 + nth - 1)/nth;
    const int grid_y = ne1;
    const int grid_z = ne3;
    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=lightning_indexer kernel=%s tensor=%s parallel=%d staged_q=%d staged_q_bytes=%zu q_type=%s k_type=%s dst_type=%s q_ne0=%lld q_ne1=%lld q_ne2=%lld q_ne3=%lld k_ne0=%lld k_ne1=%lld k_ne2=%lld k_ne3=%lld dst_ne0=%lld dst_ne1=%lld dst_ne2=%lld dst_ne3=%lld grid_x=%d grid_y=%d grid_z=%d threads_x=%d\n",
            staged_q ? "staged_q" : parallel ? "parallel" : "serial",
            ggml_metal_tensor_name(op),
            parallel ? 1 : 0,
            staged_q ? 1 : 0,
            staged_q_bytes,
            ggml_type_name(op->src[0]->type),
            ggml_type_name(op->src[1]->type),
            ggml_type_name(op->type),
            (long long) ne00,
            (long long) ne01,
            (long long) ne02,
            (long long) ne03,
            (long long) ne10,
            (long long) ne11,
            (long long) ne12,
            (long long) ne13,
            (long long) ne0,
            (long long) ne1,
            (long long) ne2,
            (long long) ne3,
            grid_x,
            grid_y,
            grid_z,
            nth);
    }
    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, nth, 1, 1);

    return 1;
}

int ggml_metal_op_dsa_sparse_mask(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_dsa_sparse_mask args = {
        /*.n_kv     =*/ ne01,
        /*.n_batch  =*/ ne02,
        /*.n_stream =*/ ne03,
        /*.n_top_k  =*/ ne10,
        /*.n_top_stream =*/ ne12,
        /*.elem_size =*/ (int32_t) ggml_type_size(op->type),
        /*._pad1    =*/ 0,
        /*._pad2    =*/ 0,
        /*.nb01     =*/ nb01,
        /*.nb02     =*/ nb02,
        /*.nb03     =*/ nb03,
        /*.nb10     =*/ nb10,
        /*.nb11     =*/ nb11,
        /*.nb12     =*/ nb12,
        /*.nb0      =*/ nb0,
        /*.nb1      =*/ nb1,
        /*.nb2      =*/ nb2,
        /*.nb3      =*/ nb3,
    };

    auto pipeline_fill = ggml_metal_library_get_pipeline_dsa_sparse_mask_fill(lib);
    auto pipeline_set  = ggml_metal_library_get_pipeline_dsa_sparse_mask_set(lib);

    ggml_metal_encoder_set_pipeline(enc, pipeline_fill);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),          0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    const int nth_fill = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_fill));
    const int fill_grid_x = (ne01 + nth_fill - 1)/nth_fill;
    const int fill_grid_y = ne02;
    const int fill_grid_z = ne03;
    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=dsa_sparse_mask kernel=fill tensor=%s src_type=%s top_k_type=%s dst_type=%s kv=%lld batch=%lld stream=%lld top_k=%lld top_stream=%lld grid_x=%d grid_y=%d grid_z=%d threads_x=%d\n",
            ggml_metal_tensor_name(op),
            ggml_type_name(op->src[0]->type),
            ggml_type_name(op->src[1]->type),
            ggml_type_name(op->type),
            (long long) ne01,
            (long long) ne02,
            (long long) ne03,
            (long long) ne10,
            (long long) ne12,
            fill_grid_x,
            fill_grid_y,
            fill_grid_z,
            nth_fill);
    }
    ggml_metal_encoder_dispatch_threadgroups(enc, fill_grid_x, fill_grid_y, fill_grid_z, nth_fill, 1, 1);

    ggml_metal_op_concurrency_reset(ctx);

    ggml_metal_encoder_set_pipeline(enc, pipeline_set);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth_set = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_set));
    const int set_grid_x = (ne10 + nth_set - 1)/nth_set;
    const int set_grid_y = ne11;
    const int set_grid_z = ne03;
    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=dsa_sparse_mask kernel=set tensor=%s src_type=%s top_k_type=%s dst_type=%s kv=%lld batch=%lld stream=%lld top_k=%lld top_stream=%lld grid_x=%d grid_y=%d grid_z=%d threads_x=%d\n",
            ggml_metal_tensor_name(op),
            ggml_type_name(op->src[0]->type),
            ggml_type_name(op->src[1]->type),
            ggml_type_name(op->type),
            (long long) ne01,
            (long long) ne02,
            (long long) ne03,
            (long long) ne10,
            (long long) ne12,
            set_grid_x,
            set_grid_y,
            set_grid_z,
            nth_set);
    }
    ggml_metal_encoder_dispatch_threadgroups(enc, set_grid_x, set_grid_y, set_grid_z, nth_set, 1, 1);

    return 1;
}

int ggml_metal_op_dsa_sparse_attn(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_dsa_sparse_attn args = {
        /*.ne00 =*/ ne00, /*.ne01 =*/ ne01, /*.ne02 =*/ ne02, /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00, /*.nb01 =*/ nb01, /*.nb02 =*/ nb02, /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10, /*.ne11 =*/ ne11, /*.ne12 =*/ ne12, /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10, /*.nb11 =*/ nb11, /*.nb12 =*/ nb12, /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20, /*.ne21 =*/ ne21, /*.ne22 =*/ ne22, /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20, /*.nb21 =*/ nb21, /*.nb22 =*/ nb22, /*.nb23 =*/ nb23,
        /*.ne30 =*/ ne30, /*.ne31 =*/ ne31, /*.ne32 =*/ ne32, /*.ne33 =*/ ne33,
        /*.nb30 =*/ nb30, /*.nb31 =*/ nb31, /*.nb32 =*/ nb32, /*.nb33 =*/ nb33,
        /*.ne40 =*/ ne40, /*.ne41 =*/ ne41, /*.ne42 =*/ ne42, /*.ne43 =*/ ne43,
        /*.nb40 =*/ nb40, /*.nb41 =*/ nb41, /*.nb42 =*/ nb42, /*.nb43 =*/ nb43,
        /*.ne0  =*/ ne0,  /*.ne1  =*/ ne1,  /*.ne2  =*/ ne2,  /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,  /*.nb1  =*/ nb1,  /*.nb2  =*/ nb2,  /*.nb3  =*/ nb3,
        /*.scale =*/ ggml_get_op_params_f32(op, 0),
    };

    const bool use_selected_row =
        ggml_metal_glm_dsa_selected_row_flash_enabled() &&
        op->type == GGML_TYPE_F32 &&
        op->src[0]->type == GGML_TYPE_F32 &&
        op->src[1]->type == GGML_TYPE_F16 &&
        op->src[2]->type == GGML_TYPE_F16 &&
        op->src[4]->type == GGML_TYPE_I32 &&
        ne00 == 576 &&
        ne20 == 512 &&
        ne01 > 0 &&
        ne02 > 0 &&
        ne03 > 0 &&
        ne12 == 1 &&
        ne22 == 1 &&
        ne40 > 0 &&
        ne40 <= 4096 &&
        ne41 == ne01 &&
        ne42 > 0 &&
        ne03 % ne42 == 0;
    if (use_selected_row) {
        const bool use_tiled =
            ggml_metal_glm_dsa_selected_row_flash_tiled_selected(op) &&
            ne02 >= 8 && ne02 % 8 == 0;
        const int32_t nwg = use_tiled
            ? ggml_metal_glm_dsa_selected_row_flash_tiled_nwg_for_top_k(ne40)
            : 1;
        const int32_t nsg = use_tiled ? 4 : 1;
        auto pipeline = use_tiled
            ? ggml_metal_library_get_pipeline_selected_row_flash_tiled(lib, nwg)
            : ggml_metal_library_get_pipeline_selected_row_flash_vec(lib, nsg, nwg);
        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

#define DSA_SELECTED_ROW_VEC_SMEM (GGML_PAD(((GGML_PAD(args.ne00, 128) + 4*OP_FLASH_ATTN_EXT_VEC_NCPSG + 2*GGML_PAD(args.ne20, 128))*nsg)*(sizeof(float)/2), 16))
#define DSA_SELECTED_ROW_TILED_SMEM (GGML_PAD((8*576 + 2*8*512 + 2*8*64 + 4*4*16*8)*sizeof(uint16_t), 16))
        const size_t smem = use_tiled ? DSA_SELECTED_ROW_TILED_SMEM : DSA_SELECTED_ROW_VEC_SMEM;
#undef DSA_SELECTED_ROW_TILED_SMEM
#undef DSA_SELECTED_ROW_VEC_SMEM

        const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);
        ggml_metal_buffer_id bid_tmp = bid_dst;
        bid_tmp.offs += ggml_nbytes(op);

        ggml_metal_op_concurrency_reset(ctx);
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                       0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]),      1); // q
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]),      2); // k
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]),      3); // v
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]),      4); // top_k
        ggml_metal_encoder_set_buffer  (enc, nwg == 1 ? bid_dst : bid_tmp,              5); // dst
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]),      6); // kq_mask_rows
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        const int grid_x = use_tiled ? (ne02 + 7)/8 : ne01;
        const int grid_y = use_tiled ? ne01 : ne02;
        const int grid_z = ne03*nwg;
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);

        if (nwg > 1) {
            ggml_metal_op_concurrency_reset(ctx);

            ggml_metal_kargs_flash_attn_ext_vec_reduce reduce_args = {
                (int32_t) (ne1*ne2*ne3),
            };
            auto reduce_pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);
            GGML_ASSERT(32*nwg <= ggml_metal_pipeline_max_theads_per_threadgroup(reduce_pipeline));
            ggml_metal_encoder_set_pipeline(enc, reduce_pipeline);
            ggml_metal_encoder_set_bytes(enc, &reduce_args, sizeof(reduce_args), 0);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer(enc, bid_dst, 2);
            ggml_metal_encoder_dispatch_threadgroups(enc, reduce_args.nrows, 1, 1, 32*nwg, 1, 1);
        }

        if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
            GGML_LOG_INFO(
                "ggml: glm_dsa_metal_dispatch op=dsa_sparse_attn kernel=%s tensor=%s q_type=%s k_type=%s v_type=%s mask_type=%s top_k_type=%s dst_type=%s q_width=%lld v_width=%lld batch=%lld heads=%lld stream=%lld kv=%lld top_k=%lld top_stream=%lld nwg=%d q_nb=%llu,%llu,%llu,%llu k_nb=%llu,%llu,%llu,%llu v_nb=%llu,%llu,%llu,%llu top_k_nb=%llu,%llu,%llu,%llu grid_x=%d grid_y=%d grid_z=%d threads_x=32 threads_y=%d\n",
                use_tiled ? "selected_row_tiled" : "selected_row",
                ggml_metal_tensor_name(op),
                ggml_type_name(op->src[0]->type),
                ggml_type_name(op->src[1]->type),
                ggml_type_name(op->src[2]->type),
                ggml_type_name(op->src[3]->type),
                ggml_type_name(op->src[4]->type),
                ggml_type_name(op->type),
                (long long) ne00,
                (long long) ne20,
                (long long) ne1,
                (long long) ne2,
                (long long) ne3,
                (long long) ne11,
                (long long) ne40,
                (long long) ne42,
                nwg,
                (unsigned long long) nb00,
                (unsigned long long) nb01,
                (unsigned long long) nb02,
                (unsigned long long) nb03,
                (unsigned long long) nb10,
                (unsigned long long) nb11,
                (unsigned long long) nb12,
                (unsigned long long) nb13,
                (unsigned long long) nb20,
                (unsigned long long) nb21,
                (unsigned long long) nb22,
                (unsigned long long) nb23,
                (unsigned long long) nb40,
                (unsigned long long) nb41,
                (unsigned long long) nb42,
                (unsigned long long) nb43,
                grid_x,
                grid_y,
                grid_z,
                nsg);
        }

        return 1;
    }

    const int requested_head_group = ggml_metal_glm_dsa_sparse_attn_decode_group_heads_requested();
    const bool use_decode_grouped =
        requested_head_group > 1 &&
        ne1 == 1 &&
        ne3 == 1 &&
        ne40 <= 1024 &&
        ne2 % requested_head_group == 0;
    const bool use_cached_topk =
        !use_decode_grouped &&
        ggml_metal_glm_dsa_sparse_attn_cache_topk_enabled() &&
        ne40 <= 1024;
    auto pipeline = use_decode_grouped
        ? ggml_metal_library_get_pipeline_dsa_sparse_attn_decode_grouped(lib, op)
        : use_cached_topk
        ? ggml_metal_library_get_pipeline_dsa_sparse_attn_cached_topk(lib, op)
        : ggml_metal_library_get_pipeline_dsa_sparse_attn(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // kq_mask_rows
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // top_k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst

    const int nth_requested = ggml_metal_glm_dsa_sparse_attn_threads_for_shape(
            ggml_metal_glm_dsa_sparse_attn_threads_requested(), ne1, ne40);
    const int head_group = use_decode_grouped ? requested_head_group : 1;
    const int max_threads_per_group = std::max(1, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/head_group);
    const int nth = std::min(nth_requested, max_threads_per_group);
    const int grid_x = ne1;
    const int grid_y = use_decode_grouped ? (ne2 + head_group - 1)/head_group : ne2;
    const int grid_z = ne3;
    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=dsa_sparse_attn kernel=%s tensor=%s q_type=%s k_type=%s v_type=%s mask_type=%s top_k_type=%s dst_type=%s q_width=%lld v_width=%lld batch=%lld heads=%lld stream=%lld kv=%lld top_k=%lld top_stream=%lld grid_x=%d grid_y=%d grid_z=%d threads_x=%d threads_y=%d\n",
            use_decode_grouped ? "decode_grouped" : use_cached_topk ? "cached_topk" : "default",
            ggml_metal_tensor_name(op),
            ggml_type_name(op->src[0]->type),
            ggml_type_name(op->src[1]->type),
            ggml_type_name(op->src[2]->type),
            ggml_type_name(op->src[3]->type),
            ggml_type_name(op->src[4]->type),
            ggml_type_name(op->type),
            (long long) ne00,
            (long long) ne20,
            (long long) ne1,
            (long long) ne2,
            (long long) ne3,
            (long long) ne11,
            (long long) ne40,
            (long long) ne42,
            grid_x,
            grid_y,
            grid_z,
            nth,
            head_group);
    }
    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, nth, head_group, 1);

    return 1;
}

int ggml_metal_op_dsa_top1_attn(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_dsa_top1_attn args = {
        /*.ne00 =*/ ne00, /*.ne01 =*/ ne01, /*.ne02 =*/ ne02, /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00, /*.nb01 =*/ nb01, /*.nb02 =*/ nb02, /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10, /*.ne11 =*/ ne11, /*.ne12 =*/ ne12, /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10, /*.nb11 =*/ nb11, /*.nb12 =*/ nb12, /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20, /*.ne21 =*/ ne21, /*.ne22 =*/ ne22, /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20, /*.nb21 =*/ nb21, /*.nb22 =*/ nb22, /*.nb23 =*/ nb23,
        /*.ne0  =*/ ne0,  /*.ne1  =*/ ne1,  /*.ne2  =*/ ne2,  /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,  /*.nb1  =*/ nb1,  /*.nb2  =*/ nb2,  /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsa_top1_attn(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // top_k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst

    const int nth = std::min(64, std::max(1, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)));
    const int grid_x = (ne0 + nth - 1)/nth;
    const int grid_y = ne1*ne2;
    const int grid_z = ne3;
    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=dsa_top1_attn tensor=%s q_type=%s v_type=%s top_k_type=%s dst_type=%s v_width=%lld batch=%lld heads=%lld stream=%lld kv=%lld value_heads=%lld top_stream=%lld grid_x=%d grid_y=%d grid_z=%d threads_x=%d\n",
            ggml_metal_tensor_name(op),
            ggml_type_name(op->src[0]->type),
            ggml_type_name(op->src[1]->type),
            ggml_type_name(op->src[2]->type),
            ggml_type_name(op->type),
            (long long) ne10,
            (long long) ne1,
            (long long) ne2,
            (long long) ne3,
            (long long) ne11,
            (long long) ne12,
            (long long) ne22,
            grid_x,
            grid_y,
            grid_z,
            nth);
    }
    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, nth, 1, 1);

    return 1;
}

int ggml_metal_op_moe_route_weights(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_moe_route_weights args = {
        /*.n_expert      =*/ ne01,
        /*.n_tokens      =*/ ne11,
        /*.n_expert_used =*/ ne10,
        /*.norm          =*/ ggml_get_op_params_i32(op, 2),
        /*.clamp_min     =*/ ggml_get_op_params_f32(op, 0),
        /*.scale         =*/ ggml_get_op_params_f32(op, 1),
        /*._pad0         =*/ ggml_metal_glm_dsa_moe_route_weights_slot0_enabled() ? 1 : 0,
        /*._pad1         =*/ 0,
        /*.probs_nb1     =*/ nb01,
        /*.probs_nb2     =*/ nb02,
        /*.ids_nb0       =*/ nb10,
        /*.ids_nb1       =*/ nb11,
        /*.dst_nb1       =*/ nb1,
        /*.dst_nb2       =*/ nb2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_moe_route_weights(lib);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                     0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]),    1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]),    2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),            3);

    const int nth = 1;

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=moe_route_weights tensor=%s probs=%s ids=%s experts=%d tokens=%d used_experts=%d norm=%d scale=%g force_slot0=%d grid_x=%d grid_y=1 grid_z=1 threads_x=%d\n",
            ggml_metal_tensor_name(op),
            ggml_metal_tensor_name(op->src[0]),
            ggml_metal_tensor_name(op->src[1]),
            args.n_expert,
            args.n_tokens,
            args.n_expert_used,
            args.norm,
            (double) args.scale,
            args._pad0,
            args.n_tokens,
            nth);
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, args.n_tokens, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_moe_weighted_sum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_moe_weighted_sum args = {
        /*.n_embd        =*/ ne00,
        /*.n_tokens      =*/ ne02,
        /*.n_expert_used =*/ ne01,
        /*.already_weighted =*/ ggml_get_op_params_i32(op, 0) != 0 ? 1 : 0,
        /*.experts_nb0   =*/ nb00,
        /*.experts_nb1   =*/ nb01,
        /*.experts_nb2   =*/ nb02,
        /*.weights_nb1   =*/ nb11,
        /*.weights_nb2   =*/ nb12,
        /*.dst_nb0       =*/ nb0,
        /*.dst_nb1       =*/ nb1,
    };

    const bool use_x4 =
        ne00 % 4 == 0 &&
        nb00 == sizeof(float) &&
        nb0 == sizeof(float) &&
        nb01 % (4*sizeof(float)) == 0 &&
        nb02 % (4*sizeof(float)) == 0 &&
        nb1  % (4*sizeof(float)) == 0;
    auto pipeline = use_x4 ?
        ggml_metal_library_get_pipeline_moe_weighted_sum_x4(lib) :
        ggml_metal_library_get_pipeline_moe_weighted_sum(lib);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                     0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]),    1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]),    2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),            3);

    const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int ncols = use_x4 ? ne00/4 : ne00;
    const int grid_x = (ncols + nth - 1)/nth;
    const int grid_y = ne02;

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=moe_weighted_sum kernel=%s tensor=%s experts=%s weights=%s embd=%lld tokens=%lld used_experts=%lld grid_x=%d grid_y=%d grid_z=1 threads_x=%d\n",
            args.already_weighted ? "already_weighted" : (use_x4 ? "f32x4" : "f32"),
            ggml_metal_tensor_name(op),
            ggml_metal_tensor_name(op->src[0]),
            ggml_metal_tensor_name(op->src[1]),
            (long long) ne00,
            (long long) ne02,
            (long long) ne01,
            grid_x,
            grid_y,
            nth);
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, 1, nth, 1, 1);

    return 1;
}

static int ggml_metal_encode_moe_mul_mat_id(
        ggml_metal_op_t ctx,
        ggml_tensor * op,
        const ggml_metal_glm_moe_private_bindings * private_bindings) {
    const bool private_path = private_bindings != nullptr;

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const bool use_glm52_slot_parallel =
        op->src[0]->type == GGML_TYPE_Q3_K &&
        ne00 == 2048 && ne01 == 6144 && ne02 == 256 &&
        op->src[1]->type == GGML_TYPE_F32 &&
        ne20 == 8 && ne21 == 1;
    auto pipeline = use_glm52_slot_parallel ?
        ggml_metal_library_get_pipeline_mul_mv_id_q3_weighted_reduce_slots_sg_r8_nb8_w0(lib, op) :
        ggml_metal_library_get_pipeline_mul_mv_id_weighted_reduce(lib, op);
    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;

    ggml_metal_kargs_mul_mv_id args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb1  =*/ nb1,
        /*.nr0  =*/ nr0,
    };
    ggml_metal_kargs_mul_mv_id_weighted_reduce_extra extra = {
        /*.weights_nb1 =*/ nb31,
        /*.weights_nb2 =*/ nb32,
        /*.dst_nb0 =*/ nb0,
        /*.dst_nb1 =*/ nb1,
        /*.already_weighted =*/ 0,
        /*._pad0 =*/ 0,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_bytes(enc, &extra, sizeof(extra), 6);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, private_path ? private_bindings->activation : ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         3);
    ggml_metal_encoder_set_buffer(enc, private_path ? private_bindings->ids : ggml_metal_get_buffer_id(op->src[2]), 4);
    ggml_metal_encoder_set_buffer(enc, private_path ? private_bindings->weights : ggml_metal_get_buffer_id(op->src[3]), 5);

    const int grid_x = use_glm52_slot_parallel ?
        (ne01 + nr0 - 1)/nr0 :
        (ne01 + nr0*nsg - 1)/(nr0*nsg);
    const int grid_y = ne21;
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, 1, 32, nsg, 1);

    if (ggml_metal_glm_dsa_dispatch_log_enabled()) {
        GGML_LOG_INFO(
            "ggml: glm_dsa_metal_dispatch op=moe_mul_mat_id tensor=%s experts=%s input=%s ids=%s weights=%s kernel=%s private_scratch=%d src0_type=%s n_ff=%d n_embd=%d experts_total=%d used_experts=%d tokens=%d nr0=%d nsg=%d grid_x=%d grid_y=%d grid_z=1 threads_x=32 threads_y=%d fused_nodes=1\n",
            ggml_metal_tensor_name(op),
            ggml_metal_tensor_name(op->src[0]),
            ggml_metal_tensor_name(op->src[1]),
            ggml_metal_tensor_name(op->src[2]),
            ggml_metal_tensor_name(op->src[3]),
            use_glm52_slot_parallel ? "q3_r8_nb8_w0" : "sequential_slots",
            private_path ? 1 : 0,
            ggml_type_name(op->src[0]->type),
            ne00,
            ne01,
            ne02,
            ne20,
            ne21,
            nr0,
            nsg,
            grid_x,
            grid_y,
            nsg);
    }

    return 1;
}

int ggml_metal_op_moe_mul_mat_id(ggml_metal_op_t ctx, int idx) {
    return ggml_metal_encode_moe_mul_mat_id(ctx, ctx->node(idx), nullptr);
}

int ggml_metal_op_top_k(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    const int npr = (ne00 + nth - 1)/nth;

    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += sizeof(int32_t)*ggml_nelements(op->src[0]);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    const int top_k = ne0;

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ std::min(nth, top_k), // for each block, keep just the top_k indices
    };

    if (npr > 1) {
        args.ne0 = (npr - 1)*args.top_k + std::min(ne00 - (npr - 1)*nth, args.top_k);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_top_k_merge(lib, op);

    int len = args.top_k;

    while (len < args.ne0) {
        ggml_metal_op_concurrency_reset(ctx);

        // merges per row
        const int nm = (args.ne0 + 2*len - 1) / (2*len);

        const int nth = std::min(512, std::min(len, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge)));

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ args.ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ nm == 1 ? top_k : args.ne0, // the final merge outputs top_k elements
            /*.len   =*/ len,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

int ggml_metal_op_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_tri args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_tri(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_adamw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_sgd(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_count_equal(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset args = { /*.val =*/ 0 };

        auto pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    {
        ggml_metal_kargs_count_equal args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
        };

        auto pipeline = ggml_metal_library_get_pipeline_count_equal(lib, op);

        const size_t smem = pipeline.smem;

        const int nth = 32*pipeline.nsg;

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}
