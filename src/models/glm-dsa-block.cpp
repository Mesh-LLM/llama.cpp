#include "glm-dsa-block.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace {

struct full_block_span {
    int full_layer;
    int layer_end;
};

[[noreturn]] void throw_bad_group_size(
        int full_layer,
        int actual_size,
        int expected_size) {
    std::ostringstream message;
    message << "GLM_DSA IndexShare group beginning at Full layer " << full_layer
            << " has " << actual_size << " layer(s), expected " << expected_size;
    throw std::runtime_error(message.str());
}

} // namespace

llama_glm_dsa_block_plan llama_glm_dsa_make_block_plan(
        int n_layers,
        int execution_begin,
        int execution_end,
        int repeating_group_begin,
        int expected_repeating_group_size,
        const std::function<bool(int)> & layer_is_full) {
    if (n_layers <= 0 || execution_begin < 0 || execution_end < execution_begin || execution_end > n_layers) {
        throw std::runtime_error("GLM_DSA IndexShare block plan has an invalid layer span");
    }
    if (!layer_is_full) {
        throw std::runtime_error("GLM_DSA IndexShare block plan requires a layer-role resolver");
    }

    std::vector<full_block_span> full_blocks;
    int current_full = -1;
    for (int il = 0; il < n_layers; ++il) {
        if (layer_is_full(il)) {
            if (current_full >= 0) {
                full_blocks.push_back({ current_full, il });
            }
            current_full = il;
        } else if (current_full < 0) {
            std::ostringstream message;
            message << "GLM_DSA IndexShare Shared layer " << il << " has no preceding Full layer";
            throw std::runtime_error(message.str());
        }
    }
    if (current_full < 0) {
        throw std::runtime_error("GLM_DSA IndexShare block plan contains no Full layer");
    }
    full_blocks.push_back({ current_full, n_layers });

    llama_glm_dsa_block_plan plan;
    int ordinal = 0;
    for (const full_block_span & block : full_blocks) {
        const int group_size = block.layer_end - block.full_layer;
        const bool repeating_group = block.full_layer >= repeating_group_begin;
        const bool trailing_group = block.layer_end == n_layers;
        const bool invalid_repeating_group = expected_repeating_group_size > 0 &&
            (group_size > expected_repeating_group_size ||
             (!trailing_group && group_size != expected_repeating_group_size));
        if (repeating_group && invalid_repeating_group) {
            throw_bad_group_size(block.full_layer, group_size, expected_repeating_group_size);
        }

        const int clipped_begin = std::max(block.full_layer, execution_begin);
        const int clipped_end = std::min(block.layer_end, execution_end);
        if (clipped_begin >= clipped_end) {
            ++ordinal;
            continue;
        }

        const bool producer_in_execution = clipped_begin == block.full_layer;
        plan.blocks.push_back({
            ordinal,
            block.full_layer,
            block.full_layer,
            block.layer_end,
            clipped_begin,
            clipped_end,
            repeating_group,
            clipped_begin == block.full_layer && clipped_end == block.layer_end,
            producer_in_execution,
            !producer_in_execution,
        });
        ++ordinal;
    }

    if (execution_begin < execution_end && plan.blocks.empty()) {
        throw std::runtime_error("GLM_DSA IndexShare block plan does not cover the execution span");
    }
    return plan;
}
