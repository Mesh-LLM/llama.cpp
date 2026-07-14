#pragma once

#include <functional>
#include <vector>

struct llama_glm_dsa_block_span {
    int ordinal;
    int full_layer;
    int layer_begin;
    int layer_end;
    int execution_begin;
    int execution_end;
    bool repeating_group;
    bool complete_execution;
    bool producer_in_execution;
    bool needs_input_top_k;
};

struct llama_glm_dsa_block_plan {
    std::vector<llama_glm_dsa_block_span> blocks;
};

llama_glm_dsa_block_plan llama_glm_dsa_make_block_plan(
        int n_layers,
        int execution_begin,
        int execution_end,
        int repeating_group_begin,
        int expected_repeating_group_size,
        const std::function<bool(int)> & layer_is_full);
